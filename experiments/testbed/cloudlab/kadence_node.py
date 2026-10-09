"""KADENCE per-node runtime: the unit that runs on each CloudLab worker.

A node hosts the jobs resident on it as a ring of real UDP phase-oscillator
agents (one agent per job), using the committed kernel
(core.neighbor_gossip.local_correction + limit_local_displacement = the
rate-bound guard). Coordination is entirely within this node: agents bind
127.0.0.1 and talk only to their two ring neighbours. K rings on one node
(``--rings K``) are independent, so a run with N nodes is N samples of a
single-node experiment, not a cross-node scaling result.

Three arms (``--impl``):
  python  counted copy of experiments/simulation/desync_distributed.agent (kadence_agent_counted)
  rust    the fixed kadence-rs binary, same initial phases, same parameters
  c       admission-path cost of the src/native/ C reservation kernel (no network)

Everything the JSON reports as a message count is counted by the agents, not
estimated from seconds / tick:
  datagrams_sent, datagrams_received, ticks_executed, and the cross-node
  counters (datagrams whose source or destination was not loopback).
Two convergence criteria are recorded side by side and never merged:
  loose  final_gap_error < 0.1 * (2*pi/n)   (what this runtime used before)
  strict final_gap_error < 1e-6 rad          (what the simulation uses)

    python3 kadence_node.py --node-id w07 --jobs 32 --rings 1 --seconds 8 \
        --loss 0.1 --impl rust --out /tmp/kadence/w07.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from multiprocessing import Process

import random as _random


def _percentile(data, pct):
    s = sorted(data)
    if not s:
        return 0.0
    k = (len(s) - 1) * pct / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


HERE = os.path.dirname(os.path.abspath(__file__))
# repo root = three levels up from experiments/testbed/cloudlab/
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
from experiments.simulation.desync_distributed import _gap_error             # committed helper
from kadence_agent_counted import counted_agent           # counting agent

TWO_PI = 2 * math.pi
CRIT_LOOSE_FRAC = 0.1
CRIT_STRICT_ABS = 1e-6
DEFAULT_RUST = os.path.join(HERE, "bin", "kadence")
DEFAULT_C = os.path.join(HERE, "bin", "c_admit_bench")


# ---------------------------------------------------------------- kernel UDP
def _read_snmp_udp():
    """System-wide kernel UDP counters (InDatagrams, OutDatagrams, RcvbufErrors...)."""
    try:
        lines = open("/proc/net/snmp").read().splitlines()
        hdr = [l.split()[1:] for l in lines if l.startswith("Udp:")]
        return dict(zip(hdr[0], map(int, hdr[1])))
    except Exception:
        return {}


def _read_nic_tx():
    """Non-loopback transmitted packets, all protocols (informational only)."""
    tot = 0
    try:
        for l in open("/proc/net/dev").read().splitlines()[2:]:
            name, rest = l.split(":", 1)
            if name.strip() != "lo":
                tot += int(rest.split()[9])
    except Exception:
        return None
    return tot


# ------------------------------------------------------------------ analysis
def _order_preserved(final):
    p = [x % TWO_PI for x in final]
    n = len(p)
    return sum(1 for i in range(n) if p[(i + 1) % n] < p[i]) <= 1


def _err_trace(traces, target):
    """Ring gap error at each async round k (each agent's k-th update)."""
    kmin = min(len(t) for t in traces)
    return [_gap_error([t[k] for t in traces], target) for k in range(kmin)]


def analyze(*, n, seconds, tick, jitter, loss, final, ticks_pj, recv_pj,
            err_trace, target):
    """Common metrics + classification for one ring, independent of the arm."""
    final_err = _gap_error(final, target)
    loose = bool(final_err < CRIT_LOOSE_FRAC * target)
    strict = bool(final_err < CRIT_STRICT_ABS)
    # Ticks the budget should have allowed: first tick at t=0, then one per
    # (tick + mean jitter) while t < seconds.
    mean_period = tick + jitter / 2.0
    expected_pj = int(math.ceil(seconds / mean_period))
    nominal_pj = int(math.ceil(seconds / tick))
    executed = sum(ticks_pj)
    deficit = expected_pj * n - executed
    r = {
        "final_gap_error": float(final_err),
        "final_pct_of_fair": round(100 * float(final_err) / target, 3),
        "converged_loose_0p1_target": loose,
        "converged_strict_1e6": strict,
        "order_preserved": _order_preserved(final),
        "ticks_executed": executed,
        "ticks_expected_per_job": expected_pj,
        "ticks_nominal_per_job_no_jitter": nominal_pj,
        "ticks_min_per_job": min(ticks_pj),
        "ticks_max_per_job": max(ticks_pj),
        "tick_deficit_total": deficit,
        "tick_deficit_frac": round(deficit / (expected_pj * n), 4),
        "recv_min_per_job": min(recv_pj),
    }
    first_loose = next((k for k, e in enumerate(err_trace)
                        if e < CRIT_LOOSE_FRAC * target), None)
    first_strict = next((k for k, e in enumerate(err_trace)
                         if e < CRIT_STRICT_ABS), None)
    r["tick_of_first_loose"] = first_loose
    r["tick_of_first_strict"] = first_strict
    r["classification"] = classify(r, err_trace, ticks_pj, recv_pj, loss, expected_pj)
    r["err_trace_stride10"] = [round(e, 6) for e in err_trace[::10]]
    return r


def classify(r, err_trace, ticks_pj, recv_pj, loss, expected_pj):
    """Why a ring did not meet the loose criterion. Precedence, most to least
    specific:

    converged        loose criterion met
    order-broken     final cyclic order of jobs differs from the initial order
                     (the safety clip should make this impossible)
    starved          some job executed < 50% of the median job's ticks, or
                     received < 25% of the datagrams a lossless neighbour
                     would have delivered (CPU or socket starvation)
    budget-cutoff    error still falling at the end (last-20% error < 0.9 x
                     error at 80% of the run): the ring was cut off, not stuck.
                     ``budget_cutoff_cause`` says whether executed ticks fell
                     short of the budget (tick_deficit) or the budget itself is
                     too short for this alpha (horizon_too_short)
    stalled          error flat or rising at the end with a full tick budget
                     and intact order: a real convergence failure
    """
    if r["converged_loose_0p1_target"]:
        return {"class": "converged"}
    if not r["order_preserved"]:
        return {"class": "order-broken"}
    med = _percentile(ticks_pj, 50)
    exp_recv = 2.0 * (1.0 - loss) * med
    if r["ticks_min_per_job"] < 0.5 * med or (exp_recv > 20 and r["recv_min_per_job"] < 0.25 * exp_recv):
        return {"class": "starved"}
    k = len(err_trace)
    if k >= 10:
        e80, eend = err_trace[int(0.8 * k)], err_trace[-1]
        decaying = eend < 0.9 * e80
    else:
        decaying = False
    if decaying:
        cause = "tick_deficit" if r["tick_deficit_frac"] > 0.05 else "horizon_too_short"
        return {"class": "budget-cutoff", "budget_cutoff_cause": cause}
    return {"class": "stalled"}


# -------------------------------------------------------------------- rings
def _phases(n, seed):
    rng = _random.Random(seed)
    return sorted(rng.uniform(0, TWO_PI) for _ in range(n))


def run_ring_python(n, seconds, tick, base_port, seed, loss, jitter, alpha, outdir):
    os.makedirs(outdir, exist_ok=True)
    for f in os.listdir(outdir):
        os.remove(os.path.join(outdir, f))
    target = TWO_PI / n
    phase0 = _phases(n, seed)
    init_err = _gap_error(phase0, target)
    procs = [Process(target=counted_agent,
                     args=(i, n, base_port, float(phase0[i]), target, seconds,
                           tick, outdir, loss, jitter, alpha, seed))
             for i in range(n)]
    t0 = time.time()
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    wall = time.time() - t0
    ag = [json.load(open(os.path.join(outdir, f"{i}.json"))) for i in range(n)]
    return init_err, wall, ag


def run_ring_rust(n, seconds, tick, base_port, seed, loss, jitter, alpha, outdir,
                  rust_bin):
    os.makedirs(outdir, exist_ok=True)
    target = TWO_PI / n
    phase0 = _phases(n, seed)
    pf = os.path.join(outdir, "phases.txt")
    with open(pf, "w") as f:
        f.write(" ".join(repr(float(p)) for p in phase0))
    out = os.path.join(outdir, "rust.json")
    cmd = [rust_bin, "--jobs", str(n), "--seconds", str(seconds),
           "--tick-ms", str(int(round(tick * 1000))),
           "--jitter-ms", str(jitter * 1000.0), "--loss", str(loss),
           "--alpha", str(alpha), "--seed", str(seed), "--reps", "1",
           "--base-port", str(base_port), "--phases-file", pf,
           "--trace-every", "1", "--out", out]
    t0 = time.time()
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    wall = time.time() - t0
    return _gap_error(phase0, target), wall, json.load(open(out))["reps"][0]


def run_ring(impl, n, seconds, tick, base_port, seed, loss, jitter, alpha,
             outdir, rust_bin):
    """One ring of n UDP job-agents on this node; returns measured metrics."""
    target = TWO_PI / n
    snmp0, nic0 = _read_snmp_udp(), _read_nic_tx()
    if impl == "python":
        init_err, wall, ag = run_ring_python(n, seconds, tick, base_port, seed,
                                             loss, jitter, alpha, outdir)
        final = [a["phase"] for a in ag]
        ticks_pj = [a["ticks"] for a in ag]
        recv_pj = [a["received"] for a in ag]
        trace = _err_trace([a["trace"] for a in ag], target)
        counts = {
            "datagrams_sent": sum(a["sent"] for a in ag),
            "datagrams_received": sum(a["received"] for a in ag),
            "datagrams_malformed": sum(a["malformed"] for a in ag),
            "sends_dropped_by_loss_injection": sum(a["loss_dropped"] for a in ag),
            "send_errors": sum(a["send_errors"] for a in ag),
            "unread_at_exit": sum(a["residual_at_exit"] for a in ag),
            "cross_node_recv_nonloopback_src": sum(a["recv_nonloopback_src"] for a in ag),
            "cross_node_sent_nonloopback_dst": sum(a["sent_nonloopback_dst"] for a in ag),
        }
    else:
        init_err, wall, rr = run_ring_rust(n, seconds, tick, base_port, seed,
                                           loss, jitter, alpha, outdir, rust_bin)
        final = rr["final_phases"]
        ticks_pj = rr["ticks_per_job"]
        recv_pj = rr["recv_per_job"]
        trace = [e for _, e in rr["err_trace"]]
        counts = {k: rr[k] for k in (
            "datagrams_sent", "datagrams_received", "datagrams_malformed",
            "sends_dropped_by_loss_injection", "send_errors", "unread_at_exit",
            "cross_node_recv_nonloopback_src", "cross_node_sent_nonloopback_dst")}
    snmp1, nic1 = _read_snmp_udp(), _read_nic_tx()
    a = analyze(n=n, seconds=seconds, tick=tick, jitter=jitter, loss=loss,
                final=final, ticks_pj=ticks_pj, recv_pj=recv_pj,
                err_trace=trace, target=target)
    d = lambda k: (snmp1.get(k, 0) - snmp0.get(k, 0)) if snmp0 else None
    res = {
        "impl": impl, "jobs": n, "seconds": seconds, "tick": tick, "loss": loss,
        "jitter": jitter, "alpha": alpha, "seed": seed,
        "initial_phases": _phases(n, seed),
        "final_phases": final,
        "init_gap_error": round(float(init_err), 6),
        "wall_s": round(wall, 3),
        "messages_per_job_round_nominal": 2,
        "measured": counts,
        "cross_node_messages": counts["cross_node_recv_nonloopback_src"]
                               + counts["cross_node_sent_nonloopback_dst"],
        "kernel_udp_delta_system_wide": {
            "InDatagrams": d("InDatagrams"), "OutDatagrams": d("OutDatagrams"),
            "RcvbufErrors": d("RcvbufErrors"), "InErrors": d("InErrors"),
            "note": "all UDP on the box during the ring, loopback included; "
                    "includes any unrelated UDP such as DNS"},
        "nic_tx_packets_delta_all_traffic": (nic1 - nic0) if nic0 is not None and nic1 is not None else None,
        "estimate_that_was_reported_before": 2 * n * max(1, int(seconds / tick)),
    }
    res["estimate_error_vs_measured_sent"] = (
        res["estimate_that_was_reported_before"] - counts["datagrams_sent"])
    res.update(a)
    return res


# ---------------------------------------------------------------- C arm
def run_c_arm(c_bin, calls, days, cpu, node_id):
    """Admission-path cost of the native C reservation kernel on this node."""
    args = [c_bin, "--calls", str(calls), "--days", str(days)]
    if cpu is not None:
        args += ["--cpu", str(cpu)]
    out = subprocess.run(args, check=True, capture_output=True, text=True).stdout
    res = json.loads(out)
    res["binary_size_bytes"] = os.path.getsize(c_bin)
    try:
        res["ldd"] = subprocess.run(["ldd", c_bin], capture_output=True,
                                    text=True).stdout.strip() or \
            subprocess.run(["ldd", c_bin], capture_output=True, text=True).stderr.strip()
    except Exception as e:
        res["ldd"] = f"unavailable: {e}"
    res["interpreter_needed"] = False
    res["loadavg"] = os.getloadavg()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node-id", default="node0")
    ap.add_argument("--impl", choices=["python", "rust", "c"], default="python")
    ap.add_argument("--jobs", type=int, default=32, help="resident jobs per ring")
    ap.add_argument("--rings", type=int, default=1, help="rings on this node, run one after another")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--tick", type=float, default=0.02)
    ap.add_argument("--loss", type=float, default=0.0)
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="update rate, paper convention: displacement = alpha/2 * corr")
    ap.add_argument("--base-port", type=int, default=49000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rust-bin", default=DEFAULT_RUST)
    ap.add_argument("--c-bin", default=DEFAULT_C)
    ap.add_argument("--c-calls", type=int, default=200000)
    ap.add_argument("--c-days", type=int, default=3)
    ap.add_argument("--cpu", type=int, default=None)
    ap.add_argument("--out", default="/tmp/kadence/node.json")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.impl == "c":
        c = run_c_arm(a.c_bin, a.c_calls, a.c_days, a.cpu, a.node_id)
        out = {"node_id": a.node_id, "impl": "c", "c_admission": c,
               "node_summary": {"impl": "c", "cross_node_messages": 0,
                                "note": "no network in this arm"}}
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"[{a.node_id}] c admission p50={c['arms']['reserve_harmonic_k4']['p50_ns']}ns "
              f"-> {a.out}", flush=True)
        return

    rings = []
    for k in range(a.rings):
        base_port = a.base_port + k * (a.jobs + 2)
        outdir = os.path.join("/tmp", "kadence_dist", f"{a.node_id}_{a.impl}_r{k}")
        rings.append(run_ring(a.impl, a.jobs, a.seconds, a.tick, base_port,
                              a.seed + k, a.loss, a.jitter, a.alpha, outdir,
                              a.rust_bin))
        shutil.rmtree(outdir, ignore_errors=True)

    errs = [r["final_pct_of_fair"] for r in rings]
    classes = [r["classification"]["class"] for r in rings]
    out = {
        "node_id": a.node_id,
        "impl": a.impl,
        "rings": rings,
        "n_rings": a.rings,
        "jobs_per_ring": a.jobs,
        "node_summary": {
            "max_final_pct_of_fair": round(max(errs), 3),
            "median_final_pct_of_fair": round(_percentile(errs, 50), 3),
            "all_converged_loose_0p1_target": all(r["converged_loose_0p1_target"] for r in rings),
            "all_converged_strict_1e6": all(r["converged_strict_1e6"] for r in rings),
            "classes": {c: classes.count(c) for c in sorted(set(classes))},
            "total_messages_sent": sum(r["measured"]["datagrams_sent"] for r in rings),
            "total_messages_received": sum(r["measured"]["datagrams_received"] for r in rings),
            "total_ticks_executed": sum(r["ticks_executed"] for r in rings),
            "total_tick_deficit": sum(r["tick_deficit_total"] for r in rings),
            "total_messages_old_estimate": sum(r["estimate_that_was_reported_before"] for r in rings),
            "cross_node_messages": sum(r["cross_node_messages"] for r in rings),
        },
        "criteria": {
            "loose": "final_gap_error < 0.1 * (2*pi/n); the pre-existing runtime criterion",
            "strict": "final_gap_error < 1e-6 radians; the simulation criterion"},
        "provenance": {
            "kernel": "core.neighbor_gossip" if a.impl == "python" else "kadence-rs (Rust port of the same kernel)",
            "transport": "UDP loopback (jobs are agents on this node)",
            "global_clock": False, "alpha": a.alpha,
            "alpha_convention": "displacement = alpha/2 * correction"},
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    ns = out["node_summary"]
    print(f"[{a.node_id}] {a.impl} rings={a.rings} jobs/ring={a.jobs} "
          f"median={ns['median_final_pct_of_fair']}% loose={ns['all_converged_loose_0p1_target']} "
          f"strict={ns['all_converged_strict_1e6']} sent={ns['total_messages_sent']} "
          f"recv={ns['total_messages_received']} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
