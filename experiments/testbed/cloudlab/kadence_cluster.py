"""KADENCE cluster orchestrator (head-node side).

Runs N per-node runtimes and aggregates the cluster result the paper needs:
the ACROSS-NODE distribution of convergence (median/p95/p99 across nodes),
measured message counts (sent, received, ticks, cross-node), and seeds+reps+95% CIs.

Two modes:
  local   spawn N node-runtimes as concurrent processes on this box (logical
          multiplexing; immediate end-to-end validation and the real-hardware
          flat-vs-N stand-in when only one machine is available).
  ssh     fan out to real CloudLab workers listed in --hostfile over the
          internal LAN, run kadence_node.py on each, collect the per-node JSON.

Difficulty ladder via --loss/--jitter (async, lossy network); churn/attack arms
reuse experiments/simulation/churn_evaluation.py and experiments/simulation/desync_adversary.py per node.

    # local, immediate:
    python3 kadence_cluster.py --mode local --nodes 16 --jobs 32 --reps 3
    # CloudLab (from the head node):
    python3 kadence_cluster.py --mode ssh --hostfile workers.txt --jobs 48 \
        --rings 50 --reps 5 --remote-root /proj/PROJECT/kadence
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

def _percentile(data, pct):
    s = sorted(data)
    if not s:
        return 0.0
    k = (len(s) - 1) * pct / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
NODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kadence_node.py")


def ci95(x):
    x = [float(v) for v in x]
    n = len(x)
    if n == 0:
        return (0.0, 0.0, 0.0)
    m = sum(x) / n
    if n < 2:
        return (round(m, 4), round(m, 4), round(m, 4))
    var = sum((v - m) ** 2 for v in x) / (n - 1)
    half = 1.96 * math.sqrt(var) / math.sqrt(n)
    return (round(m, 4), round(m - half, 4), round(m + half, 4))


def run_local(nodes, jobs, rings, seconds, tick, loss, jitter, seed, impl, alpha):
    """Spawn `nodes` node-runtimes concurrently on this box; return per-node JSONs."""
    outdir = os.path.join("/tmp", "kadence_cluster")
    os.makedirs(outdir, exist_ok=True)
    procs, outs = [], []
    for i in range(nodes):
        nid = f"n{i:04d}"
        out = os.path.join(outdir, f"{nid}.json")
        outs.append(out)
        base_port = 50000 + i * (jobs * rings + 4)
        cmd = [sys.executable, NODE, "--node-id", nid, "--jobs", str(jobs),
               "--rings", str(rings), "--seconds", str(seconds), "--tick", str(tick),
               "--loss", str(loss), "--jitter", str(jitter),
               "--impl", impl, "--alpha", str(alpha),
               "--base-port", str(base_port), "--seed", str(seed + i), "--out", out]
        procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL))
    for p in procs:
        p.wait()
    res = []
    for o in outs:
        if os.path.exists(o):
            res.append(json.load(open(o)))
    return res


def run_ssh(hostfile, jobs, rings, seconds, tick, loss, jitter, seed,
            remote_root, key, user, impl, alpha, c_calls):
    hosts = [h.strip() for h in open(hostfile) if h.strip() and not h.startswith("#")]
    ssh_base = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15"]
    if key:
        ssh_base += ["-i", key]
    node_remote = f"{remote_root}/experiments/testbed/cloudlab/kadence_node.py"
    procs = []
    for i, h in enumerate(hosts):
        tgt = f"{user}@{h}" if user else h
        remote_out = f"/tmp/kadence/{h.replace('.', '_')}.json"
        base_port = 49000 + i * 2  # distinct across hosts (irrelevant; per-host localhost)
        cmd = (f"python3 {node_remote} --node-id {h} --jobs {jobs} --rings {rings} "
               f"--seconds {seconds} --tick {tick} --loss {loss} --jitter {jitter} "
               f"--impl {impl} --alpha {alpha} --c-calls {c_calls} "
               f"--seed {seed + i} --out {remote_out} && cat {remote_out}")
        procs.append((h, subprocess.Popen(ssh_base + [tgt, cmd],
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)))
    res = []
    for h, p in procs:
        out, err = p.communicate()
        try:
            res.append(json.loads(out.strip().splitlines()[-1]) if out.strip()
                       else json.loads(out))
        except Exception:
            # node printed a human line before JSON; try to find the JSON blob
            try:
                res.append(json.loads(out[out.index("{"):]))
            except Exception:
                print(f"[warn] no JSON from {h}: {err[:200]}", flush=True)
    return res


def aggregate(per_node, meta):
    """Cross-node aggregate. Every count here is summed from agent counters."""
    if not per_node:
        return {"nodes_reporting": 0, "error": "no nodes reported", **meta}
    if per_node[0].get("impl") == "c":
        arms = {}
        for arm in per_node[0]["c_admission"]["arms"]:
            for key in ("p50_ns", "p99_ns", "p999_ns", "max_ns"):
                v = [n["c_admission"]["arms"][arm][key] for n in per_node]
                arms.setdefault(arm, {})[key] = {
                    "median_across_nodes": _percentile(v, 50),
                    "min": min(v), "max": max(v)}
        return {"nodes_reporting": len(per_node), "impl": "c", "arms": arms,
                "binary_size_bytes": per_node[0]["c_admission"]["binary_size_bytes"],
                "interpreter_needed": False, **meta}
    S = [n["node_summary"] for n in per_node]
    med = [s["median_final_pct_of_fair"] for s in S]
    mx = [s["max_final_pct_of_fair"] for s in S]
    classes = {}
    for s in S:
        for c, k in s["classes"].items():
            classes[c] = classes.get(c, 0) + k
    n_rings = sum(sum(s["classes"].values()) for s in S)
    sent = sum(s["total_messages_sent"] for s in S)
    old = sum(s["total_messages_old_estimate"] for s in S)
    return {
        "nodes_reporting": len(per_node),
        "impl": per_node[0].get("impl"),
        "rings_total": n_rings,
        "across_node_final_pct_of_fair": {
            "median": round(_percentile(med, 50), 3),
            "p95": round(_percentile(med, 95), 3),
            "p99": round(_percentile(med, 99), 3),
            "worst_node_max": round(max(mx), 3)},
        "fraction_nodes_converged_loose_0p1_target":
            round(sum(s["all_converged_loose_0p1_target"] for s in S) / len(S), 4),
        "fraction_nodes_converged_strict_1e6":
            round(sum(s["all_converged_strict_1e6"] for s in S) / len(S), 4),
        "classification_counts": classes,
        "measured_datagrams_sent": sent,
        "measured_datagrams_received": sum(s["total_messages_received"] for s in S),
        "measured_ticks_executed": sum(s["total_ticks_executed"] for s in S),
        "measured_tick_deficit": sum(s["total_tick_deficit"] for s in S),
        "old_estimate_2n_seconds_over_tick": old,
        "old_estimate_minus_measured_sent": old - sent,
        "cross_node_messages_measured": sum(s["cross_node_messages"] for s in S),
        **meta,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local", "ssh"], default="local")
    ap.add_argument("--nodes", type=int, default=16, help="local mode node count")
    ap.add_argument("--hostfile", help="ssh mode: one worker host per line")
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--rings", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--tick", type=float, default=0.02)
    ap.add_argument("--loss", type=float, default=0.0)
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--impl", choices=["python", "rust", "c"], default="python")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="update rate, paper convention alpha/2 * correction")
    ap.add_argument("--c-calls", type=int, default=200000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--remote-root", default="/proj/PROJECT/kadence")
    ap.add_argument("--key")
    ap.add_argument("--user")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "kadence"))
    a = ap.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    reps_summ = []
    for rep in range(a.reps):
        t0 = time.time()
        if a.mode == "local":
            per_node = run_local(a.nodes, a.jobs, a.rings, a.seconds, a.tick,
                                 a.loss, a.jitter, a.seed + rep * 1000,
                                 a.impl, a.alpha)
        else:
            per_node = run_ssh(a.hostfile, a.jobs, a.rings, a.seconds, a.tick,
                               a.loss, a.jitter, a.seed + rep * 1000,
                               a.remote_root, a.key, a.user, a.impl, a.alpha, a.c_calls)
        agg = aggregate(per_node, {"rep": rep, "wall_s": round(time.time() - t0, 2)})
        reps_summ.append(agg)
        d = agg.get("across_node_final_pct_of_fair")
        if d:
            print(f"rep {rep}: nodes={agg['nodes_reporting']} "
                  f"across-node median={d['median']}% p95={d['p95']}% worst={d['worst_node_max']}% "
                  f"loose={agg['fraction_nodes_converged_loose_0p1_target']} "
                  f"strict={agg['fraction_nodes_converged_strict_1e6']} "
                  f"classes={agg['classification_counts']}", flush=True)
        elif agg.get("impl") == "c":
            print(f"rep {rep}: c arm nodes={agg['nodes_reporting']}", flush=True)
        else:
            print(f"rep {rep}: no nodes reported", flush=True)

    meds = [r["across_node_final_pct_of_fair"]["median"]
            for r in reps_summ if r.get("across_node_final_pct_of_fair")]
    out = {
        "provenance": {"kind": f"KADENCE cluster ({a.mode})", "impl": a.impl,
                       "kernel": "core.neighbor_gossip via kadence_node (real UDP)"
                                 if a.impl == "python" else
                                 "kadence-rs (Rust) / native C reservation kernel",
                       "mode": a.mode, "nodes": a.nodes if a.mode == "local" else "hostfile",
                       "jobs_per_ring": a.jobs, "rings_per_node": a.rings,
                       "loss": a.loss, "jitter": a.jitter, "alpha": a.alpha,
                       "reps": a.reps, "seconds": a.seconds, "tick": a.tick,
                       "global_clock": False,
                       "criteria": {"loose": "final_gap_error < 0.1*(2*pi/n)",
                                    "strict": "final_gap_error < 1e-6 rad"},
                       "scope_note": "rings are per-node and isolated by construction; "
                                     "N nodes = N samples of a single-node experiment"},
        "reps": reps_summ,
        "across_node_median_pct_of_fair_ci95": ci95(meds),
    }
    os.makedirs(a.out, exist_ok=True)
    prefix = "cloudlab" if a.mode == "ssh" else "cluster_local"
    tag = f"_{a.tag}" if a.tag else f"_{a.impl}"
    path = os.path.join(a.out, f"{prefix}{tag}_{ts}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
