"""Stage 1 head-to-head: the oscillator 'brain' vs linear baselines on a
mixed workload, apples-to-apples.

This is the Stage 1 result run in the
EXISTING phase-oscillator simulator so it produces numbers NOW. The full
k3s cluster run (real containerd runtime, kube-scheduler, ML-weighted
CFS) is deliberately deferred; the cluster protocol lives in
docs/STAGE1_HEAD_TO_HEAD.md. Everything here is sim, not hardware —
labelled as such everywhere.

Same primitive as experiments/simulation/simulate.py: every job is a multi-resource phase
oscillator ('bubble') sharing one node's cpu+io. Only the *arrangement
of phases in time* differs between policies; total work is identical, so
any win is purely the scheduling model, not doing less.

What is new here vs simulate.py:

  1. A MIX of five NAMED workload types (web, api, batch, analytics, ml)
     with distinct demand waveforms (duty cycle, peak height, dominant
     resource), so metrics can be broken down per type.

  2. A fourth policy, `ml-cfs-static`: a real static-weight baseline
     standing in for ML-weighted CFS / a request-based kube-scheduler.
     It reads each job's MEAN demand once, sets a proportional weight,
     and lays jobs out in weighted (load-proportional) order with NO
     feedback and NO resource-phase awareness. This is what a static
     right-sizing scheduler does: it spaces heavy jobs further apart than
     light ones, but it is still resource-blind (cannot interleave a
     cpu-peak into an io-valley) and never re-plans.

  3. Per-policy AND per-workload-type metrics:
       - peak contention per resource (packing / peak-pressure)
       - packing density = jobs fit under a fixed SLO ceiling
       - SLO-violation rate (fraction of period over capacity), per type
       - JCT proxy (contention-derived, see jct_proxy docstring), per type

Scope: JCT here is a *contention-derived proxy*, not a real runtime
measurement — there is no discrete-event job scheduler in this model.
It is defined deterministically from the demand traces (see below) and
is only meaningful RELATIVELY, policy-vs-policy on identical work. The
real JCT number comes from the deferred k3s run.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from kadence.oscillator import (Job, desync_arrange, peak_contention,  # noqa: E402
                             dominant_resource, TWO_PI)

RESOURCES = ["cpu", "io"]
PERIOD = 100.0
CAPACITY = 1.0

# Five named workload types with distinct demand waveforms. Ranges are
# hand-set to be plausibly heterogeneous, NOT trace-derived — this is a
# synthetic mixed workload for the sim. (cpu/io duty = fraction of period
# the type needs the resource; height = share of the resource at peak.)
WORKLOAD_TYPES = {
    # bursty, short, cpu-leaning request/response
    "web":       {"cpu_duty": (0.06, 0.14), "io_duty": (0.03, 0.07),
                  "cpu_h": (0.20, 0.35),    "io_h": (0.05, 0.12)},
    # spikier than web, cpu-dominant, very short pulses
    "api":       {"cpu_duty": (0.04, 0.10), "io_duty": (0.02, 0.05),
                  "cpu_h": (0.25, 0.45),    "io_h": (0.03, 0.08)},
    # long, wide, io-dominant throughput jobs
    "batch":     {"cpu_duty": (0.10, 0.18), "io_duty": (0.25, 0.40),
                  "cpu_h": (0.08, 0.18),    "io_h": (0.25, 0.45)},
    # io-leaning but with sustained cpu (scan + aggregate)
    "analytics": {"cpu_duty": (0.15, 0.25), "io_duty": (0.18, 0.30),
                  "cpu_h": (0.15, 0.28),    "io_h": (0.18, 0.30)},
    # cpu-dominant, wide sustained pulses (training steps)
    "ml":        {"cpu_duty": (0.25, 0.40), "io_duty": (0.06, 0.12),
                  "cpu_h": (0.30, 0.50),    "io_h": (0.05, 0.12)},
}
TYPE_NAMES = list(WORKLOAD_TYPES.keys())


def make_mixed_jobs(n: int, seed: int) -> list[Job]:
    """n jobs drawn uniformly across the five named types. The type is
    recorded on the job id (``<type>#<i>``) and as an attribute so
    metrics can be grouped per type."""
    rng = random.Random(seed)
    jobs = []
    for i in range(n):
        t = TYPE_NAMES[i % len(TYPE_NAMES)]  # even mix across types
        spec = WORKLOAD_TYPES[t]
        duty = {"cpu": rng.uniform(*spec["cpu_duty"]),
                "io":  rng.uniform(*spec["io_duty"])}
        height = {"cpu": rng.uniform(*spec["cpu_h"]),
                  "io":  rng.uniform(*spec["io_h"])}
        j = Job(jid=f"{t}#{i}", duty=duty, height=height,
                omega=rng.uniform(0.9, 1.1))
        j.wtype = t  # type: ignore[attr-defined]
        jobs.append(j)
    return jobs


# ---------------------------------------------------------------------------
# Policies. Each sets jobs[*].phase in place. Total work is identical; only
# the time-arrangement differs.
# ---------------------------------------------------------------------------

def set_linear_fixed(jobs):
    """Naive fixed schedule: everyone at phase 0 (thundering herd floor)."""
    for j in jobs:
        j.phase = {r: 0.0 for r in RESOURCES}


def set_linear_rr(jobs):
    """Round-robin / fixed-share: equal phase slots by index, resource-
    blind. The classic time-slice scheduler."""
    n = len(jobs)
    for idx, j in enumerate(jobs):
        j.phase = {r: (TWO_PI * idx / max(1, n)) for r in RESOURCES}


def _mean_load(job: Job, r: str) -> float:
    """Mean demand of a job for r over the period = height * duty (the
    time-average of the raised-cosine pulse ~ height*duty). This is the
    single scalar an ML/right-sizing scheduler would read from history."""
    return job.height.get(r, 0.0) * job.duty.get(r, 0.0)


def set_ml_cfs_static(jobs):
    """Static-weight baseline standing in for ML-weighted CFS / a
    request-based scheduler.

    Real, implemented policy (no feedback, no resource-phase awareness):
      1. Read each job's MEAN load once (weight = dominant-resource mean).
        This is the 'ML right-sizing' step: set the weight from history.
      2. Lay jobs out in phase in load-proportional slot WIDTHS: heavier
        jobs get a wider slot (are spaced further from the next job),
        lighter jobs are packed closer. This is exactly what weighted
        fair sharing (CFS with per-cgroup weights) does in time — big
        jobs get proportionally more of the cycle.
      3. Never re-plan; identical phase on cpu and io (resource-blind).

    It is strictly smarter than plain RR (it accounts for job size) but
    still cannot interleave a cpu-peak into an io-valley, because the same
    phase is used for both resources and the plan is set once."""
    weights = [max(1e-6, _mean_load(j, dominant_resource(j))) for j in jobs]
    total = sum(weights)
    # cumulative load-proportional phase: heavier => bigger step to next
    cum = 0.0
    for j, w in zip(jobs, weights):
        centre = (cum + 0.5 * w) / total
        j.phase = {r: (centre * TWO_PI) for r in RESOURCES}
        cum += w


def set_phase_coupled(jobs):
    """The oscillator brain: DESYNC among per-resource competitors, then
    greedy valley-fill for complements (core.oscillator.desync_arrange).
    Resource-phase-aware: a cpu-peak is placed in an io-valley."""
    rng = random.Random(1)
    for j in jobs:
        j.phase = {r: rng.uniform(0, TWO_PI) for r in RESOURCES}
    desync_arrange(jobs, RESOURCES)


POLICIES = {
    "linear-fixed":  set_linear_fixed,
    "linear-rr":     set_linear_rr,
    "ml-cfs-static": set_ml_cfs_static,
    "phase-coupled": set_phase_coupled,
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _total_load(jobs, r, t):
    return sum(j.demand(r, t, PERIOD) for j in jobs)


def slo_violation_rate(jobs, samples=300):
    """Fraction of the period each resource is over CAPACITY (cluster-
    wide). Lower = fewer SLO violations at identical total work."""
    viol = {}
    for r in RESOURCES:
        over = sum(1 for k in range(samples)
                   if _total_load(jobs, r, PERIOD * k / samples) > CAPACITY)
        viol[r] = over / samples
    return viol


def jct_proxy(jobs, samples=300):
    """Contention-derived JCT proxy, per job (NOT a real runtime number).

    Model: when total demand on a resource exceeds CAPACITY at instant t,
    the node cannot serve all requested work; the excess is deferred, so
    every job wanting that resource at t is throttled by the overload
    factor total/CAPACITY. A job's completion time stretches in
    proportion to the overload it sits through:

        stretch_i = 1 + (1/W_i) * integral over t of
                        demand_i(t) * max(0, total(t)/CAP - 1) dt

    where W_i is the job's total requested work. stretch_i = 1.0 means the
    job never sat in contention (ideal JCT); larger means longer JCT. It
    is deterministic from the traces and only comparable relatively across
    policies on the SAME job set. It is a proxy for the real k3s JCT."""
    dt = PERIOD / samples
    # precompute overload factor per resource per sample
    overload = {r: [] for r in RESOURCES}
    for k in range(samples):
        t = PERIOD * k / samples
        for r in RESOURCES:
            tot = _total_load(jobs, r, t)
            overload[r].append(max(0.0, tot / CAPACITY - 1.0))
    stretch = {}
    for j in jobs:
        work = 0.0
        excess = 0.0
        for k in range(samples):
            t = PERIOD * k / samples
            for r in RESOURCES:
                d = j.demand(r, t, PERIOD)
                if d <= 0:
                    continue
                work += d * dt
                excess += d * overload[r][k] * dt
        stretch[j.jid] = 1.0 + (excess / work if work > 0 else 0.0)
    return stretch


def evaluate(jobs):
    """Full metric bundle for one placed job set."""
    peaks = peak_contention(jobs, RESOURCES, PERIOD)
    viol = slo_violation_rate(jobs)
    stretch = jct_proxy(jobs)
    # group per workload type
    by_type = {t: {"count": 0, "viol_share": 0.0, "jct": []} for t in TYPE_NAMES}
    # per-type SLO violation share: fraction of a type's own demand-mass
    # that lands in over-capacity instants (a per-type violation exposure)
    dt_s = 300
    type_demand = {t: 0.0 for t in TYPE_NAMES}
    type_over = {t: 0.0 for t in TYPE_NAMES}
    for k in range(dt_s):
        t = PERIOD * k / dt_s
        totals = {r: _total_load(jobs, r, t) for r in RESOURCES}
        for j in jobs:
            wt = getattr(j, "wtype", "?")
            for r in RESOURCES:
                d = j.demand(r, t, PERIOD)
                type_demand[wt] += d
                if totals[r] > CAPACITY:
                    type_over[wt] += d
    for j in jobs:
        wt = getattr(j, "wtype", "?")
        by_type[wt]["count"] += 1
        by_type[wt]["jct"].append(stretch[j.jid])
    for t in TYPE_NAMES:
        dem = type_demand[t]
        by_type[t]["viol_share"] = round(type_over[t] / dem, 4) if dem > 0 else 0.0
        jl = by_type[t]["jct"]
        by_type[t]["jct_mean"] = round(sum(jl) / len(jl), 4) if jl else 0.0
        del by_type[t]["jct"]
    return {
        "peak": {r: round(peaks[r], 4) for r in RESOURCES},
        "slo_violation_frac": {r: round(viol[r], 4) for r in RESOURCES},
        "jct_stretch_mean": round(sum(stretch.values()) / len(stretch), 4),
        "by_type": by_type,
    }


def packing_capacity(seed, place, ceil=CAPACITY, nmax=80):
    """How many mixed jobs fit before peak on either resource breaches the
    ceiling (packing density). Grows the mix until it no longer fits."""
    n = 5  # start at one of each type
    last_ok = 0
    while n <= nmax:
        jobs = make_mixed_jobs(n, seed)
        place(jobs)
        if max(peak_contention(jobs, RESOURCES, PERIOD).values()) <= ceil:
            last_ok = n
            n += 5
        else:
            break
    return last_ok


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(n_jobs=35, seeds=24):
    """Head-to-head. n_jobs is a fixed mix (7 per type at n=35), chosen to
    LOAD the node past the point where RR alone keeps every resource under
    the SLO ceiling, so SLO-violation and JCT differences between policies
    are actually exercised (at a light n every policy trivially fits and
    all metrics collapse to zero). Metrics averaged over `seeds`
    independent draws of that mix. The packing-density metric separately
    grows n to find each policy's ceiling."""
    out = {
        "note": "SIMULATION ONLY, not a cluster measurement. JCT is a "
                "contention-derived proxy. See docs/STAGE1_HEAD_TO_HEAD.md.",
        "n_jobs": n_jobs, "seeds": seeds, "capacity": CAPACITY,
        "period": PERIOD, "resources": RESOURCES,
        "workload_types": TYPE_NAMES,
        "policies": list(POLICIES),
        "generated_at": time.time(),
        "aggregate": {}, "by_type": {}, "packing": {},
    }

    acc = {p: {"peak_cpu": [], "peak_io": [], "viol_cpu": [], "viol_io": [],
               "jct": []} for p in POLICIES}
    type_acc = {p: {t: {"viol_share": [], "jct_mean": []} for t in TYPE_NAMES}
                for p in POLICIES}

    for seed in range(seeds):
        for pname, place in POLICIES.items():
            jobs = make_mixed_jobs(n_jobs, seed)  # identical mix per policy
            place(jobs)
            ev = evaluate(jobs)
            acc[pname]["peak_cpu"].append(ev["peak"]["cpu"])
            acc[pname]["peak_io"].append(ev["peak"]["io"])
            acc[pname]["viol_cpu"].append(ev["slo_violation_frac"]["cpu"])
            acc[pname]["viol_io"].append(ev["slo_violation_frac"]["io"])
            acc[pname]["jct"].append(ev["jct_stretch_mean"])
            for t in TYPE_NAMES:
                type_acc[pname][t]["viol_share"].append(ev["by_type"][t]["viol_share"])
                type_acc[pname][t]["jct_mean"].append(ev["by_type"][t]["jct_mean"])

    def m(x):
        return round(sum(x) / len(x), 4) if x else 0.0

    for p in POLICIES:
        d = acc[p]
        out["aggregate"][p] = {
            "peak_cpu": m(d["peak_cpu"]), "peak_io": m(d["peak_io"]),
            "slo_viol_cpu": m(d["viol_cpu"]), "slo_viol_io": m(d["viol_io"]),
            "jct_stretch_mean": m(d["jct"]),
        }
        out["by_type"][p] = {
            t: {"slo_viol_share": m(type_acc[p][t]["viol_share"]),
                "jct_stretch_mean": m(type_acc[p][t]["jct_mean"])}
            for t in TYPE_NAMES
        }

    # packing density (jobs fit at ceiling), averaged over seeds
    for p, place in POLICIES.items():
        caps = [packing_capacity(s, place) for s in range(seeds)]
        out["packing"][p] = {
            "mean_jobs_fit": round(sum(caps) / len(caps), 2),
            "min": min(caps), "max": max(caps),
        }
    base = out["packing"]["linear-rr"]["mean_jobs_fit"]
    for p in POLICIES:
        v = out["packing"][p]["mean_jobs_fit"]
        out["packing"][p]["gain_vs_rr_pct"] = round(100 * (v - base) / base, 1)
    return out


def write_outputs(res, outdir):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "stage1_head_to_head.json"), "w") as f:
        json.dump(res, f, indent=2)

    # aggregate CSV
    with open(os.path.join(outdir, "stage1_aggregate.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "peak_cpu", "peak_io", "slo_viol_cpu",
                    "slo_viol_io", "jct_stretch_mean", "packing_jobs_fit",
                    "packing_gain_vs_rr_pct"])
        for p in res["policies"]:
            a = res["aggregate"][p]
            pk = res["packing"][p]
            w.writerow([p, a["peak_cpu"], a["peak_io"], a["slo_viol_cpu"],
                        a["slo_viol_io"], a["jct_stretch_mean"],
                        pk["mean_jobs_fit"], pk["gain_vs_rr_pct"]])

    # per-type CSV
    with open(os.path.join(outdir, "stage1_by_type.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "workload_type", "slo_viol_share", "jct_stretch_mean"])
        for p in res["policies"]:
            for t in res["workload_types"]:
                d = res["by_type"][p][t]
                w.writerow([p, t, d["slo_viol_share"], d["jct_stretch_mean"]])


if __name__ == "__main__":
    res = run()
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, "..", "..", "results", "stage1_head_to_head")
    write_outputs(res, outdir)

    print(json.dumps(res, indent=2))
    print("\n=== SUMMARY (sim only) ===", file=sys.stderr)
    hdr = f"{'policy':<15}{'peak_cpu':>9}{'peak_io':>9}{'viol_cpu':>9}" \
          f"{'viol_io':>9}{'jct':>7}{'fit':>6}{'gain%':>7}"
    print(hdr, file=sys.stderr)
    for p in res["policies"]:
        a = res["aggregate"][p]
        pk = res["packing"][p]
        print(f"{p:<15}{a['peak_cpu']:>9.3f}{a['peak_io']:>9.3f}"
              f"{a['slo_viol_cpu']:>9.4f}{a['slo_viol_io']:>9.4f}"
              f"{a['jct_stretch_mean']:>7.3f}{pk['mean_jobs_fit']:>6.1f}"
              f"{pk['gain_vs_rr_pct']:>7.1f}", file=sys.stderr)
