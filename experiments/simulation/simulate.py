"""Cluster simulation: phase-coupled ('bubble') scheduling vs linear baselines.

Answers the load-bearing question — is the logic sound? — with a number.

Setup: a heterogeneous job mix sharing two resources (cpu, io) on one
node. Each job has a demand waveform (a bubble). We compare three ways
of placing those waveforms in time:

  linear-fixed  : every job's phase = 0 (naive fixed schedule; all peaks
                  land together — the thundering herd).
  linear-rr     : phases spread by fixed round-robin index (equal slots,
                  resource-blind — the classic time-slice scheduler).
  phase-coupled : phases evolve under the couple law until self-organised
                  (repel competitors, attract complements).

Metric: peak simultaneous demand per resource (lower = fewer SLO
violations at identical total work), plus an SLO-violation count when
peak exceeds capacity=1.0. Total work is identical across all three —
only the *arrangement in time* differs, so any win is purely from the
scheduling model, not from doing less.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kadence.oscillator import (Job, couple_step, desync_arrange,  # noqa: E402
                             peak_contention, TWO_PI)

RESOURCES = ["cpu", "io"]
PERIOD = 100.0
CAPACITY = 1.0


def make_jobs(n: int, seed: int) -> list[Job]:
    rng = random.Random(seed)
    jobs = []
    for i in range(n):
        # three archetypes: cpu-heavy, io-heavy, balanced — heterogeneous mix
        kind = rng.choice(["cpu", "io", "bal"])
        if kind == "cpu":
            duty = {"cpu": rng.uniform(0.15, 0.30), "io": rng.uniform(0.03, 0.08)}
            height = {"cpu": rng.uniform(0.25, 0.45), "io": rng.uniform(0.03, 0.10)}
        elif kind == "io":
            duty = {"cpu": rng.uniform(0.03, 0.08), "io": rng.uniform(0.15, 0.30)}
            height = {"cpu": rng.uniform(0.03, 0.10), "io": rng.uniform(0.25, 0.45)}
        else:
            duty = {"cpu": rng.uniform(0.10, 0.18), "io": rng.uniform(0.10, 0.18)}
            height = {"cpu": rng.uniform(0.15, 0.25), "io": rng.uniform(0.15, 0.25)}
        jobs.append(Job(jid=f"j{i}", duty=duty, height=height,
                        omega=rng.uniform(0.9, 1.1)))
    return jobs


def set_phases_fixed(jobs):
    for j in jobs:
        j.phase = {r: 0.0 for r in RESOURCES}


def set_phases_rr(jobs):
    n = len(jobs)
    for idx, j in enumerate(jobs):
        j.phase = {r: (TWO_PI * idx / n) for r in RESOURCES}


def set_phases_coupled(jobs):
    rng = random.Random(1)
    for j in jobs:
        j.phase = {r: rng.uniform(0, TWO_PI) for r in RESOURCES}
    desync_arrange(jobs, RESOURCES)


def evaluate(jobs) -> dict:
    peaks = peak_contention(jobs, RESOURCES, PERIOD)
    # SLO violation = fraction of the period any resource is over capacity
    viol = {}
    for r in RESOURCES:
        over = 0
        S = 300
        for k in range(S):
            t = PERIOD * k / S
            if sum(j.demand(r, t, PERIOD) for j in jobs) > CAPACITY:
                over += 1
        viol[r] = round(over / S, 4)
    return {"peak": {r: round(peaks[r], 3) for r in RESOURCES},
            "slo_violation_frac": viol}


def run(n_jobs=24, seeds=20):
    out = {"n_jobs": n_jobs, "seeds": seeds, "capacity": CAPACITY,
           "schemes": {}}
    agg = {s: {"peak_cpu": [], "peak_io": [], "viol_cpu": [], "viol_io": []}
           for s in ("linear-fixed", "linear-rr", "phase-coupled")}
    for seed in range(seeds):
        base = make_jobs(n_jobs, seed)
        for scheme, setter in (("linear-fixed", set_phases_fixed),
                               ("linear-rr", set_phases_rr),
                               ("phase-coupled", set_phases_coupled)):
            jobs = make_jobs(n_jobs, seed)   # identical mix per scheme
            setter(jobs)
            ev = evaluate(jobs)
            agg[scheme]["peak_cpu"].append(ev["peak"]["cpu"])
            agg[scheme]["peak_io"].append(ev["peak"]["io"])
            agg[scheme]["viol_cpu"].append(ev["slo_violation_frac"]["cpu"])
            agg[scheme]["viol_io"].append(ev["slo_violation_frac"]["io"])

    def mean(x):
        return round(sum(x) / len(x), 4)
    for s, d in agg.items():
        out["schemes"][s] = {k: mean(v) for k, v in d.items()}
    return out


if __name__ == "__main__":
    res = run()
    res["generated_at"] = time.time()
    print(json.dumps(res, indent=2))
    base = res["schemes"]["linear-rr"]
    ph = res["schemes"]["phase-coupled"]
    for r in ("cpu", "io"):
        b, p = base[f"peak_{r}"], ph[f"peak_{r}"]
        print(f"peak {r}: linear-rr {b} -> phase-coupled {p} "
              f"({round(100*(b-p)/b,1)}% lower)", file=sys.stderr)
