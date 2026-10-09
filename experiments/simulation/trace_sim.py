"""Trace-calibrated evaluation: the three schemes on realistic demand.

Confirms whether the +38% packing result from the synthetic simulator
survives trace-calibrated demand (heterogeneous rhythms, diurnal
envelope, lognormal amplitudes, cpu/mem correlation). Same three
arrangements as experiments/simulation/simulate.py, now on TraceJob demand:

  linear-fixed  : all phases 0 (thundering herd)
  linear-rr     : phases spread by round-robin index (resource-blind)
  phase-coupled : DESYNC competitors + valley-fill complements

Metric: packing density — how many jobs fit under a fixed peak ceiling
before either resource is breached, over the full 24h horizon. Bootstrap
CIs over seeds. Everything reproducible; results committed as JSON.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from experiments.simulation.workload import (TraceJob, generate, diurnal_curve,  # noqa: E402
                          RESOURCES, HORIZON)

TWO_PI = 2 * np.pi
CEIL = 1.0
DUR = diurnal_curve()


def dominant(job: TraceJob) -> str:
    return max(RESOURCES, key=lambda r: job.amp[r])


def total_load(jobs: list[TraceJob], r: str) -> np.ndarray:
    acc = np.zeros(HORIZON)
    for j in jobs:
        acc += j.demand_series(r, diurnal=DUR)
    return acc


def peak(jobs: list[TraceJob]) -> float:
    return max(float(total_load(jobs, r).max()) for r in RESOURCES)


def set_rr(jobs):
    n = len(jobs)
    for i, j in enumerate(jobs):
        j.phase = {r: TWO_PI * i / max(1, n) for r in RESOURCES}


def set_fixed(jobs):
    for j in jobs:
        j.phase = {r: 0.0 for r in RESOURCES}


def _period_shift(job: TraceJob, r: str, phase: float):
    job.phase[r] = phase % TWO_PI


def set_coupled(jobs):
    """DESYNC among per-resource competitors, then valley-fill complements.

    Phase here shifts the job's rhythm along the horizon. Competitors on
    r (r is their dominant resource) are spread evenly; complements are
    placed on the running valley of r's competitor load."""
    rng = np.random.default_rng(1)
    for j in jobs:
        j.phase = {r: float(rng.uniform(0, TWO_PI)) for r in RESOURCES}
    for r in RESOURCES:
        comp = [j for j in jobs if dominant(j) == r]
        m = len(comp)
        if m >= 2:
            for k, j in enumerate(sorted(comp, key=lambda x: x.phase[r])):
                _period_shift(j, r, TWO_PI * k / m)     # even DESYNC spread
    for r in RESOURCES:
        comp = [j for j in jobs if dominant(j) == r]
        if len(comp) < 2:
            continue
        others = [j for j in jobs if dominant(j) != r]
        placed = list(comp)
        for j in others:
            base = total_load(placed, r)
            # try a coarse grid of phase shifts, pick the lowest-peak add
            best_ph, best_pk = 0.0, float("inf")
            for g in range(24):
                ph = TWO_PI * g / 24
                _period_shift(j, r, ph)
                pk = float((base + j.demand_series(r, diurnal=DUR)).max())
                if pk < best_pk:
                    best_pk, best_ph = pk, ph
            _period_shift(j, r, best_ph)
            placed.append(j)


def capacity(seed: int, place, ceil: float = CEIL) -> int:
    n = 4
    last_ok = 0
    while n <= 120:
        jobs = generate(n, seed=seed)
        place(jobs)
        if peak(jobs) <= ceil:
            last_ok = n
            n += 4
        else:
            break
    return last_ok


def bootstrap_ci(vals, iters=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.array(vals, dtype=float)
    means = [rng.choice(v, size=len(v), replace=True).mean() for _ in range(iters)]
    return round(float(np.percentile(means, 2.5)), 2), round(float(np.percentile(means, 97.5)), 2)


def run(seeds=24, ceil=CEIL):
    schemes = {"linear-fixed": set_fixed, "linear-rr": set_rr,
               "phase-coupled": set_coupled}
    out = {"ceiling": ceil, "seeds": seeds, "workload": "trace-calibrated",
           "generated_at": time.time(), "schemes": {}}
    caps = {}
    for name, place in schemes.items():
        caps[name] = [capacity(s, place, ceil) for s in range(seeds)]
        lo, hi = bootstrap_ci(caps[name])
        out["schemes"][name] = {
            "mean_jobs_fit": round(float(np.mean(caps[name])), 2),
            "ci95": [lo, hi], "min": int(min(caps[name])), "max": int(max(caps[name]))}
    rr = out["schemes"]["linear-rr"]["mean_jobs_fit"]
    ph = out["schemes"]["phase-coupled"]["mean_jobs_fit"]
    out["phase_packing_gain_pct"] = round(100 * (ph - rr) / rr, 1)
    out["_caps_raw"] = caps
    return out


def run_both():
    """Two regimes, kept separate:
      with_diurnal    : realistic shared day/night common-mode present
      rhythm_only      : diurnal removed -> only the per-job rhythm, which
                         is the component a phase scheduler can control.
    The gap between these two tells you exactly how much of the packing
    benefit is reachable by scheduling vs how much needs node autoscaling."""
    global DUR
    out = {}
    DUR = diurnal_curve()
    out["with_diurnal"] = {k: v for k, v in run().items() if k != "_caps_raw"}
    DUR = np.zeros(HORIZON)                     # remove common-mode
    out["rhythm_only"] = {k: v for k, v in run().items() if k != "_caps_raw"}
    return out


if __name__ == "__main__":
    print(json.dumps(run_both(), indent=2))
