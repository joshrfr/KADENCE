"""Higher-resolution packing: AGENTS.md open experiment 3.

experiments/simulation/trace_sim.py grows the job count in steps of 4 on a node whose ceiling
admits about 4 jobs, so its capacities take only the values 0, 4, and 8 and
the rhythm-only "+12.5%" is 4.0 against 4.5 jobs. This script repeats the
same experiment with a step of one job and on larger nodes (ceiling K times
the original), so each capacity is resolved to a single job.

Everything else is unchanged from trace_sim: the same trace-calibrated
generator, the same three arrangements (linear-fixed, linear-rr,
phase-coupled), the same two regimes (with_diurnal, rhythm_only), and the
same stopping rule (the largest n before the first n whose peak exceeds the
ceiling). Gains are paired per seed (same jobs, only the arrangement
differs) with a percentile bootstrap CI over seeds.

Usage: python3 experiments/simulation/packing_fine.py [--seeds 24] [--ceilings 1 2 4 8]
Writes results/packing_fine.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
import experiments.simulation.trace_sim as ts  # noqa: E402
from experiments.simulation.workload import HORIZON, diurnal_curve, generate  # noqa: E402

SCHEMES = {"linear-fixed": ts.set_fixed, "linear-rr": ts.set_rr,
           "phase-coupled": ts.set_coupled}


def capacity(seed: int, place, ceil: float, n_max: int) -> int:
    """Largest n before the first n (step 1) whose arranged peak exceeds ceil."""
    last_ok = 0
    for n in range(1, n_max + 1):
        jobs = generate(n, seed=seed)
        place(jobs)
        if ts.peak(jobs) <= ceil:
            last_ok = n
        else:
            break
    return last_ok


def paired_gain(base, new, iters=4000, seed=0):
    """Mean relative gain of new over base (ratio of means), bootstrap over seeds."""
    b, n = np.asarray(base, float), np.asarray(new, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(b), size=(iters, len(b)))
    ratios = 100 * (n[idx].mean(1) - b[idx].mean(1)) / b[idx].mean(1)
    point = 100 * (n.mean() - b.mean()) / b.mean()
    return {"gain_pct": round(float(point), 2),
            "ci95_pct": [round(float(np.percentile(ratios, 2.5)), 2),
                         round(float(np.percentile(ratios, 97.5)), 2)],
            "seeds_new_better": int((n > b).sum()),
            "seeds_tied": int((n == b).sum()),
            "seeds_new_worse": int((n < b).sum())}


def run(seeds: int, ceilings: list[float]) -> dict:
    out = {"experiment": "higher-resolution packing (step 1, ceilings x K)",
           "workload": "trace-calibrated generator (experiments/simulation/workload.py), not raw trace",
           "seeds": seeds, "step": 1, "generated_at": time.time(), "regimes": {}}
    for regime in ("with_diurnal", "rhythm_only"):
        ts.DUR = diurnal_curve() if regime == "with_diurnal" else np.zeros(HORIZON)
        rows = {}
        for ceil in ceilings:
            n_max = int(12 * ceil) + 12
            caps = {name: [capacity(s, place, ceil, n_max) for s in range(seeds)]
                    for name, place in SCHEMES.items()}
            row = {"ceiling": ceil, "n_max": n_max, "schemes": {}}
            for name, v in caps.items():
                row["schemes"][name] = {"mean_jobs_fit": round(float(np.mean(v)), 2),
                                        "min": int(min(v)), "max": int(max(v)),
                                        "per_seed": [int(x) for x in v]}
                assert max(v) < n_max, f"{name} hit n_max at ceiling {ceil}"
            row["coupled_vs_rr"] = paired_gain(caps["linear-rr"], caps["phase-coupled"])
            row["rr_vs_fixed"] = paired_gain(caps["linear-fixed"], caps["linear-rr"])
            rows[str(ceil)] = row
            print(regime, ceil, {k: row["schemes"][k]["mean_jobs_fit"] for k in caps},
                  row["coupled_vs_rr"], flush=True)
        out["regimes"][regime] = rows
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--ceilings", type=float, nargs="+", default=[1, 2, 4, 8])
    a = ap.parse_args()
    res = run(a.seeds, a.ceilings)
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "results", "packing_fine.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", path)
