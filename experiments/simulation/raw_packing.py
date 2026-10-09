"""Packing on raw Google ClusterData 2011 series: does the rhythm-only gain hold?

experiments/simulation/trace_sim.py reports that, with the shared diurnal component removed
("rhythm_only"), phase-coupled arrangement fits 12.5% more jobs than
round-robin on the trace-calibrated generator. This script repeats that
comparison on real per-task CPU and memory series extracted by
experiments/data_prep/extract_gct_series.py (24 h, 5-minute slots, tasks present in every
slot).

Control. The scheduler's only control is a circular time shift of each task's
series, the raw-trace counterpart of the model's phase. One shift applies to
both resources of a task: unlike the generator, a real task cannot move its
CPU and memory rhythms independently, so this is a stricter test.

Arrangements (same task stream, only the shifts differ):
  as-recorded        : shift 0, the task timing in the trace
  linear-rr          : shift i*H/n by arrival index (resource-blind)
  phase-coupled      : trace_sim.set_coupled ported to series: DESYNC each
                       resource's competitors (tasks dominant in it) so their
                       peaks are evenly spaced, then valley-fill every other
                       task on a 24-point shift grid
  greedy-valley-fill : ablation without DESYNC: valley-fill every task

Regimes:
  raw         : series as recorded, including the shared day/night component
  rhythm_only : the shared component removed per task (least-squares gain on
                the cluster-wide mean profile, clipped at 0), matching the
                generator's rhythm_only regime

Metric: capacity, the largest n before the first n whose arranged peak (max
over time and over CPU and memory) exceeds the ceiling, stepping n by one.
Tasks arrive in a seeded random order drawn from the eligible pool. Gains are
paired per seed with a percentile bootstrap CI.

Usage: python3 experiments/simulation/raw_packing.py <series.npz> [--seeds 100] [--ceilings 0.5 1]
Writes results/raw_packing.json.
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
from experiments.simulation.packing_fine import paired_gain  # noqa: E402

GRID = 24


def load(path: str):
    z = np.load(path)
    x = np.stack([z["cpu"], z["mem"]], axis=1).astype(np.float64)   # tasks x 2 x H
    return x, z["keys"]


def remove_common_mode(x: np.ndarray) -> np.ndarray:
    a = x.mean(axis=0)                                   # 2 x H cluster profile
    ac = a - a.mean(axis=1, keepdims=True)
    var = (ac ** 2).sum(axis=1)                          # 2
    xc = x - x.mean(axis=2, keepdims=True)
    g = (xc * ac[None]).sum(axis=2) / var[None]          # tasks x 2
    return np.clip(x - g[:, :, None] * ac[None], 0.0, None)


def shifted(s: np.ndarray, k: int) -> np.ndarray:
    return np.roll(s, k, axis=-1)


def arrange_fixed(x):
    return np.zeros(len(x), int)


def arrange_rr(x):
    n, h = len(x), x.shape[2]
    return np.array([round(i * h / n) % h for i in range(n)])


def _amp(x):
    return np.percentile(x, 95, axis=2) - np.percentile(x, 5, axis=2)   # tasks x 2


def _valley_fill(x, shifts, placed_idx, todo_idx):
    h = x.shape[2]
    load = np.zeros(x.shape[1:])
    for i in placed_idx:
        load += shifted(x[i], shifts[i])
    grid = [round(g * h / GRID) for g in range(GRID)]
    for i in todo_idx:
        best, best_pk = 0, np.inf
        for k in grid:
            pk = (load + shifted(x[i], k)).max()
            if pk < best_pk:
                best, best_pk = k, pk
        shifts[i] = best
        load += shifted(x[i], best)
    return shifts


def arrange_coupled(x):
    n, _, h = x.shape
    shifts = np.zeros(n, int)
    dom = _amp(x).argmax(axis=1)
    placed = []
    for r in (0, 1):
        comp = [i for i in range(n) if dom[i] == r]
        if len(comp) >= 2:
            peak_t = {i: int(x[i, r].argmax()) for i in comp}
            for k, i in enumerate(sorted(comp, key=peak_t.get)):
                shifts[i] = (round(k * h / len(comp)) - peak_t[i]) % h
            placed += comp
    rest = [i for i in range(n) if i not in set(placed)]
    return _valley_fill(x, shifts, placed, rest)


def arrange_greedy(x):
    return _valley_fill(x, np.zeros(len(x), int), [], list(range(len(x))))


SCHEMES = {"as-recorded": arrange_fixed, "linear-rr": arrange_rr,
           "phase-coupled": arrange_coupled, "greedy-valley-fill": arrange_greedy}


def peak(x, shifts) -> float:
    tot = np.zeros(x.shape[1:])
    for i, k in enumerate(shifts):
        tot += shifted(x[i], k)
    return float(tot.max())


def capacity(pool, order, arrange, ceil, n_max):
    last_ok = 0
    for n in range(1, n_max + 1):
        xs = pool[order[:n]]
        if peak(xs, arrange(xs)) <= ceil:
            last_ok = n
        else:
            break
    return last_ok


def rhythm_stats(x):
    mean = x.mean(axis=2)
    ptm = np.where(mean > 0, x.max(axis=2) / np.where(mean > 0, mean, 1), np.nan)
    cv = np.where(mean > 0, x.std(axis=2) / np.where(mean > 0, mean, 1), np.nan)
    return {"cpu_peak_to_mean_median": round(float(np.nanmedian(ptm[:, 0])), 3),
            "mem_peak_to_mean_median": round(float(np.nanmedian(ptm[:, 1])), 3),
            "cpu_cv_median": round(float(np.nanmedian(cv[:, 0])), 3),
            "mem_cv_median": round(float(np.nanmedian(cv[:, 1])), 3),
            "cpu_mean_median": round(float(np.median(mean[:, 0])), 5),
            "mem_mean_median": round(float(np.median(mean[:, 1])), 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("series")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--ceilings", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--pool", type=int, default=0, help="cap pool size (0 = all eligible)")
    a = ap.parse_args()
    x, keys = load(a.series)
    eligible = np.where((x[:, 0].mean(axis=1) > 0) & (x[:, 1].mean(axis=1) > 0))[0]
    if a.pool:
        eligible = np.random.default_rng(12345).choice(eligible, a.pool, replace=False)
    out = {"experiment": "packing on raw Google ClusterData 2011 series",
           "series_file": os.path.basename(a.series), "tasks_in_file": int(len(x)),
           "eligible_pool": int(len(eligible)), "slots": int(x.shape[2]),
           "control": "one circular time shift per task, applied to CPU and memory together",
           "seeds": a.seeds, "step": 1, "generated_at": time.time(), "regimes": {}}
    for regime in ("raw", "rhythm_only"):
        pool = x[eligible] if regime == "raw" else remove_common_mode(x[eligible])
        reg = {"rhythm_stats": rhythm_stats(pool), "ceilings": {}}
        for ceil in a.ceilings:
            per_task = pool.max(axis=2).max(axis=1)
            n_max = min(len(pool), int(4 * ceil / max(np.median(per_task), 1e-9)) + 50)
            caps = {name: [] for name in SCHEMES}
            for s in range(a.seeds):
                order = np.random.default_rng(s).permutation(len(pool))
                for name, arr in SCHEMES.items():
                    caps[name].append(capacity(pool, order, arr, ceil, n_max))
            for name, v in caps.items():
                assert max(v) < n_max, f"{regime} {name} hit n_max at ceiling {ceil}"
            row = {"n_max": n_max,
                   "schemes": {k: {"mean_jobs_fit": round(float(np.mean(v)), 2),
                                   "min": int(min(v)), "max": int(max(v)),
                                   "per_seed": [int(t) for t in v]} for k, v in caps.items()},
                   "coupled_vs_rr": paired_gain(caps["linear-rr"], caps["phase-coupled"]),
                   "coupled_vs_greedy": paired_gain(caps["greedy-valley-fill"], caps["phase-coupled"]),
                   "rr_vs_recorded": paired_gain(caps["as-recorded"], caps["linear-rr"])}
            reg["ceilings"][str(ceil)] = row
            print(regime, ceil, {k: row["schemes"][k]["mean_jobs_fit"] for k in caps},
                  "coupled_vs_rr", row["coupled_vs_rr"], flush=True)
        out["regimes"][regime] = reg
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "results", "raw_packing.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
