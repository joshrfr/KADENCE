"""EXPLORATORY (not a paper claim yet): time-varying node-capacity "bubble".

Standard packing (experiments/simulation/pack_bubble.py) fits job demand under a FLAT capacity.
Here we model a node's available capacity as a *breathing* bubble cap(t) that
varies over the horizon -- e.g., capacity ceded to background/harvested/spot
load that rises and falls on its own daily rhythm. The question this answers:

    If a node's capacity itself oscillates, does placement that KNOWS the
    capacity rhythm (and shifts job demand into capacity peaks / out of
    capacity troughs) pack more than placement that treats capacity as flat?

Arms (all on the real Google-2011 job series, circular time-shifts):
  round_robin        resource/-capacity-blind even spread of start times
  repulsive_flat     current KADENCE-style repulsion; minimises absolute peak,
                     admits while peak <= mean(cap(t))  [capacity-SHAPE-blind]
  repulsive_capaware minimises the running demand/capacity RATIO against cap(t),
                     admits while that ratio <= 1        [capacity-bubble-aware]
  oracle_capaware    best capacity-aware count over several insertion orders

Metric: jobs admitted before the summed demand violates the time-varying
envelope cap(t) on either resource. Headline: gain of capaware over flat.
Writes results/kadence/capacity_bubble.json with paired bootstrap CIs.

This is a design probe to see how a bubble-capacity model performs; it does not
change any existing paper claim.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from experiments.simulation.gct_common import load_series, bootstrap_ci, ROOT

GRID = 48
TWO_PI = 2 * math.pi


def capacity_bubble(H, base, amp, n_res=2, seed=0):
    """Breathing per-resource capacity cap[r,t] = base_r * (1 + amp*sin(...)).

    Each resource gets its own phase so the two capacity rhythms are not
    aligned, which is the realistic case (CPU and memory headroom breathe
    independently)."""
    rng = np.random.default_rng(seed)
    t = np.arange(H)
    cap = np.zeros((n_res, H))
    for r in range(n_res):
        phase = rng.uniform(0, TWO_PI)
        cap[r] = base[r] * (1.0 + amp * np.sin(TWO_PI * t / H + phase))
    return np.maximum(cap, 1e-9)


def _fits(total, cap):
    """True while summed demand is under the time-varying envelope everywhere."""
    return bool(np.all(total <= cap + 1e-12))


def _max_ratio(total, cap):
    return float((total / cap).max())


def cap_round_robin(series, order, cap):
    H = series[0].shape[1]
    admitted = []
    for idx in order:
        admitted.append(idx)
        n = len(admitted)
        shifts = (np.arange(n) * H // n)
        total = sum(np.roll(series[j], int(sh), axis=1)
                    for j, sh in zip(admitted, shifts))
        if not _fits(total, cap):
            return n - 1
    return len(admitted)


def cap_repulsive_flat(series, order, cap):
    """Capacity-SHAPE-blind PLACEMENT, scored on the TRUE envelope (fair).

    The placement policy ignores the shape of cap(t): it picks the shift that
    minimises the absolute demand peak (what current KADENCE repulsion does,
    treating capacity as a flat ceiling). Admission is checked against the SAME
    true time-varying cap(t) used for every arm, so the only thing that differs
    from the capacity-aware arm is whether the policy knows the capacity rhythm.
    """
    H = series[0].shape[1]
    cand = (np.arange(GRID) * H // GRID)
    total = np.zeros_like(cap)
    n = 0
    for idx in order:
        job = series[idx]
        best_s, best_pk = 0, np.inf
        for s in cand:
            pk = (total + np.roll(job, int(s), axis=1)).max()   # cap-shape-blind
            if pk < best_pk:
                best_pk, best_s = pk, s
        cand_total = total + np.roll(job, int(best_s), axis=1)
        if not _fits(cand_total, cap):                          # scored on TRUE cap(t)
            break
        total = cand_total
        n += 1
    return n


def cap_repulsive_capaware(series, order, cap, return_total=False):
    """Capacity-bubble-aware: minimise running demand/capacity ratio vs cap(t)."""
    H = series[0].shape[1]
    cand = (np.arange(GRID) * H // GRID)
    total = np.zeros_like(cap)
    n = 0
    for idx in order:
        job = series[idx]
        best_s, best_ratio = 0, np.inf
        for s in cand:
            ratio = _max_ratio(total + np.roll(job, int(s), axis=1), cap)
            if ratio < best_ratio:
                best_ratio, best_s = ratio, s
        if best_ratio > 1.0:
            break
        total = total + np.roll(job, int(best_s), axis=1)
        n += 1
    return (n, total) if return_total else n


def cap_oracle_capaware(series, order, cap, tries=8, seed=0):
    rng = np.random.default_rng(seed)
    best = cap_repulsive_capaware(series, order, cap)
    for _ in range(tries - 1):
        best = max(best, cap_repulsive_capaware(series, rng.permutation(order), cap))
    return best


ARMS = {
    "round_robin": cap_round_robin,
    "repulsive_flat": cap_repulsive_flat,
    "repulsive_capaware": cap_repulsive_capaware,
    "oracle_capaware": cap_oracle_capaware,
}


def run_amp(series, base, amp, seeds, pool):
    rng = np.random.default_rng(1234)
    counts = {a: np.zeros(seeds) for a in ARMS}
    for si in range(seeds):
        order = rng.choice(len(series), pool, replace=False)
        cap = capacity_bubble(series[0].shape[1], base, amp, seed=si)
        for a, fn in ARMS.items():
            counts[a][si] = fn(series, order, cap)
    res = {"amp": amp, "seeds": seeds, "pool": pool, "per_arm": {}}
    flat = counts["repulsive_flat"]
    rr = counts["round_robin"]
    for a in ARMS:
        c = counts[a]
        entry = {"mean_fit": round(float(c.mean()), 2),
                 "std": round(float(c.std()), 2)}
        if a not in ("round_robin", "repulsive_flat"):
            gain = 100.0 * (c - flat) / np.maximum(flat, 1)   # vs capacity-blind
            m, lo, hi = bootstrap_ci(gain, seed=7)
            entry["gain_vs_flat_pct"] = round(m, 2)
            entry["gain_vs_flat_ci95"] = [round(lo, 2), round(hi, 2)]
        res["per_arm"][a] = entry
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amps", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75])
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--pool", type=int, default=200)
    ap.add_argument("--base", type=float, default=1.0,
                    help="mean capacity per resource (series units)")
    ap.add_argument("--output", default=os.path.join(ROOT, "results", "kadence",
                                                     "capacity_bubble.json"))
    a = ap.parse_args()

    series, task_ids, prov = load_series()
    series = np.asarray(series, float)
    base = np.array([a.base, a.base], float)

    out = {"exploratory": True,
           "note": "time-varying node-capacity bubble; design probe, not a paper claim",
           "provenance": {"trace": prov, "grid": GRID, "base": a.base,
                          "arms": list(ARMS.keys()),
                          "gain_reference": "repulsive_flat (capacity-shape-blind)"},
           "by_amplitude": []}
    for amp in a.amps:
        r = run_amp(series, base, amp, a.seeds, a.pool)
        out["by_amplitude"].append(r)
        ca = r["per_arm"]["repulsive_capaware"]
        print(f"amp={amp:>4}: capaware mean_fit={ca['mean_fit']:>6} "
              f"gain_vs_flat={ca.get('gain_vs_flat_pct')}% "
              f"CI={ca.get('gain_vs_flat_ci95')}", flush=True)

    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", a.output)


if __name__ == "__main__":
    main()
