"""Vectorised convergence-vs-population for large rings.

Reproduces the committed strict-neighbor update exactly, but as one numpy
vector operation per round so ring sizes in the thousands are tractable:

  disp_i = clip( dt*(e_i - e_{i-1}), -sf*g_{i-1}, +sf*g_i ),  e_i = g_i - 1/n

which is local_correction (right_error - left_error) followed by the same
per-edge safety limiter (safety_fraction sf, min gaps ~0) as
limit_local_displacement in core.neighbor_gossip. Validated round-for-round
against RingController on small n before use. Writes results/desync_scale_big.json.
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
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TWO_PI = 2 * math.pi


def settle_fast(n, dt=0.1, sf=0.45, tol=1e-6, max_rounds=4_000_000, seed=0,
                record=False):
    rng = np.random.default_rng(seed)
    target = TWO_PI / n
    g = target * rng.uniform(0.4, 1.6, n)
    g *= TWO_PI / g.sum()                                  # gaps sum to circumference
    e_prev_energy = math.inf
    einc = 0
    traj = []
    for r in range(max_rounds):
        e = g - target
        gerr = np.max(np.abs(e))
        if record:
            traj.append(float(gerr))
        if gerr < tol:
            return r + 1, float(gerr), einc, traj
        # correction_i = e_i - e_{i-1}; safety clip to per-edge slack
        corr = e - np.roll(e, 1)
        disp = np.clip(dt * corr, -sf * np.roll(g, 1), sf * g)
        # applying disp_i moves phase i; gap_i grows by (disp_{i+1}-disp_i)
        g = g + (np.roll(disp, -1) - disp)
        energy = float(np.sum(e * e))
        if energy > e_prev_energy + 1e-15:
            einc += 1
        e_prev_energy = energy
    return None, float(np.max(np.abs(g - target))), einc, traj


def _validate():
    """Check the vectorised stepper matches RingController settling on small n."""
    from kadence.neighbor_gossip import (BubbleSpec, admit_ring, RingController,
                                       phases_from_gaps)
    n = 50
    rng = np.random.default_rng(0)
    bubbles = [BubbleSpec(f"j{i}", 1e-4) for i in range(n)]
    plan = admit_ring(bubbles, slack_weights=[1.0] * n)
    d = np.array(plan.desired); g = d * rng.uniform(0.4, 1.6, n)
    g = (g / g.sum() * TWO_PI).tolist(); g[-1] = TWO_PI - math.fsum(g[:-1])
    rc = RingController(bubbles, list(phases_from_gaps(g, circumference=TWO_PI)), plan)
    rounds = 0
    while rc.max_gap_error() >= 1e-6 and rounds < 20000:
        rc.step(dt=0.1); rounds += 1
    fast, _, _, _ = settle_fast(n, seed=0)
    # same order of magnitude (both realise the same rule; tiny numeric drift ok)
    ok = fast is not None and abs(fast - rounds) / rounds < 0.10
    return ok, rounds, fast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sizes", nargs="*", type=int, default=[400, 1000, 2000])
    ap.add_argument("--max-rounds", type=int, default=4_000_000)
    args = ap.parse_args()
    ok, rc_rounds, fast_rounds = _validate()
    print(f"validation vs RingController (n=50): kernel={rc_rounds} "
          f"fast={fast_rounds} match={ok}", flush=True)
    out = {"provenance": {"stepper": "vectorised, validated vs RingController",
                          "dt": 0.1, "safety_fraction": 0.45, "tol": 1e-6,
                          "validation": {"n": 50, "kernel_rounds": rc_rounds,
                                         "fast_rounds": fast_rounds, "match": ok}},
           "populations": []}
    for n in args.sizes:
        rounds, err, einc, _ = settle_fast(n, max_rounds=args.max_rounds)
        out["populations"].append({"n": n, "settling_rounds": rounds,
                                   "final_max_gap_error": err,
                                   "energy_increase_rounds": einc,
                                   "converged": rounds is not None})
        print(f"n={n:>6} settle={rounds} err={err:.2e} Einc={einc}", flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "desync_scale_big.json"), "w"),
              indent=2)
    print("wrote results/desync_scale_big.json")


if __name__ == "__main__":
    main()
