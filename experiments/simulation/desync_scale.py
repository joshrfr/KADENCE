"""Convergence vs population for the strict-neighbor desync kernel.

Drives the committed RingController (src/kadence/neighbor_gossip.py, the same kernel
behind results/churn_evaluation.json) across ring sizes n and measures how many
local rounds it takes to reach an even splay (max gap error below tol), the
final error, and whether the spacing energy ever increased. Two messages per
job per round throughout. Writes results/desync_scale.json.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from kadence.neighbor_gossip import (
    BubbleSpec, admit_ring, RingController, phases_from_gaps,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TWO_PI = 2 * math.pi


def settle(n, dt=0.1, tol=1e-6, max_rounds=400000, seed=0):
    rng = np.random.default_rng(seed)
    bubbles = [BubbleSpec(f"j{i}", 1e-4) for i in range(n)]
    plan = admit_ring(bubbles, slack_weights=[1.0] * n)   # uniform desired gaps
    d = np.array(plan.desired, float)
    g = d * rng.uniform(0.4, 1.6, n)
    g = g / g.sum() * TWO_PI                               # valid winding, off-splay
    g = g.tolist()
    g[-1] = TWO_PI - math.fsum(g[:-1])                     # exact closure
    phases = phases_from_gaps(g, circumference=TWO_PI)
    rc = RingController(bubbles, phases, plan)
    e_prev = rc.energy(); e_inc = 0
    for r in range(max_rounds):
        rc.step(dt=dt)
        e = rc.energy()
        if e > e_prev + 1e-15:
            e_inc += 1
        e_prev = e
        if rc.max_gap_error() < tol:
            return r + 1, rc.max_gap_error(), e_inc
    return None, rc.max_gap_error(), e_inc


def main(sizes):
    out = {"provenance": {"kernel": "core.neighbor_gossip.RingController",
                          "dt": 0.1, "tol": 1e-6,
                          "messages_per_job_round": 2},
           "populations": []}
    for n in sizes:
        rounds, err, einc = settle(n)
        out["populations"].append({"n": n, "settling_rounds": rounds,
                                   "final_max_gap_error": err,
                                   "energy_increase_rounds": einc})
        print(f"n={n} settle={rounds} err={err:.2e} Einc={einc}", flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "desync_scale.json"), "w"),
              indent=2)
    print("wrote results/desync_scale.json")


if __name__ == "__main__":
    sizes = [int(x) for x in sys.argv[1:]] or [10, 25, 50, 100, 200]
    main(sizes)
