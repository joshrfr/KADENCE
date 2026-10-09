"""Population-scale convergence: large n (500, 1000, 2000).

Extends desync_scale.py using the identical settle() logic; just runs bigger
ring sizes that are impractical locally.  Uses core.neighbor_gossip.RingController
directly (same kernel as the committed desync_scale.json).

Writes results/desync_scale_large.json with the same schema as desync_scale.json.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from kadence.neighbor_gossip import BubbleSpec, admit_ring, RingController, phases_from_gaps

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TWO_PI = 2 * math.pi


def settle(n, dt=0.1, tol=1e-6, max_rounds=2_000_000, seed=0):
    rng = np.random.default_rng(seed)
    bubbles = [BubbleSpec(f"j{i}", 1e-4) for i in range(n)]
    plan = admit_ring(bubbles, slack_weights=[1.0] * n)
    d = np.array(plan.desired, float)
    g = d * rng.uniform(0.4, 1.6, n)
    g = g / g.sum() * TWO_PI
    g = g.tolist()
    g[-1] = TWO_PI - math.fsum(g[:-1])
    phases = phases_from_gaps(g, circumference=TWO_PI)
    rc = RingController(bubbles, phases, plan)
    e_prev = rc.energy(); e_inc = 0
    t0 = time.time()
    for r in range(max_rounds):
        rc.step(dt=dt)
        e = rc.energy()
        if e > e_prev + 1e-15:
            e_inc += 1
        e_prev = e
        if rc.max_gap_error() < tol:
            return {
                "n": n, "settling_rounds": r + 1,
                "final_max_gap_error": rc.max_gap_error(),
                "energy_increase_rounds": e_inc,
                "wall_seconds": round(time.time() - t0, 1),
                "converged": True,
            }
    return {
        "n": n, "settling_rounds": max_rounds,
        "final_max_gap_error": rc.max_gap_error(),
        "energy_increase_rounds": e_inc,
        "wall_seconds": round(time.time() - t0, 1),
        "converged": False,
    }


def main():
    sizes = [500, 1000, 2000]
    out = {
        "provenance": {"kernel": "core.neighbor_gossip.RingController",
                       "dt": 0.1, "tol": 1e-6, "messages_per_job_round": 2},
        "populations": [],
    }
    for n in sizes:
        print(f"n={n} ...", flush=True)
        r = settle(n)
        print(f"  n={n}: rounds={r['settling_rounds']} err={r['final_max_gap_error']:.2e} "
              f"wall={r['wall_seconds']}s converged={r['converged']}", flush=True)
        out["populations"].append(r)

    path = os.path.join(ROOT, "results", "desync_scale_large.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
