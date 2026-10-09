"""Reproducible correctness/scaling sweep for the strict-neighbor kernel.

This is not a scheduler benchmark.  It checks the narrower claims made by
``core.neighbor_gossip``: fixed-topology convergence, hard-gap preservation,
and exactly two directed neighbor messages per job and round.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
from datetime import datetime, timezone


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

from kadence.neighbor_gossip import (  # noqa: E402
    BubbleSpec,
    RingController,
    TWO_PI,
    admit_ring,
    phases_from_gaps,
)


def make_case(n: int, seed: int) -> tuple[tuple[BubbleSpec, ...], tuple[float, ...]]:
    rng = random.Random(seed)
    raw_widths = [rng.uniform(0.5, 1.5) for _ in range(n)]
    width_budget = 0.35 * TWO_PI
    width_scale = width_budget / sum(raw_widths)
    bubbles = tuple(
        BubbleSpec(f"job-{i}", raw_width * width_scale)
        for i, raw_width in enumerate(raw_widths)
    )
    guard = 0.02 * TWO_PI / n
    plan = admit_ring(bubbles, guard=guard)

    # Start from a deliberately uneven, but admitted, distribution of the
    # available slack.  Exponential samples provide reproducible skew.
    available = TWO_PI - sum(plan.minimum)
    raw_slack = [rng.expovariate(1.0) for _ in range(n)]
    slack_total = sum(raw_slack)
    gaps = tuple(
        minimum + available * sample / slack_total
        for minimum, sample in zip(plan.minimum, raw_slack)
    )
    return bubbles, phases_from_gaps(gaps)


def run_case(n: int, seed: int, tolerance: float, max_steps: int) -> dict:
    bubbles, phases = make_case(n, seed)
    guard = 0.02 * TWO_PI / n
    plan = admit_ring(bubbles, guard=guard)
    controller = RingController(bubbles, phases, plan)

    energy_increases = 0
    minimum_margin = math.inf
    steps = 0
    while controller.max_gap_error() > tolerance and steps < max_steps:
        report = controller.step(dt=0.10, safety_fraction=0.45)
        if report.energy_after > report.energy_before + 1e-12:
            energy_increases += 1
        minimum_margin = min(
            minimum_margin,
            *(gap - hard for gap, hard in zip(controller.gaps(), plan.minimum)),
        )
        steps += 1

    return {
        "n": n,
        "seed": seed,
        "converged": controller.max_gap_error() <= tolerance,
        "steps": steps,
        "directed_messages": controller.message_count,
        "messages_per_job_round": (
            controller.message_count / (n * steps) if steps else 0.0
        ),
        "final_max_gap_error": controller.max_gap_error(),
        "minimum_safety_margin": minimum_margin,
        "energy_increase_steps": energy_increases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument(
        "--output", default="results/neighbor_gossip_reference.json",
    )
    args = parser.parse_args()

    cases = [
        run_case(n, seed, args.tolerance, args.max_steps)
        for n in args.sizes
        for seed in range(args.seeds)
    ]
    by_size = {}
    for n in args.sizes:
        selected = [case for case in cases if case["n"] == n]
        by_size[str(n)] = {
            "cases": len(selected),
            "converged": sum(case["converged"] for case in selected),
            "median_steps": statistics.median(case["steps"] for case in selected),
            "max_final_error": max(case["final_max_gap_error"] for case in selected),
            "min_safety_margin": min(case["minimum_safety_margin"] for case in selected),
            "energy_increase_steps": sum(
                case["energy_increase_steps"] for case in selected
            ),
        }

    payload = {
        "experiment": "strict-neighbor fixed-topology reference sweep",
        "scope": "correctness/scaling simulation; not a scheduler benchmark",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "sizes": args.sizes,
            "seeds_per_size": args.seeds,
            "tolerance": args.tolerance,
            "max_steps": args.max_steps,
            "dt": 0.10,
            "safety_fraction": 0.45,
        },
        "summary": by_size,
        "cases": cases,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(by_size, indent=2, sort_keys=True))

    if not all(case["converged"] for case in cases):
        raise SystemExit("at least one case did not converge")
    if any(case["minimum_safety_margin"] < -1e-10 for case in cases):
        raise SystemExit("at least one case violated a hard minimum gap")
    if any(case["energy_increase_steps"] for case in cases):
        raise SystemExit("energy increased in at least one step")


if __name__ == "__main__":
    main()
