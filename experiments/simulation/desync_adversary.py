"""Adversarial-resilience sweep on the real strict-neighbor kernel.

Uses the committed kernel primitives (local_correction, limit_local_displacement,
forward_gap from kadence.neighbor_gossip), so honest jobs run exactly the deployed
update and the safety limiter is exactly the deployed defense. Malicious jobs
mount a phase-spoofing / slot-stealing attack: each reports to its neighbors a
phase shifted toward its successor, so the successor retreats and the attacker's
true slot grows. We measure, versus the malicious fraction, how much slot an
attacker steals (slot inflation, 1.0 = fair share) and how honest fairness
holds, with the rate-limit defense ON vs OFF. Writes results/desync_adversary.json.

Thesis check: the same per-round safety limiter that bounds joint displacement
under churn should also bound how fast a liar can steal, flattening the attack.
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
    BubbleSpec, admit_ring, NeighborSnapshot, forward_gap,
    local_correction, limit_local_displacement, phases_from_gaps,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TWO_PI = 2 * math.pi
SPOOF = 0.45          # attacker claims it is 45% further along its true gap


def _snap(jid, phase):
    return NeighborSnapshot(jid=str(jid), phase=phase % TWO_PI, width=0.0,
                            epoch=1, sequence=0)


def run(n, frac_adv, defense, dt=0.1, rounds=3000, seed=0):
    rng = np.random.default_rng(seed)
    bubbles = [BubbleSpec(f"j{i}", 1e-4) for i in range(n)]
    plan = admit_ring(bubbles, slack_weights=[1.0] * n)      # uniform desired
    desired = list(plan.desired)
    minimum = list(plan.minimum)
    # start near the even splay so we isolate the attack, not convergence
    g = np.array(desired) * rng.uniform(0.9, 1.1, n)
    g = (g / g.sum() * TWO_PI).tolist()
    g[-1] = TWO_PI - math.fsum(g[:-1])
    phases = list(phases_from_gaps(g, circumference=TWO_PI))
    n_adv = int(frac_adv * n)
    # spread adversaries evenly so they are not adjacent
    adv = set((np.arange(n_adv) * (n // max(1, n_adv))) % n) if n_adv else set()

    # Continuity detector (defense only): a neighbor's advertised phase may not
    # move more than the honesty envelope per round, because honest jobs are
    # themselves rate-bounded by the safety limiter. prev_good is seeded with
    # the admission-attested initial phase (epoch-authenticated).
    def circdelta(a, b):
        return ((a - b + math.pi) % TWO_PI) - math.pi
    prev_good = list(phases)
    envelope = 0.15 * (TWO_PI / n)

    for _ in range(rounds):
        order = np.argsort([p % TWO_PI for p in phases])
        pos_of = {int(j): k for k, j in enumerate(order)}
        # exposed phase each job advertises (adversaries spoof toward successor)
        exposed = [phases[j] for j in range(n)]
        for j in adv:
            k = pos_of[j]
            succ = phases[int(order[(k + 1) % n])]
            fg = forward_gap(phases[j], succ, TWO_PI)
            exposed[j] = (phases[j] + SPOOF * fg) % TWO_PI
        # detector: accept an advertised move only within the envelope, else
        # pin the neighbor to its last attested phase (spoof neutralized)
        if defense:
            seen = list(exposed)
            for m in range(n):
                if abs(circdelta(exposed[m], prev_good[m])) <= envelope:
                    prev_good[m] = exposed[m]
                else:
                    seen[m] = prev_good[m]
        else:
            seen = exposed
        new_phases = list(phases)
        for k, jj in enumerate(order):
            j = int(jj)
            if j in adv:
                continue                                     # attacker holds + keeps spoofing
            li = int(order[(k - 1) % n]); ri = int(order[(k + 1) % n])
            left = _snap(li, seen[li])
            cur = _snap(j, phases[j])
            right = _snap(ri, seen[ri])
            corr = local_correction(left, cur, right,
                                    left_target=desired[(k - 1) % n],
                                    right_target=desired[k],
                                    circumference=TWO_PI)
            req = dt * corr
            if defense:
                disp = limit_local_displacement(
                    req, left, cur, right,
                    left_minimum=minimum[(k - 1) % n], right_minimum=minimum[k],
                    safety_fraction=0.45, circumference=TWO_PI)
            else:
                # no rate limit: only keep the ring ordered (do not cross neighbors)
                rgap = forward_gap(phases[j], phases[ri], TWO_PI)
                lgap = forward_gap(phases[li], phases[j], TWO_PI)
                disp = float(np.clip(req, -0.98 * lgap, 0.98 * rgap))
            new_phases[j] = (phases[j] + disp) % TWO_PI
        phases = new_phases

    # measure true forward gaps per job
    order = np.argsort([p % TWO_PI for p in phases])
    gaps = np.array([forward_gap(phases[int(order[k])],
                                 phases[int(order[(k + 1) % n])], TWO_PI)
                     for k in range(n)])
    fair = TWO_PI / n
    adv_pos = [k for k in range(n) if int(order[k]) in adv]
    hon_pos = [k for k in range(n) if int(order[k]) not in adv]
    infl = float(np.mean(gaps[adv_pos]) / fair) if adv_pos else 1.0
    hon = gaps[hon_pos]
    jain = float(hon.sum() ** 2 / (len(hon) * np.sum(hon ** 2))) if len(hon) else 1.0
    return {"attacker_slot_inflation": round(infl, 3),
            "honest_jain": round(jain, 3),
            "honest_min_ratio": round(float(hon.min() / fair), 3) if len(hon) else 1.0}


def main():
    n = 200
    out = {"provenance": {"kernel": "core.neighbor_gossip primitives",
                          "n": n, "spoof_fraction_of_gap": SPOOF, "rounds": 1000,
                          "attack": "phase-spoofing / slot-stealing",
                          "defense": "rate-limit + continuity detector (envelope 0.15 fair-gap)"},
           "fractions": []}
    for frac in (0.0, 0.05, 0.10, 0.20, 0.30):
        row = {"malicious_frac": frac}
        for defense in (False, True):
            reps = [run(n, frac, defense, seed=s, rounds=1000) for s in range(3)]
            key = "defense" if defense else "no_defense"
            row[key] = {m: round(float(np.mean([r[m] for r in reps])), 3)
                        for m in ("attacker_slot_inflation", "honest_jain",
                                  "honest_min_ratio")}
        out["fractions"].append(row)
        d0, d1 = row["no_defense"], row["defense"]
        print(f"f={frac:.2f}  no-def infl={d0['attacker_slot_inflation']} "
              f"jain={d0['honest_jain']} | def infl={d1['attacker_slot_inflation']} "
              f"jain={d1['honest_jain']}", flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "desync_adversary.json"), "w"),
              indent=2)
    print("wrote results/desync_adversary.json")


if __name__ == "__main__":
    main()
