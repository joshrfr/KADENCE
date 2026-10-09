"""Coupling-radius sweep: does reading more than two neighbors help?

The committed rule couples each job to its two immediate ring neighbors (k=1 on
each side), which is exactly gradient descent on the spacing energy because a gap
touches only adjacent phases. This experiment widens the coupling to k neighbors
on each side and measures the trade-off: convergence rounds and final gap error
versus the message cost, which is 2k messages per job per round. A wider window
averages the gap-error signal over more gaps per step, shrinking the effective
diameter of the ring, so we expect fewer rounds at a higher message cost. k=1
reproduces the committed strict-neighbor rule exactly.

Writes results/coupling_radius.json. Every number is produced here.
"""
from __future__ import annotations
import json, math, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TWO_PI = 2 * math.pi


def settle_k(n, k, dt=0.1, sf=0.45, tol=1e-6, max_rounds=4_000_000, seed=0):
    """Windowed repulsion over k neighbors each side; k=1 == committed rule."""
    rng = np.random.default_rng(seed)
    target = TWO_PI / n
    g = target * rng.uniform(0.4, 1.6, n)
    g *= TWO_PI / g.sum()
    einc = 0; prev = math.inf
    for r in range(max_rounds):
        e = g - target
        if np.max(np.abs(e)) < tol:
            return r + 1, float(np.max(np.abs(e))), einc
        # windowed signal: (mean gap-error of k gaps ahead) - (k gaps behind)
        ahead = np.zeros(n); behind = np.zeros(n)
        for j in range(k):
            ahead += np.roll(e, -j)           # e_i, e_{i+1}, ... e_{i+k-1}
            behind += np.roll(e, j + 1)        # e_{i-1}, e_{i-2}, ... e_{i-k}
        corr = (ahead - behind) / k
        disp = np.clip(dt * corr, -sf * np.roll(g, 1), sf * g)
        g = g + (np.roll(disp, -1) - disp)
        energy = float(np.sum(e * e))
        if energy > prev + 1e-15:
            einc += 1
        prev = energy
    return max_rounds, float(np.max(np.abs(g - target))), einc


def main():
    out = {"populations": [], "note": "k=1 is the committed two-neighbor rule; "
           "messages/round = 2k"}
    for n in (50, 200):
        row = {"n": n, "by_k": []}
        for k in (1, 2, 3, 4):
            rounds, gerr, einc = settle_k(n, k)
            row["by_k"].append({"k": k, "neighbors_each_side": k,
                                "messages_per_round": 2 * k,
                                "settling_rounds": rounds,
                                "final_gap_error": gerr,
                                "energy_increase_rounds": einc})
        base = row["by_k"][0]["settling_rounds"]
        for e in row["by_k"]:
            e["rounds_vs_k1"] = round(e["settling_rounds"] / base, 3)
        out["populations"].append(row)
        print(f"n={n}: " + ", ".join(
            f"k={e['k']}:{e['settling_rounds']}r({e['rounds_vs_k1']}x,{e['messages_per_round']}msg)"
            for e in row["by_k"]))
    json.dump(out, open(os.path.join(ROOT, "results", "coupling_radius.json"), "w"), indent=2)
    print("wrote results/coupling_radius.json")


if __name__ == "__main__":
    main()
