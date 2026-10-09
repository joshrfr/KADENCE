"""Safety/cost frontier for phase-inference admission vs simple peak reservation.

Sweeps the trust threshold tau and safety margin z. For each operating point we
record mean nodes used and the worst-case real overload across held-out days. A
phase-inference point only beats peak reservation if it uses fewer nodes AND
keeps overload at ~0. Peak reservation is the anchor (0 overload by
construction). Writes results/phase_frontier.json and prints whether any point
dominates. Every number is produced here.
"""
from __future__ import annotations
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from experiments.simulation.oos_placement import learn_bubble, place_peak
from experiments.simulation.oos_phase_infer import dominant_confidence, place_gated
from experiments.simulation.gct_common import ROOT

DAYS = os.path.join(ROOT, "data", "gct_days.npz")
D, K, NT, SEEDS = 3, 24, 1200, 4


def main():
    d = np.load(DAYS, allow_pickle=True)
    days, full = d["series"], d["full"]
    prov = json.loads(str(d["provenance"]))
    H = days.shape[3]; KK = min(K, H // 2); seeds = list(range(SEEDS))

    # precompute per-window learned models (shared across the tau/z sweep)
    wins = []
    for test in range(D, days.shape[0]):
        cand = np.all(full[test - D:test + 1], axis=0)
        if int(cand.sum()) < 100:
            continue
        train = days[test - D:test][:, cand]; testday = days[test][cand]
        nt = min(NT, int(cand.sum()))
        learned = learn_bubble(train, K, H)
        amp, ph, conf, conf_job = dominant_confidence(train, KK, H)
        ph_job = ph.mean(axis=1)
        peak = float(np.mean([place_peak(testday, nt, s)["nodes"] for s in seeds]))
        wins.append(dict(learned=learned, conf_job=conf_job, ph_job=ph_job,
                         testday=testday, nt=nt, peak=peak))
    peak_mean = float(np.mean([w["peak"] for w in wins]))

    grid = []
    for tau in (0.5, 0.7, 0.85, 0.95, 1.01):        # 1.01 -> trust nobody (== peak)
        for z in (3, 4, 5, 6):
            nlist, olist = [], []
            for w in wins:
                rs = [place_gated(w["learned"], w["conf_job"], w["ph_job"],
                                  w["testday"], z, w["nt"], tau, s) for s in seeds]
                nlist.append(np.mean([r["nodes"] for r in rs]))
                olist.append(max(r["overload"] for r in rs))
            grid.append({"tau": tau, "z": z,
                         "nodes_mean": float(np.mean(nlist)),
                         "max_overload": float(max(olist)),
                         "nodes_vs_peak_pct": float(100 * (1 - np.mean(nlist) / peak_mean))})

    # a point "dominates" peak reservation if it is safe (<=0.5% overload) and cheaper
    safe = [g for g in grid if g["max_overload"] <= 0.005]
    dominating = [g for g in safe if g["nodes_mean"] < peak_mean - 1e-6]
    best_safe = min(safe, key=lambda g: g["nodes_mean"]) if safe else None
    out = {"dataset": prov.get("dataset"), "history": D, "K": K, "n_tasks": NT,
           "seeds": SEEDS, "peak_nodes_mean": peak_mean, "grid": grid,
           "safe_points": len(safe), "dominates_peak": bool(dominating),
           "best_safe_point": best_safe}
    json.dump(out, open(os.path.join(ROOT, "results", "phase_frontier.json"), "w"), indent=2)
    print(f"peak reservation: {peak_mean:.1f} nodes, 0 overload")
    print(f"safe (<=0.5% overload) phase points: {len(safe)}; dominates peak: {bool(dominating)}")
    if best_safe:
        print("best safe phase point:", json.dumps(best_safe, indent=2))
    print("wrote results/phase_frontier.json")


if __name__ == "__main__":
    main()
