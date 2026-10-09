"""Out-of-sample placement: learn a bubble from history, judge on an unseen day.

Each policy learns per-task phasors by averaging each harmonic's complex
coefficient across the D days before a test day (a rhythm whose phase is stable
keeps its amplitude; one that drifts cancels and shrinks). The noise variance
is the variance of all D training days around the learned periodic profile, so
day-to-day phase drift is charged to the safety margin. The learned bubble is
then used for admission on the test day, whose real series decides overload.

Requires the multi-day artifact from experiments/data_prep/extract_gct_days.py. Writes
results/oos_placement_h{D}.json. This is the most data-hungry experiment (many
trace parts); until the multi-day artifact is built it is a runnable stub.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from experiments.simulation.raw_placement import _periodic_peak, task_features, CAP
from experiments.simulation.gct_common import ROOT


def place_peak(test_day, n_tasks, seed):
    """Kubernetes baseline: requests = peak, first-fit. Zero overload by
    construction (summed peaks <= capacity)."""
    rng = np.random.default_rng(seed)
    N, H = test_day.shape[0], test_day.shape[2]
    order = rng.choice(N, min(n_tasks, N), replace=False)
    peaks = test_day.max(axis=2)                         # (N,2)
    nodes = []
    for idx in order:
        for nd in nodes:
            if (nd + peaks[idx] <= CAP).all():
                nd += peaks[idx]; break
        else:
            nodes.append(peaks[idx].copy())
    real_over = 0
    return {"nodes": len(nodes), "overload": 0.0}


def learn_bubble(train_days, K, H):
    """train_days: (D,N,2,H). Return learned mean, Z-phasors, noise var."""
    D, N = train_days.shape[:2]
    KK = min(K, H // 2)
    X = np.fft.rfft(train_days, axis=3)                  # (D,N,2,H//2+1)
    mean = X[:, :, :, 0].real.mean(axis=0) / H           # (N,2)
    amp = np.zeros((N, 2, max(KK, 1))); ph = np.zeros((N, 2, max(KK, 1)))
    for k in range(1, KK + 1):
        coef = (2 * X[:, :, :, k] / H).mean(axis=0)      # complex mean over days
        amp[:, :, k - 1] = np.abs(coef)
        ph[:, :, k - 1] = np.angle(coef)
    # residual variance around the learned periodic profile over all train days
    t = np.arange(H)
    recon = np.repeat(mean[:, :, None], H, axis=2).astype(float)
    for k in range(1, KK + 1):
        recon += amp[:, :, k - 1][:, :, None] * np.cos(
            2 * np.pi * k * t / H + ph[:, :, k - 1][:, :, None])
    nvar = np.mean([np.var(train_days[d] - recon, axis=2) for d in range(D)],
                   axis=0)
    return dict(mean=mean, amp=amp, ph=ph, noise_var=nvar, KK=KK)


def place_oos(learned, test_day, z, n_tasks, seed):
    rng = np.random.default_rng(seed)
    N, H = test_day.shape[0], test_day.shape[2]
    order = rng.choice(N, min(n_tasks, N), replace=False)
    KK = learned["KK"]
    nodes = []
    for idx in order:
        placed = False
        for nd in nodes:
            M = nd["M"] + learned["mean"][idx]
            Z = nd["Z"] + (learned["amp"][idx] * np.exp(1j * learned["ph"][idx])
                           if KK > 0 else 0)
            nvar = nd["nvar"] + learned["noise_var"][idx]
            load = _periodic_peak(M, Z, KK, H) + z * np.sqrt(nvar)
            if (load <= CAP).all():
                nd["M"], nd["Z"], nd["nvar"] = M, Z, nvar
                nd["members"].append(idx); placed = True
                break
        if not placed:
            Z0 = (learned["amp"][idx] * np.exp(1j * learned["ph"][idx])
                  if KK > 0 else np.zeros((2, max(KK, 1)), complex))
            nodes.append({"M": learned["mean"][idx].copy(), "Z": Z0,
                          "nvar": learned["noise_var"][idx].copy(),
                          "members": [idx]})
    over = tot = 0
    for nd in nodes:
        real = test_day[nd["members"]].sum(axis=0)
        over += int((real > CAP + 1e-9).sum()); tot += real.size
    return {"nodes": len(nodes), "overload": over / max(1, tot)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days-file", default=os.path.join(ROOT, "data", "gct_days.npz"))
    ap.add_argument("--history", type=int, default=7)
    ap.add_argument("--n-tasks", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--K", type=int, default=24)
    args = ap.parse_args()
    if not os.path.exists(args.days_file):
        print(f"[stub] multi-day artifact {args.days_file} not built yet; "
              f"run experiments/data_prep/extract_gct_days.py first. No OOS result written.")
        return
    d = np.load(args.days_file, allow_pickle=True)
    days = d["series"]                                   # (Ndays,M,2,H)
    full = d["full"]                                     # (Ndays,M) bool
    prov = json.loads(str(d["provenance"]))
    H = days.shape[3]
    D = args.history
    results = {}
    windows = []

    def record(key, rs):
        results.setdefault(key, []).append(
            {"nodes": float(np.mean([r["nodes"] for r in rs])),
             "overload": float(np.mean([r["overload"] for r in rs]))})

    for test in range(D, days.shape[0]):
        # tasks present in every slot on each day of the [test-D, test] window
        cand = np.all(full[test - D:test + 1], axis=0)
        ncand = int(cand.sum())
        if ncand < 100:
            continue
        windows.append({"test_day": test, "candidate_tasks": ncand})
        train = days[test - D:test][:, cand]             # (D,ncand,2,H)
        testday = days[test][cand]                       # (ncand,2,H)
        nt = min(args.n_tasks, ncand)
        learned = learn_bubble(train, args.K, H)          # phase-aware
        blind = learn_bubble(train, 0, H)                 # phase-blind
        oracle = task_features(testday, args.K, H)        # knows test day
        sd = list(range(args.seeds))
        record("requests_peak", [place_peak(testday, nt, s) for s in sd])
        for z in (3, 4):
            record(f"kuramoto_K{args.K}_z{z}",
                   [place_oos(learned, testday, z, nt, s) for s in sd])
        record("phase_blind_z4", [place_oos(blind, testday, 4, nt, s) for s in sd])
        record("oracle_testday_z3", [place_oos(oracle, testday, 3, nt, s) for s in sd])
    if not results:
        print("no test window had >=100 tasks full across the history; "
              "try a smaller --history or more days.")
        return
    summary = {k: {"nodes": float(np.mean([x["nodes"] for x in v])),
                   "overload": float(np.mean([x["overload"] for x in v]))}
               for k, v in results.items()}
    peak_nodes = summary["requests_peak"]["nodes"]
    kur = summary.get(f"kuramoto_K{args.K}_z4", summary.get(f"kuramoto_K{args.K}_z3"))
    summary["_node_saving_vs_peak_pct"] = round(
        100 * (peak_nodes - kur["nodes"]) / peak_nodes, 1)
    out = {"provenance": {**prov, "history": D, "n_tasks": args.n_tasks,
                          "seeds": args.seeds, "test_windows": windows,
                          "n_test_days": len(windows)}, "summary": summary}
    path = os.path.join(ROOT, "results", f"oos_placement_h{D}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(summary, indent=2), "->", path)


if __name__ == "__main__":
    main()
