"""Phase-aware admission without time shifts (v4 placement regime).

A service cannot move its daily load; the scheduler only chooses placement.
Phases are then observed, not chosen, and the node order parameter becomes an
admission test. Rhythms of co-located jobs add coherently (as phasors), while
independent noise adds in variance (as sqrt of the sum). The v4 test is

    max_t [ M^r + sum_{k<=K} Re(Z_k^r e^{j 2pi k t/H}) ] + z sqrt(sum_i sigma_i^{r2}) <= 1

for both resources. K=0 removes the phase information and leaves the standard
mean+z*sigma statistical test, the control that isolates what phase adds.

Baselines: Kubernetes requests=peak (safe, wasteful), requests=mean
(overcommit), and a full-real-series oracle. Overload is measured on the real
summed series: the fraction of node-slots above capacity. Also runs the join-
rule ablation (least order-parameter growth vs best-fit vs first-fit).

Writes results/raw_placement.json and results/raw_placement_ablation.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from experiments.simulation.gct_common import load_series, ROOT

CAP = 1.0


def task_features(series, K, H):
    """Per-task quantities used by the admission tests (fully vectorised).

    Uses one real FFT and Parseval's relation, so there are no Python loops
    over harmonics or tasks; this runs on all 82k tasks in a second or two.
    """
    N = len(series)
    X = np.fft.rfft(series, axis=2)                      # (N,2,H//2+1)
    mean = X[:, :, 0].real / H                           # (N,2)
    KK = min(K, H // 2)
    ampfull = 2 * np.abs(X[:, :, 1:]) / H                # (N,2,H//2) all harmonics
    if KK > 0:
        amp = ampfull[:, :, :KK].copy()
        ph = np.angle(X[:, :, 1:KK + 1])
    else:
        amp = np.zeros((N, 2, 1)); ph = np.zeros((N, 2, 1))
    var_total = np.var(series, axis=2)                   # (N,2)
    var_model = 0.5 * (ampfull[:, :, :KK] ** 2).sum(axis=2) if KK > 0 else 0.0
    noise_var = np.maximum(var_total - var_model, 0.0)   # power above K
    peak = series.max(axis=2)                            # (N,2)
    tail_amp = ampfull[:, :, KK:].sum(axis=2)            # sum_{k>K} a_ik
    return dict(mean=mean, amp=amp, ph=ph, noise_var=noise_var, peak=peak,
                tail_amp=tail_amp, KK=KK)


_BASIS = {}


def _basis(KK, H, grid=48):
    """Cache cos/sin(2 pi k t/H) for k=1..KK on a fixed grid (grid, KK)."""
    key = (KK, H, grid)
    if key not in _BASIS:
        ts = np.linspace(0, H, grid, endpoint=False)
        k = np.arange(1, KK + 1)
        ang = 2 * np.pi * np.outer(ts, k) / H            # (grid, KK)
        _BASIS[key] = (np.cos(ang), np.sin(ang))
    return _BASIS[key]


def _periodic_peak(M, Zsum, KK, H, grid=48):
    """max_t M + sum_k Re(Z_k e^{j2pi k t/H}) over a time grid, per resource."""
    if KK == 0:
        return M
    cos, sin = _basis(KK, H, grid)                       # (grid,KK)
    # vals[r,t] = M[r] + sum_k Re(Z)_k cos - Im(Z)_k sin
    contrib = Zsum.real @ cos.T - Zsum.imag @ sin.T      # (2,grid)
    return (M[:, None] + contrib).max(axis=1)            # (2,)


def admit_run(series, feat, rule, z, join, seed, n_tasks):
    """Place n_tasks (seeded order) by an admission rule; return nodes+overload."""
    rng = np.random.default_rng(seed)
    order = rng.choice(len(series), min(n_tasks, len(series)), replace=False)
    H = series.shape[2]
    KK = feat["KK"]
    nodes = []           # each: dict with aggregates + member list
    for idx in order:
        cand = []
        for ni, nd in enumerate(nodes):
            ok, slack = _fits(nd, feat, idx, rule, z, KK, H)
            if ok:
                cand.append((slack, ni))
        if cand:
            if join == "first":
                ni = min(c[1] for c in cand)
            elif join == "best":
                ni = min(cand)[1]                        # least slack = tightest
            else:                                        # least-Z growth
                ni = max(cand)[1]                        # most slack after add
            _add(nodes[ni], feat, idx)
        else:
            nd = _new_node(H, KK)
            _add(nd, feat, idx)
            nodes.append(nd)
    # overload on the real summed series
    over_slots = 0
    tot_slots = 0
    for nd in nodes:
        real = series[nd["members"]].sum(axis=0)         # (2,H)
        over_slots += int((real > CAP + 1e-9).sum())
        tot_slots += real.size
    return {"nodes": len(nodes),
            "overload": over_slots / max(1, tot_slots)}


def _new_node(H, KK):
    return {"members": [], "M": np.zeros(2), "Z": np.zeros((2, max(KK, 1)), complex),
            "nvar": np.zeros(2), "peak": np.zeros(2), "tail": np.zeros(2)}


def _add(nd, feat, idx):
    nd["members"].append(idx)
    nd["M"] += feat["mean"][idx]
    nd["nvar"] += feat["noise_var"][idx]
    nd["peak"] += feat["peak"][idx]
    nd["tail"] += feat["tail_amp"][idx]
    KK = feat["KK"]
    if KK > 0:
        nd["Z"] += feat["amp"][idx] * np.exp(1j * feat["ph"][idx])


def _fits(nd, feat, idx, rule, z, KK, H):
    """Would adding task idx keep the node safe under `rule`? Return (ok, slack)."""
    M = nd["M"] + feat["mean"][idx]
    if rule == "k8s_peak":
        load = nd["peak"] + feat["peak"][idx]
        slack = (CAP - load).min()
        return bool((load <= CAP).all()), slack
    if rule == "mean":
        slack = (CAP - M).min()
        return bool((M <= CAP).all()), slack
    Z = nd["Z"] + (feat["amp"][idx] * np.exp(1j * feat["ph"][idx]) if KK > 0
                   else 0)
    nvar = nd["nvar"] + feat["noise_var"][idx]
    if rule == "worst_bound":
        tail = nd["tail"] + feat["tail_amp"][idx]
        load = M + np.abs(Z).sum(axis=1) + tail
        slack = (CAP - load).min()
        return bool((load <= CAP).all()), slack
    # rule == "zsigma": phase-aware (KK>0) or phase-blind (KK==0)
    pp = _periodic_peak(M, Z, KK, H)
    load = pp + z * np.sqrt(nvar)
    slack = (CAP - load).min()
    return bool((load <= CAP).all()), slack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tasks", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    series, ids, prov = load_series()
    H = series.shape[2]
    feat0 = task_features(series, 0, H)
    feat4 = task_features(series, 4, H)
    feat24 = task_features(series, 24, H)

    def avg(rule, feat, z, join="lz"):
        rs = [admit_run(series, feat, rule, z, join, s, args.n_tasks)
              for s in range(args.seeds)]
        return {"nodes": float(np.mean([r["nodes"] for r in rs])),
                "overload": float(np.mean([r["overload"] for r in rs]))}

    res = {"provenance": {**prov, "n_tasks": args.n_tasks, "seeds": args.seeds},
           "rules": {
               "k8s_peak": avg("k8s_peak", feat0, 0),
               "mean_overcommit": avg("mean", feat0, 0),
               "worst_bound_K24": avg("worst_bound", feat24, 0),
               "phase_blind_z3": avg("zsigma", feat0, 3),
               "phase_blind_z4": avg("zsigma", feat0, 4),
               "phase_blind_z5": avg("zsigma", feat0, 5),
               "kuramoto_K4_z3": avg("zsigma", feat4, 3),
               "kuramoto_K4_z4": avg("zsigma", feat4, 4),
               "kuramoto_K24_z3": avg("zsigma", feat24, 3),
               "kuramoto_K24_z4": avg("zsigma", feat24, 4),
           }}
    path = os.path.join(ROOT, "results", "raw_placement.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print("placement:", json.dumps(res["rules"], indent=2), "->", path)

    abl = {"provenance": res["provenance"], "K24_z4": {
        j: avg("zsigma", feat24, 4, join=j) for j in ("lz", "best", "first")}}
    apath = os.path.join(ROOT, "results", "raw_placement_ablation.json")
    with open(apath, "w") as f:
        json.dump(abl, f, indent=2)
    print("ablation:", json.dumps(abl["K24_z4"]), "->", apath)


if __name__ == "__main__":
    main()
