"""A better out-of-sample admission: per-task, confidence-calibrated margins.

The study (experiments/simulation/oos_study.py) shows the task population is heterogeneous: most
rhythms recur (median day-to-day phase agreement 0.86, 56% with R>0.8) but a
tail (~18%) drifts and a tail of peaks swings widely. A single global safety
margin must be sized for the volatile tail, so it over-provisions the
predictable majority. That is why phase-aware admission barely beats a global
mean+z-sigma test out of sample.

New method (v2): learn each task's periodic profile by a cross-day complex mean
(drifting harmonics cancel and shrink on their own), then size that task's
margin from ITS OWN cross-day prediction error, not a global within-day sigma.
Stable tasks get tight margins and pack densely; volatile tasks get wide
margins and stay safe. Node admits a set when, per resource,

    max_t [ M + sum_k Re(Z_k e^{...}) ] + z * sqrt( sum_i s_i^2 ) <= 1,

with s_i the per-task day-to-day residual scale (v2) instead of the within-day
noise sigma (current phase-aware) or a global sigma (phase-blind, K=0).

Compares peak / phase-blind / phase-aware(current) / v2 / oracle on the real
multi-day trace. Writes results/oos_v2.json.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAP = 1.0


def _phasors(day_stack, K, H):
    """day_stack (D,n,H) -> complex-mean coef (n,K), mean (n)."""
    Xf = np.fft.rfft(day_stack, axis=2)
    mean = Xf[:, :, 0].real.mean(axis=0) / H
    coef = (2 * Xf[:, :, 1:K + 1] / H).mean(axis=0)          # (n,K) complex mean
    return mean, coef


def _periodic_series(mean, coef, H):
    """Reconstruct (n,H) from mean+coef."""
    n, K = coef.shape
    t = np.arange(H)
    ang = 2 * np.pi * np.outer(np.arange(1, K + 1), t) / H   # (K,H)
    return mean[:, None] + (coef.real @ np.cos(ang) - coef.imag @ np.sin(ang))


def learn(train, K, H):
    """train (D,n,2,H). Returns per-resource features incl. margins."""
    D, n, _, _ = train.shape
    feat = {}
    for r in range(2):
        mean, coef = _phasors(train[:, :, r, :], K, H)
        prof = _periodic_series(mean, coef, H)               # (n,H) learned profile
        within = np.sqrt(np.mean((train[:, :, r, :] - prof[None]) ** 2, axis=(0, 2)))
        # cross-day residual: how far each training day's peak deviates from the
        # learned profile's peak, per task (the predictive error that matters OOS)
        day_peak = train[:, :, r, :].max(axis=2)             # (D,n)
        prof_peak = prof.max(axis=1)                         # (n,)
        cross = np.std(day_peak - prof_peak[None], axis=0)   # (n,)
        feat[r] = dict(mean=mean, coef=coef, prof_peak=prof_peak,
                       within=within, cross=cross)
    return feat


def _fits(node, feat, i, policy, z, K, H, grid=48):
    ok = True
    for r in (0, 1):
        f = feat[r]
        M = node[r]["M"] + f["mean"][i]
        if policy == "peak":
            load = node[r]["peak"] + f["prof_peak"][i]       # requests = profile peak
            ok &= load <= CAP; node_slack = CAP - load
            node[r]["_M"] = M; node[r]["_peak"] = load; continue
        Z = node[r]["Z"] + (f["coef"][i] if K > 0 else 0)
        # periodic peak over grid
        if K > 0:
            t = np.linspace(0, H, grid, endpoint=False)
            ang = 2 * np.pi * np.outer(np.arange(1, K + 1), t) / H
            pk = float((M + (Z.real @ np.cos(ang) - Z.imag @ np.sin(ang))).max())
        else:
            pk = M
        if policy == "blind":
            var = node[r]["within"] + f["within"][i] ** 2
        elif policy == "aware":
            var = node[r]["within"] + f["within"][i] ** 2    # within-day sigma (current)
        else:  # v2
            var = node[r]["cross"] + f["cross"][i] ** 2      # per-task cross-day sigma
        load = pk + z * math.sqrt(var)
        ok &= load <= CAP
        node[r]["_M"], node[r]["_Z"], node[r]["_pk"] = M, Z, pk
        node[r]["_within"] = node[r]["within"] + f["within"][i] ** 2
        node[r]["_cross"] = node[r]["cross"] + f["cross"][i] ** 2
        node[r]["_peak"] = node[r]["peak"] + f["prof_peak"][i]
    return ok


def _commit(node, feat, i, policy, K):
    for r in (0, 1):
        f = feat[r]
        node[r]["M"] += f["mean"][i]
        node[r]["peak"] += f["prof_peak"][i]
        node[r]["within"] += f["within"][i] ** 2
        node[r]["cross"] += f["cross"][i] ** 2
        if K > 0 and policy not in ("peak",):
            node[r]["Z"] = node[r]["Z"] + f["coef"][i]
        node[r]["members"].append(i)


def _new(K):
    return {r: {"M": 0.0, "peak": 0.0, "within": 0.0, "cross": 0.0,
                "Z": np.zeros(K, complex), "members": []} for r in (0, 1)}


def place(test, feat, policy, z, K, H, n_tasks, seed):
    rng = np.random.default_rng(seed)
    n = test.shape[0]
    order = rng.choice(n, min(n_tasks, n), replace=False)
    nodes = []
    for i in order:
        placed = False
        for nd in nodes:
            if _fits(nd, feat, i, policy, z, K, H):
                _commit(nd, feat, i, policy, K); placed = True; break
        if not placed:
            nd = _new(K); _commit(nd, feat, i, policy, K); nodes.append(nd)
    over = tot = 0
    for nd in nodes:
        real = test[nd[0]["members"]].sum(axis=0)            # (2,H) real test day
        over += int((real > CAP + 1e-9).sum()); tot += real.size
    return len(nodes), over / max(1, tot)


def _nodes_at_overload(frontier, target):
    """Interpolate nodes needed to hit `target` overload from a (nodes,over) set
    sorted by nodes descending (more nodes -> less overload)."""
    pts = sorted(frontier, key=lambda p: p[0], reverse=True)  # nodes desc
    for (n1, o1), (n2, o2) in zip(pts, pts[1:]):
        if (o1 - target) * (o2 - target) <= 0 and o1 != o2:
            w = (target - o1) / (o2 - o1)
            return n1 + w * (n2 - n1)
    # target not bracketed: return the safest point that meets it, else NaN
    ok = [n for n, o in pts if o <= target]
    return min(ok) if ok else float("nan")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", type=int, default=3)
    ap.add_argument("--n-tasks", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--zs", nargs="+", type=float, default=[3, 4, 5, 6, 8])
    ap.add_argument("--target", type=float, default=0.5, help="overload %% to match")
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "gct_days.npz"),
                    help="npz with series/full arrays")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "oos_v2.json"))
    args = ap.parse_args()
    d = np.load(args.data, allow_pickle=True)
    days = d["series"]; full = d["full"]; H = days.shape[3]
    D = args.history
    policies = [("blind", "blind", 0), ("aware", "aware", args.K),
                ("v2", "v2", args.K)]
    # frontier[policy] = list of (nodes, overload%) across z
    frontier = {p[0]: [] for p in policies}
    peak_nodes = []
    for z in args.zs:
        acc = {p[0]: {"n": [], "o": []} for p in policies}
        for test in range(D, days.shape[0]):
            cand = np.all(full[test - D:test + 1], axis=0)
            if cand.sum() < 100:
                continue
            train = days[test - D:test][:, cand]; testday = days[test][cand]
            nt = min(args.n_tasks, cand.sum())
            feat = learn(train, args.K, H)
            if z == args.zs[0]:
                for s in range(args.seeds):
                    pn, _ = place(testday, feat, "peak", 0, 0, H, nt, s)
                    peak_nodes.append(pn)
            for name, pol, K in policies:
                for s in range(args.seeds):
                    n, o = place(testday, feat, pol, z, K, H, nt, s)
                    acc[name]["n"].append(n); acc[name]["o"].append(o)
        for name in acc:
            if acc[name]["n"]:
                frontier[name].append((round(float(np.mean(acc[name]["n"])), 1),
                                       round(100 * float(np.mean(acc[name]["o"])), 3)))
    tgt = args.target
    at = {name: round(float(_nodes_at_overload(frontier[name], tgt)), 1)
          for name in frontier}
    pk = round(float(np.mean(peak_nodes)), 1) if peak_nodes else float("nan")
    out = {"provenance": {"history": D, "K": args.K, "zs": args.zs,
                          "n_tasks": args.n_tasks, "seeds": args.seeds,
                          "target_overload_pct": tgt,
                          "dataset": os.path.basename(args.data),
                          "data_path": args.data},
           "frontier": frontier, "peak_nodes": pk,
           "nodes_at_target": at,
           "v2_saving_vs_blind_pct": round(100 * (at["blind"] - at["v2"]) / at["blind"], 1)
           if at.get("blind") and at["blind"] == at["blind"] else None,
           "v2_saving_vs_peak_pct": round(100 * (pk - at["v2"]) / pk, 1)
           if pk == pk else None}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps({"nodes_at_%.2f%%_overload" % tgt: at, "peak": pk,
                      "v2_vs_blind_%": out["v2_saving_vs_blind_pct"],
                      "frontier": frontier}, indent=2))


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
