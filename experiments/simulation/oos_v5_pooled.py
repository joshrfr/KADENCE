"""Out-of-sample admission v5: calibrated cooperative-pooling quantile reservation.

Round-2 attempt to finally beat the baselines on the NOW-8-DAY Google trace.

Setting (identical harness to oos_v2 / oos_v3, run on a copied npz to avoid
races): for each test day learn only from the prior `history` days; per window
use tasks present in every slot of every day in [test-history, test]; greedily
first-fit N=1000 tasks onto nodes with a per-resource cap of 1.0 (CPU and mem);
a node admits a set if the per-resource PREDICTED requirement stays <= 1.0;
overload = fraction of node-slots where the SUMMED REAL test-day load exceeds
1.0. Sweep the safety knob, trace the frontier, report NODES AT 0.5% OVERLOAD
averaged over all usable test windows.

Baselines (computed inline on the same data, lower nodes = better):
  * peak-requests        : reserve each task's max requirement over history.
  * phase-blind mean+zσ  : reserve summed mean + z * sqrt(sum within-day var).
  * phase-aware Fourier   : same but on a K-harmonic periodic profile peak.

v5 method (SOTA recipe: Google Autopilot / MS Resource Central per-entity
quantile recommenders + prior cooperative pooling variance discount):

  1. PER-TASK PER-SLOT DISTRIBUTION.  For each task and slot, build the demand
     distribution over the history days, recency-weighted (half-life), and
     reserve a per-task quantile q_i of it -- NOT a mean plus one global z.

  2. CALIBRATE q TO THE OVERLOAD BUDGET (the cooperative-pooling step).  On a
     validation split of the history days (hold out the most recent history day
     as pseudo-test, learn from the earlier ones), raise a single global
     quantile level q until the POOLED (summed) out-of-sample overload on that
     validation day just meets the 0.5% target with the smallest reservation.
     Because independent per-task tail reservations are SUMMED (not fleet-wide
     margined), the pooled tail concentrates: the calibrated q needed is lower
     than a per-task worst case, so nodes pack denser at the same risk.  We
     search a grid of q and pick the smallest q whose validation pooled overload
     <= target (interpolating between grid points), giving one calibrated q per
     window that is then applied to the held-out TEST day.

  3. REGIME GATE.  Per task, if the most recent history day diverges from the
     earlier-day mean beyond a threshold (mean-normalised L1), predict that task
     from the recent day alone so drifted tasks do not carry stale tails.

The frontier knob for v5 is the TARGET the calibration aims at (we sweep a set
of validation targets so the frontier is traced the same way as the baselines'
z / the raw quantile), then read nodes at the 0.5% test overload.

The test day is strictly held out: calibration only ever sees the
history days.  If v5 does not beat peak-requests we say so and report the margin
vs phase-blind / phase-aware.  Writes results/oos_v5_pooled.json.
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


# ---------------------------------------------------------------------------
# Baseline features (peak / blind / aware) -- lifted verbatim in spirit from
# oos_v2 / oos_v3 so numbers are computed on the SAME copied data.
# ---------------------------------------------------------------------------
def _phasors(day_stack, K, H):
    Xf = np.fft.rfft(day_stack, axis=2)
    mean = Xf[:, :, 0].real.mean(axis=0) / H
    coef = (2 * Xf[:, :, 1:K + 1] / H).mean(axis=0)
    return mean, coef


def _periodic_series(mean, coef, H):
    n, K = coef.shape
    t = np.arange(H)
    ang = 2 * np.pi * np.outer(np.arange(1, K + 1), t) / H
    return mean[:, None] + (coef.real @ np.cos(ang) - coef.imag @ np.sin(ang))


def learn_baselines(train, K, H):
    _, n, _, _ = train.shape
    feat = {}
    for r in range(2):
        mean, coef = _phasors(train[:, :, r, :], K, H)
        prof = _periodic_series(mean, coef, H)
        within = np.sqrt(np.mean((train[:, :, r, :] - prof[None]) ** 2, axis=(0, 2)))
        prof_peak = prof.max(axis=1)
        feat[r] = dict(mean=mean, coef=coef, prof_peak=prof_peak, within=within)
    return feat


# ---------------------------------------------------------------------------
# v5: per-task per-slot recency-weighted quantile with a regime gate.
# ---------------------------------------------------------------------------
def _weighted_percentile_days(X, w, q):
    """X (D,n,H), weights w (D,) -> weighted q-th percentile over D, per (n,H)."""
    order = np.argsort(X, axis=0)
    Xs = np.take_along_axis(X, order, axis=0)
    ws = np.take_along_axis(np.broadcast_to(w[:, None, None], X.shape), order, axis=0)
    cw = np.cumsum(ws, axis=0)
    total = cw[-1]
    tgt = (q / 100.0) * total
    ge = cw >= tgt[None]
    idx = np.argmax(ge, axis=0)
    return np.take_along_axis(Xs, idx[None], axis=0)[0]


def _regime_gate(X, cp_thresh):
    """X (D,n,H) -> boolean (n,) tasks whose most-recent day drifted."""
    D = X.shape[0]
    if D < 2:
        return np.zeros(X.shape[1], bool)
    recent = X[-1]
    earlier = X[:-1].mean(axis=0)
    denom = np.maximum(earlier.mean(axis=1), 1e-6)
    div = np.abs(recent - earlier).mean(axis=1) / denom
    return div > cp_thresh


def learn_v5(train, q, H, half_life, cp_thresh):
    """train (D,n,2,H) at quantile q -> per-resource predicted (n,H)."""
    D = train.shape[0]
    ages = np.arange(D)[::-1]
    w = 0.5 ** (ages / max(1e-9, half_life))
    feat = {}
    for r in range(2):
        X = train[:, :, r, :]
        pred = _weighted_percentile_days(X, w, q)
        gate = _regime_gate(X, cp_thresh)
        if gate.any():
            pred[gate] = X[-1][gate]
        feat[r] = dict(pred=np.ascontiguousarray(pred))
    return feat


# ---------------------------------------------------------------------------
# Placement (first-fit) -- shared shape with oos_v3.
# ---------------------------------------------------------------------------
def _new_node(K, H, is_vec):
    if is_vec:
        return {r: {"sum": np.zeros(H)} for r in (0, 1)} | {"members": []}
    return {r: {"M": 0.0, "peak": 0.0, "within": 0.0,
                "Z": np.zeros(K, complex)} for r in (0, 1)} | {"members": []}


def _fits_baseline(node, feat, i, policy, z, K, H, grid=48):
    stash = {}
    for r in (0, 1):
        f = feat[r]
        if policy == "peak":
            load = node[r]["peak"] + f["prof_peak"][i]
            if load > CAP:
                return False, None
            stash[r] = dict(peak=load)
            continue
        M = node[r]["M"] + f["mean"][i]
        Z = node[r]["Z"] + (f["coef"][i] if K > 0 else 0)
        if K > 0:
            t = np.linspace(0, H, grid, endpoint=False)
            ang = 2 * np.pi * np.outer(np.arange(1, K + 1), t) / H
            pk = float((M + (Z.real @ np.cos(ang) - Z.imag @ np.sin(ang))).max())
        else:
            pk = M
        var = node[r]["within"] + f["within"][i] ** 2
        load = pk + z * math.sqrt(var)
        if load > CAP:
            return False, None
        stash[r] = dict(M=M, Z=Z, within=var)
    return True, stash


def _commit_baseline(node, feat, i, policy, K, stash):
    for r in (0, 1):
        f = feat[r]
        if policy == "peak":
            node[r]["peak"] = stash[r]["peak"]
        else:
            node[r]["M"] = stash[r]["M"]
            node[r]["within"] = stash[r]["within"]
            if K > 0:
                node[r]["Z"] = stash[r]["Z"]
            node[r]["peak"] += f["prof_peak"][i]
    node["members"].append(i)


def _fits_vec(node, feat, i):
    new0 = node[0]["sum"] + feat[0]["pred"][i]
    if new0.max() > CAP:
        return False, None
    new1 = node[1]["sum"] + feat[1]["pred"][i]
    if new1.max() > CAP:
        return False, None
    return True, (new0, new1)


def place(testday, feat, policy, z, K, H, n_tasks, seed):
    """Returns (nodes, real overload fraction on `testday`)."""
    rng = np.random.default_rng(seed)
    n = testday.shape[0]
    order = rng.choice(n, min(n_tasks, n), replace=False)
    is_vec = policy == "v5"
    nodes = []
    for i in order:
        placed = False
        for nd in nodes:
            if is_vec:
                ok, stash = _fits_vec(nd, feat, i)
                if ok:
                    nd[0]["sum"], nd[1]["sum"] = stash
                    nd["members"].append(i)
                    placed = True
                    break
            else:
                ok, stash = _fits_baseline(nd, feat, i, policy, z, K, H)
                if ok:
                    _commit_baseline(nd, feat, i, policy, K, stash)
                    placed = True
                    break
        if not placed:
            nd = _new_node(K, H, is_vec)
            if is_vec:
                nd[0]["sum"] = feat[0]["pred"][i].copy()
                nd[1]["sum"] = feat[1]["pred"][i].copy()
                nd["members"].append(i)
            else:
                stash = {}
                for r in (0, 1):
                    f = feat[r]
                    if policy == "peak":
                        stash[r] = dict(peak=f["prof_peak"][i])
                    else:
                        stash[r] = dict(M=f["mean"][i],
                                        Z=(f["coef"][i] if K > 0 else np.zeros(K, complex)),
                                        within=f["within"][i] ** 2)
                _commit_baseline(nd, feat, i, policy, K, stash)
            nodes.append(nd)
    over = tot = 0
    for nd in nodes:
        real = testday[nd["members"]].sum(axis=0)
        over += int((real > CAP + 1e-9).sum())
        tot += real.size
    return len(nodes), over / max(1, tot)


# ---------------------------------------------------------------------------
# Calibration: pick the smallest global quantile q whose POOLED out-of-sample
# overload on a validation split of the HISTORY days meets `val_target`.
# The validation split holds out the most-recent history day as pseudo-test and
# learns from the earlier history days -- the test day is never touched.
# ---------------------------------------------------------------------------
def calibrate_q(train, H, half_life, cp_thresh, val_target, q_grid,
                n_tasks, seeds):
    """train (D,n,2,H). Returns calibrated q (float).

    D-1 earlier days -> learn; the most recent history day -> pseudo-test.
    Needs D>=2; caller guarantees history>=2.
    """
    D = train.shape[0]
    inner_train = train[:-1]                      # (D-1,n,2,H)
    pseudo_test = train[-1]                       # (n,2,H)
    best_q = q_grid[-1]
    prev = None                                   # (q, over_pct)
    for q in q_grid:
        fv = learn_v5(inner_train, q, H, half_life, cp_thresh)
        os_ = []
        for s in range(seeds):
            _, o = place(pseudo_test, fv, "v5", 0, 0, H, n_tasks, s)
            os_.append(o)
        over = 100.0 * float(np.mean(os_))
        if over <= val_target:
            if prev is not None and prev[1] > val_target and prev[1] != over:
                # interpolate q between prev (too hot) and this (safe)
                pq, po = prev
                w = (val_target - po) / (over - po) if over != po else 0.0
                # over<=target<po: move from q toward pq
                return float(q + w * (pq - q))
            return float(q)
        prev = (q, over)
    return float(best_q)                          # nothing met target -> safest


def _nodes_at_overload(frontier, target):
    pts = sorted(frontier, key=lambda p: p[0], reverse=True)  # nodes desc
    for (n1, o1), (n2, o2) in zip(pts, pts[1:]):
        if (o1 - target) * (o2 - target) <= 0 and o1 != o2:
            w = (target - o1) / (o2 - o1)
            return n1 + w * (n2 - n1)
    ok = [n for n, o in pts if o <= target]
    return min(ok) if ok else float("nan")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", type=int, default=3)
    ap.add_argument("--n-tasks", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--zs", nargs="+", type=float, default=[3, 4, 5, 6, 8],
                    help="safety multiplier for blind/aware baselines")
    ap.add_argument("--val-targets", nargs="+", type=float,
                    default=[2.0, 1.0, 0.5, 0.25, 0.1],
                    help="frontier knob: validation overload targets for v5 calib")
    ap.add_argument("--q-grid", nargs="+", type=float,
                    default=[80, 85, 90, 93, 95, 97, 98, 99, 99.5, 100],
                    help="quantile grid searched during calibration")
    ap.add_argument("--half-life", type=float, default=1.5)
    ap.add_argument("--cp-thresh", type=float, default=0.6)
    ap.add_argument("--target", type=float, default=0.5,
                    help="TEST overload %% at which nodes are reported")
    ap.add_argument("--data", default="/tmp/gct_days_v5.npz")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "oos_v5_pooled.json"))
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    days = d["series"]; full = d["full"]; H = days.shape[3]
    D = args.history
    assert D >= 2, "v5 calibration needs history>=2 (inner train + pseudo-test)"

    frontier = {"blind": [], "aware": [], "v5": []}
    peak_nodes = []
    windows_used = 0
    calib_qs = []

    # ---- baselines: sweep z ----
    for zi, z in enumerate(args.zs):
        acc = {"blind": {"n": [], "o": []}, "aware": {"n": [], "o": []}}
        for test in range(D, days.shape[0]):
            cand = np.all(full[test - D:test + 1], axis=0)
            if cand.sum() < 100:
                continue
            if zi == 0:
                windows_used += 1
            train = days[test - D:test][:, cand]; testday = days[test][cand]
            nt = min(args.n_tasks, int(cand.sum()))
            fb = learn_baselines(train, args.K, H)
            if zi == 0:
                for s in range(args.seeds):
                    pn, _ = place(testday, fb, "peak", 0, 0, H, nt, s)
                    peak_nodes.append(pn)
            for name, K in (("blind", 0), ("aware", args.K)):
                for s in range(args.seeds):
                    n, o = place(testday, fb, name, z, K, H, nt, s)
                    acc[name]["n"].append(n); acc[name]["o"].append(o)
        for name in acc:
            if acc[name]["n"]:
                frontier[name].append(
                    (round(float(np.mean(acc[name]["n"])), 1),
                     round(100 * float(np.mean(acc[name]["o"])), 3)))

    # ---- v5: sweep validation targets, calibrate q per window, apply to test ----
    for vt in args.val_targets:
        n_all, o_all = [], []
        for test in range(D, days.shape[0]):
            cand = np.all(full[test - D:test + 1], axis=0)
            if cand.sum() < 100:
                continue
            train = days[test - D:test][:, cand]; testday = days[test][cand]
            nt = min(args.n_tasks, int(cand.sum()))
            q = calibrate_q(train, H, args.half_life, args.cp_thresh, vt,
                            args.q_grid, nt, args.seeds)
            calib_qs.append((round(vt, 3), test, round(q, 3)))
            fv = learn_v5(train, q, H, args.half_life, args.cp_thresh)
            for s in range(args.seeds):
                n, o = place(testday, fv, "v5", 0, 0, H, nt, s)
                n_all.append(n); o_all.append(o)
        if n_all:
            frontier["v5"].append((round(float(np.mean(n_all)), 1),
                                   round(100 * float(np.mean(o_all)), 3)))

    tgt = args.target
    at = {name: round(float(_nodes_at_overload(frontier[name], tgt)), 1)
          for name in frontier}
    pk = round(float(np.mean(peak_nodes)), 1) if peak_nodes else float("nan")

    def saving(base):
        b = pk if base == "peak" else at.get(base)
        v = at["v5"]
        if b is None or b != b or v != v or b == 0:
            return None
        return round(100 * (b - v) / b, 1)

    out = {"provenance": {"history": D, "K": args.K, "zs": args.zs,
                          "val_targets": args.val_targets, "q_grid": args.q_grid,
                          "half_life": args.half_life, "cp_thresh": args.cp_thresh,
                          "n_tasks": args.n_tasks, "seeds": args.seeds,
                          "target_overload_pct": tgt,
                          "test_windows_used": windows_used,
                          "dataset": "Google 2011 multi-day, 8-day trace (gct_days.npz copy)"},
           "frontier": frontier, "peak_nodes": pk, "nodes_at_target": at,
           "calibrated_qs": calib_qs,
           "v5_saving_vs_peak_pct": saving("peak"),
           "v5_saving_vs_blind_pct": saving("blind"),
           "v5_saving_vs_aware_pct": saving("aware")}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)

    print("=" * 68)
    print("v5 calibrated cooperative-pooling quantile admission (history=%d)" % D)
    print("test windows used: %d   seeds: %d   n_tasks: %d"
          % (windows_used, args.seeds, args.n_tasks))
    print("-" * 68)
    print("frontier blind (nodes, over%%):", frontier["blind"])
    print("frontier aware (nodes, over%%):", frontier["aware"])
    print("frontier v5    (nodes, over%%):", frontier["v5"])
    print("-" * 68)
    print("NODES AT %.2f%% OVERLOAD (lower is better):" % tgt)
    print("  peak-requests    : %s" % pk)
    print("  phase-blind mean+zσ : %s" % at["blind"])
    print("  phase-aware Fourier  : %s" % at["aware"])
    print("  v5 (this method)  : %s" % at["v5"])
    print("-" * 68)
    print("v5 delta vs peak  : %s%%  (positive = v5 uses fewer nodes)"
          % out["v5_saving_vs_peak_pct"])
    print("v5 delta vs blind : %s%%" % out["v5_saving_vs_blind_pct"])
    print("v5 delta vs aware : %s%%" % out["v5_saving_vs_aware_pct"])
    beat = [b for b, v in (("peak", pk), ("blind", at["blind"]),
                           ("aware", at["aware"]))
            if at["v5"] == at["v5"] and v == v and at["v5"] < v]
    if beat:
        print("VERDICT: v5 beats: %s" % ", ".join(beat))
    else:
        print("VERDICT: v5 does NOT beat any baseline at %.2f%% test overload." % tgt)
    print("=" * 68)


if __name__ == "__main__":
    main()
