"""Out-of-sample admission v3: recency-weighted per-task percentile recommender.

Baselines to beat (oos_v2.py, history=3, nodes at 0.5%% overload; lower better):
  peak-requests        = 31.3
  phase-blind mean+zσ  = 33.4
  phase-aware Fourier   = 34.2

Idea (in the spirit of Google Autopilot's percentile/limit recommender):
predict each task's next-day per-slot requirement as a HIGH PERCENTILE of its
own recent history, recency-weighted over the `history` training days, rather
than a mean plus a single global z-sigma. A node admits a set when, per
resource and per slot, the SUMMED per-task predicted requirement <= CAP; we
enforce the tightest binding slot (the max over slots of the summed prediction).

Two levers matter:
  * Recency weighting -- the most recent day counts more, so a task that has
    settled into a new level is predicted at that level, not the stale average.
  * Change-point gating -- if the most recent day deviates strongly from the
    earlier days (per-task L1 divergence over a threshold), we discard the
    older days for that task and predict from the recent day alone, so stale
    history does not inflate margins for tasks that just shifted regime.

The safety multiplier here is the PERCENTILE q swept as the frontier knob
(higher q -> safer, more nodes). We also add a small z * per-task day-to-day
scale term so single-day (post-gate) tasks are not left with a degenerate
percentile. Predictable tasks pack tight; volatile ones stay safe.

Phase-aware phasor smoothing is available as an option (--smooth-k) but off by
default: the per-slot percentile already captures the rhythm directly.

Compares peak / blind / aware / v3 on the real multi-day trace at the requested
history. Writes results/oos_v3_recency.json.

Note: gct_days.npz has 5 days and day index 4 is empty, so history=3
yields exactly ONE usable test window (day 3) -- the frontier is a single trace,
underpowered. history=1 yields three windows (days 1,2,3).
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


# ----------------------------------------------------------------------------
# Baseline features (peak / blind / aware) -- lifted from oos_v2 so the numbers
# are computed on the SAME copied data and directly comparable.
# ----------------------------------------------------------------------------
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
    D, n, _, _ = train.shape
    feat = {}
    for r in range(2):
        mean, coef = _phasors(train[:, :, r, :], K, H)
        prof = _periodic_series(mean, coef, H)
        within = np.sqrt(np.mean((train[:, :, r, :] - prof[None]) ** 2, axis=(0, 2)))
        prof_peak = prof.max(axis=1)
        feat[r] = dict(mean=mean, coef=coef, prof_peak=prof_peak, within=within)
    return feat


# ----------------------------------------------------------------------------
# v3 features: per-task recency-weighted percentile prediction, per slot.
# ----------------------------------------------------------------------------
def learn_v3(train, q, H, half_life, cp_thresh, z_extra, smooth_k=0):
    """train (D,n,2,H) -> per-resource predicted (n,H) requirement + per-task scale.

    q         : percentile (0..100) taken across the recency-weighted day samples
    half_life : recency half-life in days (weight = 0.5 ** (age / half_life))
    cp_thresh : change-point gate; if the most recent day's mean-normalised L1
                distance from the earlier-day average exceeds this, keep only
                the recent day for that task.
    z_extra   : extra safety = z_extra * per-task day-to-day peak scale.
    """
    D, n, _, _ = train.shape
    ages = np.arange(D)[::-1]                      # day 0 oldest -> highest age
    w = 0.5 ** (ages / max(1e-9, half_life))       # (D,) recency weights
    feat = {}
    for r in range(2):
        X = train[:, :, r, :]                       # (D,n,H)
        # change-point gate (per task): compare most-recent day to earlier mean
        if D >= 2:
            recent = X[-1]                          # (n,H)
            earlier = X[:-1].mean(axis=0)           # (n,H)
            denom = np.maximum(earlier.mean(axis=1), 1e-6)   # (n,)
            div = np.abs(recent - earlier).mean(axis=1) / denom  # (n,)
            gate = div > cp_thresh                  # tasks that shifted regime
        else:
            gate = np.zeros(n, bool)

        # weighted percentile across days, per (task, slot)
        pred = _weighted_percentile_days(X, w, q)   # (n,H)
        if D >= 2 and gate.any():
            # for gated tasks, predict from the recent day alone (its own q over
            # the day is just that day; use the day values directly)
            pred[gate] = X[-1][gate]

        # per-task day-to-day scale of the per-slot peak (safety cushion)
        day_peak = X.max(axis=2)                    # (D,n)
        scale = np.std(day_peak, axis=0)            # (n,)
        pred = pred + z_extra * scale[:, None]

        if smooth_k > 0:
            pred = _lowpass(pred, smooth_k, H)

        feat[r] = dict(pred=np.ascontiguousarray(pred))
    return feat


def _weighted_percentile_days(X, w, q):
    """X (D,n,H), weights w (D,) -> weighted q-th percentile over D, per (n,H).

    Weighted percentile via the standard CDF interpolation on sorted samples.
    Vectorised over (n,H); D is small (<=3) so the sort is cheap.
    """
    D, n, H = X.shape
    order = np.argsort(X, axis=0)                   # (D,n,H)
    Xs = np.take_along_axis(X, order, axis=0)
    ws = np.take_along_axis(np.broadcast_to(w[:, None, None], X.shape), order, axis=0)
    cw = np.cumsum(ws, axis=0)
    total = cw[-1]                                   # (n,H)
    # target cumulative weight for percentile q
    tgt = (q / 100.0) * total                        # (n,H)
    # find first index where cw >= tgt
    ge = cw >= tgt[None]
    idx = np.argmax(ge, axis=0)                      # (n,H) first True
    out = np.take_along_axis(Xs, idx[None], axis=0)[0]
    return out


def _lowpass(pred, K, H):
    """Keep DC + first K harmonics of each row (n,H)."""
    Xf = np.fft.rfft(pred, axis=1)
    Xf[:, K + 1:] = 0
    return np.fft.irfft(Xf, n=H, axis=1)


# ----------------------------------------------------------------------------
# First-fit placement. Node state per resource:
#   baselines carry M/peak/within/Z; v3 carries a running per-slot sum vector.
# ----------------------------------------------------------------------------
def _new_node(K, H, is_v3):
    if is_v3:
        return {r: {"sum": np.zeros(H), "members_first": True} for r in (0, 1)} | \
               {"members": []}
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


def _fits_v3(node, feat, i):
    new0 = node[0]["sum"] + feat[0]["pred"][i]
    if new0.max() > CAP:
        return False, None
    new1 = node[1]["sum"] + feat[1]["pred"][i]
    if new1.max() > CAP:
        return False, None
    return True, (new0, new1)


def _commit_v3(node, i, stash):
    node[0]["sum"] = stash[0]
    node[1]["sum"] = stash[1]
    node["members"].append(i)


def place(testday, feat, policy, z, K, H, n_tasks, seed):
    rng = np.random.default_rng(seed)
    n = testday.shape[0]
    order = rng.choice(n, min(n_tasks, n), replace=False)
    is_v3 = policy == "v3"
    nodes = []
    for i in order:
        placed = False
        for nd in nodes:
            if is_v3:
                ok, stash = _fits_v3(nd, feat, i)
                if ok:
                    _commit_v3(nd, i, stash); placed = True; break
            else:
                ok, stash = _fits_baseline(nd, feat, i, policy, z, K, H)
                if ok:
                    _commit_baseline(nd, feat, i, policy, K, stash); placed = True; break
        if not placed:
            nd = _new_node(K, H, is_v3)
            if is_v3:
                nd[0]["sum"] = feat[0]["pred"][i].copy()
                nd[1]["sum"] = feat[1]["pred"][i].copy()
                nd["members"].append(i)
            else:
                # fresh node always accepts its first task (matches v2 _new/_commit,
                # which does not re-test fit); build the stash unconditionally.
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
        real = testday[nd["members"]].sum(axis=0)   # (2,H) real test day
        over += int((real > CAP + 1e-9).sum()); tot += real.size
    return len(nodes), over / max(1, tot)


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
                    help="safety multiplier for the blind/aware baselines")
    ap.add_argument("--qs", nargs="+", type=float,
                    default=[90, 95, 98, 99, 99.5],
                    help="percentile frontier knob for v3")
    ap.add_argument("--half-life", type=float, default=1.5)
    ap.add_argument("--cp-thresh", type=float, default=0.6)
    ap.add_argument("--z-extra", type=float, default=0.0)
    ap.add_argument("--smooth-k", type=int, default=0)
    ap.add_argument("--target", type=float, default=0.5)
    ap.add_argument("--data", default="/tmp/gct_days_v3.npz")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    days = d["series"]; full = d["full"]; H = days.shape[3]
    D = args.history

    frontier = {"blind": [], "aware": [], "v3": []}
    peak_nodes = []
    windows_used = 0

    # baselines swept over z; v3 swept over q. Align by index length for output.
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

    for q in args.qs:
        n_all, o_all = [], []
        for test in range(D, days.shape[0]):
            cand = np.all(full[test - D:test + 1], axis=0)
            if cand.sum() < 100:
                continue
            train = days[test - D:test][:, cand]; testday = days[test][cand]
            nt = min(args.n_tasks, int(cand.sum()))
            fv = learn_v3(train, q, H, args.half_life, args.cp_thresh,
                          args.z_extra, args.smooth_k)
            for s in range(args.seeds):
                n, o = place(testday, fv, "v3", 0, 0, H, nt, s)
                n_all.append(n); o_all.append(o)
        if n_all:
            frontier["v3"].append((round(float(np.mean(n_all)), 1),
                                   round(100 * float(np.mean(o_all)), 3)))

    tgt = args.target
    at = {name: round(float(_nodes_at_overload(frontier[name], tgt)), 1)
          for name in frontier}
    pk = round(float(np.mean(peak_nodes)), 1) if peak_nodes else float("nan")

    def saving(base):
        b = at.get(base) if base != "peak" else pk
        v = at["v3"]
        if b is None or b != b or v != v or b == 0:
            return None
        return round(100 * (b - v) / b, 1)

    out = {"provenance": {"history": D, "K": args.K, "zs": args.zs, "qs": args.qs,
                          "half_life": args.half_life, "cp_thresh": args.cp_thresh,
                          "z_extra": args.z_extra, "smooth_k": args.smooth_k,
                          "n_tasks": args.n_tasks, "seeds": args.seeds,
                          "target_overload_pct": tgt,
                          "test_windows_used": windows_used,
                          "dataset": "Google 2011 multi-day (gct_days.npz copy)"},
           "frontier": frontier, "peak_nodes": pk, "nodes_at_target": at,
           "baselines_reference_h3": {"peak": 31.3, "blind": 33.4, "aware": 34.2},
           "v3_saving_vs_peak_pct": saving("peak"),
           "v3_saving_vs_blind_pct": saving("blind"),
           "v3_saving_vs_aware_pct": saving("aware")}

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", "oos_v3_recency.json"), "w"),
              indent=2)

    print("=" * 66)
    print("v3 recency-weighted percentile admission  (history=%d)" % D)
    print("test windows used: %d   seeds: %d   n_tasks: %d"
          % (windows_used, args.seeds, args.n_tasks))
    print("-" * 66)
    print("frontier v3 (nodes, overload%%):", frontier["v3"])
    print("-" * 66)
    print("nodes at %.2f%% overload (lower is better):" % tgt)
    print("  peak-requests   : %s" % pk)
    print("  phase-blind     : %s" % at["blind"])
    print("  phase-aware     : %s" % at["aware"])
    print("  v3 (this method): %s" % at["v3"])
    print("-" * 66)
    print("v3 vs peak  : %s%%" % out["v3_saving_vs_peak_pct"])
    print("v3 vs blind : %s%%" % out["v3_saving_vs_blind_pct"])
    print("v3 vs aware : %s%%" % out["v3_saving_vs_aware_pct"])
    beat = [b for b, v in (("peak", pk), ("blind", at["blind"]), ("aware", at["aware"]))
            if at["v3"] == at["v3"] and v == v and at["v3"] < v]
    if beat:
        print("VERDICT: v3 beats: %s" % ", ".join(beat))
    else:
        print("VERDICT: v3 does NOT beat any baseline at %.2f%% overload." % tgt)
    print("=" * 66)


if __name__ == "__main__":
    main()
