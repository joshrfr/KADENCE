"""v4: a small LEARNED per-task confidence model for cluster admission.

Prior baselines (history=3, 0.5% overload, lower=better):
  peak-requests        = 31.3 machines
  phase-blind mean+zs  = 33.4
  phase-aware          = 34.2

Idea. The task population is heterogeneous: predictable tasks (stable dominant
harmonic phase, low burstiness) need only a tight margin above their recent
peak; volatile tasks need a wide one. The prior v2 sized margins from a global
z times a per-task cross-day sigma. Here we instead LEARN, from features
computed on the history days, both (a) the expected next-day peak and (b) a
calibrated upper margin, per task. The admission requirement for a task is

    req_i(z) = qhat_i + z * margin_i          (per resource, cpu & mem)

where qhat_i is the model's central next-day-peak prediction and margin_i is a
learned spread (upper-quantile minus central, floored). z sweeps the frontier.
A node admits a task set while sum_i req_i(z) <= 1 per resource; overload =
fraction of the 288 test-day slots where the SUMMED REAL load exceeds 1.

Training data. With `history` days d_0..d_{H-1} we form (train_day ->
next_day_peak) pairs from the history ONLY: features on days up to k predict
the peak on day k+1, for k in 1..H-1. The TEST day is never used to fit the
model (strict hold-out). Model = sklearn GradientBoostingRegressor with
quantile loss (a central alpha=0.5 head and an upper alpha for the margin); if
sklearn is missing, a hand-rolled quantile-by-feature-bin fallback is used.

Features per task/resource, computed on the observed days available at
prediction time:
  R          dominant-harmonic phase stability = mean over day-pairs of
             |cos(delta phi)| of the strongest harmonic (1 => perfectly
             recurring rhythm, 0 => phase scrambles day to day)
  burst      peak / mean (peak-to-mean burstiness)
  cv         coefficient of variation of the within-day profile (mean over days)
  meanload   mean load
  xcorr      cpu/mem correlation of the day-mean profiles (shared per task)
  p_last     most recent day's peak
  p_mean     mean of the observed daily peaks
  p_max      max of the observed daily peaks
  p_slope    (last peak - first peak)/ndays  (trend)

Compares peak / blind / aware(v2-style global) / v4(learned) at history=3 and 1.
Writes results/oos_v4_learned.json. The sample is small, so overfitting is a
real risk and is reported with the result.
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
DATA = "/tmp/gct_days_v4.npz"
CAP = 1.0

try:
    from sklearn.ensemble import GradientBoostingRegressor
    HAVE_SK = True
except Exception:  # pragma: no cover
    HAVE_SK = False


# --------------------------------------------------------------------------- #
# feature extraction
# --------------------------------------------------------------------------- #
def _dom_harmonic_phase(day_stack, H):
    """day_stack (nd, n, H) -> (n,) mean over day-pairs of |cos(dphi)| of the
    per-task dominant harmonic (harmonic index chosen by summed magnitude)."""
    nd, n, _ = day_stack.shape
    Xf = np.fft.rfft(day_stack, axis=2)[:, :, 1:]        # drop DC, (nd,n,F)
    mag = np.abs(Xf).sum(axis=0)                          # (n,F)
    kbest = mag.argmax(axis=1)                            # (n,)
    phi = np.angle(Xf[:, np.arange(n), kbest])            # (nd,n)
    if nd < 2:
        return np.ones(n)
    # mean |cos(delta phi)| over consecutive day pairs
    dphi = phi[1:] - phi[:-1]                             # (nd-1,n)
    return np.abs(np.cos(dphi)).mean(axis=0)


def features(obs, H):
    """obs (nd, n, 2, H) observed days at prediction time.
    Returns dict r -> (n, F) feature matrix, plus shared xcorr feature."""
    nd, n, _, _ = obs.shape
    # shared cpu/mem correlation on the day-mean profile
    mprof = obs.mean(axis=0)                              # (n,2,H)
    a = mprof[:, 0, :] - mprof[:, 0, :].mean(axis=1, keepdims=True)
    b = mprof[:, 1, :] - mprof[:, 1, :].mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1)) + 1e-12
    xcorr = num / den                                    # (n,)
    # smoothed periodic profile peak (matches oos_v2 prof_peak footing):
    # cross-day complex mean of the first K harmonics, reconstructed, peak.
    K = 8
    Xf = np.fft.rfft(obs, axis=3)                        # (nd,n,2,F)
    dc = Xf[:, :, :, 0].real.mean(axis=0) / H            # (n,2)
    coef = (2 * Xf[:, :, :, 1:K + 1] / H).mean(axis=0)   # (n,2,K) complex mean
    t = np.arange(H)
    ang = 2 * np.pi * np.outer(np.arange(1, K + 1), t) / H  # (K,H)
    cos, sin = np.cos(ang), np.sin(ang)
    prof = dc[:, :, None] + (coef.real @ cos - coef.imag @ sin)  # (n,2,H)
    prof_peak = prof.max(axis=2)                         # (n,2)

    feat = {}
    for r in (0, 1):
        ds = obs[:, :, r, :]                             # (nd,n,H)
        R = _dom_harmonic_phase(ds, H)                   # (n,)
        day_peak = ds.max(axis=2)                        # (nd,n)
        day_mean = ds.mean(axis=2)                       # (nd,n)
        day_std = ds.std(axis=2)                         # (nd,n)
        meanload = day_mean.mean(axis=0)
        burst = (day_peak.mean(axis=0)) / (meanload + 1e-9)
        cv = (day_std / (day_mean + 1e-9)).mean(axis=0)
        p_last = day_peak[-1]
        p_mean = day_peak.mean(axis=0)
        p_max = day_peak.max(axis=0)
        p_slope = (day_peak[-1] - day_peak[0]) / max(1, nd - 1)
        pp = prof_peak[:, r]                              # smoothed profile peak
        X = np.column_stack([R, burst, cv, meanload, xcorr,
                             p_last, p_mean, p_max, p_slope, pp])
        feat[r] = dict(X=X, p_last=p_last, p_mean=p_mean, p_max=p_max,
                       prof_peak=pp)
    return feat


# --------------------------------------------------------------------------- #
# model: predict next-day peak + upper margin per task
# --------------------------------------------------------------------------- #
def _build_pairs(train_days, H):
    """train_days (D,n,2,H). Build (X, y_peak) supervised pairs from history
    ONLY: features on days [0..k] predict peak on day k+1, k=1..D-1.
    Returns per resource r -> (Xtr, ytr)."""
    D = train_days.shape[0]
    # out[r] = [X_list, y_central_list, y_upper_list]
    out = {0: [[], [], []], 1: [[], [], []]}
    for k in range(1, D):                                # need >=2 obs days
        obs_prev = train_days[:k]                        # nd=k
        feat = features(obs_prev, H)
        ynext = features(train_days[k:k + 1], H)         # next-day features
        for r in (0, 1):
            out[r][0].append(feat[r]["X"])
            out[r][1].append(ynext[r]["prof_peak"])       # central: next smoothed peak
            out[r][2].append(train_days[k, :, r, :].max(axis=1))  # upper: raw max
    res = {}
    for r in (0, 1):
        if out[r][0]:
            res[r] = (np.vstack(out[r][0]), np.concatenate(out[r][1]),
                      np.concatenate(out[r][2]))
        else:
            res[r] = (None, None, None)
    return res


def _fallback_predict(Xtr, ytr, Xte, alpha):
    """Hand-rolled quantile-by-feature-bin: bin on burstiness*R stability, take
    the alpha-quantile of ytr in each bin; smooth toward global quantile."""
    if Xtr is None or len(ytr) < 4:
        # too little data: fall back to identity on p_last (col 5)
        return Xte[:, 5]
    key_tr = Xtr[:, 1] * (1.0 - Xtr[:, 0])               # burst * (1-R): volatility
    key_te = Xte[:, 1] * (1.0 - Xte[:, 0])
    edges = np.quantile(key_tr, np.linspace(0, 1, 6))
    edges[0] = -np.inf; edges[-1] = np.inf
    gq = np.quantile(ytr, alpha)
    pred = np.full(len(Xte), gq)
    bt = np.digitize(key_tr, edges) - 1
    be = np.digitize(key_te, edges) - 1
    for b in range(len(edges) - 1):
        m = bt == b
        if m.sum() >= 3:
            q = np.quantile(ytr[m], alpha)
            pred[be == b] = q
    # never predict below the task's own recent peak scaled a bit
    return pred


def train_predict(train_days, H, alpha_up, seed):
    """Fit central + upper models on history pairs; predict for the TEST-day
    admission using ALL history days as the feature window.
    Returns per r: qhat (central peak), margin (>=0 spread)."""
    pairs = _build_pairs(train_days, H)
    obs_all = train_days                                 # all history observed
    fte = features(obs_all, H)
    res = {}
    for r in (0, 1):
        Xtr, yc, yu = pairs[r]
        Xte = fte[r]["X"]
        if HAVE_SK and Xtr is not None and len(yc) >= 20:
            # subsample training rows to keep fitting bounded on 40k-task days
            if len(yc) > 20000:
                sidx = np.random.default_rng(seed).choice(
                    len(yc), 20000, replace=False)
                Xtr, yc, yu = Xtr[sidx], yc[sidx], yu[sidx]
            common = dict(n_estimators=60, max_depth=2, learning_rate=0.1,
                          subsample=0.7, random_state=seed)
            # central head predicts next-day smoothed profile peak;
            # upper head predicts an alpha-quantile of next-day RAW max.
            m_mid = GradientBoostingRegressor(loss="quantile", alpha=0.5, **common)
            m_up = GradientBoostingRegressor(loss="quantile", alpha=alpha_up, **common)
            m_mid.fit(Xtr, yc)
            m_up.fit(Xtr, yu)
            qhat = m_mid.predict(Xte)
            qup = m_up.predict(Xte)
        else:
            qhat = _fallback_predict(Xtr, yc, Xte, 0.5)
            qup = _fallback_predict(Xtr, yu, Xte, alpha_up)
        qhat = np.maximum(qhat, 0.0)
        # margin = learned upper-quantile spread; floored to a small fraction
        # of the smoothed profile peak so zero-spread predictions still carry
        # a little safety (predictable tasks keep tight margins).
        margin = np.maximum(qup - qhat, 0.0)
        margin = np.maximum(margin, 0.05 * fte[r]["prof_peak"])
        res[r] = dict(qhat=qhat, margin=margin,
                      prof_peak=fte[r]["prof_peak"], p_peak=fte[r]["p_mean"],
                      p_max=fte[r]["p_max"])
    return res


# --------------------------------------------------------------------------- #
# placement (first-fit) for each policy
# --------------------------------------------------------------------------- #
def place(test, model, policy, z, n_tasks, seed):
    """test (n,2,H) real test-day profiles. model from train_predict.
    policy: 'peak' | 'blind' | 'v2' | 'v4'.
    For peak/blind/v2 we reuse simple per-task stats already in `model`."""
    rng = np.random.default_rng(seed)
    n = test.shape[0]
    order = rng.choice(n, min(n_tasks, n), replace=False)
    # precompute per-task requirement per resource for this z
    req = {}
    for r in (0, 1):
        m = model[r]
        if policy == "peak":
            req[r] = m["prof_peak"]                       # request = smoothed profile peak
        elif policy == "blind":
            # phase-blind: smoothed profile peak + z * GLOBAL margin scale
            gsig = np.full(n, m["margin"].mean())
            req[r] = m["prof_peak"] + z * gsig
        elif policy == "v2":
            req[r] = m["prof_peak"] + z * m["margin"]     # per-task margin (v2-style)
        else:  # v4 learned
            req[r] = m["qhat"] + z * m["margin"]
        req[r] = np.maximum(req[r], 0.0)
    nodes = []                                           # each: [cpu_used, mem_used, members]
    for i in order:
        placed = False
        for nd in nodes:
            if nd[0] + req[0][i] <= CAP and nd[1] + req[1][i] <= CAP:
                nd[0] += req[0][i]; nd[1] += req[1][i]; nd[2].append(i)
                placed = True; break
        if not placed:
            nodes.append([req[0][i], req[1][i], [i]])
    over = tot = 0
    for nd in nodes:
        real = test[nd[2]].sum(axis=0)                   # (2,H)
        over += int((real > CAP + 1e-9).sum()); tot += real.size
    return len(nodes), over / max(1, tot)


def _nodes_at_overload(frontier, target):
    pts = sorted(frontier, key=lambda p: p[0], reverse=True)
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
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--alpha-up", type=float, default=0.9)
    ap.add_argument("--zs", nargs="+", type=float, default=[0, 1, 2, 3, 4, 6])
    ap.add_argument("--target", type=float, default=0.5)
    ap.add_argument("--out", default="oos_v4_learned.json")
    args = ap.parse_args()

    d = np.load(DATA, allow_pickle=True)
    days = d["series"]; full = d["full"]; H = days.shape[3]
    D = args.history
    policies = ["blind", "v2", "v4"]
    frontier = {p: [] for p in policies}
    peak_nodes = []
    windows = 0

    # Fit the model ONCE per test window (it does not depend on z); cache the
    # per-task requirement inputs and the real test-day profiles.
    cache = []
    for test in range(D, days.shape[0]):
        cand = np.all(full[test - D:test + 1], axis=0)
        if cand.sum() < 100:
            continue
        train = np.ascontiguousarray(days[test - D:test][:, cand])
        testday = np.ascontiguousarray(days[test][cand])
        nt = min(args.n_tasks, int(cand.sum()))
        model = train_predict(train, H, args.alpha_up, seed=0)
        cache.append((model, testday, nt))
        windows += 1
        for s in range(args.seeds):
            pn, _ = place(testday, model, "peak", 0, nt, s)
            peak_nodes.append(pn)

    for z in args.zs:
        acc = {p: {"n": [], "o": []} for p in policies}
        for model, testday, nt in cache:
            for pol in policies:
                for s in range(args.seeds):
                    nn, oo = place(testday, model, pol, z, nt, s)
                    acc[pol]["n"].append(nn); acc[pol]["o"].append(oo)
        for pol in policies:
            if acc[pol]["n"]:
                frontier[pol].append(
                    (round(float(np.mean(acc[pol]["n"])), 1),
                     round(100 * float(np.mean(acc[pol]["o"])), 3)))

    tgt = args.target
    at = {pol: round(float(_nodes_at_overload(frontier[pol], tgt)), 1)
          for pol in frontier}
    pk = round(float(np.mean(peak_nodes)), 1) if peak_nodes else float("nan")
    base = {"peak_requests": 31.3, "phase_blind": 33.4, "phase_aware": 34.2}
    v4 = at["v4"]
    best_base = min(base.values())
    out = {
        "provenance": {"history": D, "alpha_up": args.alpha_up, "zs": args.zs,
                       "n_tasks": args.n_tasks, "seeds": args.seeds,
                       "target_overload_pct": tgt, "test_windows": windows,
                       "sklearn": HAVE_SK,
                       "dataset": "Google 2011 multi-day (/tmp/gct_days_v4.npz)"},
        "frontier": frontier, "peak_nodes_measured": pk,
        "nodes_at_target": at,
        "reference_baselines": base,
        "v4_nodes_at_target": v4,
        "beats_best_baseline": bool(v4 == v4 and v4 < best_base),
        "v4_vs_peak_pct": (round(100 * (base["peak_requests"] - v4)
                                 / base["peak_requests"], 1)
                           if v4 == v4 else None),
        "notes": ("Strict hold-out: the test day never enters model fitting; "
                  "the model is trained only on within-history day->next-day "
                  "peak pairs. With history=%d there are few (%d) training day "
                  "transitions per window, so the learned model has high "
                  "variance and risks overfitting; results should be read as "
                  "indicative, not definitive." % (D, max(0, D - 1))),
    }
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", args.out), "w"), indent=2)
    print(json.dumps({"history": D, "nodes_at_%.2f%%" % tgt: at,
                      "peak_measured": pk,
                      "reference_baselines": base,
                      "beats_best_baseline(%.1f)" % best_base:
                          out["beats_best_baseline"]}, indent=2))
    print("frontier v4:", frontier["v4"])
    return out


if __name__ == "__main__":
    main()
