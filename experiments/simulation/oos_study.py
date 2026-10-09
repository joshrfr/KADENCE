"""Low-level study: what in a task's history actually predicts its next day?

For the multi-day trace (gct_days.npz), for tasks present on all days, measure:
 - per-harmonic cross-day PHASE STABILITY, the circular concentration
   R_k = |mean_d e^{j phi_k^d}| in [0,1]; R_k~1 = a rhythm you can trust to
   recur, R_k~0 = phase drifts and the rhythm is useless for prediction;
 - how much of a task's daily-peak variability is periodic (arrangeable) vs
   day-to-day residual (must be covered by margin);
 - whether phase stability is heterogeneous across tasks (some predictable,
   some not) -- the signal a better model should exploit.

Prints a summary; writes results/oos_study.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    d = np.load(os.path.join(ROOT, "data", "gct_days.npz"), allow_pickle=True)
    days = d["series"]                 # (D, M, 2, H)
    full = d["full"]                   # (D, M)
    D, M, _, H = days.shape
    K = 8
    # For each adjacent day pair (learn d -> predict d+1), on tasks present both
    # days, measure per-harmonic phase agreement between the two days. This is
    # exactly the drift that limits out-of-sample phase prediction.
    R_dom_all, R1_all, peak_ratio_all = [], [], []
    for d in range(D - 1):
        m = full[d] & full[d + 1]
        if m.sum() < 50:
            continue
        A = days[d][m, 0, :]; B = days[d + 1][m, 0, :]     # (n,H) CPU both days
        Af = np.fft.rfft(A, axis=1)[:, 1:K + 1]
        Bf = np.fft.rfft(B, axis=1)[:, 1:K + 1]
        ampA = 2 * np.abs(Af) / H
        dphi = np.angle(Af) - np.angle(Bf)
        agree = np.abs(np.cos(dphi))                        # 1=stable, 0=drifted
        dom = ampA.argmax(axis=1)
        R_dom_all.append(agree[np.arange(len(dom)), dom])
        R1_all.append(agree[:, 0])
        pa = A.max(axis=1); pb = B.max(axis=1)
        peak_ratio_all.append(pb / np.maximum(pa, 1e-9))    # next/prev peak
    R_dom = np.concatenate(R_dom_all); R1 = np.concatenate(R1_all)
    peak_ratio = np.concatenate(peak_ratio_all)
    N = len(R_dom)
    peak_cv = np.abs(np.log(np.maximum(peak_ratio, 1e-3)))  # dispersion of day-to-day peak
    out = {
        "days": int(D), "task_day_pairs": int(N), "K": K,
        "phase_stability_dominant_harmonic": {
            "median_R": round(float(np.median(R_dom)), 3),
            "frac_stable_R>0.8": round(float(np.mean(R_dom > 0.8)), 3),
            "frac_unstable_R<0.4": round(float(np.mean(R_dom < 0.4)), 3),
        },
        "phase_stability_daily_harmonic_k1": {
            "median_R": round(float(np.median(R1)), 3),
        },
        "daily_peak_cv": {
            "median": round(float(np.median(peak_cv)), 3),
            "p90": round(float(np.percentile(peak_cv, 90)), 3),
        },
        "reading": ("R_dom near 1 means the rhythm recurs and phase helps; "
                    "near 0 means it drifts and only a per-task margin helps. "
                    "Heterogeneity across tasks is what a better model exploits."),
    }
    json.dump(out, open(os.path.join(ROOT, "results", "oos_study.json"), "w"),
              indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
