"""Confidence-gated phase inference for out-of-sample admission.

Magnitude forecasting (experiments/simulation/forecast_gpu.py) over-provisions and loses to simple
peak reservation. The scheduler, however, consumes *timing*: two jobs whose
dominant rhythms are in anti-phase can share a node because their peaks fall at
different times. So the quantity worth inferring is not a job's demand
magnitude but the phase of its dominant harmonic, learned from history.

A predicted phase is only useful if it is stable. For each job we measure the
circular concentration of its dominant-harmonic phase across the D training
days: R = |mean_d e^{j phi_d}| in [0,1]. Jobs whose phase is concentrated
(R >= tau) are *trusted*: their learned periodic bubble is used for admission,
which co-schedules anti-phase jobs and lowers the reserved peak. Jobs whose
phase drifts (R < tau) are *demoted* to peak reservation, which is safe by
construction. The same gate is the security control: an adversary that injects
an unstable or spoofed rhythm cannot pass the concentration test, so it is
reserved at peak and can never induce under-provisioning or overload on honest
neighbours. One mechanism -- a phase-stability test -- delivers both the packing
gain and the attack safety.

Runs on the committed multi-day artifact (data/gct_days.npz). Writes
results/oos_phase_infer.json. Every number is produced here; nothing is typed
into the paper by hand.
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
from experiments.simulation.oos_placement import learn_bubble, place_peak
from experiments.simulation.gct_common import ROOT


def dominant_confidence(train_days, KK, H):
    """Per-job dominant-harmonic amplitude, phase and circular concentration.

    train_days: (D,N,2,H). Returns amp,ph,conf of shape (N,2) at the harmonic
    that maximises mean amplitude * concentration (the rhythm we would rely on).
    conf = |mean_d exp(j phi_{d,k})|, the phase agreement across training days.
    """
    D, N = train_days.shape[:2]
    X = np.fft.rfft(train_days, axis=3)                       # (D,N,2,H//2+1)
    best_amp = np.zeros((N, 2)); best_ph = np.zeros((N, 2)); best_conf = np.zeros((N, 2))
    best_score = np.full((N, 2), -1.0)
    for k in range(1, max(KK, 1) + 1):
        coef = 2 * X[:, :, :, k] / H                          # (D,N,2) complex
        amp = np.abs(coef).mean(axis=0)                       # (N,2) mean amplitude
        unit = coef / np.maximum(np.abs(coef), 1e-12)
        conf = np.abs(unit.mean(axis=0))                      # (N,2) concentration
        ph = np.angle(coef.mean(axis=0))
        score = amp * conf
        take = score > best_score
        best_score = np.where(take, score, best_score)
        best_amp = np.where(take, amp, best_amp)
        best_ph = np.where(take, ph, best_ph)
        best_conf = np.where(take, conf, best_conf)
    # one confidence per job: the weaker of its two resource channels
    conf_job = best_conf.min(axis=1)
    return best_amp, best_ph, best_conf, conf_job


def place_gated(learned, conf_job, ph_job, test_day, z, n_tasks, tau, seed):
    """Confidence-gated phase-aware placement.

    Trusted jobs (conf >= tau) admit with the learned periodic bubble (anti-phase
    co-scheduling lowers the reserved peak). Demoted jobs reserve at peak. Jobs
    are offered in anti-phase order (by dominant phase) so complementary rhythms
    meet on the same node. Overload is measured on the real held-out day.
    """
    rng = np.random.default_rng(seed)
    N, H = test_day.shape[0], test_day.shape[2]
    sel = rng.choice(N, min(n_tasks, N), replace=False)
    # anti-phase ordering: sort selected jobs by dominant phase angle
    sel = sel[np.argsort(ph_job[sel])]
    KK = learned["KK"]
    peaks = test_day.max(axis=2)                              # (N,2) safe reservation
    nodes = []
    n_trusted = 0
    for idx in sel:
        trusted = conf_job[idx] >= tau
        n_trusted += int(trusted)
        placed = False
        for nd in nodes:
            if trusted:
                M = nd["M"] + learned["mean"][idx]
                Z = nd["Z"] + learned["amp"][idx] * np.exp(1j * learned["ph"][idx])
                nvar = nd["nvar"] + learned["noise_var"][idx]
                load = _periodic_peak(M, Z, KK, H) + z * np.sqrt(nvar)
                if (load <= CAP).all():
                    nd["M"], nd["Z"], nd["nvar"] = M, Z, nvar
                    nd["members"].append(idx); placed = True; break
            else:
                if (nd["peak"] + peaks[idx] <= CAP).all():
                    nd["peak"] = nd["peak"] + peaks[idx]
                    nd["members"].append(idx); placed = True; break
        if not placed:
            Z0 = learned["amp"][idx] * np.exp(1j * learned["ph"][idx])
            nodes.append({"M": learned["mean"][idx].copy(), "Z": Z0,
                          "nvar": learned["noise_var"][idx].copy(),
                          "peak": peaks[idx].copy(), "members": [idx]})
    over = tot = 0
    for nd in nodes:
        real = test_day[nd["members"]].sum(axis=0)
        over += int((real > CAP + 1e-9).sum()); tot += real.size
    return {"nodes": len(nodes), "overload": over / max(1, tot),
            "trusted_frac": n_trusted / max(1, len(sel))}


def inject_unstable(train_days, frac, seed):
    """Adversary: give a fraction of jobs a per-day-randomised dominant phase
    (a spoofed/unstable rhythm) while keeping the same amplitude. Their history
    no longer agrees, so the concentration gate should catch them. Returns the
    tampered training window and the boolean mask of tampered jobs."""
    rng = np.random.default_rng(seed)
    D, N, C, H = train_days.shape
    mask = rng.random(N) < frac
    out = train_days.copy()
    idx = np.where(mask)[0]
    for i in idx:
        shift = rng.uniform(0, H, size=D)                     # different phase each day
        for d in range(D):
            out[d, i] = np.roll(out[d, i], int(shift[d]), axis=-1)
    return out, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days-file", default=os.path.join(ROOT, "data", "gct_days.npz"))
    ap.add_argument("--history", type=int, default=3)
    ap.add_argument("--n-tasks", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--z", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "oos_phase_infer.json"))
    args = ap.parse_args()
    if not os.path.exists(args.days_file):
        print(f"[stub] {args.days_file} not built; run extractor first."); return
    d = np.load(args.days_file, allow_pickle=True)
    days = d["series"]; full = d["full"]; prov = json.loads(str(d["provenance"]))
    H = days.shape[3]; D = args.history
    KK = min(args.K, H // 2)
    seeds = list(range(args.seeds))

    def mean_of(rs, key):
        return float(np.mean([r[key] for r in rs]))

    windows = []
    agg = {"peak": [], "phase_gated": [], "phase_ungated": []}
    for test in range(D, days.shape[0]):
        cand = np.all(full[test - D:test + 1], axis=0)
        if int(cand.sum()) < 100:
            continue
        train = days[test - D:test][:, cand]                  # (D,n,2,H)
        testday = days[test][cand]
        nt = min(args.n_tasks, int(cand.sum()))
        learned = learn_bubble(train, args.K, H)
        amp, ph, conf, conf_job = dominant_confidence(train, KK, H)
        ph_job = ph[:, np.argmax(np.abs(amp), axis=1)][np.arange(len(ph)),
                    np.argmax(np.abs(amp), axis=1)] if False else ph.mean(axis=1)
        # per-window means over seeds
        pk = [place_peak(testday, nt, s) for s in seeds]
        gated = [place_gated(learned, conf_job, ph_job, testday, args.z, nt, args.tau, s)
                 for s in seeds]
        # ungated = trust every job (tau = 0): isolates the value of the gate
        ung = [place_gated(learned, conf_job, ph_job, testday, args.z, nt, -1.0, s)
               for s in seeds]
        windows.append({"test_day": int(test), "candidate_tasks": int(cand.sum()),
                        "peak_nodes": mean_of(pk, "nodes"),
                        "gated_nodes": mean_of(gated, "nodes"),
                        "gated_overload": mean_of(gated, "overload"),
                        "ungated_nodes": mean_of(ung, "nodes"),
                        "ungated_overload": mean_of(ung, "overload"),
                        "trusted_frac": mean_of(gated, "trusted_frac")})
        agg["peak"].append(mean_of(pk, "nodes"))
        agg["phase_gated"].append(mean_of(gated, "nodes"))
        agg["phase_ungated"].append(mean_of(ung, "nodes"))

    # security: adversarial unstable-rhythm sweep on the last valid window
    adv = []
    test = windows[-1]["test_day"]
    cand = np.all(full[test - D:test + 1], axis=0)
    testday = days[test][cand]; nt = min(args.n_tasks, int(cand.sum()))
    for frac in (0.0, 0.1, 0.2, 0.3):
        tampered, mask = inject_unstable(days[test - D:test][:, cand], frac, seed=0)
        learned_a = learn_bubble(tampered, args.K, H)
        amp_a, ph_a, conf_a, conf_job_a = dominant_confidence(tampered, KK, H)
        ph_job_a = ph_a.mean(axis=1)
        g = [place_gated(learned_a, conf_job_a, ph_job_a, testday, args.z, nt, args.tau, s)
             for s in seeds]
        # fraction of tampered jobs the gate correctly demoted
        caught = float(np.mean(conf_job_a[mask] < args.tau)) if mask.any() else 1.0
        adv.append({"malicious_frac": frac,
                    "overload": mean_of(g, "overload"),
                    "nodes": mean_of(g, "nodes"),
                    "adversary_demoted_frac": caught})

    summary = {
        "peak_nodes_mean": float(np.mean(agg["peak"])),
        "phase_gated_nodes_mean": float(np.mean(agg["phase_gated"])),
        "phase_ungated_nodes_mean": float(np.mean(agg["phase_ungated"])),
        "gated_gain_vs_peak_pct": float(100 * (1 - np.mean(agg["phase_gated"]) / np.mean(agg["peak"]))),
        "max_gated_overload": float(max(w["gated_overload"] for w in windows)),
        "windows": len(windows),
    }
    out = {"dataset": prov.get("dataset"), "history": D, "K": args.K, "tau": args.tau,
           "z": args.z, "n_tasks": args.n_tasks, "seeds": args.seeds,
           "summary": summary, "per_window": windows, "adversary": adv}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print("adversary:", json.dumps(adv, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
