"""Do real tasks carry rhythms worth arranging? (v4 spectral analysis)

For every full-day task, measure the fraction of its CPU variance carried by
its single dominant harmonic (against the white-noise floor 2/H), the period
of that harmonic, the memory coefficient of variation, and the CPU/memory
phase agreement when they share a dominant period. Writes
``results/raw_rhythm_analysis.json``. No scheduling here, only whether the
signal the mechanism exploits exists in the real trace.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from experiments.simulation.gct_common import load_series, ROOT


def analyse(series: np.ndarray, H: int) -> dict:
    N = len(series)
    dom_frac = np.zeros(N)          # variance share of dominant CPU harmonic
    dom_period_h = np.zeros(N)      # its period in hours
    mem_cv = np.zeros(N)
    phase_agree = []                # |cos(dphi)| when cpu/mem share a period
    for i, s in enumerate(series):
        cpu, mem = s[0], s[1]
        Xc = np.fft.rfft(cpu - cpu.mean())
        powc = np.abs(Xc) ** 2
        total = powc[1:].sum()
        if total <= 0:
            continue
        k = 1 + int(np.argmax(powc[1:]))
        dom_frac[i] = powc[k] / total
        dom_period_h[i] = (H / k) * (24.0 / H)         # slots->hours
        mem_cv[i] = mem.std() / mem.mean() if mem.mean() > 0 else 0.0
        Xm = np.fft.rfft(mem - mem.mean())
        powm = np.abs(Xm) ** 2
        km = 1 + int(np.argmax(powm[1:]))
        if km == k:
            dphi = np.angle(Xc[k]) - np.angle(Xm[k])
            phase_agree.append(abs(np.cos(dphi)))
    noise_floor = 2.0 / H
    period_bins = {
        "<1h": float(np.mean(dom_period_h < 1)),
        "1-12h": float(np.mean((dom_period_h >= 1) & (dom_period_h < 12))),
        "12-24h": float(np.mean((dom_period_h >= 12) & (dom_period_h <= 24))),
    }
    return {
        "n_tasks": int(N),
        "H": H,
        "white_noise_floor": noise_floor,
        "dom_harmonic_var_share": {
            "median": float(np.median(dom_frac)),
            "mean": float(np.mean(dom_frac)),
            "x_over_noise_floor": float(np.median(dom_frac) / noise_floor),
            "frac_tasks_over_0.30": float(np.mean(dom_frac > 0.30)),
        },
        "dominant_period_hours": {
            "median": float(np.median(dom_period_h)),
            "share_daily_12_24h": period_bins["12-24h"],
            "bins": period_bins,
        },
        "memory_cv": {
            "median": float(np.median(mem_cv)),
        },
        "cpu_mem_phase_agreement_median_abscos": (
            float(np.median(phase_agree)) if phase_agree else None),
    }


def main():
    series, ids, prov = load_series()
    H = prov["H"]
    out = {"provenance": prov, "analysis": analyse(series, H)}
    path = os.path.join(ROOT, "results", "raw_rhythm_analysis.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["analysis"], indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
