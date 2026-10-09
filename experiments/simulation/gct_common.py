"""Shared helpers for the v4 experiment drivers: load the extracted series,
build bubbles, and bootstrap confidence intervals.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

from kadence.kuramoto_packing import fit_bubble

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_SERIES = os.path.join(ROOT, "data", "gct_day0_series.npz")


def load_series(path: str = DEFAULT_SERIES):
    """Return (series (N,2,H), task_ids (N,), provenance dict)."""
    d = np.load(path, allow_pickle=True)
    prov = json.loads(str(d["provenance"]))
    return d["series"], d["task_ids"], prov


def make_bubbles(series: np.ndarray, K: int):
    return [fit_bubble(s, K) for s in series]


def bootstrap_ci(deltas: np.ndarray, iters: int = 10000, seed: int = 0,
                 pct=(2.5, 97.5)):
    """Percentile bootstrap CI of the mean of a paired difference array."""
    rng = np.random.default_rng(seed)
    n = len(deltas)
    means = np.array([rng.choice(deltas, n, replace=True).mean()
                      for _ in range(iters)])
    lo, hi = np.percentile(means, pct)
    return float(deltas.mean()), float(lo), float(hi)
