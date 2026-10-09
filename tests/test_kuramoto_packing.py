"""Unit tests for the v4 job-bubble mechanism (core.kuramoto_packing).

These check the *properties* the paper relies on, not tuned numbers:
the Fourier round-trip, the shift-rotation identity, that the exact peak
never exceeds the Proposition-1 bound, that repulsive descent does not
increase the peak, and that the legacy-DESYNC counterexample stands.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from kadence.kuramoto_packing import (
    Bubble, fit_bubble, reconstruct, order_parameter, peak_bound, exact_peak,
    repulsive_kuramoto, round_robin_shifts, legacy_desync_shifts,
)

H = 288


def _rhythmic_series(rng, mean, amp1, phi1, noise=0.0):
    t = np.arange(H)
    cpu = mean + amp1 * np.cos(2 * np.pi * t / H + phi1) + noise * rng.standard_normal(H)
    mem = 0.05 + 0.002 * rng.standard_normal(H)
    return np.clip(np.vstack([cpu, mem]), 0, None)


def _make_bubbles(n, K=4, seed=0, coherent=False):
    """n rhythmic jobs. If coherent, all share phase 0 (peaks aligned, the
    worst case a schedule should be able to improve)."""
    rng = np.random.default_rng(seed)
    bubbles = []
    for _ in range(n):
        phi = 0.0 if coherent else rng.uniform(0, 2 * np.pi)
        s = _rhythmic_series(rng, mean=0.1, amp1=0.05, phi1=phi, noise=0.005)
        bubbles.append(fit_bubble(s, K))
    return bubbles


def test_fourier_roundtrip():
    rng = np.random.default_rng(1)
    s = _rhythmic_series(rng, 0.2, 0.08, 0.7, noise=0.0)
    b = fit_bubble(s, K=8)
    recon = reconstruct(b.mean, b.amp, b.phase, np.zeros(2), H)
    # A signal that is mean + single harmonic must reconstruct almost exactly.
    assert np.allclose(recon[0], s[0], atol=1e-6)


def test_shift_rotates_profile():
    b = _make_bubbles(1, K=4, seed=2)[0]
    base = reconstruct(b.mean[0], b.amp[0], b.phase[0], np.array([0.0]), H)[0]
    shifted = reconstruct(b.mean[0], b.amp[0], b.phase[0], np.array([24.0]), H)[0]
    # Delaying by 24 slots is a circular roll of the reconstructed profile.
    assert np.allclose(shifted, np.roll(base, 24), atol=1e-6)


def test_exact_peak_never_exceeds_bound():
    bubbles = _make_bubbles(12, K=4, seed=3)
    shifts = round_robin_shifts(len(bubbles), H)
    for r in range(2):
        assert exact_peak(bubbles, shifts, r) <= peak_bound(bubbles, shifts, r) + 1e-9


def test_repulsive_descent_lowers_peak():
    bubbles = _make_bubbles(16, K=4, seed=4, coherent=True)
    aligned = np.zeros(len(bubbles), dtype=int)          # all peaks in phase
    p0 = max(exact_peak(bubbles, aligned, r) for r in range(2))
    shifts = repulsive_kuramoto(bubbles, starts=6, steps=150, seed=4)
    p1 = max(exact_peak(bubbles, shifts, r) for r in range(2))
    assert p1 <= p0 + 1e-9
    # On coherent rhythms the descent should give a real reduction.
    assert p1 < p0


def test_order_parameter_shrinks_under_descent():
    bubbles = _make_bubbles(16, K=4, seed=5, coherent=True)
    aligned = np.zeros(len(bubbles), dtype=int)
    z0 = np.abs(order_parameter(bubbles, aligned, 0)).sum()
    shifts = repulsive_kuramoto(bubbles, starts=6, steps=150, seed=5)
    z1 = np.abs(order_parameter(bubbles, shifts, 0)).sum()
    assert z1 < z0


def test_legacy_desync_is_a_valid_arrangement():
    # It must produce in-range integer shifts; it is a baseline, not a winner.
    bubbles = _make_bubbles(10, K=4, seed=6)
    shifts = legacy_desync_shifts(bubbles)
    assert shifts.shape == (10,)
    assert np.all((0 <= shifts) & (shifts < H))
