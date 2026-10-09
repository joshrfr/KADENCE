"""Job-bubble mechanism: Fourier phasors + repulsive Kuramoto.

A job's demand for a resource over a cycle of ``H`` slots is a Fourier sum

    x_i^r(t) = m_i^r + sum_{k>=1} a_ik^r cos(2 pi k t / H + phi_ik^r).

Its *bubble* is the triple (mean, the first ``K`` harmonic phasors
``a_ik e^{j phi_ik}``, the residual noise variance). Delaying the job by
``s_i`` slots rotates every phasor: ``theta_ik = phi_ik - 2 pi k s_i / H``.
CPU and memory share one shift (a job cannot move one rhythm without the
other).

A node's excess over its mean is bounded, harmonic by harmonic, by the
magnitude of its amplitude-weighted Kuramoto order parameter
``Z_k^r = sum_i a_ik^r e^{j theta_ik^r}`` (Proposition 1):

    L^r(t) = M^r + sum_k Re(Z_k^r e^{j 2 pi k t / H}) <= M^r + sum_k |Z_k^r|.

Minimising that bound over the shifts is gradient descent on ``sum_k |Z_k^r|``
with a *negative* (repulsive) Kuramoto coupling: each job's phase is pushed
away from the node's collective phase ``Psi_k = angle(Z_k)``, large rhythms
hardest. The rule needs only the node mean field ``Z_k``.

This is a distinct contribution from the strict-neighbour / pressure-fire
line in ``core.neighbor_gossip`` / ``core.pressure_fire``; it is not imported
here and shares no state.

Everything is deterministic given a seed; results are reproducible via the
drivers in ``experiments/simulation/``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

RESOURCES = ("cpu", "mem")
TWO_PI = 2.0 * math.pi


@dataclass
class Bubble:
    """The v4 representation of one job, learned from its per-slot series.

    ``amp``/``phase`` are (R, K) real arrays: harmonic amplitudes and phases
    for k = 1..K per resource. ``mean`` and ``noise_var`` are length-R.
    ``series`` keeps the raw (R, H) telemetry so the *exact* discretised peak
    can be evaluated rather than only the bound.
    """
    mean: np.ndarray        # (R,)
    amp: np.ndarray         # (R, K)
    phase: np.ndarray       # (R, K)  radians
    noise_var: np.ndarray   # (R,)
    series: np.ndarray      # (R, H)
    H: int

    @property
    def K(self) -> int:
        return self.amp.shape[1]


def fit_bubble(series: np.ndarray, K: int) -> Bubble:
    """Fit a bubble to a real per-slot demand series, shape (R, H).

    Uses the real FFT. For a real signal the cosine-form amplitude of
    harmonic k is ``2|X[k]|/H`` and its phase is ``angle(X[k])`` where
    ``X = rfft(x)``. The noise variance is the variance of the residual after
    subtracting the mean and the first K harmonics, i.e. the power in
    harmonics above K (what a phase schedule cannot arrange).
    """
    series = np.asarray(series, dtype=float)
    R, H = series.shape
    K = min(K, H // 2)
    X = np.fft.rfft(series, axis=1)
    mean = X[:, 0].real / H
    amp = np.zeros((R, K))
    phase = np.zeros((R, K))
    for k in range(1, K + 1):
        amp[:, k - 1] = 2.0 * np.abs(X[:, k]) / H
        phase[:, k - 1] = np.angle(X[:, k])
    recon = reconstruct(mean, amp, phase, np.zeros(R), H)
    noise_var = np.var(series - recon, axis=1)
    return Bubble(mean=mean, amp=amp, phase=phase, noise_var=noise_var,
                  series=series, H=H)


def reconstruct(mean, amp, phase, shift_slots, H) -> np.ndarray:
    """Rebuild the (R, H) periodic profile from mean+harmonics at a shift.

    A shift of ``s`` slots rotates harmonic k's phase by ``-2 pi k s / H``.
    """
    mean = np.atleast_1d(mean)
    amp = np.atleast_2d(amp)
    phase = np.atleast_2d(phase)
    shift_slots = np.atleast_1d(shift_slots)
    R, K = amp.shape
    t = np.arange(H)
    out = np.repeat(mean[:, None], H, axis=1).astype(float)
    for r in range(R):
        for k in range(1, K + 1):
            th = phase[r, k - 1] - TWO_PI * k * shift_slots[r % shift_slots.size] / H
            out[r] += amp[r, k - 1] * np.cos(TWO_PI * k * t / H + th)
    return out


def shifted_phase(bubble: Bubble, s: float) -> np.ndarray:
    """Harmonic phases (R, K) after delaying the whole job by s slots."""
    k = np.arange(1, bubble.K + 1)[None, :]
    return bubble.phase - TWO_PI * k * s / bubble.H


def _stack(bubbles: list[Bubble]):
    """Stack a job list into arrays for vectorised computation.

    Returns amp,phase of shape (n,R,K), mean (n,R), and H. Cached on the list
    identity so the greedy admission loop does not rebuild it every step.
    """
    amp = np.stack([b.amp for b in bubbles])
    phase = np.stack([b.phase for b in bubbles])
    mean = np.stack([b.mean for b in bubbles])
    return amp, phase, mean, bubbles[0].H


def _v_shifted_phase(phase, shifts, H):
    """phase (n,R,K) rotated by per-job shift: theta = phi - 2 pi k s/H."""
    K = phase.shape[2]
    k = np.arange(1, K + 1)[None, None, :]          # (1,1,K)
    return phase - TWO_PI * k * shifts[:, None, None] / H


def order_parameter(bubbles: list[Bubble], shifts: np.ndarray, r: int) -> np.ndarray:
    """Amplitude-weighted Kuramoto order parameters Z_k for resource r.

    Returns a complex array of length K (one phasor sum per harmonic).
    """
    amp, phase, mean, H = _stack(bubbles)
    th = _v_shifted_phase(phase, np.asarray(shifts, float), H)   # (n,R,K)
    return (amp[:, r, :] * np.exp(1j * th[:, r, :])).sum(axis=0)


def peak_bound(bubbles: list[Bubble], shifts: np.ndarray, r: int) -> float:
    """Proposition 1 bound: mean load + sum_k |Z_k^r|."""
    M = sum(b.mean[r] for b in bubbles)
    Z = order_parameter(bubbles, shifts, r)
    return M + float(np.abs(Z).sum())


def _v_total(amp, phase, mean, shifts, r, H):
    """Total reconstructed load series (length H) on resource r, vectorised."""
    t = np.arange(H)
    th = _v_shifted_phase(phase, shifts, H)[:, r, :]      # (n,K)
    K = amp.shape[2]
    k = np.arange(1, K + 1)
    ang = TWO_PI * np.outer(k, t) / H                     # (K,H)
    # sum_n mean + sum_{n,k} amp cos(ang_k(t) + th_nk)
    contrib = np.einsum("nk,kt->t", amp[:, r, :] * np.cos(th),  np.cos(ang)) \
            - np.einsum("nk,kt->t", amp[:, r, :] * np.sin(th),  np.sin(ang))
    return mean[:, r].sum() + contrib


def exact_peak(bubbles: list[Bubble], shifts: np.ndarray, r: int) -> float:
    """Exact discretised peak of total load on resource r at these shifts."""
    amp, phase, mean, H = _stack(bubbles)
    return float(_v_total(amp, phase, mean, np.asarray(shifts, float), r, H).max())


def _peak_resource(bubbles: list[Bubble], shifts: np.ndarray) -> int:
    """Index of the resource that currently sets the node peak."""
    return int(np.argmax([exact_peak(bubbles, shifts, r)
                          for r in range(bubbles[0].series.shape[0])]))


def _bound_grad(bubbles: list[Bubble], shifts: np.ndarray, r: int) -> np.ndarray:
    """Analytic gradient of sum_k |Z_k^r| w.r.t. each shift s_i (vectorised).

    d|Z_k|/ds_i = (2 pi k / H) a_ik ( Zx sin th_ik - Zy cos th_ik ) / |Z_k|,
    the repulsive law: it pushes theta_ik away from Psi_k = angle(Z_k),
    weighted by amplitude and by harmonic number.
    """
    amp, phase, mean, H = _stack(bubbles)
    return _v_bound_grad(amp, phase, np.asarray(shifts, float), r, H)


def _v_bound_grad(amp, phase, shifts, r, H):
    K = amp.shape[2]
    kk = np.arange(1, K + 1)[None, :]                     # (1,K)
    th = _v_shifted_phase(phase, shifts, H)[:, r, :]      # (n,K)
    a = amp[:, r, :]
    Z = (a * np.exp(1j * th)).sum(axis=0)                 # (K,)
    mag = np.abs(Z); mag[mag < 1e-12] = 1e-12
    dZx = a * np.sin(th) * (TWO_PI * kk / H)              # (n,K)
    dZy = -a * np.cos(th) * (TWO_PI * kk / H)
    return ((Z.real * dZx + Z.imag * dZy) / mag).sum(axis=1)   # (n,)


def repulsive_kuramoto(bubbles: list[Bubble], starts: int = 6,
                       steps: int = 200, eta0: float = 4.0,
                       seed: int = 0) -> np.ndarray:
    """Minimise the node peak by repulsive-Kuramoto descent on the shifts.

    Annealed gradient descent on the peak-setting resource's bound from
    ``starts`` random initialisations; the arrangement with the lowest *exact*
    discretised peak is returned. Shifts are returned as integer slot delays.
    """
    n = len(bubbles)
    if n <= 1:
        return np.zeros(n, dtype=int)
    amp, phase, mean, H = _stack(bubbles)
    R = amp.shape[1]
    rng = np.random.default_rng(seed)
    best_shift, best_peak = None, math.inf
    for st in range(starts):
        s = rng.uniform(0, H, size=n) if st else np.zeros(n)
        for it in range(steps):
            # Steepest descent on the peak-setting resource's bound. The step
            # is scaled to the slot axis and the gradient is normalised to its
            # largest component, because raw |Z| gradients are tiny relative to
            # H; eta anneals from a broad search to fine adjustment.
            eta = eta0 * (H / 8.0) * (1.0 - it / steps)
            r = int(np.argmax([_v_total(amp, phase, mean, s, rr, H).max()
                               for rr in range(R)]))
            g = _v_bound_grad(amp, phase, s, r, H)
            gmax = np.abs(g).max()
            if gmax < 1e-12:
                break                                    # at an equilibrium
            s = (s - eta * g / gmax) % H                 # descend the bound
        s_int = np.round(s).astype(float) % H
        pk = max(_v_total(amp, phase, mean, s_int, rr, H).max() for rr in range(R))
        if pk < best_peak:
            best_peak, best_shift = pk, s_int.astype(int) % H
    return best_shift


def round_robin_shifts(n: int, H: int) -> np.ndarray:
    """Resource-blind even spread of start times (the strong simple baseline)."""
    return (np.arange(n) * H // max(1, n)) % H


def legacy_desync_shifts(bubbles: list[Bubble]) -> np.ndarray:
    """Refuted baseline: space each job's peak *time* evenly.

    Uses the argmax slot of each job's dominant-resource profile as its phase,
    then lays those peaks out on evenly spaced TDMA slots. Ignores amplitude
    and mixes periods, which is why it does not beat round-robin on real tasks.
    """
    H = bubbles[0].H
    n = len(bubbles)
    peak_slot = []
    for b in bubbles:
        r = int(np.argmax(b.mean + b.amp.sum(axis=1)))
        prof = reconstruct(b.mean[r], b.amp[r], b.phase[r], np.array([0.0]), H)[0]
        peak_slot.append(int(np.argmax(prof)))
    order = np.argsort(peak_slot)
    target = (np.arange(n) * H // max(1, n))
    shifts = np.zeros(n, dtype=int)
    for rank, i in enumerate(order):
        shifts[i] = (peak_slot[i] - target[rank]) % H
    return shifts


def oracle_shifts(bubbles: list[Bubble], starts: int = 24, seed: int = 0) -> np.ndarray:
    """Centralised upper bound: the best arrangement many restarts can find."""
    return repulsive_kuramoto(bubbles, starts=starts, steps=300,
                              eta0=5.0, seed=seed)
