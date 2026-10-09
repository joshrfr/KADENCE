"""numpy reference for every kernel in src/ (the correctness oracle).

Inputs are the same float32 arrays the C code sees; the maths is float64.  The
harmonic arm deliberately uses np.fft.rfft/irfft, a different algorithm from
the direct 5-bin projection in src/reserve.c, so agreement is evidence for the
method as well as the arithmetic.  tests/gate.py compares; bench/py_bench.py
also times these functions as the Python baseline.
"""
from __future__ import annotations

import struct

import numpy as np

SLOTS, CH, K = 288, 2, 4
SSM_DIAG, SSM_HAS_D = 1, 2


def reserve_peak(hist, z=0.0):
    """hist (B, D, 2, 288) -> (B, 2, 288): per-task, per-channel max."""
    h = np.asarray(hist, dtype=np.float64)
    return np.broadcast_to(h.max(axis=(1, 3))[:, :, None], (h.shape[0], CH, SLOTS)).copy()


def reserve_slot(hist, z):
    h = np.asarray(hist, dtype=np.float64)
    return h.mean(axis=1) + z * h.std(axis=1)          # ddof = 0


def reserve_harmonic(hist, z):
    h = np.asarray(hist, dtype=np.float64)
    spec = np.fft.rfft(h.mean(axis=1), axis=-1)
    spec[..., K + 1:] = 0.0
    return np.fft.irfft(spec, n=SLOTS, axis=-1) + z * h.std(axis=1)


ARMS = {"peak": reserve_peak, "slot": reserve_slot, "harmonic": reserve_harmonic}


def write_ssm(path, A, B, C, D=None, diag=False):
    """Write the flat SFNS v1 weight file read by sfn_ssm_load()."""
    A, B, C = (np.asarray(a, dtype="<f4") for a in (A, B, C))
    n, d_in = B.shape
    d_out = C.shape[0]
    flags = (SSM_DIAG if diag else 0) | (SSM_HAS_D if D is not None else 0)
    with open(path, "wb") as f:
        f.write(b"SFNS" + struct.pack("<7I", 1, n, d_in, d_out, flags, 0, 0))
        parts = [A, B, C] + ([np.asarray(D, dtype="<f4")] if D is not None else [])
        for p in parts:
            f.write(np.ascontiguousarray(p).tobytes())


def ssm_step(A, B, C, D, diag, H, X):
    """Batched fixed-A step.  H (nb, n), X (nb, d_in) -> (H', Y)."""
    A, B, C, H, X = (np.asarray(a, dtype=np.float64) for a in (A, B, C, H, X))
    Hn = (H * A[None, :] if diag else H @ A.T) + X @ B.T
    Y = Hn @ C.T
    if D is not None:
        Y = Y + X @ np.asarray(D, dtype=np.float64).T
    return Hn, Y


def ssm_scan(A, B, C, D, diag, h, X):
    """Single-sequence scan.  h (n,), X (T, d_in) -> (h', Y (T, d_out))."""
    h = h[None, :]
    ys = []
    for t in range(X.shape[0]):
        h, y = ssm_step(A, B, C, D, diag, h, X[t][None, :])
        ys.append(y[0])
    return h[0], np.stack(ys)


def ssm_step_gain(B, C, D, h, x, gain):
    h = np.asarray(gain, np.float64) * h + np.asarray(B, np.float64) @ x
    y = np.asarray(C, np.float64) @ h
    if D is not None:
        y = y + np.asarray(D, np.float64) @ x
    return h, y


def delta_step(S, q, k, v, beta):
    """S (..., dk, dv); q, k (..., dk); v (..., dv); beta scalar or (...,)."""
    S, q, k, v = (np.asarray(a, dtype=np.float64) for a in (S, q, k, v))
    beta = np.asarray(beta, dtype=np.float64)[..., None]
    pred = np.einsum("...i,...ij->...j", k, S)
    Sn = S + np.einsum("...i,...j->...ij", k, beta * (v - pred))
    return Sn, np.einsum("...i,...ij->...j", q, Sn)
