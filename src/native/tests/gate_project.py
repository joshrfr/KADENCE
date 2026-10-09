"""Gate against the project's own Python rule (experiments/simulation/forecast_models.py).

    python3 tests/gate_project.py [--lib build/libsfn.so] [--jobs 1000]

The oracle composes exactly what build_examples + experiments/forecasting/stage1_cheap.py do
for the harmonic_z arm: per-task float32 window (days, 2, 288), m = win.mean(0),
s = win.std(0), hb = harmonic_recon(m, 4) (float32 rFFT/irFFT), reservation
hb + z * s in float32.  harmonic_recon is copied verbatim from
experiments/simulation/forecast_models.py.  The same jobs are also scored against a float64
oracle, which separates "C differs from float32 numpy" from "either differs
from the exact value".  Exit 1 on any NaN or any breach.
"""
from __future__ import annotations

import argparse
import ctypes as C
import os

import warnings

import numpy as np

warnings.simplefilter("ignore")   # the float32 oracle overflows on purpose-built huge inputs

HERE = os.path.dirname(os.path.abspath(__file__))
F32P = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
TOL_ULP = 64        # float32 ulps of the job's output scale
SUBN = 2.0 ** -149  # smallest float32 subnormal; the ulp budget has this absolute floor


def harmonic_recon(profile, K=4):   # verbatim from experiments/simulation/forecast_models.py
    f = np.fft.rfft(profile, axis=-1)
    if f.shape[-1] > K + 1:
        f[..., K + 1:] = 0
    return np.fft.irfft(f, n=profile.shape[-1], axis=-1).astype(np.float32)


def oracle_f32(win, z):
    m = win.mean(axis=0)
    s = win.std(axis=0)
    return harmonic_recon(m, 4) + np.float32(z) * s            # float32 throughout


def oracle_f64(win, z):
    w = win.astype(np.float64)
    spec = np.fft.rfft(w.mean(axis=0), axis=-1)
    spec[..., 5:] = 0
    return np.fft.irfft(spec, n=288, axis=-1) + z * w.std(axis=0)


def jobs(rng, n):
    out = []
    for i in range(n):
        d = int(rng.choice([2, 3, 5, 7, 14]))
        kind = i % 4
        t = np.arange(288) / 288.0
        if kind == 0:
            w = rng.uniform(0, 1, (d, 2, 288))
        elif kind == 1:
            base = 0.4 + 0.3 * np.sin(2 * np.pi * t)[None, None] * rng.uniform(0.5, 1, (1, 2, 1))
            w = np.clip(base + rng.normal(0, 0.05, (d, 2, 288)), 0, None)
        elif kind == 2:
            w = rng.exponential(1.0, (d, 2, 288)) * (rng.random((d, 2, 288)) < 0.2)
        else:
            w = rng.uniform(0, 1, (d, 2, 288)) * 10.0 ** rng.uniform(-3, 3)
        out.append((w.astype(np.float32), float(rng.choice([0.0, 1.0, 1.645, 3.0, -1.5]))))
    d = 5
    edge = {
        "all_zero": np.zeros((d, 2, 288)),
        "single_slot": np.zeros((d, 2, 288)),
        "constant": np.full((d, 2, 288), 0.37),
        "constant_big": np.full((d, 2, 288), 123456.0),
        "huge_1e30": rng.uniform(0, 1, (d, 2, 288)) * 1e30,
        "tiny_1e-30": rng.uniform(0, 1, (d, 2, 288)) * 1e-30,
        "denormal_1e-40": rng.uniform(0, 1, (d, 2, 288)) * 1e-40,
        "one_day": rng.uniform(0, 1, (1, 2, 288)),
        "spike": np.zeros((d, 2, 288)),
        "mixed_scale": rng.uniform(0, 1, (d, 2, 288)) * np.array([1e-6, 1e6])[None, :, None],
    }
    edge["single_slot"][:, 0, 100] = 7.0
    edge["spike"][2, 1, 287] = 1e4
    for name, w in edge.items():
        for z in (0.0, 3.0):
            out.append((w.astype(np.float32), z, f"{name} z={z}"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lib", default=os.path.join(HERE, "..", "build", "libsfn.so"))
    ap.add_argument("--jobs", type=int, default=1000)
    a = ap.parse_args()
    lib = C.CDLL(a.lib)
    lib.sfn_trig_init.argtypes = [C.c_void_p]
    lib.sfn_reserve_harmonic.argtypes = [C.c_void_p, F32P, C.c_uint32, C.c_float, F32P]
    lib.sfn_reserve_batch.argtypes = [C.c_int, C.c_void_p, F32P, C.c_uint32, C.c_float,
                                      C.c_uint32, F32P]
    trig = C.create_string_buffer(2 * 288 * 8)
    lib.sfn_trig_init(trig)
    rng = np.random.default_rng(20261008)
    J = jobs(rng, a.jobs)
    ulp = np.finfo(np.float32).eps
    maxdev = np.zeros((2, 288))          # per output element, over all jobs, vs float32 oracle
    maxdev64 = np.zeros((2, 288))
    worst_rel, worst_name, bad, dev_o = 0.0, "", 0, 0.0
    worst_rel64 = 0.0
    nan_n = 0
    n32 = 0
    for i, j in enumerate(J):
        w, z = j[0], j[1]
        name = j[2] if len(j) > 2 else f"rand{i}"
        d = w.shape[0]
        out = np.empty((2, 288), np.float32)
        lib.sfn_reserve_harmonic(trig, np.ascontiguousarray(w), d, z, out)
        r32 = oracle_f32(w, z).astype(np.float64)
        r64 = oracle_f64(w, z)
        o = out.astype(np.float64)
        if not np.isfinite(o).all() or not np.isfinite(r64).all():
            nan_n += 1
            bad += 1
            print("NONFINITE", name)
            continue
        dv, dv64 = np.abs(o - r32), np.abs(o - r64)
        scale = max(float(np.abs(r64).max()), 1e-45)
        budget = TOL_ULP * ulp * scale + 4 * SUBN
        # the float32 numpy oracle squares float32 values inside std(), so it overflows
        # above ~1e19 and loses std below ~1e-19; it is only a valid oracle in range
        amax = float(np.abs(w).max())
        f32_valid = (amax == 0.0 or 1e-15 <= amax <= 1e15) and np.isfinite(r32).all()
        if f32_valid:
            maxdev = np.maximum(maxdev, dv)
            n32 += 1
            if dv.max() > budget:
                bad += 1
                print("BREACH vs float32 oracle", name, f"{dv.max() / (ulp * scale):.1f} ulp")
            if dv.max() / (ulp * scale) > worst_rel:
                worst_rel, worst_name = dv.max() / (ulp * scale), name
        maxdev64 = np.maximum(maxdev64, dv64)
        worst_rel64 = max(worst_rel64, float(dv64.max()) / (ulp * scale))
        if dv64.max() > budget:
            bad += 1
            print("BREACH vs float64 oracle", name, f"{dv64.max() / (ulp * scale):.1f} ulp")
    # the batch entry point must equal the single-call path bit for bit
    wb = [j[0] for j in J[:1000] if j[0].shape[0] == 5][:200]
    H = np.ascontiguousarray(np.stack(wb))
    ob = np.empty((len(wb), 2, 288), np.float32)
    lib.sfn_reserve_batch(2, trig, H, 5, 1.645, len(wb), ob)
    os1 = np.empty((2, 288), np.float32)
    batch_eq = True
    for k in range(len(wb)):
        lib.sfn_reserve_harmonic(trig, np.ascontiguousarray(wb[k]), 5, 1.645, os1)
        batch_eq &= bool(np.array_equal(os1, ob[k]))
    print(f"jobs: {len(J)} ({a.jobs} random + {len(J) - a.jobs} edge cases); "
          f"float32 oracle valid on {n32}, float64 oracle on all")
    print(f"max |C - float32 project oracle| per element (valid jobs): {maxdev.max():.6e} "
          f"at {tuple(int(i) for i in np.unravel_index(maxdev.argmax(), maxdev.shape))}; "
          f"absolute, scales with magnitude")
    print(f"  worst in float32 ulps of the job's output scale: {worst_rel:.2f} ({worst_name}); "
          f"budget {TOL_ULP} ulp")
    print(f"max |C - float64 exact oracle| per element (all jobs): {maxdev64.max():.6e}; "
          f"worst in ulps: {worst_rel64:.2f}")
    print(f"batch == single bitwise: {batch_eq}; non-finite C outputs: {nan_n}")
    if bad or not batch_eq:
        print("PROJECT GATE FAIL")
        raise SystemExit(1)
    print("PROJECT GATE PASS")


if __name__ == "__main__":
    main()
