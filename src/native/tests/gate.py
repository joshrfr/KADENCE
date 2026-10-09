"""Correctness gate: every C kernel against its numpy reference.

    python3 tests/gate.py [--lib build/libsfn.so] [--selftest]

For each kernel it reports the maximum absolute deviation per output element
(reservation arms: a (2, 288) map of the worst case over all test inputs).  Any
deviation above tolerance, any NaN, or any exception exits 1 with FAIL.  The
float32 outputs of the C code are compared with float64 reference values, so
the floor is float32 rounding (about 6e-8 relative); tolerances are set as a
small multiple of that and the measured deviation is always printed.
--selftest sabotages one output by 1e-3 and requires the gate to catch it.
"""
from __future__ import annotations

import argparse
import ctypes as C
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reference as ref  # noqa: E402

F32P = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
RESERVE_TOL = 4e-6      # relative to max(1, max|ref|); float32 store is ~6e-8
SSM_TOL = 5e-5          # relative to max(1, max|ref|); float32 accumulation


def load(path):
    lib = C.CDLL(path)
    u32, f32, vp = C.c_uint32, C.c_float, C.c_void_p
    lib.sfn_trig_init.argtypes = [vp]
    lib.sfn_reserve_batch.argtypes = [C.c_int, vp, F32P, u32, f32, u32, F32P]
    lib.sfn_reserve_peak.argtypes = [F32P, u32, F32P]
    lib.sfn_reserve_slot.argtypes = [F32P, u32, f32, F32P]
    lib.sfn_reserve_harmonic.argtypes = [vp, F32P, u32, f32, F32P]
    lib.sfn_ssm_sizeof.restype = C.c_size_t
    lib.sfn_ssm_load.argtypes = [vp, C.c_char_p]
    lib.sfn_ssm_free.argtypes = [vp]
    lib.sfn_ssm_scratch_floats.argtypes = [vp, u32]
    lib.sfn_ssm_scratch_floats.restype = C.c_size_t
    lib.sfn_ssm_step.argtypes = [vp, F32P, F32P, F32P, F32P]
    lib.sfn_ssm_scan.argtypes = [vp, F32P, F32P, F32P, u32, F32P]
    lib.sfn_ssm_step_batch.argtypes = [vp, F32P, F32P, F32P, u32, F32P]
    lib.sfn_ssm_step_gain.argtypes = [vp, F32P, F32P, F32P, F32P]
    lib.sfn_delta_step.argtypes = [F32P, u32, u32, F32P, F32P, F32P, f32, F32P, F32P]
    return lib


class Gate:
    def __init__(self, sabotage=False):
        self.rows, self.sabotage, self.failed = [], sabotage, False

    def record(self, name, got, want, tol_rel, elem_max=None):
        got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
        if self.sabotage and name == self.sabotage:
            got = got.copy()
            got.flat[0] += 1e-3
        dev = np.abs(got - want)
        scale = max(1.0, float(np.abs(want).max()))
        bad = (not np.isfinite(got).all()) or (not np.isfinite(dev).all())
        bad = bad or float(np.nanmax(dev)) > tol_rel * scale     # per-case, own scale
        return dev, scale, bad

    def finish(self, name, dev_map, scale_max, tol_rel, bad, note=""):
        mx = float(np.nanmax(dev_map)) if dev_map.size else 0.0
        where = np.unravel_index(int(np.nanargmax(dev_map)), dev_map.shape) if dev_map.size else ()
        tol = tol_rel * scale_max      # shown for the largest-scale case
        ok = (not bad) and mx <= tol
        self.failed |= not ok
        self.rows.append((name, mx, tol, tuple(int(i) for i in where), ok, note))

    def print(self):
        print(f"{'kernel':<34}{'max |C-ref|':>13}{'tolerance':>12}  worst element   result")
        for name, mx, tol, where, ok, note in self.rows:
            print(f"{name:<34}{mx:>13.3e}{tol:>12.3e}  {str(where):<14}  "
                  f"{'ok' if ok else 'FAIL'} {note}")


def make_histories(rng, B, D, kind):
    t = np.arange(ref.SLOTS) / ref.SLOTS
    if kind == "uniform":
        h = rng.uniform(0, 1, (B, D, 2, ref.SLOTS))
    elif kind == "diurnal":
        base = rng.uniform(0.1, 0.6, (B, 1, 2, 1))
        amp = rng.uniform(0.0, 0.3, (B, 1, 2, 1))
        ph = rng.uniform(0, 2 * np.pi, (B, 1, 2, 1))
        h = base + amp * np.sin(2 * np.pi * t + ph) + 0.05 * rng.standard_normal((B, D, 2, ref.SLOTS))
    elif kind == "large":
        h = 1000.0 * rng.uniform(0, 1, (B, D, 2, ref.SLOTS))
    elif kind == "constant":
        h = np.broadcast_to(rng.uniform(0, 1, (B, 1, 2, 1)), (B, D, 2, ref.SLOTS))
    elif kind == "zeros":
        h = np.zeros((B, D, 2, ref.SLOTS))
    else:
        raise ValueError(kind)
    return np.ascontiguousarray(h, dtype=np.float32)


def gate_reserve(lib, g, rng):
    trig = C.create_string_buffer(16 * ref.SLOTS)
    lib.sfn_trig_init(trig)
    arms = [("peak", 0), ("slot", 1), ("harmonic", 2)]
    for name, arm_id in arms:
        maps = {"unit": np.zeros((ref.CH, ref.SLOTS)), "x1000": np.zeros((ref.CH, ref.SLOTS))}
        scales, bad = {"unit": 1.0, "x1000": 1.0}, False
        for D in (1, 3, 7):
            for kind in ("uniform", "diurnal", "large", "constant", "zeros"):
                for z in (0.0, 3.0, -1.5):
                    hist = make_histories(rng, 20, D, kind)
                    want = ref.ARMS[name](hist, z)
                    out = np.zeros((20, ref.CH, ref.SLOTS), np.float32)
                    lib.sfn_reserve_batch(arm_id, trig, hist, D, z, 20, out)
                    # also the single-task entry points for task 0
                    one = np.zeros((ref.CH, ref.SLOTS), np.float32)
                    if arm_id == 0:
                        lib.sfn_reserve_peak(hist[0], D, one)
                    elif arm_id == 1:
                        lib.sfn_reserve_slot(hist[0], D, z, one)
                    else:
                        lib.sfn_reserve_harmonic(trig, hist[0], D, z, one)
                    if not np.array_equal(one, out[0]):
                        bad = True
                    dev, sc, b = g.record(f"reserve_{name}", out, want, RESERVE_TOL)
                    bad |= b
                    grp = "x1000" if kind == "large" else "unit"
                    scales[grp] = max(scales[grp], sc)
                    maps[grp] = np.maximum(maps[grp], dev.max(axis=0))
        for grp in maps:        # values O(1) and values O(1000) reported apart
            g.finish(f"reserve_{name} [{grp}]", maps[grp], scales[grp], RESERVE_TOL, bad,
                     "single==batch" if not bad else "")


def rand_stable(rng, n, diag):
    if diag:
        return rng.uniform(0.5, 0.98, n)
    M = rng.standard_normal((n, n))
    return 0.9 * M / max(abs(np.linalg.eigvals(M)))      # spectral radius 0.9


def gate_ssm(lib, g, rng):
    tmp = tempfile.mkdtemp()
    for diag in (False, True):
        for n, d_in, d_out, has_d in ((64, 8, 2, False), (128, 8, 2, True), (7, 3, 5, True)):
            A = rand_stable(rng, n, diag)
            B = rng.standard_normal((n, d_in)) / np.sqrt(d_in)
            Cm = rng.standard_normal((d_out, n)) / np.sqrt(n)
            Dm = rng.standard_normal((d_out, d_in)) if has_d else None
            path = os.path.join(tmp, "w.sfns")
            ref.write_ssm(path, A, B, Cm, Dm, diag)
            m = C.create_string_buffer(int(lib.sfn_ssm_sizeof()))
            rc = lib.sfn_ssm_load(m, path.encode())
            if rc != 0:
                g.failed = True
                g.rows.append((f"ssm load n={n}", float("nan"), 0.0, (), False, f"rc={rc}"))
                continue
            # the reference sees the float32-rounded weights the file holds
            A32, B32, C32 = (np.float32(x).astype(np.float64) for x in (A, B, Cm))
            D32 = None if Dm is None else np.float32(Dm).astype(np.float64)
            tag = f"{'diag' if diag else 'dense'} n={n}"
            scratch = np.zeros(int(lib.sfn_ssm_scratch_floats(m, 1000)), np.float32)

            # single step
            h0 = rng.standard_normal(n).astype(np.float32)
            x0 = rng.standard_normal(d_in).astype(np.float32)
            h, y = h0.copy(), np.zeros(d_out, np.float32)
            lib.sfn_ssm_step(m, h, x0, y, scratch)
            Hr, Yr = ref.ssm_step(A32, B32, C32, D32, diag, h0[None], x0[None])
            for nm, got, want in (("step h", h, Hr[0]), ("step y", y, Yr[0])):
                dev, sc, bad = g.record(f"ssm {tag} {nm}", got, want, SSM_TOL)
                g.finish(f"ssm {tag} {nm}", dev, sc, SSM_TOL, bad)

            # 288-step scan
            X = rng.standard_normal((288, d_in)).astype(np.float32)
            h, Y = h0.copy(), np.zeros((288, d_out), np.float32)
            lib.sfn_ssm_scan(m, h, X, Y, 288, scratch)
            hr, Yr = ref.ssm_scan(A32, B32, C32, D32, diag, h0.astype(np.float64), X.astype(np.float64))
            dev, sc, bad = g.record(f"ssm {tag} scan288 y", Y, Yr, SSM_TOL)
            g.finish(f"ssm {tag} scan288 y", dev, sc, SSM_TOL, bad)
            dev, sc, bad = g.record(f"ssm {tag} scan288 h", h, hr, SSM_TOL)
            g.finish(f"ssm {tag} scan288 h", dev, sc, SSM_TOL, bad)

            # batch of 1000
            H = rng.standard_normal((1000, n)).astype(np.float32)
            Xb = rng.standard_normal((1000, d_in)).astype(np.float32)
            Hb, Yb = H.copy(), np.zeros((1000, d_out), np.float32)
            lib.sfn_ssm_step_batch(m, Hb, Xb, Yb, 1000, scratch)
            Hr, Yr = ref.ssm_step(A32, B32, C32, D32, diag, H, Xb)
            dev, sc, bad = g.record(f"ssm {tag} batch1000 y", Yb, Yr, SSM_TOL)
            g.finish(f"ssm {tag} batch1000 y", dev, sc, SSM_TOL, bad)

            # input-dependent gain (diagonal models only)
            if diag:
                gain = rng.uniform(0.3, 0.99, n).astype(np.float32)
                h, y = h0.copy(), np.zeros(d_out, np.float32)
                lib.sfn_ssm_step_gain(m, h, x0, gain, y)
                hr, yr = ref.ssm_step_gain(B32, C32, D32, h0.astype(np.float64),
                                           x0.astype(np.float64), gain.astype(np.float64))
                dev, sc, bad = g.record(f"ssm {tag} gain y", y, yr, SSM_TOL)
                g.finish(f"ssm {tag} input-gain y", dev, sc, SSM_TOL, bad)
            lib.sfn_ssm_free(m)

    # a corrupt image must be rejected, not parsed
    m = C.create_string_buffer(int(lib.sfn_ssm_sizeof()))
    bad_path = os.path.join(tmp, "bad.sfns")
    with open(bad_path, "wb") as f:
        f.write(b"XXXX" + bytes(60))
    rc = lib.sfn_ssm_load(m, bad_path.encode())
    ok = rc < 0
    g.failed |= not ok
    g.rows.append(("ssm loader rejects bad magic", 0.0, 0.0, (), ok, f"rc={rc}"))


def gate_delta(lib, g, rng):
    for dk, dv in ((64, 64), (16, 32)):
        dev_max, bad, scale = np.zeros((dv,)), False, 1.0
        S = (0.1 * rng.standard_normal((dk, dv))).astype(np.float32)
        Sr = S.astype(np.float64)
        scratch = np.zeros(dv, np.float32)
        for t in range(500):                      # a long run so drift would show
            q, k = (rng.standard_normal(dk).astype(np.float32) for _ in range(2))
            k /= np.linalg.norm(k)
            v = rng.standard_normal(dv).astype(np.float32)
            beta = float(np.float32(rng.uniform(0.05, 1.0)))
            y = np.zeros(dv, np.float32)
            lib.sfn_delta_step(S, dk, dv, q, k, v, beta, y, scratch)
            Sr, yr = ref.delta_step(Sr, q, k, v, beta)
            dev, sc, b = g.record(f"delta {dk}x{dv} y", y, yr, SSM_TOL)
            dev_max, bad, scale = np.maximum(dev_max, dev), bad | b, max(scale, sc)
        dev_s, sc, b = g.record(f"delta {dk}x{dv} S", S, Sr, SSM_TOL)
        g.finish(f"delta rule {dk}x{dv} y (500 steps)", dev_max, scale, SSM_TOL, bad)
        g.finish(f"delta rule {dk}x{dv} S (500 steps)", dev_s, sc, SSM_TOL, b)


def run(lib_path, sabotage=False):
    lib = load(lib_path)
    g = Gate(sabotage)
    rng = np.random.default_rng(20261008)
    for fn in (gate_reserve, gate_ssm, gate_delta):
        try:
            fn(lib, g, rng)
        except Exception as e:                          # a crash is a failure
            g.failed = True
            g.rows.append((fn.__name__, float("nan"), 0.0, (), False, repr(e)))
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lib", default=os.path.join(HERE, "..", "build", "libsfn.so"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        g = run(a.lib, sabotage="reserve_slot")
        g.print()
        if not g.failed:
            print("SELFTEST FAIL: sabotaged output was not caught")
            sys.exit(1)
        print("SELFTEST ok: sabotaged output was caught (the FAIL rows above are expected)")
        return
    g = run(a.lib)
    g.print()
    if g.failed:
        print("\nGATE FAIL: a native kernel does not reproduce its reference", file=sys.stderr)
        sys.exit(1)
    print("\nGATE PASS: all kernels reproduce their numpy reference within tolerance")


if __name__ == "__main__":
    main()
