"""Python/numpy baseline timed the same way as bench.c (one impl, one batch).

    python3 bench/py_bench.py --impl reserve_slot --batch 1000 [--calls N] [--cpu C]

Runs the float64 numpy reference from tests/reference.py, so it is the thing the
C kernels are checked against, not a tuned numpy implementation.  Same rotating
input pool and the same nearest-rank percentiles as the C harness; the clock is
time.perf_counter_ns (monotonic).  RSS is read from /proc after the timed loop
and includes the interpreter and numpy, which is the real cost of the Python
path.  Prints one JSON object with the same keys as bench.c.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))
import reference as ref  # noqa: E402

POOL, DAYS, D_IN, D_OUT = 1000, 3, 8, 2


def proc_kb(key):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(key):
                return int(line.split()[1])
    return -1


def loaded_libs():
    seen = {}
    with open("/proc/self/maps") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 6 and ".so" in parts[5] and parts[5].startswith("/"):
                seen[parts[5]] = os.path.getsize(parts[5])
    return seen


def stable_dense(rng, n):
    return (rng.uniform(-1, 1, (n, n)) * 0.9 * math.sqrt(3.0 / n)).astype(np.float32)


def build(impl, rng):
    """Return run(i, batch) for the impl."""
    if impl.startswith("reserve_"):
        hist = rng.uniform(0, 1, (POOL, DAYS, 2, ref.SLOTS)).astype(np.float32)
        fn = {"reserve_peak": ref.reserve_peak, "reserve_slot": ref.reserve_slot,
              "reserve_harmonic_k4": ref.reserve_harmonic}[impl]
        return lambda i, b: fn(hist[i % POOL:i % POOL + 1], 3.0) if b == 1 else fn(hist, 3.0)
    if impl.startswith("ssm_"):
        diag = "_diag_" in impl or impl.startswith("ssm_gain")
        n = int(impl.split("_n")[1].split("_")[0])
        A = rng.uniform(0.5, 0.98, n) if diag else stable_dense(rng, n)
        B = rng.uniform(-0.3, 0.3, (n, D_IN)).astype(np.float32)
        Cm = rng.uniform(-0.3, 0.3, (D_OUT, n)).astype(np.float32)
        H = np.zeros((POOL, n))
        X = rng.uniform(-1, 1, (POOL, D_IN)).astype(np.float32)
        Xseq = rng.uniform(-1, 1, (288, D_IN)).astype(np.float32)
        gain = rng.uniform(0.5, 0.98, n)
        if impl.endswith("scan288"):
            return lambda i, b: ref.ssm_scan(A, B, Cm, None, diag, H[i % POOL], Xseq)
        if impl.startswith("ssm_gain"):
            Bd, Cd = B.astype(np.float64), Cm.astype(np.float64)

            def gain_run(i, b):
                s = slice(i % POOL, i % POOL + 1) if b == 1 else slice(None)
                Hn = gain * H[s] + X[s] @ Bd.T
                H[s] = Hn
                return Hn @ Cd.T
            return gain_run

        def step_run(i, b):
            s = slice(i % POOL, i % POOL + 1) if b == 1 else slice(None)
            Hn, Y = ref.ssm_step(A, B, Cm, None, diag, H[s], X[s])
            H[s] = Hn
            return Y
        return step_run
    if impl == "deltanet_64x64_step":
        dk = 64
        S = np.zeros((POOL, dk, dk))
        Q, V = (rng.uniform(-1, 1, (POOL, dk)) for _ in range(2))
        K = rng.uniform(-1, 1, (POOL, dk))
        K /= np.linalg.norm(K, axis=1, keepdims=True)

        def delta_run(i, b):
            s = slice(i % POOL, i % POOL + 1) if b == 1 else slice(None)
            Sn, y = ref.delta_step(S[s], Q[s], K[s], V[s], 0.5)
            S[s] = Sn
            return y
        return delta_run
    raise SystemExit(f"unknown impl {impl}")


def pct(sorted_s, p):
    r = int(p * len(sorted_s) + 0.999999999)
    return sorted_s[max(r, 1) - 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--impl", required=True)
    ap.add_argument("--batch", type=int, default=1, choices=(1, 1000))
    ap.add_argument("--calls", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--cpu", type=int, default=-1)
    a = ap.parse_args()
    if a.cpu >= 0:
        os.sched_setaffinity(0, {a.cpu})
    calls = a.calls or (20000 if a.batch == 1 else 10000)
    warm = a.warmup or (2000 if a.batch == 1 else 50)
    run = build(a.impl, np.random.default_rng(1))
    rss0 = proc_kb("VmRSS:")
    for i in range(warm):
        run(i, a.batch)
    clk = time.perf_counter_ns
    s = []
    for i in range(calls):
        t0 = clk()
        run(i, a.batch)
        s.append(clk() - t0)
    rss1, hwm = proc_kb("VmRSS:"), proc_kb("VmHWM:")
    fl = sorted(clk() - clk() for _ in range(2001))
    mean = sum(s) / len(s)
    s.sort()
    libs = loaded_libs()
    print(json.dumps({
        "impl": a.impl, "batch": a.batch, "calls": calls, "warmup": warm, "pool": POOL,
        "p50_ns": pct(s, .5), "p99_ns": pct(s, .99), "p999_ns": pct(s, .999),
        "min_ns": s[0], "max_ns": s[-1], "mean_ns": round(mean, 1),
        "timer_floor_ns": fl[1000], "rss_kb_before": rss0, "rss_kb_after": rss1,
        "hwm_kb": hwm, "loaded_shared_objects": len(libs),
        "loaded_shared_object_bytes": sum(libs.values()),
        "runtime": f"python{sys.version.split()[0]}+numpy{np.__version__}"}))


if __name__ == "__main__":
    main()
