"""One command: build, measure every implementation, emit JSON plus a table.

    python3 bench/run_bench.py [--quick] [--cpu N] [--no-python] [--out FILE]

Runs build/bench (C) and bench/py_bench.py (numpy baseline) once per
(implementation, batch) in a fresh process pinned to one CPU, then measures
stripped size and the ldd closure of one-kernel probe executables.  CPU only,
no network.  Meaning of every column is in README.md.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BUILD = os.path.join(ROOT, "build")

PROBE = {"reserve_peak": 1, "reserve_slot": 2, "reserve_harmonic_k4": 3,
         "deltanet_64x64_step": 5}          # everything ssm_* -> probe 4


def probe_for(impl):
    return PROBE.get(impl, 4)


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout


def probe_info(tmp, idx, name=None):
    src = os.path.join(BUILD, name or f"probe{idx}")
    dst = os.path.join(tmp, os.path.basename(src))
    shutil.copy(src, dst)
    subprocess.run(["strip", "--strip-all", dst], check=True)
    r = subprocess.run(["ldd", dst], capture_output=True, text=True)
    out = (r.stdout + r.stderr)
    libs = [re.split(r"\s+", ln.strip())[0] for ln in out.splitlines()
            if ln.strip() and "statically linked" not in ln and "not a dynamic" not in ln]
    text = int(sh(["size", dst]).splitlines()[1].split()[0])
    return {"stripped_bytes": os.path.getsize(dst), "text_bytes": text, "ldd": libs,
            "static": "statically linked" in out or "not a dynamic" in out}


COLD_PY = ("import sys; sys.path.insert(0, %r); import numpy as np, reference as r, time; "
           "h = np.random.default_rng(1).uniform(0, 1, (1, 3, 2, 288)).astype(np.float32); "
           "r.reserve_slot(h, 3.0); print(time.monotonic_ns())")


def cold_start(cpu, runs, tmp):
    """Median/max ms from spawn to first prediction, one fresh process each time."""
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import numpy as np
    import reference as ref
    w = os.path.join(tmp, "w.sfns")
    rng = np.random.default_rng(1)
    ref.write_ssm(w, 0.1 * rng.standard_normal((64, 64)), rng.standard_normal((64, 8)),
                  rng.standard_normal((2, 64)))

    def one(cmd):
        t0 = time.monotonic_ns()
        out = subprocess.run(["taskset", "-c", str(cpu)] + cmd, check=True,
                             capture_output=True, text=True).stdout
        return (int(out.split()[0]) - t0) / 1e6

    res = {}
    cases = {"c:reserve_peak": [os.path.join(BUILD, "probe1")],
             "c:reserve_slot": [os.path.join(BUILD, "probe2")],
             "c:reserve_harmonic_k4": [os.path.join(BUILD, "probe3")],
             "c:reserve_harmonic_k4_static": [os.path.join(BUILD, "probe3_static")],
             "c:ssm_dense_n64_step": [os.path.join(BUILD, "probe4"), w],
             "c:deltanet_64x64_step": [os.path.join(BUILD, "probe5")],
             "python+numpy:reserve_slot": [sys.executable, "-c", COLD_PY % os.path.join(ROOT, "tests")]}
    for name, cmd in cases.items():
        t = sorted(one(cmd) for _ in range(runs))
        res[name] = {"runs": runs, "median_ms": t[len(t) // 2], "p90_ms": t[int(.9 * len(t))],
                     "max_ms": t[-1], "min_ms": t[0]}
    return res


def idlest_cpu(skip=2):
    """CPU with the lowest busy fraction over one second (host is shared)."""
    def snap():
        out = {}
        for ln in open("/proc/stat"):
            if re.match(r"cpu\d+ ", ln):
                f = [int(x) for x in ln.split()[1:]]
                out[int(ln.split()[0][3:])] = (sum(f), f[3] + f[4])
        return out
    a = snap()
    time.sleep(1.0)
    b = snap()
    busy = {c: 1 - (b[c][1] - a[c][1]) / max(1, b[c][0] - a[c][0]) for c in b if c >= skip}
    return min(busy, key=busy.get)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="smoke run, far below the 10k-call floor")
    ap.add_argument("--cpu", type=int, default=-1, help="CPU to pin to (default: idlest)")
    ap.add_argument("--no-python", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "native_bench.json"))
    a = ap.parse_args()

    if a.cpu < 0:
        a.cpu = idlest_cpu()
    sh(["make", "-C", ROOT, "-s", "all"])
    impls = [ln.split() for ln in sh([os.path.join(BUILD, "bench"), "--list"]).splitlines()]
    extra = (["--calls", "300", "--warmup", "20"] if a.quick else [])
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

    runs = []
    for name, max_batch in impls:
        for batch in (1, 1000):
            if batch > int(max_batch):
                continue
            for runtime in ("c",) + (() if a.no_python else ("python",)):
                if runtime == "c":
                    cmd = [os.path.join(BUILD, "bench"), "--impl", name, "--batch", str(batch),
                           "--cpu", str(a.cpu)] + extra
                else:
                    cmd = [sys.executable, os.path.join(HERE, "py_bench.py"), "--impl", name,
                           "--batch", str(batch), "--cpu", str(a.cpu)] + extra
                t0 = time.time()
                rec = json.loads(sh(cmd, env=env))
                rec["runtime"] = rec.get("runtime", "c")
                rec["lang"] = runtime
                runs.append(rec)
                print(f"  {runtime:6} {name:24} batch {batch:<5} {time.time() - t0:6.1f}s",
                      file=sys.stderr, flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        probes = {"baseline_empty_main": probe_info(tmp, 0)}
        for name, _ in impls:
            probes[name] = probe_info(tmp, probe_for(name))
        probes["reserve_harmonic_k4_static"] = probe_info(tmp, 3, "probe3_static")
        cold = cold_start(a.cpu, 10 if a.quick else 50, tmp)
    py_ldd = sh(["ldd", sys.executable]).split("\n")

    meta = {"host": platform.node(), "cpu": next((l.split(":")[1].strip() for l in open("/proc/cpuinfo")
            if l.startswith("model name")), "?"), "kernel": platform.release(),
            "compiler": sh(["cc", "--version"]).splitlines()[0], "pinned_cpu": a.cpu,
            "loadavg": os.getloadavg(), "quick": a.quick, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cblas": "not built (no cblas.h on host)", "python": sys.version.split()[0]}
    doc = {"meta": meta, "runs": runs, "probes": probes, "cold_start": cold,
           "python_ldd": [l.split()[0] for l in py_ldd if l.strip()]}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=1)

    # ---- readable table (microseconds) ----
    idx = {(r["impl"], r["batch"], r["lang"]): r for r in runs}
    print(f"\nhost: {meta['cpu']}, pinned to cpu {a.cpu}, loadavg {meta['loadavg'][0]:.2f}, {meta['compiler']}")
    print("latency in microseconds per call (batch 1000 = one call over 1000 tasks)\n")
    hdr = f"{'implementation':<24}{'lang':<7}{'batch':>6}{'p50':>11}{'p99':>11}{'p99.9':>11}{'max':>11}{'p50/task':>10}{'RSS MB':>8}{'calls':>7}"
    print(hdr)
    print("-" * len(hdr))
    for name, _ in impls:
        for batch in (1, 1000):
            for lang in ("c", "python"):
                r = idx.get((name, batch, lang))
                if not r:
                    continue
                per = r["p50_ns"] / 1000.0 / batch
                print(f"{name:<24}{lang:<7}{batch:>6}{r['p50_ns'] / 1e3:>11,.2f}{r['p99_ns'] / 1e3:>11,.2f}"
                      f"{r['p999_ns'] / 1e3:>11,.2f}{r['max_ns'] / 1e3:>11,.1f}{per:>10,.3f}"
                      f"{r['rss_kb_after'] / 1024:>8.1f}{r['calls']:>7}")
    tf = [r["timer_floor_ns"] for r in runs if r["lang"] == "c"]
    print(f"\nC timer floor (clock_gettime pair, not subtracted): median {sorted(tf)[len(tf) // 2]} ns")
    tbase = probes["baseline_empty_main"]["text_bytes"]
    print("\nlinked cost of one kernel (stripped probe executable, dynamic glibc)")
    print(f"{'implementation':<28}{'stripped KB':>12}{'.text KB':>10}{'.text over empty':>18}  ldd closure")
    for k, v in probes.items():
        print(f"{k:<28}{v['stripped_bytes'] / 1024:>12.1f}{v['text_bytes'] / 1024:>10.1f}"
              f"{(v['text_bytes'] - tbase) / 1024:>18.1f}"
              f"  {'static' if v['static'] else ', '.join(os.path.basename(x) for x in v['ldd'])}")
    print("\ncold start: spawn to first prediction, ms (fresh process each, includes fork/exec from the harness)")
    print(f"{'case':<34}{'median':>9}{'p90':>9}{'max':>9}{'min':>9}")
    for k, v in cold.items():
        print(f"{k:<34}{v['median_ms']:>9.2f}{v['p90_ms']:>9.2f}{v['max_ms']:>9.2f}{v['min_ms']:>9.2f}")
    py = next((r for r in runs if r["lang"] == "python"), None)
    if py:
        print(f"\npython baseline closure: {len(doc['python_ldd'])} direct libs for the interpreter, "
              f"{py['loaded_shared_objects']} shared objects mapped after numpy import "
              f"({py['loaded_shared_object_bytes'] / 1e6:.0f} MB on disk)")
    print(f"\nJSON: {a.out}")


if __name__ == "__main__":
    main()
