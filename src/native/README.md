# src/native - C reservation and recurrence kernels

C kernels that could be linked into a pure-C scheduler, the numpy reference
each must reproduce, and the harness that measures them. CPU only, no network.

This directory does not claim native inference is faster in any setting that
matters. At a five-minute tick with 10,000 tasks the forecaster needs about 33
inferences per second, a small fraction of one core, so throughput is not the
argument. The arguments are (a) proactive preemption puts inference on the
scheduling path where its p99 adds to scheduling latency, (b) a single static
binary with no interpreter, and (c) tail predictability with no garbage
collector or allocator spikes. The numbers below measure kernels in isolation;
none of them is a measurement of inference on the scheduling path.

The intended measurement protocol is batch 1 and 1000, p50/p99/p99.9 over at
least 10,000 calls, one pinned core, RSS, stripped size, ldd closure, cold
start, and deviation from the reference. Two parts of it are not met here: the
machine is shared rather than idle, and the reference is numpy because PyTorch
is not installed on this host. Only the reservation arms and the linear
recurrences are implemented; no convolutional or attention decoder head is, and
no weights from a trained forecaster have been exported into this format.

## Layout

| Path | What |
|------|------|
| `include/sfn.h` | the whole API |
| `src/reserve.c` | peak, slot mean + z*std, K=4 harmonic reservation arms |
| `src/ssm.c` | linear recurrence, weight loader, input-gain step, delta rule |
| `src/probe.c` | one-kernel executables used to measure link cost |
| `bench/bench.c` | C latency harness for one (impl, batch) |
| `bench/py_bench.py` | the same measurement for the numpy reference |
| `bench/run_bench.py` | the single command: runs everything, emits JSON + table |
| `tests/reference.py` | numpy reference for every kernel, and the weight-file writer |
| `tests/gate.py` | correctness gate, fails loudly |

## Run

```bash
cd src/native
make                 # warnings are errors (-Wall -Wextra -Wpedantic -Wconversion -Werror ...)
make test            # correctness gate, then the gate's own sabotage self-test
make bench           # == python3 bench/run_bench.py ; about 40 minutes
python3 bench/run_bench.py --quick   # smoke run, 300 calls: NOT valid latency data
```

`make bench` writes `results/native_bench.json` and prints the table. It
needs gcc, python3 and numpy; `strip`, `size` and `ldd` come from binutils/libc.
Defaults pin each measurement to the idlest CPU (override with `--cpu N`);
the host is shared, so check `loadavg` in the JSON before trusting a tail.

CBLAS is optional. This host has no `cblas.h`, so the default build uses plain
loops and `make CBLAS=1` (which defines `SFN_USE_CBLAS` and links `-lcblas`)
is written but has not been compiled or run here. Nothing was installed.

## The kernels

Reservation arms take a history (days, 2, 288) float32 and return (2, 288).
Arithmetic is double, only the store rounds to float32.

* `sfn_reserve_peak`: per channel max over all days and slots, broadcast.
* `sfn_reserve_slot`: per-slot cross-day mean + z * population std (ddof 0).
* `sfn_reserve_harmonic`: per-slot mean projected onto harmonics 0..4 by a
  direct 5-bin DFT (not a full FFT; 288 is not a power of two and five bins
  cost 2*5*288 MACs), reconstructed, plus z * per-slot std. Twiddles come from
  one caller-owned 288-entry cos/sin table indexed by `(k*s) mod 288`, so the
  angle is reduced exactly in integers. Equivalent to `irfft` of the `rfft`
  with bins above 4 zeroed. The output is not clamped at zero.

The linear recurrence is `h = A h + B x`, `y = C h (+ D x)` with dense or
diagonal A, loaded from a flat file (format in `sfn.h`, writer
`tests/reference.py:write_ssm`). `sfn_ssm_from_buffer` parses in place without
allocating; `sfn_ssm_load` allocates once at init. A fixed A is only one case:
`sfn_ssm_step_gain` takes an input-dependent diagonal gain (selective SSM), and
`sfn_delta_step` is the DeltaNet delta rule on a matrix state,
`S <- S + beta k (v - S^T k)^T`, whose effective transition depends on the
input and cannot be written as a fixed A. The delta kernel is the recurrence
only; the q/k/v/beta projections belong to the network, and no trained model
has been exported to this format yet.

## What the bench numbers mean

Each row is one (implementation, batch) in a fresh process, 20,000 timed calls
at batch 1 (warmup 5,000) and 10,000 at batch 1000 (warmup 200); the Python
rows use warmup 2,000 / 50. All times are microseconds.

* `p50 / p99 / p99.9 / max`: per-call latency, nearest-rank percentiles over
  the timed calls, `CLOCK_MONOTONIC` (`time.perf_counter_ns` in Python). p99.9 of
  10,000 calls is the 10th slowest, so treat it as indicative.
* `batch 1000`: one call that does 1000 independent tasks (one SSM step each,
  one reservation each). `p50/task` is p50 divided by 1000.
* Inputs rotate through a pool of 1000 distinct tasks (about 7 MB of history
  for the reservation arms), so repeated calls are not served from one hot
  line. State persists across calls, so the recurrences run in steady state.
* `timer floor`: cost of a back-to-back `clock_gettime` pair. It is included in
  every C sample and not subtracted; it matters only for the sub-microsecond rows.
* `RSS MB`: `VmRSS` of the measuring process after the timed loop. For C this
  is the process including the pool it allocated; for Python it includes the
  interpreter and numpy, which is the real cost of that path.
* `stripped KB`, `.text KB`: size of a stripped one-kernel probe executable
  (`src/probe.c`, dynamic glibc, `-Wl,--gc-sections`). `.text over empty` subtracts an empty
  `main`. The reservation arms share an object file and the three SSM entry
  points share one, so figures are per link, not per function. A fully static
  link adds glibc, shown separately.
* `cold start`: wall time from the harness spawning a fresh probe process
  (via `taskset`) to the probe printing its timestamp right after its first
  prediction, 50 runs. It includes fork/exec and the dynamic loader. The
  Python row imports numpy and runs one reservation.
* `ldd closure`: the probe's shared-library dependencies. The Python row
  reports the interpreter's libs and the number of shared objects mapped after
  importing numpy.

Python rows run the float64 reference in `tests/reference.py`, not a tuned
numpy implementation, with one BLAS thread. Do not read the C/Python gap as a
speedup claim: the inputs, precision and batching differ, and it is a kernel
in isolation.

## The correctness gate

`tests/gate.py` loads `build/libsfn.so` with ctypes and compares every kernel
against `tests/reference.py` on seeded random cases (days 1/3/7, z of 0/3/-1.5,
uniform, diurnal, large-magnitude, constant and all-zero histories; stable
dense and diagonal models of three sizes loaded through the real file loader;
batch of 1000; 288-step scans; the input-gain step; a 500-step delta-rule run).
It prints the maximum absolute deviation per kernel, the worst element, and a
tolerance of a few float32 ulps relative to the output scale, enforced per
case. Any breach, NaN or exception exits 1 with `GATE FAIL`. `--selftest`
perturbs one output by 1e-3 and requires the gate to catch it. The gate runs
against the same object code the bench uses, built from `src/` via the shared
library.

## Limits

* Single host (Xeon E5-2630 v3, shared, no isolation), no frequency pinning.
* Weights are random and stable, not exported from a trained forecaster.
* `ssm_*_scan288` is batch 1 only: 288 steps x 1000 states per call would make
  10,000 calls take hours.
* SSE2 baseline code; AVX2 is available on this CPU but would need `-mavx2` or
  runtime dispatch, which costs portability.
