# KADENCE: companion artifact

This repository accompanies a paper on KADENCE, a decentralized scheduler for
periodic bursty jobs on shared resources. Each admitted job is a phase
oscillator on a per-node resource ring. Neighbor-only messages drive the ring
toward an evenly spaced layout, so that bursts occupy disjoint parts of their
period without a central coordinator. Topology changes are fenced by epochs,
and a continuity detector bounds what a misbehaving job can claim.

The artifact holds everything behind the reported numbers: a Python reference
implementation and simulator, a Rust prototype of the same rule over UDP
sockets, C reservation kernels, the experiment drivers, the committed raw
results, the figures as they appear in the paper, and the analysis scripts that
turn results into the paper's reported values.

This copy is prepared for double-anonymous review and carries no author,
institution, host or account identifiers.

## Requirements

| Component | Needs |
|---|---|
| Simulators, analysis, tests | Python 3.10 or later with NumPy (`requirements.txt`); pytest for the test suite (`requirements-dev.txt`) |
| Rust prototype | Rust stable with `cargo`; crates are fetched from crates.io |
| C kernels | `gcc`, `make`, and Python 3 with NumPy for the correctness gate |
| Forecaster experiments | PyTorch and a CUDA GPU (`requirements-gpu.txt`); no other result depends on them |
| Physical run | An account on a public research testbed and 16 worker nodes |

The reference environment for the checks below was Linux with Python 3.13.3,
NumPy 2.4.3, Rust 1.92.0 and gcc on a 16-core host, without a GPU.

## Quick start

Run everything from the repository root.

```
pip install -r requirements-dev.txt
python3 -m pytest -q tests                           # 60 tests, about 2 s
python3 experiments/simulation/desync_adversary.py   # about 1 min
```

The adversary run rewrites `results/desync_adversary.json` and reproduces the
committed file byte for byte on the reference host, which is the quickest
end-to-end check that the code and the committed results agree. The unit tests
need no dataset; 54 of the 60 run under plain `unittest`, and the remaining
module needs pytest. `make help` lists shortcuts for the common commands.

## Layout

```
src/                     implementation
  kadence/               Python reference: oscillator model, neighbor-gossip rule,
                         epoch-fenced topology, continuity (safety) detector
  kadence-rs/            Rust/tokio prototype, and equiv/ (Python equivalence harness)
  native/                C admission kernels, benchmarks, correctness gate
experiments/             drivers that produce results/
  simulation/            simulators and trace analyses, one script per result file
  forecasting/           forecaster training and head-to-head comparison (GPU)
  testbed/               physical and cluster runs: cloudlab/, nrp/
  data_prep/             builders for the large trace caches
  evaluate_all.py        orchestrator for the head-to-head comparison
analysis/                turns results/ into the paper's reported values
figures/                 the figures as they appear in the paper (PDF)
results/                 committed raw outputs (JSON, CSV, logs); see results/README.md
data/                    the shipped trace slice; see data/README.md
tests/                   unit tests for src/kadence and the simulators
```

Dependencies run one way: `src/` is imported by `experiments/`, `experiments/`
writes `results/`, and `analysis/` reads `results/`. Data is kept apart from
code, so `results/` and `data/` hold no scripts.

Python scripts put the repository root and `src/` on `sys.path` themselves, so
there is no installation step. The library is imported as `kadence` and the
simulator modules as `experiments.simulation.<name>`.

## From the paper to this repository

Each row names the command behind a reported result and what it writes. The
status column says what a reader needs in order to run it. **Reproduced** means
it was rerun on the reference host and the output matched the committed file,
byte for byte unless noted.

| Paper item | Command | Writes | Status |
|---|---|---|---|
| Convergence and scale | `python3 experiments/simulation/desync_scale.py` | `results/desync_scale.json` | Reproduced, 109 s |
| Convergence at large n | `python3 experiments/simulation/desync_scale_fast.py` | `results/desync_scale_big.json` | Not rerun in this pass |
| Coupling radius | `python3 experiments/simulation/coupling_radius.py` | `results/coupling_radius.json` | Started but did not finish inside a 15 min limit, so it is not verified here |
| Churn tables | `python3 experiments/simulation/churn_evaluation.py --output results/churn_evaluation.json` | `results/churn_evaluation.json` | Reproduced, 1 s, apart from the `generated_at` field |
| Adversary table and figure | `python3 experiments/simulation/desync_adversary.py` | `results/desync_adversary.json` | Reproduced, 62 s |
| Packing, two-resource model | `python3 experiments/simulation/packing.py > results/packing.json` | `results/packing.json` | Reproduced, 44 s; the script prints to stdout |
| Real-trace fairness table | `python3 experiments/simulation/raw_trace_replay.py evaluate --data-kind raw-official-slice` | `results/raw_trace_replay.json` | Reproduced in under 1 s from the shipped slice, apart from `generated_at` |
| Asynchrony and packet loss | `python3 experiments/simulation/desync_distributed.py` | `results/desync_async.json`, `results/desync_distributed.json`, `results/_dist/` | Runs in 8 s over real UDP sockets. The regenerated files differ from the committed ones and were not compared quantitatively, so reproduction is not claimed |
| Packing gain on the real trace | `python3 experiments/simulation/pack_bubble.py` | `results/raw_packing_c*.json` | Needs `data/gct_day0_series.npz` |
| Placement and rhythm statistics | `experiments/simulation/raw_placement.py`, `raw_rhythm_analysis.py` | `results/raw_placement*.json`, `results/raw_rhythm_analysis.json` | Needs `data/gct_day0_series.npz` |
| Out-of-sample placement | `experiments/simulation/oos_placement.py`, `oos_v2.py`, `oos_v3_recency.py`, `oos_v4_learned.py`, `oos_v5_pooled.py` | `results/oos_*.json` | Needs `data/gct_days.npz` |
| Inference versus reservation frontier | `python3 experiments/simulation/phase_frontier.py` | `results/phase_frontier.json` | Needs `data/gct_days.npz` |
| Forecaster comparison | `experiments/forecasting/` (`stage1_cheap.py`, `stage2_tsfm.py`, `train_forecaster.py`, `fair_compare/`) | `results/fair_compare/`, `results/s1nested/`, `results/forecaster_leaderboard.md` | Training needs a GPU and the trace caches. The summaries, `analysis/aggregate_fair_compare.py` and `analysis/aggregate_s1nested.py`, were rerun and reproduce the committed files |
| Head-to-head against baselines | `python3 experiments/evaluate_all.py` | `results/EVALUATION.md`, `results/evaluation_summary.json`, `results/stage1_head_to_head/` | Needs `data/gct_day0_series.npz`; not rerun |
| Reported values for the paper | `python3 analysis/ipdps_snapshot.py`, `python3 analysis/make_snapshot.py` | `analysis/generated/results_snapshot.tex` | Both run without error; the generated file was not diffed against the paper's own macro file |
| Rust prototype | `cd src/kadence-rs && cargo build --release`, then the command below | `results/kadence/rust_16jobs_*` | Builds and runs; see below |
| C kernels | `make -C src/native test` | none | Reproduced: the correctness gate and its sabotage self-test pass |
| C kernel benchmark | `make -C src/native bench` | `src/native/results/native_bench.json` | Not rerun in this pass; the script documents about 40 min |
| Physical-cluster table | `experiments/testbed/cloudlab/` | `results/kadence/cloudlab_*.json`, `results/cloudlab16oct09/` | Needs testbed access; the raw outputs are committed |

The Rust prototype, as run:

```
cd src/kadence-rs && cargo build --release
./target/release/kadence --jobs 16 --rings 1 --loss 0.0 --tick-ms 20 \
    --seconds 15 --reps 5 --out out.json
```

The build succeeds offline from a populated cargo cache and the five replicates
take 75 s. Replicates depend on thread and socket scheduling, so the per-run
convergence count varies; the committed file
`results/kadence/rust_16jobs_quiet_20261007.json` comes from an earlier
revision and uses a different output schema.

The physical run measures protocol behavior with independent localhost UDP
rings on each worker. It is a paired replication of a node-local experiment and
does not measure cross-node scheduling, job completion time or utilization.

## Datasets

A 15 KB slice of the Google trace is shipped so that the trace-replay path runs
out of the box. The two large derived caches are not redistributed; both are
rebuilt from public traces, and `data/README.md` records the details.

| File | Bytes | SHA-256 | Licence | Shipped |
|---|---|---|---|---|
| `data/google_cluster_2011_task_usage_slice.csv` | 15,614 | `54a2d8d2fd3142e5f0008fe2507e770a876cd25c1db8a4cd6b78d839a3b03e98` | CC-BY-4.0 | yes |
| `data/gct_days.npz` | 487,395,383 | `81ee817361f57a913a7275000392f49fb32564c9a0c5700819153316634026a8` | CC-BY-4.0, derived from Google ClusterData 2011 v2.1 | no |
| `data/alibaba_days.npz` | 52,456,447 | `baa1839a8dbfabc82da1c3c4f0284adfa716e63d83ac312067b71969c4241a61` | Publisher's terms, Alibaba cluster trace 2018 | no |
| `data/gct_day0_series.npz` | not recorded | not recorded | CC-BY-4.0, derived from Google ClusterData 2011 v2.1 | no |

The Google trace is documented at
https://github.com/google/cluster-data/blob/master/ClusterData2011_2.md and the
Alibaba trace at https://github.com/alibaba/clusterdata. Every result derived
from the absent caches is committed under `results/`, so the reported numbers
can be inspected and checked without rebuilding them.

## Notes for readers

* Scripts marked "Needs ..." above read a trace cache that is not
  redistributed, and the rebuild path in this artifact is incomplete.
  `experiments/data_prep/extract_gct_days.py` imports `extract_gct_day0`,
  which is not included, and needs pandas, which is not in
  `requirements.txt`. No script here builds `data/gct_day0_series.npz`, and
  `extract_newtrace.py` writes `data/newtrace_days.npz` by default rather than
  `alibaba_days.npz`. A reader without the original caches cannot regenerate
  the rows marked "Needs ...".
* `data/gct_day0_series.npz` is required by several scripts and has no
  recorded checksum.
* The figures are shipped as the PDFs that appear in the paper. The scripts
  that generated them are not included, so the figures cannot be rebuilt from
  this artifact; the results they are drawn from are all committed under
  `results/`.
* `src/kadence-rs/equiv/` compares against a second binary from an earlier
  revision, passed through `KADENCE_OLD_BIN`. That binary is not shipped and
  the harness was not run.
* Some of those scripts default to cache paths outside the artifact, such as
  `/tmp/gct_days_v3.npz`; pass the path explicitly.
* The testbed and cluster manifests under `experiments/testbed/` carry
  site-specific placeholders (`PROJECT`, `USER`, `NAMESPACE`, `<head>`) and
  expect prebuilt binaries in `cloudlab/bin/`.
* Runs over real sockets, in Rust and in `desync_distributed.py`, depend on
  thread and socket timing and are not expected to reproduce bit for bit.
* Provenance strings inside the committed JSON name the module paths in use
  when the results were produced. The results were left untouched so that
  regenerated files stay byte-identical.
* `results/README.md` lists each committed file and the script that produced it.

## Licence and citation

The code is released under the MIT licence (`LICENSE`). The Google trace slice
is CC-BY-4.0. `CITATION.cff` holds a citation stub that stays anonymous until
the review ends.
