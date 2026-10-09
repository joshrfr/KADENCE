# Results

Committed raw outputs. Everything here is data, not code. Files are written by
the scripts named in the mapping table of the top-level README; each JSON file
carries a `provenance` or `meta` block naming its inputs where the script
records one.

`analysis/ipdps_snapshot.py` reads only these files, and every value reported in
the paper comes from one of them:

`churn_evaluation.json`, `coupling_radius.json`, `desync_adversary.json`,
`desync_async.json`, `desync_distributed.json`, `desync_scale.json`,
`oos_phase_infer.json`, `phase_frontier.json`, `phase_velocity_plot.json`,
`raw_packing_c1p0.json`, `raw_placement_ablation.json`, `raw_trace_replay.json`,
`rhythmic_task_count.json`, `fair_compare/summary.json`,
`s1nested/summary.json`, `s1robust/stage1_robustness_summary.json`,
`cloudlab16oct09/kadence16oct09-results-v2/{manifest,summary}.json` and
`src/native/results/native_bench.json`.

Everything else here is supporting or exploratory. In particular
`forecaster_leaderboard.md` and `stage2_real_baselines.md` are small sweeps that
sized the search; the forecaster comparison the paper reports is
`fair_compare/` and `s1nested/`.

| Path | Contents | Produced by |
|---|---|---|
| `desync_scale.json`, `desync_scale_big.json` | settling rounds and energy monotonicity vs ring size | `experiments/simulation/desync_scale.py`, `desync_scale_fast.py` |
| `desync_async.json`, `desync_distributed.json` | asynchrony and packet loss | `experiments/simulation/desync_distributed.py` |
| `coupling_radius.json` | coupling-radius sweep | `experiments/simulation/coupling_radius.py` |
| `churn_evaluation.json` | epoch-fenced join, leave, repair, swap | `experiments/simulation/churn_evaluation.py` |
| `desync_adversary.json` | malicious-fraction sweep, with and without the continuity detector | `experiments/simulation/desync_adversary.py` |
| `raw_packing_c*.json` | packing gain at four capacity ceilings | `experiments/simulation/pack_bubble.py` |
| `packing.json` | two-resource packing at ceiling 1.0 (stdout of the script) | `experiments/simulation/packing.py` |
| `raw_trace_replay.json` | fluid replay of the shipped Google slice | `experiments/simulation/raw_trace_replay.py` |
| `raw_placement*.json`, `raw_rhythm_analysis.json` | placement and rhythm statistics on day-0 series | `raw_placement.py`, `raw_rhythm_analysis.py` |
| `oos_*.json`, `phase_frontier.json` | out-of-sample placement and inference frontier | `experiments/simulation/oos_*.py`, `phase_frontier.py` |
| `fair_compare/`, `s1nested/` | per-seed forecaster comparison runs and their summaries | `experiments/forecasting/fair_compare/fair_compare.py` (runs), `analysis/aggregate_fair_compare.py`, `analysis/aggregate_s1nested.py` (summaries) |
| `stage1_head_to_head/` | head-to-head simulator output (JSON and CSV) | `experiments/simulation/stage1_head_to_head.py` |
| `s1robust/` | per-seed held-out forecaster runs, same record format as `s1nested/` | `experiments/forecasting/`; the exact driver is not identified |
| `kadence/` | Rust prototype and physical-testbed raw outputs | `src/kadence-rs`, `experiments/testbed/cloudlab/` |
| `cloudlab16oct09/` | 16-worker physical campaign, raw and summarised | `experiments/testbed/cloudlab/deploy/` |
| `_dist/` | per-agent final gap error from the localhost UDP run | `experiments/simulation/desync_distributed.py` |
| `EVALUATION.md`, `forecaster_leaderboard.md`, `stage2_real_baselines.md` | human-readable summaries | `experiments/evaluate_all.py`, `experiments/forecasting/train_forecaster.py` |

No producing script is included, under the names these files carry, for
`alibaba_oos_baselines.json`, `duty_sweep.json`, `rapid_*.json`,
`sim_v1.json`, `trace_packing.json`, `stage2_real_baselines.json` and
`oos_v4_learned_h3.json`. They are earlier runs, kept as a record, and no
reported result rests on them.
