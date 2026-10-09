# Forecaster sweep

_A sweep over model configurations. It sizes the search; the reported comparison against reservation is the matched run under `results/fair_compare/`._

_epochs=8, max_tasks=120, 1/1 runs ok, 50.3s._

Ranking rule: a config only 'beats peak' if it uses no more nodes than peak-requests at no worse overload on the held-out day.

| config | learned nodes | overload % | peak nodes | beats peak | quant nodes | weakest-decile err | removable features |
|---|---|---|---|---|---|---|---|
| google/gru/h3/q0.95 | 35 | 0.0 | 5 | False | - | 0.02236 | raw,xday_mean,xday_std,harmonic |

**Best:** google/gru/h3/q0.95 (35 nodes, beats_peak=False).

