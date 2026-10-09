# Fair comparison, Google cluster, history 3, 12 configurations

| ld | seed | held | train win | peak | mean+z | harm+z | neural (q) | neural ovl | winner |
|---|---|---|---|---|---|---|---|---|---|
| 7 | 0 | 6 | 78910 | 50 | 33* | 33 | 34 (q=0.9) | 0.00000 | harmonic_z |
| 7 | 1 | 6 | 78910 | 47 | 34 | 31 | 30* (q=0.8) | 0.00104 | harmonic_z |
| 7 | 2 | 6 | 78910 | 49* | 34* | 32* | 34* (q=0.9) | 0.00005 | peak |
| 6 | 0 | 5 | 40481 | 46 | 31 | 32 | 33 (q=0.9) | 0.00000 | mean_z |
| 6 | 1 | 5 | 40481 | 50 | 34* | 36 | 36 (q=0.9) | 0.00000 | harmonic_z/neural |
| 6 | 2 | 5 | 40481 | 53 | 36 | 36 | 37 (q=0.8) | 0.00000 | mean_z/harmonic_z |
| 5 | 0 | 4 | - | 55 | 39* | 40 | n/a | - | harmonic_z |
| 5 | 1 | 4 | - | 53 | 37 | 36 | n/a | - | harmonic_z |
| 5 | 2 | 4 | - | 49 | 36 | 36 | n/a | - | mean_z/harmonic_z |
| 4 | 0 | 3 | - | 50 | n/a | n/a | n/a | - | peak |
| 4 | 1 | 3 | - | 54 | n/a | n/a | n/a | - | peak |
| 4 | 2 | 3 | - | 50 | n/a | n/a | n/a | - | peak |

`*` = held-out overload above zero.

## all configurations (arms averaged over the configs where they exist) (n=12)

| arm | n | mean nodes | range | mean % vs peak | pooled % vs peak | zero-overload | wins | ties |
|---|---|---|---|---|---|---|---|---|
| peak | 12 | 50.5 | 46-55 | 0.0 | 0.0 | 11/12 | 4 | 0 |
| mean_z | 9 | 34.89 | 31-39 | -30.53 | -30.53 | 5/9 | 1 | 2 |
| harmonic_z | 9 | 34.67 | 31-40 | -31.01 | -30.97 | 8/9 | 4 | 3 |
| neural | 6 | 34 | 30-37 | -30.87 | -30.85 | 4/6 | 0 | 1 |

mean vs harmonic: {"n_paired": 9, "mean_fewer_nodes": 3, "harmonic_fewer_nodes": 3, "tied": 3}
neural comparisons: {"n_paired": 6, "mean_z_fewer_than_neural": 4, "mean_z_equal_neural": 1, "mean_z_more_than_neural": 1, "harmonic_z_fewer_than_neural": 4, "harmonic_z_equal_neural": 1, "harmonic_z_more_than_neural": 1, "neural_fewer_than_peak": 6, "neural_zero_overload_and_fewer_than_peak": 4}

## configs where all four arms are defined (last_day 6, 7) (n=6)

| arm | n | mean nodes | range | mean % vs peak | pooled % vs peak | zero-overload | wins | ties |
|---|---|---|---|---|---|---|---|---|
| peak | 6 | 49.17 | 46-53 | 0.0 | 0.0 | 5/6 | 1 | 0 |
| mean_z | 6 | 33.67 | 31-36 | -31.49 | -31.53 | 3/6 | 1 | 1 |
| harmonic_z | 6 | 33.33 | 31-36 | -32.21 | -32.2 | 5/6 | 2 | 2 |
| neural | 6 | 34 | 30-37 | -30.87 | -30.85 | 4/6 | 0 | 1 |

mean vs harmonic: {"n_paired": 6, "mean_fewer_nodes": 2, "harmonic_fewer_nodes": 2, "tied": 2}
neural comparisons: {"n_paired": 6, "mean_z_fewer_than_neural": 4, "mean_z_equal_neural": 1, "mean_z_more_than_neural": 1, "harmonic_z_fewer_than_neural": 4, "harmonic_z_equal_neural": 1, "harmonic_z_more_than_neural": 1, "neural_fewer_than_peak": 6, "neural_zero_overload_and_fewer_than_peak": 4}

Notes:
- winner = fewest held-out nodes among arms whose held-out overload is no worse than peak's own held-out overload (peak always included); zero-overload counts are reported separately
- last_day 5: usable days [3,4] -> val 3, held 4, no training day, neural infeasible
- last_day 4: usable days [3] -> no validation day, only the peak anchor can be scored
- seed changes only the sampled task order and net init; the data day split is fixed by last_day
