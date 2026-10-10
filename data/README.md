# Data

Only one small input is shipped. The large derived trace caches are not
redistributed; this file records what each one is, how it is obtained, and the
checksums of the copies used for the paper.

## Shipped

| File | Bytes | SHA-256 | Licence |
|---|---|---|---|
| `google_cluster_2011_task_usage_slice.csv` | 15,614 | `54a2d8d2fd3142e5f0008fe2507e770a876cd25c1db8a4cd6b78d839a3b03e98` | CC-BY-4.0 |
| `google_cluster_2011_task_usage_slice.metadata.json` | | `2a0a23df273705c63566b125ae18173e32ae42fedd699178f2c854a5e5bbeb16` | CC-BY-4.0 |

The slice is 102 rows (6 tasks, 17 reporting intervals) of the Google
ClusterData 2011 v2.1 `task_usage` table. The metadata file records the
extraction parameters and the source object. It is the only trace input to
`experiments/simulation/raw_trace_replay.py` and to the corresponding test.

## Not shipped

| File | Bytes | SHA-256 | Source |
|---|---|---|---|
| `gct_days.npz` | 487,395,383 | `81ee817361f57a913a7275000392f49fb32564c9a0c5700819153316634026a8` | Google ClusterData 2011 v2.1, multi-day per-task series (5 days per the provenance in `results/oos_placement_h3.json`) |
| `alibaba_days.npz` | 52,456,447 | `baa1839a8dbfabc82da1c3c4f0284adfa716e63d83ac312067b71969c4241a61` | Alibaba cluster trace 2018 |
| `gct_day0_series.npz` | 80,968,854 | `bca336f381f3edbc3093c08253f3f541ffa9886f376f4152d0768c27199cd68c` | Google ClusterData 2011 v2.1, day 0, 5-minute slots |
| `gct32.npz` | 1,520,194,106 | `1e4bb514b1f7fb8f74410b9f3896faf71eadd95eac0d81f72c52ee79310e472c` | float32 projection of `gct_days.npz`, holding `series.npy` and `full.npy` |
| `series_f32.npy` | 1,519,534,208 | `8741e5a16e8f114f79e51982dbb257209e718048fe6a06191e2af130f8c78140` | the same float32 `series` array standalone, as a memmap |

Licences. The Google trace is published under CC-BY-4.0
(https://github.com/google/cluster-data/blob/master/ClusterData2011_2.md). The
Alibaba trace is distributed under the publisher's own terms of use
(https://github.com/alibaba/clusterdata); obtain it from the publisher.

Rebuild entry points are in `experiments/data_prep/`; see the reader notes in
the top-level README for what each one needs. Expected locations are
`data/<name>.npz`. A rebuilt file should be
compared against the SHA-256 above; byte equality is not guaranteed across
NumPy versions.

Every result file derived from these caches is committed under `results/`, so
the numbers in the paper can be inspected without the caches.

The last two rows are caches of a cache and are not inputs. Both are produced
by streaming `gct_days.npz:series` down from float64 to float32, which takes
about 90 seconds and holds little memory; float32 is not a loss of precision
here because `build_examples` already casts every window to float32 before
use. They are listed only so that a 3 GB pair found on a disk is recognisable
as regenerable rather than irreplaceable. Delete them first when space is
needed.
