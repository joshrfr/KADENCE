"""Stream the Google 2011 trace part-by-part into per-day task series.

For out-of-sample placement we need several days of the same tasks. This
streams each downloaded ``task_usage`` part, assigns its rows to a (day, slot)
grid, accumulates per (task, day, slot) means, and optionally deletes each part
after reading to bound disk use. A task is kept for a day only if it reports in
every slot of that day.

    python3 experiments/data_prep/extract_gct_days.py --parts 'data/gct_raw/part-*.csv.gz' \
        --days 8 --out data/gct_days.npz [--delete-after]

Output ``.npz``: ``series`` (D, N, 2, H) for the tasks present on *all* D days,
``task_ids`` (N,), and provenance. This is the expensive artifact (many parts);
the single-day extractor covers the day-0 headline results on its own.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd

from extract_gct_day0 import C_START, C_JOB, C_TASK, C_CPU, C_MEM, DAY_US, _sha256

BASE_US = 600_000_000                    # day 0 begins at 600 s


def extract_days(parts, H, days, allowed_keys, delete_after=False, verbose=True):
    """Incremental, memory-bounded multi-day extraction.

    Restricts to a fixed candidate task set (``allowed_keys``, the day-0
    full-day tasks) and accumulates directly into pre-sized dense
    (days, n_candidate, H) arrays as each chunk is read, so no giant per-row
    lists are held in memory. Candidate index is found by searchsorted.
    """
    slot_us = DAY_US / H
    end_us = BASE_US + days * DAY_US
    allowed = np.sort(np.asarray(allowed_keys, dtype=np.int64))
    M = len(allowed)
    cpu_s = np.zeros((days, M, H)); mem_s = np.zeros((days, M, H))
    cnt = np.zeros((days, M, H))
    part_sha, rows = {}, 0
    cols, names = [C_START, C_JOB, C_TASK, C_CPU, C_MEM], \
        ["start", "job", "task", "cpu", "mem"]
    for p in sorted(parts):
        part_sha[p.split("/")[-1]] = _sha256(p)
        for ch in pd.read_csv(p, header=None, usecols=cols, names=names,
                              compression="gzip", chunksize=4_000_000,
                              dtype={"job": "int64", "task": "int64"}):
            ch = ch[(ch["start"] >= BASE_US) & (ch["start"] < end_us)]
            if ch.empty:
                continue
            key = ch["job"].to_numpy() * (1 << 20) + ch["task"].to_numpy()
            pos = np.searchsorted(allowed, key)
            pos = np.clip(pos, 0, M - 1)
            keep = allowed[pos] == key                    # only candidate tasks
            if not keep.any():
                continue
            off = ch["start"].to_numpy()[keep] - BASE_US
            dd = (off // DAY_US).astype(np.int64)
            sl = ((off % DAY_US) // slot_us).astype(np.int64)
            ci = pos[keep]
            np.add.at(cpu_s, (dd, ci, sl), np.nan_to_num(ch["cpu"].to_numpy(float)[keep]))
            np.add.at(mem_s, (dd, ci, sl), np.nan_to_num(ch["mem"].to_numpy(float)[keep]))
            np.add.at(cnt, (dd, ci, sl), 1.0)
            rows += int(keep.sum())
        if verbose:
            print(f"  read {p.split('/')[-1]} kept_rows={rows}", flush=True)
        if delete_after:
            os.remove(p)
    uniq = allowed
    # Per-day fullness: is this task present in every slot of this day? Real
    # tasks start and stop, so few are full across *all* days; the OOS driver
    # selects, per (history, test-day) window, the tasks full on those days.
    full = np.all(cnt > 0, axis=2)                        # (days, M) bool
    with np.errstate(invalid="ignore", divide="ignore"):
        cpu_m = np.where(cnt > 0, cpu_s / cnt, 0.0)
        mem_m = np.where(cnt > 0, mem_s / cnt, 0.0)
    series = np.stack([cpu_m, mem_m], axis=2)             # (days, M, 2, H)
    ids = np.array([f"{int(uniq[i] >> 20)}:{int(uniq[i] & ((1 << 20) - 1))}"
                    for i in range(len(uniq))])
    prov = {"dataset": "Google ClusterData 2011 v2.1 task_usage",
            "license": "CC-BY-4.0", "H": H, "days": days, "rows": rows,
            "candidate_tasks": int(len(uniq)),
            "full_task_days": int(full.sum()),
            "parts": sorted(part_sha), "part_sha256": part_sha,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    return series, full, ids, prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", type=int, default=8)
    ap.add_argument("--slots", type=int, default=288)
    ap.add_argument("--day0", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "gct_day0_series.npz"),
        help="day-0 npz; its task set is the candidate pool")
    ap.add_argument("--delete-after", action="store_true")
    args = ap.parse_args()
    parts = []
    for p in args.parts:
        parts.extend(sorted(glob.glob(p)))
    d0 = np.load(args.day0, allow_pickle=True)
    allowed = np.array([int(s.split(":")[0]) * (1 << 20) + int(s.split(":")[1])
                        for s in d0["task_ids"]], dtype=np.int64)
    print(f"candidate pool = {len(allowed)} day-0 tasks", flush=True)
    series, full, ids, prov = extract_days(parts, args.slots, args.days,
                                            allowed, args.delete_after)
    np.savez_compressed(args.out, series=series, full=full, task_ids=ids,
                        provenance=json.dumps(prov))
    print(json.dumps(prov, indent=2))
    print("saved", series.shape, "to", args.out)


if __name__ == "__main__":
    main()
