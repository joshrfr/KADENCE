"""Reshape the Alibaba cluster-trace-v2018 ``machine_usage`` table into the same
per-entity multi-day array layout the OOS driver expects.

Motivation
----------
The Google 2011 slice (``data/gct_days.npz``) is only 5 days long, so the
out-of-sample admission experiment has just 1-2 held-out test windows and is
underpowered. Alibaba cluster-trace-v2018 gives ~4000 *machines*, each with a
continuous CPU **and** memory utilisation series over a full **8 days**
(cpu_util_percent, mem_util_percent), sampled roughly every ~10 s. That is one
more day than we have now and, unlike the time-sliced Azure VM traces, it lives
in a single 1.7 GB file whose rows are sorted by machine, so the first N
machines already carry their whole 8-day history -> disk-safe.

Output layout (matches ``experiments/data_prep/extract_gct_days.py`` / ``sim/oos_v2.py``):
    series : (D_days, N_entities, 2, H)   float64   channel 0 = CPU, 1 = mem, in [0,1]
    full   : (D_days, N_entities)          bool      entity present in EVERY slot of that day
    entity_ids : (N,)                      str
    provenance : json str

machine_usage schema (see cluster-trace-v2018/schema.txt), 0-indexed columns:
    0 machine_id  1 time_stamp(s)  2 cpu_util_percent[0-100]  3 mem_util_percent[0-100]
    4 mem_gps  5 mkpi  6 net_in  7 net_out  8 disk_io_percent

Usage
-----
Prototype against the committed 31 MB sample (125 machines, all 8 days) that
was pulled with an HTTP Range request into ``data/newtrace_raw/``::

    python3 experiments/data_prep/extract_newtrace.py \
        --csv data/newtrace_raw/machine_usage_sample.csv \
        --out data/newtrace_days.npz

Full run (after acquiring the real file; see the acquisition plan in the task
report -- do NOT run a multi-GB download without checking `df -h` first)::

    # on a host with >12 GB free:
    #   wget http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/machine_usage.tar.gz
    #   tar xzf machine_usage.tar.gz          # -> machine_usage.csv (~9 GB)
    python3 experiments/data_prep/extract_newtrace.py --csv machine_usage.csv \
        --out data/newtrace_days.npz --max-entities 4000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd

DAY_S = 86_400                      # seconds per day
INVALID = {-1.0, 101.0}            # documented abnormal sentinels


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def extract(csv_path, H, days, max_entities, chunksize=2_000_000, verbose=True,
            full_frac=1.0):
    """Bin machine_usage rows into per-(entity, day, slot) means.

    First pass over the file discovers the entity set (first ``max_entities``
    distinct machine ids, in file order) and the global start time; second pass
    accumulates sums/counts into dense (days, N, H) grids so nothing per-row is
    held. Two channels: CPU and memory utilisation, rescaled from [0,100] to
    [0,1] to match the Google arrays (which are already fractional).
    """
    slot_s = DAY_S / H
    names = ["mid", "ts", "cpu", "mem", "mem_gps", "mkpi",
             "nin", "nout", "dio"]
    usecols = [0, 1, 2, 3]

    # ---- pass 1: entity order + t0 ------------------------------------------
    order, seen, t0 = [], set(), None
    for ch in pd.read_csv(csv_path, header=None, usecols=usecols,
                          names=["mid", "ts", "cpu", "mem"],
                          chunksize=chunksize):
        if t0 is None:
            t0 = int(ch["ts"].min())
        else:
            t0 = min(t0, int(ch["ts"].min()))
        for mid in ch["mid"].tolist():
            if mid not in seen:
                seen.add(mid); order.append(mid)
                if len(order) >= max_entities:
                    break
        if len(order) >= max_entities:
            break
    idx = {m: i for i, m in enumerate(order)}
    N = len(order)
    if verbose:
        print(f"pass1: N={N} entities, t0={t0}", flush=True)

    cpu_s = np.zeros((days, N, H)); mem_s = np.zeros((days, N, H))
    cnt = np.zeros((days, N, H))
    end_s = t0 + days * DAY_S
    rows = 0

    # ---- pass 2: accumulate --------------------------------------------------
    for ch in pd.read_csv(csv_path, header=None, usecols=usecols,
                          names=["mid", "ts", "cpu", "mem"],
                          chunksize=chunksize):
        ch = ch[ch["mid"].isin(idx)]
        ch = ch[(ch["ts"] >= t0) & (ch["ts"] < end_s)]
        if ch.empty:
            continue
        ei = ch["mid"].map(idx).to_numpy()
        off = ch["ts"].to_numpy() - t0
        dd = (off // DAY_S).astype(np.int64)
        sl = ((off % DAY_S) // slot_s).astype(np.int64)
        cpu = ch["cpu"].to_numpy(float); mem = ch["mem"].to_numpy(float)
        # blank / sentinel values -> NaN so they don't pollute the mean or count
        cpu = np.where(np.isin(cpu, list(INVALID)), np.nan, cpu)
        mem = np.where(np.isin(mem, list(INVALID)), np.nan, mem)
        good = ~(np.isnan(cpu) | np.isnan(mem))
        dd, sl, ei = dd[good], sl[good], ei[good]
        cpu, mem = cpu[good] / 100.0, mem[good] / 100.0     # [0,100] -> [0,1]
        np.add.at(cpu_s, (dd, ei, sl), cpu)
        np.add.at(mem_s, (dd, ei, sl), mem)
        np.add.at(cnt, (dd, ei, sl), 1.0)
        rows += int(good.sum())
    if verbose:
        print(f"pass2: binned {rows} rows", flush=True)

    # "present that day" = at least full_frac of the H slots have >=1 sample.
    # full_frac=1.0 reproduces the strict np.all(cnt>0) behaviour.
    frac_filled = (cnt > 0).mean(axis=2)                    # (days, N)
    full = frac_filled >= full_frac                         # (days, N) bool
    with np.errstate(invalid="ignore", divide="ignore"):
        cpu_m = np.where(cnt > 0, cpu_s / cnt, 0.0)
        mem_m = np.where(cnt > 0, mem_s / cnt, 0.0)
    series = np.stack([cpu_m, mem_m], axis=2)               # (days, N, 2, H)
    ids = np.array(order)
    prov = {"dataset": "Alibaba cluster-trace-v2018 machine_usage",
            "source_url": "http://aliopentrace.oss-cn-beijing.aliyuncs.com/"
                          "v2018Traces/machine_usage.tar.gz",
            "license": "see github.com/alibaba/clusterdata (research use, "
                       "survey/attribution); cite the v2018 trace",
            "channels": ["cpu_util_percent", "mem_util_percent"],
            "H": H, "days": days, "rows": rows, "entities": N,
            "full_entity_days": int(full.sum()),
            "csv_sha256": _sha256(csv_path),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    return series, full, ids, prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="machine_usage.csv (or sample)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", type=int, default=8)
    ap.add_argument("--slots", type=int, default=288, help="H, 5-min slots/day")
    ap.add_argument("--max-entities", type=int, default=4000)
    ap.add_argument("--full-frac", type=float, default=1.0,
                    help="fraction of H slots that must be present for an "
                         "entity-day to count as 'full' (1.0 = strict all-slots)")
    args = ap.parse_args()
    series, full, ids, prov = extract(args.csv, args.slots, args.days,
                                      args.max_entities, full_frac=args.full_frac)
    prov["full_frac"] = args.full_frac
    np.savez_compressed(args.out, series=series, full=full, entity_ids=ids,
                        provenance=json.dumps(prov))
    print(json.dumps(prov, indent=2))
    print("saved", series.shape, "to", args.out)


if __name__ == "__main__":
    main()
