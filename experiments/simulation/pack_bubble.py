"""Packing by time-shift on the real trace (v4 raw regime), fast version.

Deferrable tasks may be delayed. Tasks arrive in a seeded random order; the
metric is how many fit on one node before the exact peak of the summed *real*
per-slot series exceeds capacity. Four arrangements are compared:

  round_robin      resource-blind even spread of start times by arrival index
  legacy_desync    space each job's peak *time* evenly (the refuted rule)
  repulsive        online: each job takes the shift that minimises the running
                   peak, the realizable form of repulsive-Kuramoto placement
  oracle           best arrangement found over many random insertion orders

Everything is computed on the raw series with circular shifts, so peaks are
exact and the run is cheap. Writes results/raw_packing_c{scale}.json with
paired bootstrap CIs of the gain over round-robin.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from experiments.simulation.gct_common import load_series, bootstrap_ci, ROOT

GRID = 48                    # candidate shift positions for online placement


def _peak(total):
    return float(total.max())


def _peak_time(job):
    """Slot of the dominant-resource peak of a raw job series (2,H)."""
    r = int(np.argmax(job.mean(axis=1)))
    return int(np.argmax(job[r]))


def cap_round_robin(series, order, cap):
    H = series.shape[2]
    admitted = []
    for idx in order:
        admitted.append(idx)
        n = len(admitted)
        shifts = (np.arange(n) * H // n)
        total = sum(np.roll(series[j], sh, axis=1)
                    for j, sh in zip(admitted, shifts))
        if _peak(total) > cap:
            return n - 1
    return len(admitted)


def cap_legacy(series, order, cap):
    H = series.shape[2]
    admitted = []
    for idx in order:
        admitted.append(idx)
        n = len(admitted)
        peaks = np.array([_peak_time(series[j]) for j in admitted])
        rank = np.argsort(peaks)
        target = np.arange(n) * H // n
        shifts = np.zeros(n, dtype=int)
        for r, j in enumerate(rank):
            shifts[j] = (target[r] - peaks[j]) % H
        total = sum(np.roll(series[admitted[j]], int(shifts[j]), axis=1)
                    for j in range(n))
        if _peak(total) > cap:
            return n - 1
    return len(admitted)


def cap_online(series, order, cap, return_total=False):
    """Greedy online repulsive placement on a shift grid."""
    H = series.shape[2]
    cand = (np.arange(GRID) * H // GRID)
    total = np.zeros((2, H))
    n = 0
    for idx in order:
        job = series[idx]
        best_s, best_pk = 0, np.inf
        for s in cand:
            pk = np.maximum((total + np.roll(job, int(s), axis=1)).max(axis=1),
                            0).max()
            if pk < best_pk:
                best_pk, best_s = pk, s
        if best_pk > cap:
            break
        total = total + np.roll(job, int(best_s), axis=1)
        n += 1
    return (n, total) if return_total else n


def cap_oracle(series, order, cap, tries=8, seed=0):
    """Best count found over several random insertion orders (upper bound)."""
    rng = np.random.default_rng(seed)
    best = cap_online(series, order, cap)
    for _ in range(tries - 1):
        best = max(best, cap_online(series, rng.permutation(order), cap))
    return best


ARRANGERS = {
    "round_robin": cap_round_robin,
    "legacy_desync": cap_legacy,
    "repulsive": cap_online,
    "oracle": cap_oracle,
}


def run_scale(series, scale, seeds, pool):
    cap = float(scale)
    rng = np.random.default_rng(1234)
    counts = {a: np.zeros(seeds) for a in ARRANGERS}
    for si in range(seeds):
        order = rng.choice(len(series), pool, replace=False)
        for a, fn in ARRANGERS.items():
            counts[a][si] = fn(series, order, cap)
    res = {"scale": scale, "capacity": cap, "seeds": seeds, "per_scheme": {}}
    rr = counts["round_robin"]
    for a in ARRANGERS:
        c = counts[a]
        res["per_scheme"][a] = {"mean_fit": round(float(c.mean()), 2),
                                "std": round(float(c.std()), 2)}
        if a != "round_robin":
            gain = 100.0 * (c - rr) / np.maximum(rr, 1)
            m, lo, hi = bootstrap_ci(gain, seed=7)
            res["per_scheme"][a]["gain_vs_rr_pct"] = round(m, 2)
            res["per_scheme"][a]["gain_ci95"] = [round(lo, 2), round(hi, 2)]

    # Per-run gain-to-optimal bracket: how close KADENCE (repulsive) gets to the
    # oracle upper bound on the SAME trace+seed. Paired per seed so the bracket
    # is trace-identical (Sparrow/Apollo/Tarcil-style "within X% of optimal").
    orc = counts["oracle"]
    rep = counts["repulsive"]
    pct_of_oracle = 100.0 * rep / np.maximum(orc, 1)
    m, lo, hi = bootstrap_ci(pct_of_oracle, seed=11)
    res["bracket"] = {
        "lower_ref": "round_robin",
        "upper_ref": "oracle",
        "repulsive_pct_of_oracle": round(m, 2),
        "repulsive_pct_of_oracle_ci95": [round(lo, 2), round(hi, 2)],
        "rr_pct_of_oracle": round(float((100.0 * rr /
                                   np.maximum(orc, 1)).mean()), 2),
    }
    return res


def _jain(values):
    """Jain fairness index: (sum x)^2 / (n * sum x^2); matches raw_trace_replay."""
    v = np.asarray([x for x in values if x >= 0.0], dtype=float)
    denom = len(v) * float(np.sum(v * v))
    if len(v) == 0 or denom <= 1e-12:
        return 1.0
    return float(v.sum() ** 2 / denom)


def across_node_dist(series, scale, nodes, seeds, pool, base_seed=4242):
    """Spread a candidate pool across N nodes; each node packs its share with
    the repulsive (KADENCE) rule under capacity. Report the ACROSS-NODE
    distribution of achieved packing (fit count) and of utilization, plus a
    Jain fairness index across nodes. Averaged over seeds for stability.
    """
    cap = float(scale)
    rng = np.random.default_rng(base_seed + int(round(scale * 10)) + nodes)
    per_node_fit_all = []
    per_node_util_all = []
    jains_fit = []
    jains_util = []
    for _ in range(seeds):
        order = rng.choice(len(series), pool, replace=False)
        # Deal the pool round-robin across nodes (shard the candidate stream).
        shards = [order[i::nodes] for i in range(nodes)]
        fits, utils = [], []
        for shard in shards:
            n, total = cap_online(series, shard, cap, return_total=True)
            fits.append(float(n))
            # Achieved utilization = realized peak load / capacity on this node.
            utils.append(float(total.max() / cap) if cap > 0 else 0.0)
        per_node_fit_all.append(fits)
        per_node_util_all.append(utils)
        jains_fit.append(_jain(fits))
        jains_util.append(_jain(utils))

    fit_arr = np.concatenate([np.asarray(f) for f in per_node_fit_all])
    util_arr = np.concatenate([np.asarray(u) for u in per_node_util_all])

    def _pctl(a):
        return {
            "median": round(float(np.percentile(a, 50)), 4),
            "p5": round(float(np.percentile(a, 5)), 4),
            "p95": round(float(np.percentile(a, 95)), 4),
            "p99": round(float(np.percentile(a, 99)), 4),
            "mean": round(float(a.mean()), 4),
        }

    return {
        "nodes": nodes,
        "seeds": seeds,
        "pool": pool,
        "share_per_node": round(pool / nodes, 2),
        "rule": "repulsive",
        "fit_across_nodes": _pctl(fit_arr),
        "util_across_nodes": _pctl(util_arr),
        "jain_fit_mean": round(float(np.mean(jains_fit)), 4),
        "jain_util_mean": round(float(np.mean(jains_util)), 4),
    }


def run_bracketed(series, prov, scales, seeds, pool, node_counts):
    out = {"mode": "bracketed", "seeds": seeds, "per_scale": {},
           "across_node": {}}
    for scale in scales:
        res = run_scale(series, scale, seeds, pool)
        tag = str(scale).replace(".", "p")
        out["per_scale"][tag] = {
            "scale": scale,
            "capacity": res["capacity"],
            "mean_fit": {a: res["per_scheme"][a]["mean_fit"] for a in ARRANGERS},
            "bracket": res["bracket"],
        }
    # Use the unit-capacity scale for the across-node report (most informative).
    anchor = 1.0 if 1.0 in scales else scales[0]
    for nodes in node_counts:
        out["across_node"][str(nodes)] = across_node_dist(
            series, anchor, nodes, seeds, pool)
    out["across_node_scale"] = anchor
    out["provenance"] = {**prov, "pool": pool, "grid": GRID,
                         "jain_convention": "sum(x)^2 / (n*sum(x^2))"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--pool", type=int, default=200,
                    help="candidate tasks drawn per seed (>= expected fit)")
    ap.add_argument("--bracketed", action="store_true",
                    help="write results/kadence/packing_bracketed.json with "
                         "gain-to-optimal brackets and across-node distribution")
    ap.add_argument("--nodes", nargs="+", type=int, default=[8, 16, 32],
                    help="node counts for the across-node distribution report")
    args = ap.parse_args()
    series, ids, prov = load_series()

    if args.bracketed:
        pool = min(args.pool, len(series))
        out = run_bracketed(series, prov, args.scales, args.seeds, pool,
                            args.nodes)
        outdir = os.path.join(ROOT, "results", "kadence")
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, "packing_bracketed.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        for tag, d in out["per_scale"].items():
            b = d["bracket"]
            print(f"scale {d['scale']}: rr={d['mean_fit']['round_robin']} "
                  f"rep={d['mean_fit']['repulsive']} "
                  f"orc={d['mean_fit']['oracle']} | "
                  f"rep={b['repulsive_pct_of_oracle']}% of oracle "
                  f"CI{b['repulsive_pct_of_oracle_ci95']}", flush=True)
        for nk, nd in out["across_node"].items():
            print(f"nodes={nk}: fit med={nd['fit_across_nodes']['median']} "
                  f"p5={nd['fit_across_nodes']['p5']} "
                  f"p95={nd['fit_across_nodes']['p95']} | "
                  f"jain_fit={nd['jain_fit_mean']} "
                  f"jain_util={nd['jain_util_mean']}", flush=True)
        print(f"-> {path}", flush=True)
        return

    for scale in args.scales:
        pool = min(args.pool, len(series))
        res = run_scale(series, scale, args.seeds, pool)
        res["provenance"] = {**prov, "pool": pool, "grid": GRID}
        tag = str(scale).replace(".", "p")
        path = os.path.join(ROOT, "results", f"raw_packing_c{tag}.json")
        with open(path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"scale {scale}: " + ", ".join(
            f"{a}={res['per_scheme'][a]['mean_fit']}" for a in ARRANGERS)
            + f" -> {path}", flush=True)


if __name__ == "__main__":
    main()
