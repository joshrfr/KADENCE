#!/usr/bin/env python3
"""Validate and summarize a 16-worker UDP-ring campaign without imputing rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

from run_16_worker_campaign import atomic_json, cell_name, valid_result


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate percentile of empty data")
    index = (len(ordered) - 1) * pct / 100
    lo = int(index)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def summarize(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    rows = manifest["schedule"]
    groups: dict[tuple[int, float], list[dict]] = {}
    failed = []
    for row in rows:
        name = cell_name(row)
        result_file = root / "raw" / f"{name}.json"
        status_file = root / "status" / f"{name}.json"
        errors = []
        if not status_file.exists():
            errors.append("missing status")
        else:
            try:
                status = json.loads(status_file.read_text())
                errors.extend(status.get("errors", []))
                if status.get("returncode") != 0:
                    errors.append(f"nonzero return code: {status.get('returncode')}")
            except (OSError, ValueError, TypeError):
                errors.append("invalid status JSON")
        if not result_file.exists():
            errors.append("missing raw JSON")
        else:
            try:
                data = json.loads(result_file.read_text())
                errors.extend(valid_result(data, row))
            except (OSError, ValueError, TypeError, KeyError):
                errors.append("invalid raw JSON")
        if errors:
            failed.append({"cell": name, "errors": sorted(set(errors))})
            continue
        rep = data["reps"][0]
        med = rep["across_node_final_pct_of_fair"]["median"]
        p95 = rep["across_node_final_pct_of_fair"]["p95"]
        groups.setdefault((row["jobs"], row["loss"]), []).append({
            "cell": name, "rep": row["rep"], "median_error_pct_of_fair": med,
            "p95_node_error_pct_of_fair": p95,
            "fraction_nodes_converged_loose_0p1_target":
                rep["fraction_nodes_converged_loose_0p1_target"],
            "fraction_nodes_converged_strict_1e6":
                rep["fraction_nodes_converged_strict_1e6"],
            "classification_counts": rep["classification_counts"],
            "wall_s": rep["wall_s"],
            "measured_datagrams_sent": rep["measured_datagrams_sent"],
            "measured_datagrams_received": rep["measured_datagrams_received"],
            "measured_ticks_executed": rep["measured_ticks_executed"],
            "cross_node_messages_measured": rep["cross_node_messages_measured"],
        })
    conditions = []
    for jobs in manifest["jobs_per_ring"]:
        for loss in manifest["loss_levels"]:
            observations = sorted(groups.get((jobs, loss), []), key=lambda o: o["rep"])
            medians = [x["median_error_pct_of_fair"] for x in observations]
            tails = [x["p95_node_error_pct_of_fair"] for x in observations]
            converged = [x["fraction_nodes_converged_loose_0p1_target"] for x in observations]
            strict = [x["fraction_nodes_converged_strict_1e6"] for x in observations]
            classes: dict[str, int] = {}
            for obs in observations:
                for name, count in obs["classification_counts"].items():
                    classes[name] = classes.get(name, 0) + count
            conditions.append({
                "jobs_per_ring": jobs, "loss": loss,
                "n_valid": len(observations),
                "n_expected": manifest["repetitions_per_condition"],
                "complete": len(observations) == manifest["repetitions_per_condition"],
                "observations": observations,
                "mean_median_error_pct_of_fair": statistics.mean(medians) if medians else None,
                "median_median_error_pct_of_fair": statistics.median(medians) if medians else None,
                "p95_median_error_pct_of_fair": percentile(medians, 95) if medians else None,
                "mean_p95_node_error_pct_of_fair": statistics.mean(tails) if tails else None,
                "mean_fraction_nodes_converged_loose_0p1_target":
                    statistics.mean(converged) if converged else None,
                "mean_fraction_nodes_converged_strict_1e6":
                    statistics.mean(strict) if strict else None,
                "classification_counts": classes,
            })
    return {"experiment_id": manifest["experiment_id"], "kind": manifest["kind"],
            "not_measured": manifest["not_measured"], "expected_cells": len(rows),
            "valid_cells": sum(x["n_valid"] for x in conditions),
            "complete": not failed and all(x["complete"] for x in conditions),
            "failures": failed, "conditions": conditions}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("campaign_dir", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    result = summarize(args.campaign_dir)
    target = args.out or args.campaign_dir / "summary.json"
    atomic_json(target, result)
    print(f"{result['valid_cells']}/{result['expected_cells']} valid cells; "
          f"complete={result['complete']}; {len(result['failures'])} failed/missing; {target}")
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
