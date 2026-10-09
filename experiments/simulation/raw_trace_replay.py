"""Streaming Google Cluster 2011 trace adapter and replay scaffold.

The repository's older ``trace_sim.py`` is trace-*calibrated*: it generates
synthetic demand with published Alibaba-like properties.  This module is the
separate raw-data path.  It reads the official Google ClusterData 2011 v2.1
``task_usage`` schema without loading a shard into memory, can extract an
exact deterministic slice, and runs conservation-oriented fluid replay.

Important evidence boundary: task-usage rows report work that was actually
observed on Google's cluster.  Treating mean CPU usage as offered work is a
replay surrogate, not proof of counterfactual SLOs or scheduler performance.
The bundled slice is small and deliberately not claimed representative.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Sequence, TextIO


GOOGLE_TRACE_NAME = "Google ClusterData 2011 v2.1 task_usage"
GOOGLE_TRACE_DOC = (
    "https://github.com/google/cluster-data/blob/master/ClusterData2011_2.md"
)
GOOGLE_LICENSE = "CC-BY-4.0"
GOOGLE_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
GOOGLE_TASK_USAGE_COLUMNS = (
    "start_time",
    "end_time",
    "job_id",
    "task_index",
    "machine_id",
    "mean_cpu_usage_rate",
    "canonical_memory_usage",
    "assigned_memory_usage",
    "unmapped_page_cache",
    "total_page_cache",
    "maximum_memory_usage",
    "mean_disk_io_time",
    "mean_local_disk_space_used",
    "maximum_cpu_usage",
    "maximum_disk_io_time",
    "cycles_per_instruction",
    "memory_accesses_per_instruction",
    "sample_portion",
    "aggregation_type",
    "sampled_cpu_usage",
)
DEFAULT_SLICE = Path("data/google_cluster_2011_task_usage_slice.csv")
_TOL = 1e-9


class TraceFormatError(ValueError):
    """A trace row is malformed or violates the replay contract."""


@dataclass(frozen=True)
class UsageSample:
    """One Google task usage interval in normalized trace units."""

    start_us: int
    end_us: int
    job_id: int
    task_index: int
    mean_cpu: float
    canonical_memory: float

    @property
    def task_key(self) -> str:
        return f"{self.job_id}:{self.task_index}"

    @property
    def duration_seconds(self) -> float:
        return (self.end_us - self.start_us) / 1_000_000.0

    @property
    def cpu_work(self) -> float:
        return self.mean_cpu * self.duration_seconds


@dataclass(frozen=True)
class ExtractionReport:
    rows: int
    tasks: tuple[str, ...]
    first_start_us: int
    last_end_us: int
    sha256: str


Allocator = Callable[[dict[str, float], float, tuple[str, ...], int], dict[str, float]]


def _open_text(path: str | os.PathLike[str]) -> TextIO:
    filename = os.fspath(path)
    if filename.endswith(".gz"):
        return gzip.open(filename, "rt", encoding="utf-8", newline="")
    return open(filename, "r", encoding="utf-8", newline="")


def _finite_nonnegative(raw: str, field: str, row_number: int) -> float:
    if raw == "":
        raise TraceFormatError(f"row {row_number}: missing {field}")
    try:
        value = float(raw)
    except ValueError as exc:
        raise TraceFormatError(
            f"row {row_number}: invalid {field} {raw!r}"
        ) from exc
    if not math.isfinite(value) or value < 0.0:
        raise TraceFormatError(
            f"row {row_number}: {field} must be finite and non-negative"
        )
    return value


def parse_google_task_usage(
    path: str | os.PathLike[str],
    *,
    task_filter: frozenset[str] | None = None,
    start_us: int | None = None,
    end_us: int | None = None,
) -> Iterator[UsageSample]:
    """Yield valid v2.1 task-usage rows from a CSV or ``.csv.gz`` shard.

    Empty CPU or canonical-memory fields are not silently imputed: a replay
    cannot conserve work it invents.  Callers that need an imputation study
    should produce a separately labelled derived dataset.
    """

    prior_start = -1
    with _open_text(path) as handle:
        reader = csv.reader(handle)
        for row_number, row in enumerate(reader, 1):
            if not row or (row[0].lstrip().startswith("#")):
                continue
            if len(row) != len(GOOGLE_TASK_USAGE_COLUMNS):
                raise TraceFormatError(
                    f"row {row_number}: expected {len(GOOGLE_TASK_USAGE_COLUMNS)} "
                    f"columns, found {len(row)}"
                )
            try:
                row_start = int(row[0])
                row_end = int(row[1])
                job_id = int(row[2])
                task_index = int(row[3])
            except ValueError as exc:
                raise TraceFormatError(
                    f"row {row_number}: invalid integer identity/time field"
                ) from exc
            if row_start < 0 or row_end <= row_start:
                raise TraceFormatError(
                    f"row {row_number}: interval must have 0 <= start < end"
                )
            if row_start < prior_start:
                raise TraceFormatError(
                    f"row {row_number}: trace is not ordered by start_time"
                )
            prior_start = row_start
            if start_us is not None and row_start < start_us:
                continue
            if end_us is not None and row_start >= end_us:
                continue
            key = f"{job_id}:{task_index}"
            if task_filter is not None and key not in task_filter:
                continue
            yield UsageSample(
                start_us=row_start,
                end_us=row_end,
                job_id=job_id,
                task_index=task_index,
                mean_cpu=_finite_nonnegative(
                    row[5], "mean_cpu_usage_rate", row_number
                ),
                canonical_memory=_finite_nonnegative(
                    row[6], "canonical_memory_usage", row_number
                ),
            )


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_google_slice(
    source: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    tasks: Sequence[str],
    start_us: int,
    end_us: int,
) -> ExtractionReport:
    """Write exact source rows for selected tasks/window, preserving 20 fields."""

    selected = frozenset(tasks)
    if not selected:
        raise ValueError("at least one task key is required")
    if start_us < 0 or end_us <= start_us:
        raise ValueError("extraction requires 0 <= start_us < end_us")

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    seen_tasks: set[str] = set()
    first_start: int | None = None
    last_end: int | None = None
    # Re-serialize parsed fields would change scientific notation/precision.
    # Select the original source rows instead, and validate each selected row.
    prior_start = -1
    with _open_text(source) as source_handle, open(
        destination, "w", encoding="utf-8", newline=""
    ) as output_handle:
        reader = csv.reader(source_handle)
        writer = csv.writer(output_handle, lineterminator="\n")
        for row_number, row in enumerate(reader, 1):
            if not row or row[0].lstrip().startswith("#"):
                continue
            if len(row) != len(GOOGLE_TASK_USAGE_COLUMNS):
                raise TraceFormatError(
                    f"row {row_number}: expected 20 columns, found {len(row)}"
                )
            try:
                row_start, row_end = int(row[0]), int(row[1])
                key = f"{int(row[2])}:{int(row[3])}"
            except ValueError as exc:
                raise TraceFormatError(
                    f"row {row_number}: invalid identity/time field"
                ) from exc
            if row_start < prior_start:
                raise TraceFormatError("source trace is not start-time ordered")
            prior_start = row_start
            if row_start >= end_us:
                break
            if row_start < start_us or key not in selected:
                continue
            # Validation is explicit even though the exact source row is kept.
            _finite_nonnegative(row[5], "mean_cpu_usage_rate", row_number)
            _finite_nonnegative(row[6], "canonical_memory_usage", row_number)
            if row_end <= row_start:
                raise TraceFormatError(f"row {row_number}: invalid interval")
            writer.writerow(row)
            rows += 1
            seen_tasks.add(key)
            first_start = row_start if first_start is None else min(first_start, row_start)
            last_end = row_end if last_end is None else max(last_end, row_end)

    missing = selected - seen_tasks
    if not rows or missing:
        destination.unlink(missing_ok=True)
        raise TraceFormatError(
            "slice selection matched no rows" if not rows
            else f"slice is missing requested tasks: {sorted(missing)}"
        )
    assert first_start is not None and last_end is not None
    return ExtractionReport(
        rows=rows,
        tasks=tuple(sorted(seen_tasks)),
        first_start_us=first_start,
        last_end_us=last_end,
        sha256=sha256_file(destination),
    )


def _max_min_fair(
    backlog: dict[str, float], budget: float, order: tuple[str, ...], _: int
) -> dict[str, float]:
    """Work-conserving max-min fair fluid service (centralized baseline)."""

    allocation = {key: 0.0 for key in order}
    remaining = max(0.0, budget)
    active = {key for key in order if backlog.get(key, 0.0) > _TOL}
    while active and remaining > _TOL:
        share = remaining / len(active)
        progressed = 0.0
        completed: list[str] = []
        for key in sorted(active):
            grant = min(share, backlog[key] - allocation[key])
            allocation[key] += grant
            progressed += grant
            if backlog[key] - allocation[key] <= _TOL:
                completed.append(key)
        remaining -= progressed
        active.difference_update(completed)
        if progressed <= _TOL:
            break
    return allocation


def _max_pressure(
    backlog: dict[str, float], budget: float, order: tuple[str, ...], _: int
) -> dict[str, float]:
    """Greedy largest-backlog-first online baseline."""

    allocation = {key: 0.0 for key in order}
    remaining = max(0.0, budget)
    for key in sorted(order, key=lambda item: (-backlog.get(item, 0.0), item)):
        grant = min(backlog.get(key, 0.0), remaining)
        allocation[key] = grant
        remaining -= grant
        if remaining <= _TOL:
            break
    return allocation


def _neighbor_token(
    backlog: dict[str, float], budget: float, order: tuple[str, ...], step: int
) -> dict[str, float]:
    """Local-ring token baseline; this is explicitly not the oscillator."""

    allocation = {key: 0.0 for key in order}
    remaining = max(0.0, budget)
    if not order:
        return allocation
    start = step % len(order)
    for offset in range(len(order)):
        key = order[(start + offset) % len(order)]
        grant = min(backlog.get(key, 0.0), remaining)
        allocation[key] = grant
        remaining -= grant
        if remaining <= _TOL:
            break
    return allocation


POLICIES: Mapping[str, Allocator] = {
    "max-min-fair": _max_min_fair,
    "max-pressure": _max_pressure,
    "neighbor-token-scaffold": _neighbor_token,
}


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _jain(values: Iterable[float]) -> float:
    clean = [value for value in values if value >= 0.0]
    denominator = len(clean) * math.fsum(value * value for value in clean)
    if not clean or denominator <= _TOL:
        return 1.0
    return math.fsum(clean) ** 2 / denominator


def group_intervals(samples: Iterable[UsageSample]) -> list[tuple[int, int, dict[str, UsageSample]]]:
    grouped: dict[tuple[int, int], dict[str, UsageSample]] = defaultdict(dict)
    for sample in samples:
        key = (sample.start_us, sample.end_us)
        if sample.task_key in grouped[key]:
            raise TraceFormatError(
                f"duplicate sample for {sample.task_key} in interval {key}"
            )
        grouped[key][sample.task_key] = sample
    intervals = [
        (start, end, grouped[(start, end)])
        for start, end in sorted(grouped)
    ]
    prior_end: int | None = None
    for start, end, _ in intervals:
        if prior_end is not None and start < prior_end:
            raise TraceFormatError("overlapping reporting intervals are unsupported")
        prior_end = end
    return intervals


def replay_policy(
    intervals: Sequence[tuple[int, int, dict[str, UsageSample]]],
    policy_name: str,
    *,
    capacity: float,
) -> dict[str, object]:
    """Replay CPU usage as fluid arrivals with acknowledged service.

    ``capacity`` is normalized CPU service per second.  Each observed mean CPU
    value is multiplied by its reporting duration and enqueued as work at the
    interval boundary.  This boundary-arrival model is intentionally simple
    and recorded in the output assumptions.
    """

    if policy_name not in POLICIES:
        raise ValueError(f"unknown policy {policy_name!r}")
    if not math.isfinite(capacity) or capacity <= 0.0:
        raise ValueError("capacity must be finite and positive")
    order = tuple(sorted({key for _, _, rows in intervals for key in rows}))
    backlog = {key: 0.0 for key in order}
    arrived = {key: 0.0 for key in order}
    served = {key: 0.0 for key in order}
    backlog_series: list[float] = []
    max_backlog = 0.0
    idle_service = 0.0
    neighbor_messages = 0

    for step, (start, end, rows) in enumerate(intervals):
        duration = (end - start) / 1_000_000.0
        if duration <= 0.0:
            raise TraceFormatError("non-positive replay interval")
        for key, sample in rows.items():
            backlog[key] += sample.cpu_work
            arrived[key] += sample.cpu_work
        budget = capacity * duration
        allocation = POLICIES[policy_name](backlog, budget, order, step)
        if set(allocation) != set(order):
            raise RuntimeError("policy returned an incomplete allocation")
        used = 0.0
        for key in order:
            grant = allocation[key]
            if not math.isfinite(grant) or grant < -_TOL:
                raise RuntimeError("policy returned an invalid service grant")
            if grant > backlog[key] + _TOL:
                raise RuntimeError("policy served work that was not queued")
            grant = min(max(0.0, grant), backlog[key])
            backlog[key] -= grant
            served[key] += grant
            used += grant
        if used > budget + _TOL:
            raise RuntimeError("policy exceeded interval service capacity")
        idle_service += max(0.0, budget - used)
        total_backlog = math.fsum(backlog.values())
        backlog_series.append(total_backlog)
        max_backlog = max(max_backlog, total_backlog)
        if policy_name == "neighbor-token-scaffold":
            neighbor_messages += 2 * len(order)

    total_arrived = math.fsum(arrived.values())
    total_served = math.fsum(served.values())
    final_backlog = math.fsum(backlog.values())
    residual = total_arrived - total_served - final_backlog
    fulfillment = [
        served[key] / arrived[key]
        for key in order if arrived[key] > _TOL
    ]
    return {
        "policy": policy_name,
        "arrived_work": total_arrived,
        "acknowledged_service": total_served,
        "final_backlog": final_backlog,
        "conservation_residual": residual,
        "lost_work": 0.0,
        "idle_service": idle_service,
        "mean_interval_backlog": (
            statistics.fmean(backlog_series) if backlog_series else 0.0
        ),
        "p95_interval_backlog": _percentile(backlog_series, 95.0),
        "maximum_backlog": max_backlog,
        "fulfillment_jain_index": _jain(fulfillment),
        "minimum_task_fulfillment": min(fulfillment, default=1.0),
        "directed_neighbor_messages": neighbor_messages,
    }


def demand_characteristics(
    intervals: Sequence[tuple[int, int, dict[str, UsageSample]]]
) -> dict[str, object]:
    tasks = tuple(sorted({key for _, _, rows in intervals for key in rows}))
    aggregate_cpu: list[float] = []
    aggregate_memory: list[float] = []
    common_energy = 0.0
    total_energy = 0.0
    missing_samples = 0
    for _, _, rows in intervals:
        values = [rows[key].mean_cpu if key in rows else 0.0 for key in tasks]
        memories = [
            rows[key].canonical_memory if key in rows else 0.0 for key in tasks
        ]
        missing_samples += len(tasks) - len(rows)
        mean_value = statistics.fmean(values) if values else 0.0
        common_energy += len(values) * mean_value * mean_value
        total_energy += math.fsum(value * value for value in values)
        aggregate_cpu.append(math.fsum(values))
        aggregate_memory.append(math.fsum(memories))
    return {
        "tasks": len(tasks),
        "intervals": len(intervals),
        "missing_task_intervals_zero_filled": missing_samples,
        "mean_aggregate_cpu": (
            statistics.fmean(aggregate_cpu) if aggregate_cpu else 0.0
        ),
        "peak_aggregate_cpu": max(aggregate_cpu, default=0.0),
        "peak_aggregate_canonical_memory": max(aggregate_memory, default=0.0),
        "common_mode_energy_fraction": (
            common_energy / total_energy if total_energy > _TOL else 0.0
        ),
    }


def evaluate_trace(
    trace_path: str | os.PathLike[str],
    *,
    data_kind: str,
    capacity_fraction_of_peak: float = 0.75,
) -> dict[str, object]:
    """Evaluate replay baselines while preserving raw/fixture provenance."""

    if data_kind not in {"raw-official-slice", "synthetic-fixture"}:
        raise ValueError("data_kind must be raw-official-slice or synthetic-fixture")
    if not 0.0 < capacity_fraction_of_peak <= 2.0:
        raise ValueError("capacity fraction must be in (0, 2]")
    samples = list(parse_google_task_usage(trace_path))
    intervals = group_intervals(samples)
    if not intervals:
        raise TraceFormatError("trace contains no usable intervals")
    characteristics = demand_characteristics(intervals)
    capacity = (
        float(characteristics["peak_aggregate_cpu"])
        * capacity_fraction_of_peak
    )
    if capacity <= 0.0:
        raise TraceFormatError("trace has no positive CPU demand")
    return {
        "experiment": "raw task-usage fluid replay scaffold",
        "evidence_level": (
            "raw official trace slice; replay surrogate, not cluster measurement"
            if data_kind == "raw-official-slice"
            else "synthetic parser fixture; not raw-trace evidence"
        ),
        "data": {
            "kind": data_kind,
            "path": os.fspath(trace_path),
            "sha256": sha256_file(trace_path),
            "dataset": GOOGLE_TRACE_NAME,
            "documentation": GOOGLE_TRACE_DOC,
            "license": GOOGLE_LICENSE,
            "license_url": GOOGLE_LICENSE_URL,
            "rows": len(samples),
            **characteristics,
        },
        "assumptions": [
            "mean observed CPU usage is replayed as offered fluid work",
            "each reporting interval's work arrives at its start boundary",
            "missing task samples are zero demand; values are not imputed",
            "memory is described but not a consumable service backlog",
            "no deadline or counterfactual SLO claim is inferred",
            "neighbor-token-scaffold is a local baseline, not the oscillator",
        ],
        "parameters": {
            "capacity_fraction_of_observed_peak": capacity_fraction_of_peak,
            "normalized_cpu_capacity_per_second": capacity,
        },
        "policies": {
            name: replay_policy(intervals, name, capacity=capacity)
            for name in POLICIES
        },
    }


def _write_json(payload: object, path: str | os.PathLike[str]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--input", required=True)
    extract_parser.add_argument("--output", required=True)
    extract_parser.add_argument("--task", action="append", required=True)
    extract_parser.add_argument("--start-us", type=int, required=True)
    extract_parser.add_argument("--end-us", type=int, required=True)
    extract_parser.add_argument("--metadata-output")
    extract_parser.add_argument("--source-url", required=True)
    extract_parser.add_argument("--source-etag", required=True)
    extract_parser.add_argument("--source-sha256", required=True)

    replay_parser = subparsers.add_parser("evaluate")
    replay_parser.add_argument("--input", default=os.fspath(DEFAULT_SLICE))
    replay_parser.add_argument(
        "--data-kind",
        choices=("raw-official-slice", "synthetic-fixture"),
        required=True,
    )
    replay_parser.add_argument("--capacity-fraction", type=float, default=0.75)
    replay_parser.add_argument(
        "--output", default="results/raw_trace_replay.json"
    )

    args = parser.parse_args(argv)
    if args.command == "extract":
        report = extract_google_slice(
            args.input,
            args.output,
            tasks=args.task,
            start_us=args.start_us,
            end_us=args.end_us,
        )
        metadata = {
            "dataset": GOOGLE_TRACE_NAME,
            "data_kind": "raw-official-slice",
            "documentation": GOOGLE_TRACE_DOC,
            "license": GOOGLE_LICENSE,
            "license_url": GOOGLE_LICENSE_URL,
            "source_object": {
                "url": args.source_url,
                "etag": args.source_etag,
                "sha256": args.source_sha256,
            },
            "extraction": {
                "tasks": list(report.tasks),
                "start_us_inclusive": args.start_us,
                "end_us_exclusive": args.end_us,
                "rows": report.rows,
                "first_start_us": report.first_start_us,
                "last_end_us": report.last_end_us,
                "slice_sha256": report.sha256,
                "columns": list(GOOGLE_TASK_USAGE_COLUMNS),
            },
            "limitations": [
                "one deterministic shard/window/task subset; not representative",
                "usage is observed fulfilled demand, not unconstrained offered load",
                "Google documents missing-data and measurement caveats",
            ],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        if args.metadata_output:
            _write_json(metadata, args.metadata_output)
        print(json.dumps(metadata, indent=2, sort_keys=True))
    else:
        payload = evaluate_trace(
            args.input,
            data_kind=args.data_kind,
            capacity_fraction_of_peak=args.capacity_fraction,
        )
        payload["generated_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(payload, args.output)
        print(json.dumps(payload, indent=2, sort_keys=True))
        for result in payload["policies"].values():
            if abs(float(result["conservation_residual"])) > 1e-7:
                raise SystemExit("work-conservation check failed")


if __name__ == "__main__":
    main()
