#!/usr/bin/env python3
"""Run paired, reproducible 16-worker UDP-ring component probes.

Run on the CloudLab head with SSH agent forwarding and a 16-worker hostfile.
Each worker runs its own localhost UDP ring; this is NOT a cross-node scheduler,
trace replay, job-completion-time, SLO, or utilization experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
from datetime import datetime, timezone


JOBS = (8, 16, 48)
LOSSES = (0.0, 0.1, 0.3)
REPS = 5
NODES = 16


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def check_hosts(path: Path) -> list[str]:
    hosts = [line.strip() for line in path.read_text().splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if len(hosts) != NODES or len(set(hosts)) != NODES:
        raise ValueError(f"expected exactly {NODES} distinct workers; got {len(hosts)} entries")
    return hosts


def schedule(seed: int, alpha: float = 1.0) -> list[dict]:
    cells = list(itertools.product(JOBS, LOSSES))
    rows = []
    for rep in range(REPS):
        order = cells.copy()
        random.Random(seed + rep).shuffle(order)
        for position, (jobs, loss) in enumerate(order):
            rows.append({"rep": rep, "order": position, "jobs": jobs,
                         "loss": loss, "alpha": alpha,
                         "seed": seed + rep * 1000})
    return rows


def cell_name(row: dict) -> str:
    return f"rep{row['rep']:02d}_j{row['jobs']:02d}_loss{round(row['loss'] * 100):02d}"


def valid_result(doc: dict, row: dict) -> list[str]:
    errors = []
    p = doc.get("provenance", {})
    if p.get("mode") != "ssh" or p.get("jobs_per_ring") != row["jobs"]:
        errors.append("mode/jobs provenance mismatch")
    if p.get("rings_per_node") != 1 or p.get("reps") != 1:
        errors.append("rings/reps provenance mismatch")
    try:
        if abs(float(p.get("loss", -1)) - row["loss"]) > 1e-9:
            errors.append("loss provenance mismatch")
        if abs(float(p.get("jitter", -1)) - 0.02) > 1e-9:
            errors.append("jitter provenance mismatch")
        if abs(float(p.get("alpha", -1)) - row.get("alpha", 1.0)) > 1e-9:
            errors.append("alpha provenance mismatch")
    except (TypeError, ValueError):
        errors.append("invalid loss/jitter provenance")
    reps = doc.get("reps", [])
    if len(reps) != 1:
        errors.append(f"expected one repetition, got {len(reps)}")
    else:
        rep = reps[0]
        if rep.get("nodes_reporting") != NODES:
            errors.append(f"expected {NODES} reporting workers, got {rep.get('nodes_reporting')}")
        if rep.get("impl") != "python":
            errors.append("expected Python UDP-ring implementation")
        if not isinstance(rep.get("classification_counts"), dict):
            errors.append("missing measured failure classifications")
        for key in ("measured_datagrams_sent", "measured_datagrams_received",
                    "measured_ticks_executed", "cross_node_messages_measured"):
            if not isinstance(rep.get(key), int) or rep[key] < 0:
                errors.append(f"missing or invalid {key}")
        for key in ("fraction_nodes_converged_loose_0p1_target",
                    "fraction_nodes_converged_strict_1e6"):
            if not isinstance(rep.get(key), (int, float)) or not 0 <= rep[key] <= 1:
                errors.append(f"missing or invalid {key}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hostfile", required=True, type=Path)
    ap.add_argument("--remote-root", required=True, type=Path,
                    help="same source tree visible from head and workers")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--user", default="USER")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="phase-update gain; 1.0 matches the submitted configuration")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    hosts = check_hosts(args.hostfile)
    if args.seconds <= 0:
        ap.error("--seconds must be positive")
    if not 0 < args.alpha <= 1:
        ap.error("--alpha must lie in (0, 1]")
    cluster = args.remote_root / "experiments/testbed/cloudlab/kadence_cluster.py"
    node = args.remote_root / "experiments/testbed/cloudlab/kadence_node.py"
    counted = args.remote_root / "experiments/testbed/cloudlab/kadence_agent_counted.py"
    gossip = args.remote_root / "src/kadence/neighbor_gossip.py"
    distributed = args.remote_root / "experiments/simulation/desync_distributed.py"
    if not all(path.is_file() for path in (cluster, node, counted, gossip, distributed)):
        ap.error("--remote-root is missing a CloudLab runtime dependency")
    if not args.dry_run and not os.environ.get("SSH_AUTH_SOCK"):
        ap.error("SSH_AUTH_SOCK absent: use SSH agent forwarding to reach workers")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = schedule(args.seed, args.alpha)
    manifest = {
        "experiment_id": args.experiment_id,
        "head_hostname": socket.gethostname(), "ssh_user": args.user,
        "remote_root": str(args.remote_root),
        "kind": "16 physical workers; independent localhost UDP rings only",
        "not_measured": ["cross-node scheduler", "JCT", "SLO", "utilization"],
        "workers": hosts, "hostfile_sha256": sha256(args.hostfile),
        "source_sha256": {str(path.relative_to(args.remote_root)): sha256(path)
                          for path in (cluster, node, counted, gossip, distributed)},
        "jobs_per_ring": list(JOBS), "rings_per_node": 1,
        "loss_levels": list(LOSSES), "jitter": 0.02,
        "alpha": args.alpha,
        "seconds_per_repetition": args.seconds,
        "repetitions_per_condition": REPS,
        "paired_seed_rule": "base_seed + repetition * 1000; same across conditions",
        "base_seed": args.seed, "schedule": rows,
    }
    manifest_file = args.out / "manifest.json"
    if manifest_file.exists():
        existing = json.loads(manifest_file.read_text())
        if existing != manifest:
            ap.error("existing manifest differs; use a new output directory")
    else:
        atomic_json(manifest_file, manifest)
    if args.dry_run:
        print(f"validated {NODES} workers; {len(rows)} cells; manifest {manifest_file}")
        return 0
    failures = 0
    for row in rows:
        name = cell_name(row)
        result = args.out / "raw" / f"{name}.json"
        status = args.out / "status" / f"{name}.json"
        if result.exists() and status.exists():
            try:
                old_status = json.loads(status.read_text())
                old_result = json.loads(result.read_text())
                if (old_status.get("returncode") == 0 and not old_status.get("errors")
                        and not valid_result(old_result, row)):
                    print(f"skip validated {name}", flush=True)
                    continue
            except (OSError, ValueError, TypeError):
                pass  # retain corrupt files in scratch and rerun the cell
        cell_base = args.out / "scratch" / name
        attempt = 0
        while (cell_base / f"attempt_{attempt:03d}").exists():
            attempt += 1
        cell_dir = cell_base / f"attempt_{attempt:03d}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        result.parent.mkdir(parents=True, exist_ok=True)
        status.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(cluster), "--mode", "ssh", "--hostfile", str(args.hostfile),
               "--remote-root", str(args.remote_root), "--user", args.user,
               "--jobs", str(row["jobs"]), "--rings", "1", "--seconds", str(args.seconds),
               "--tick", "0.02", "--loss", str(row["loss"]), "--jitter", "0.02",
               "--alpha", str(row["alpha"]),
               "--reps", "1", "--seed", str(row["seed"]), "--out", str(cell_dir)]
        print(f"run {name}", flush=True)
        started = datetime.now(timezone.utc).isoformat()
        proc = subprocess.run(cmd, text=True, capture_output=True)
        log = args.out / "logs" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
        files = list(cell_dir.glob("cloudlab_*.json"))
        errors = []
        if proc.returncode != 0:
            errors.append(f"exit code {proc.returncode}")
        if len(files) != 1:
            errors.append(f"expected one raw JSON, got {len(files)}")
        if len(files) == 1:
            data = json.loads(files[0].read_text())
            errors.extend(valid_result(data, row))
            atomic_json(result, data)  # retain failed rows too, never silently drop them
        atomic_json(status, {"cell": row, "started_utc": started,
                             "finished_utc": datetime.now(timezone.utc).isoformat(),
                             "command": cmd, "returncode": proc.returncode,
                             "errors": errors, "raw": str(result) if result.exists() else None,
                             "log": str(log)})
        if errors:
            failures += 1
            print(f"FAIL {name}: {errors}", flush=True)
    print(f"completed schedule: {len(rows)} cells; {failures} failures", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
