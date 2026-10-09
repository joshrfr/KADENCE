"""Bounded, real-process workload executor for a CloudLab worker.

Reads one JSON plan from stdin. Times are relative to this process's monotonic
clock; the controller never treats clocks on different hosts as synchronized.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time


def _rss_mb(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/statm", encoding="ascii") as stream:
            return int(stream.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6
    except (FileNotFoundError, ProcessLookupError):
        return 0.0


def _job(duration: float, memory_mb: int) -> None:
    # Touch every page so this is a physical-memory workload, not a reservation.
    data = bytearray(memory_mb * 1024 * 1024)
    for offset in range(0, len(data), 4096):
        data[offset] = offset & 255
    deadline = time.monotonic() + duration
    acc = 0
    while time.monotonic() < deadline:
        for i in range(1000):
            acc = (acc * 33 + i) & 0xffffffff
    if acc < 0:  # prevent an optimizer from dropping the loop
        raise AssertionError(acc)


def execute(plan: dict) -> dict:
    jobs = sorted(plan["jobs"], key=lambda j: (j["start_offset_s"], j["job_id"]))
    cpu_cap = int(plan["cpu_capacity"])
    mem_cap = int(plan["memory_capacity_mb"])
    if cpu_cap <= 0 or mem_cap <= 0:
        raise ValueError("capacities must be positive")
    for job in jobs:
        if not 0 < job["cpu_units"] <= cpu_cap or not 0 < job["memory_mb"] <= mem_cap:
            raise ValueError(f"job {job['job_id']} exceeds worker capacity")
    origin = time.monotonic()
    waiting = list(jobs)
    active: list[tuple[dict, subprocess.Popen, float]] = []
    completed = []
    samples = []
    while waiting or active:
        now = time.monotonic()
        for entry in active[:]:
            job, proc, began = entry
            status = proc.poll()
            if status is not None:
                active.remove(entry)
                completed.append({"job_id": job["job_id"], "arrival_s": job["arrival_s"],
                                  "start_s": began - origin, "finish_s": now - origin,
                                  "jct_s": now - origin - job["arrival_s"],
                                  "slo_s": job["slo_s"], "slo_miss": now - origin - job["arrival_s"] > job["slo_s"],
                                  "exit_code": status, "cpu_units": job["cpu_units"],
                                  "memory_mb": job["memory_mb"]})
        used_cpu = sum(j["cpu_units"] for j, _, _ in active)
        used_mem = sum(j["memory_mb"] for j, _, _ in active)
        # Strict FCFS at each worker; identical resource enforcement in both arms.
        while waiting and waiting[0]["start_offset_s"] <= now - origin:
            job = waiting[0]
            if used_cpu + job["cpu_units"] > cpu_cap or used_mem + job["memory_mb"] > mem_cap:
                break
            waiting.pop(0)
            proc = subprocess.Popen([sys.executable, __file__, "job", str(job["duration_s"]),
                                     str(job["memory_mb"])], stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
            active.append((job, proc, time.monotonic()))
            used_cpu += job["cpu_units"]
            used_mem += job["memory_mb"]
        samples.append({"t_s": now - origin, "reserved_cpu": used_cpu,
                        "reserved_memory_mb": used_mem,
                        "child_rss_mb": sum(_rss_mb(p.pid) for _, p, _ in active)})
        time.sleep(0.02)
    return {"node_id": plan["node_id"], "jobs": completed, "samples": samples,
            "wall_s": time.monotonic() - origin, "worker_pid": os.getpid()}


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "job":
        _job(float(sys.argv[2]), int(sys.argv[3]))
    elif len(sys.argv) == 1:
        print(json.dumps(execute(json.load(sys.stdin))), flush=True)
    else:
        raise SystemExit("usage: worker.py [job DURATION MEMORY_MB]")
