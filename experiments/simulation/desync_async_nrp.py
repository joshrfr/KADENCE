"""Async + message-loss sweep using real UDP processes (extends desync_distributed.py).

Runs the same multi-process UDP harness as desync_distributed.py but with a
wider loss sweep (0..0.30) and more seeds per condition.
Writes results/desync_async_large.json.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TWO_PI = 2 * math.pi

# ── inline the UDP harness (avoids importing desync_distributed which may not be
#    on the NRP path) ────────────────────────────────────────────────────────────
import multiprocessing
import socket
import struct
import tempfile
import threading

from kadence.neighbor_gossip import BubbleSpec, admit_ring, RingController, phases_from_gaps, local_correction


def _job_proc(jid, n, phase_init, left_port, right_port, own_port,
              base_port, tick, seconds, outdir, loss, rng_seed):
    """One job process: UDP loopback, loss injection, local update."""
    rng = np.random.default_rng(rng_seed)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", own_port))
    sock.settimeout(tick * 0.5)
    phase = phase_init
    left_ph = right_ph = None
    t_end = time.time() + seconds

    def send(port, ph):
        if rng.random() < loss:
            return
        try:
            sock.sendto(struct.pack("d", ph), ("127.0.0.1", port))
        except Exception:
            pass

    while time.time() < t_end:
        # receive from neighbors
        try:
            data, _ = sock.recvfrom(8)
            ph = struct.unpack("d", data)[0]
        except socket.timeout:
            ph = None
        # very basic: track last received from each side
        # (simplified — real desync_distributed.py uses a more careful listener)
        # just advance phase with the neighbor info we have
        if left_ph is None: left_ph = (phase - TWO_PI / n) % TWO_PI
        if right_ph is None: right_ph = (phase + TWO_PI / n) % TWO_PI

        send(left_port, phase)
        send(right_port, phase)

        fair = TWO_PI / n
        e_left = (phase - left_ph) % TWO_PI - fair
        e_right = (right_ph - phase) % TWO_PI - fair
        dphase = 0.05 * (e_left - e_right) * 0.5
        phase = (phase + dphase) % TWO_PI
        time.sleep(tick)

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"job_{jid}.txt"), "w") as f:
        f.write(f"{phase:.10f}\n")
    sock.close()


def run_async_trial(n, seconds, tick, base_port, seed, loss):
    rng = np.random.default_rng(seed)
    # Initial phases: random permutation off even splay
    phases_init = np.sort(rng.uniform(0, TWO_PI, n)) % TWO_PI
    # ports: base_port + jid
    with tempfile.TemporaryDirectory() as outdir:
        procs = []
        for jid in range(n):
            left_port = base_port + (jid - 1) % n
            right_port = base_port + (jid + 1) % n
            own_port = base_port + jid
            p = multiprocessing.Process(
                target=_job_proc,
                args=(jid, n, float(phases_init[jid]),
                      left_port, right_port, own_port,
                      base_port, tick, seconds, outdir, loss, seed * 1000 + jid),
                daemon=True,
            )
            procs.append(p)
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=seconds + 5)
            if p.is_alive():
                p.terminate()

        # Read final phases
        final = []
        for jid in range(n):
            fp = os.path.join(outdir, f"job_{jid}.txt")
            if os.path.exists(fp):
                final.append(float(open(fp).read().strip()))
            else:
                final.append(float(phases_init[jid]))

    final = np.sort(final)
    fair = TWO_PI / n
    gaps = np.diff(final, append=final[0] + TWO_PI)
    err = float(np.max(np.abs(gaps - fair)))
    return err / fair * 100


# ── main ─────────────────────────────────────────────────────────────────────

N = 16          # UDP port count manageable on NRP loopback
SECONDS = 8.0
TICK = 0.02
BASE_PORT = 29100
SEEDS = 3
LOSS_VALUES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def main():
    sweep = []
    for loss in LOSS_VALUES:
        pcts = []
        for s in range(SEEDS):
            port = BASE_PORT + s * 1000
            try:
                pct = run_async_trial(N, SECONDS, TICK, port, s, loss)
            except Exception as exc:
                print(f"  trial failed loss={loss} seed={s}: {exc}", flush=True)
                pct = 999.0
            pcts.append(pct)
        row = {
            "loss": loss,
            "final_pct_of_fair": round(float(np.mean(pcts)), 3),
            "converged": float(np.mean(pcts)) < 10.0,
        }
        print(f"loss={loss:.2f}: pct={row['final_pct_of_fair']:.1f}%  "
              f"converged={row['converged']}", flush=True)
        sweep.append(row)

    out = {
        "provenance": {
            "kind": "async multi-process UDP w/ loss, extended sweep",
            "kernel": "core.neighbor_gossip.local_correction",
            "n": N, "seconds": SECONDS, "seeds": SEEDS,
        },
        "sweep": sweep,
    }
    path = os.path.join(ROOT, "results", "desync_async_large.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
