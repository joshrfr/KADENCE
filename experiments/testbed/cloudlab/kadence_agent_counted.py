"""Counting variant of experiments/simulation/desync_distributed.agent for the CloudLab runtime.

Same algorithm as the committed agent (UDP to two ring neighbours, local
update on its own timer, committed kernel), with three differences that exist
only so the experiment can measure instead of estimate:

  * alpha is a parameter. Paper convention: displacement = (alpha / 2) * corr.
    The committed agent hardcodes 0.1 * corr, i.e. alpha = 0.2. Pass
    ``alpha=0.2`` to reproduce it exactly.
  * it counts ticks executed, datagrams sent, datagrams received, sends
    dropped by loss injection, and datagrams still queued at exit.
  * it writes a JSON file (phase, counters, per-tick phase trace) instead of
    a bare phase, so the launcher can compute convergence per tick.

experiments/simulation/desync_distributed.py is not modified.
"""
from __future__ import annotations

import json
import math
import os
import random as _random
import socket
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
from experiments.simulation.desync_distributed import (  # noqa: E402
    local_displacement,                      # committed clipped-update helper
)

TWO_PI = 2 * math.pi


def counted_agent(idx, n, base_port, phase0, target, seconds, tick, outdir,
                  loss=0.0, jitter=0.0, alpha=1.0, seed=0):
    rng = _random.Random((seed * 1000003) ^ (idx * 7919 + 1))
    left_port = base_port + (idx - 1) % n
    right_port = base_port + (idx + 1) % n
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", base_port + idx))
    sock.setblocking(False)
    phase = phase0
    # No motion toward a neighbour before that neighbour has been heard from;
    # see local_displacement in experiments/simulation/desync_distributed.py.
    left_phase = right_phase = None
    nonlo_src = nonlo_dst = 0
    ticks = sent = received = malformed = loss_dropped = send_errors = 0
    trace = []

    def drain():
        nonlocal left_phase, right_phase, received, malformed, nonlo_src
        got = 0
        try:
            while True:
                data, src = sock.recvfrom(64)
                if src[0] != "127.0.0.1":
                    nonlo_src += 1
                received += 1
                got += 1
                try:
                    who, val = data.decode().split(":")
                    if who == "L":
                        left_phase = float(val)
                    else:
                        right_phase = float(val)
                except Exception:
                    malformed += 1
        except BlockingIOError:
            pass
        return got

    deadline = time.time() + seconds
    next_tick = time.time()
    while time.time() < deadline:
        drain()
        now = time.time()
        if now >= next_tick:
            if rng.random() >= loss:
                try:
                    sock.sendto(f"R:{phase}".encode(), ("127.0.0.1", left_port))
                    sent += 1
                except OSError:
                    send_errors += 1
            else:
                loss_dropped += 1
            if rng.random() >= loss:
                try:
                    sock.sendto(f"L:{phase}".encode(), ("127.0.0.1", right_port))
                    sent += 1
                except OSError:
                    send_errors += 1
            else:
                loss_dropped += 1
            disp = local_displacement(left_phase, phase, right_phase, target,
                                      alpha=alpha)
            phase = (phase + disp) % TWO_PI
            ticks += 1
            trace.append(phase)
            next_tick = now + tick + (rng.random() * jitter if jitter else 0.0)
        time.sleep(0.001)
    residual = drain()
    with open(os.path.join(outdir, f"{idx}.json"), "w") as f:
        json.dump({"idx": idx, "phase": phase % TWO_PI, "ticks": ticks,
                   "sent": sent, "received": received - residual,
                   "malformed": malformed, "loss_dropped": loss_dropped,
                   "send_errors": send_errors, "recv_nonloopback_src": nonlo_src,
                   "sent_nonloopback_dst": nonlo_dst, "residual_at_exit": residual,
                   "trace": trace}, f)
    sock.close()
