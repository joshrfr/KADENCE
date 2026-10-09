"""Real distributed run of the strict-neighbor desync rule (n independent
asynchronous processes over UDP sockets).

Each job is its own OS process holding only its own phase. It periodically
sends its phase to its two ring neighbors over UDP and, on its own timer,
applies the committed kernel update (core.neighbor_gossip.local_correction +
limit_local_displacement) to whatever neighbor phases it has last received.
There is no global clock and no shared memory, so this exercises the protocol
as real asynchronous distributed code, not a synchronous simulator. A launcher
starts n>=10 agents on localhost, lets them run for a fixed wall-clock budget,
then reads each agent's final phase and reports the achieved gap error.

    python3 experiments/simulation/desync_distributed.py --n 16 --seconds 8

Writes results/desync_distributed.json. Same agent containerizes to one pod per
job for a multi-node cluster; the localhost run is the portable validation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
from multiprocessing import Process

import random as _random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from kadence.neighbor_gossip import (
    NeighborSnapshot, forward_gap, local_correction, limit_local_displacement,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TWO_PI = 2 * math.pi


def _snap(phase):
    return NeighborSnapshot(jid="x", phase=phase % TWO_PI, width=0.0,
                            epoch=1, sequence=0)


def local_displacement(left_phase, phase, right_phase, target, alpha,
                       safety_fraction=0.45):
    """Clipped displacement for one tick, from the phases last received.

    ``left_phase`` or ``right_phase`` is ``None`` until that neighbor has
    actually been heard from. A job holding no snapshot of a neighbor has no
    evidence of any slack on that side, so it believes the gap there is zero
    and does not move: the only assumption that is safe before the first
    message arrives. Seeding the view with a perfectly even ring
    (``phase0 -/+ target``) instead lets ``limit_local_displacement`` clip
    against a believed gap that, on random initial phases, can be many times
    the true one, which licenses a first step straight past the neighbor.
    Neighbors are bound to ports by index, so a crossing is unrecoverable:
    afterwards each job applies the strict-neighbor rule to jobs that are no
    longer its phase neighbors and the correction points the wrong way.

    Paper convention: displacement = (alpha / 2) * correction, then clipped.
    """
    if left_phase is None or right_phase is None:
        return 0.0
    left, cur, right = _snap(left_phase), _snap(phase), _snap(right_phase)
    corr = local_correction(left, cur, right, left_target=target,
                            right_target=target, circumference=TWO_PI)
    return limit_local_displacement(0.5 * alpha * corr, left, cur, right,
                                    left_minimum=0.0, right_minimum=0.0,
                                    safety_fraction=safety_fraction,
                                    circumference=TWO_PI)


def agent(idx, n, base_port, phase0, target, seconds, tick, outdir,
          loss=0.0, jitter=0.0):
    """One job process: UDP to two neighbors, local update on its own timer.

    ``loss`` drops each outgoing message with that probability; ``jitter`` adds
    up to that many seconds of random delay to each tick, modeling an
    asynchronous, lossy network.
    """
    import random as _r
    left_port = base_port + (idx - 1) % n
    right_port = base_port + (idx + 1) % n
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", base_port + idx))
    sock.setblocking(False)
    phase = phase0
    left_phase = right_phase = None   # no motion before the first message
    deadline = time.time() + seconds
    next_tick = time.time()
    while time.time() < deadline:
        # drain any received neighbor phases
        try:
            while True:
                data, _ = sock.recvfrom(64)
                who, val = data.decode().split(":")
                if who == "L":
                    left_phase = float(val)
                else:
                    right_phase = float(val)
        except BlockingIOError:
            pass
        now = time.time()
        if now >= next_tick:
            # send my phase to both neighbors (I am their right / left resp.),
            # dropping each message with probability `loss`
            if _r.random() >= loss:
                sock.sendto(f"R:{phase}".encode(), ("127.0.0.1", left_port))
            if _r.random() >= loss:
                sock.sendto(f"L:{phase}".encode(), ("127.0.0.1", right_port))
            # 0.1 * corr is alpha = 0.2 in the paper's alpha/2 convention.
            disp = local_displacement(left_phase, phase, right_phase, target,
                                      alpha=0.2)
            phase = (phase + disp) % TWO_PI
            next_tick = now + tick + (_r.random() * jitter if jitter else 0.0)
        time.sleep(0.001)
    with open(os.path.join(outdir, f"{idx}.txt"), "w") as f:
        f.write(str(phase % TWO_PI))
    sock.close()


def _run(n, seconds, tick, base_port, seed, loss, jitter):
    outdir = os.path.join(ROOT, "results", "_dist")
    os.makedirs(outdir, exist_ok=True)
    for f in os.listdir(outdir):
        os.remove(os.path.join(outdir, f))
    rng = _random.Random(seed)
    target = TWO_PI / n
    phase0 = sorted(rng.uniform(0, TWO_PI) for _ in range(n))
    init_err = _gap_error(phase0, target)
    procs = [Process(target=agent,
                     args=(i, n, base_port, float(phase0[i]), target, seconds,
                           tick, outdir, loss, jitter))
             for i in range(n)]
    t0 = time.time()
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    wall = time.time() - t0
    final = [float(open(os.path.join(outdir, f"{i}.txt")).read())
             for i in range(n)]
    return init_err, _gap_error(final, target), wall, target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--tick", type=float, default=0.02)
    ap.add_argument("--base-port", type=int, default=48000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss", type=float, default=0.0)
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--sweep", action="store_true",
                    help="sweep message-loss rates -> results/desync_async.json")
    args = ap.parse_args()

    if args.sweep:
        rows = []
        for loss in (0.0, 0.1, 0.3, 0.5):
            ie, fe, wall, target = _run(args.n, args.seconds, args.tick,
                                        args.base_port, args.seed, loss,
                                        jitter=args.tick)   # jitter ~ one tick
            rows.append({"loss": loss, "jitter_s": args.tick,
                         "final_max_gap_error": round(fe, 6),
                         "final_pct_of_fair": round(100 * fe / target, 2),
                         "converged": bool(fe < 0.1 * target)})
            print(f"loss={loss} jitter~tick -> err={fe:.5f} "
                  f"({100*fe/target:.1f}% of fair)", flush=True)
        out = {"provenance": {"kind": "real async multi-process (UDP) w/ loss+jitter",
                              "kernel": "core.neighbor_gossip.local_correction",
                              "n": args.n, "seconds": args.seconds},
               "sweep": rows}
        json.dump(out, open(os.path.join(ROOT, "results", "desync_async.json"), "w"),
                  indent=2)
        print("wrote results/desync_async.json")
        return

    ie, fe, wall, target = _run(args.n, args.seconds, args.tick, args.base_port,
                                args.seed, args.loss, args.jitter)
    out = {"provenance": {"kind": "real asynchronous multi-process (UDP) run",
                          "kernel": "core.neighbor_gossip.local_correction",
                          "n": args.n, "seconds": args.seconds, "tick_s": args.tick,
                          "transport": "UDP/loopback, no global clock"},
           "n": args.n, "wall_seconds": round(wall, 2),
           "initial_max_gap_error": round(ie, 4),
           "final_max_gap_error": round(fe, 6),
           "converged": bool(fe < 0.05 * target)}
    json.dump(out, open(os.path.join(ROOT, "results", "desync_distributed.json"), "w"),
              indent=2)
    print(json.dumps(out, indent=2))


def _gap_error(phases, target):
    s = sorted(p % TWO_PI for p in phases)
    gaps = [s[i + 1] - s[i] for i in range(len(s) - 1)] + [s[0] + TWO_PI - s[-1]]
    return float(max(abs(g - target) for g in gaps))


if __name__ == "__main__":
    main()
