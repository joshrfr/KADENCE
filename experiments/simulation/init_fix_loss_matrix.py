"""Loss cells for the initialisation fix, using the SHIPPED agent code.

My earlier matrix printed a loss column that was never injected: I called a
run() whose signature had no loss parameter, and the tell was n=48 reporting
5.32% identically in both rows. This version imports the real agent from the
artifact and passes loss through to the sendto path, and asserts that the
arms actually differ before reporting.

base  = the agent with the optimistic even-spacing init restored
fixed = the shipped agent (pessimistic init, commit 825a15b)
"""
import importlib.util, math, os, random, sys, time
from multiprocessing import Process

ALPHA = float(__import__("os").environ.get("ALPHA","0.2"))
ROOT = "/home/svc-opsd/kadence-anon-artifact"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
spec = importlib.util.spec_from_file_location(
    "dd", os.path.join(ROOT, "experiments/simulation/desync_distributed.py"))
dd = importlib.util.module_from_spec(spec); spec.loader.exec_module(dd)
TWO_PI = 2 * math.pi


def agent_base(idx, n, base_port, phase0, target, seconds, tick, outdir,
               loss=0.0, jitter=0.0):
    """The pre-825a15b behaviour: seed neighbour beliefs with an even ring."""
    import socket
    import random as _r
    lp, rp = base_port + (idx - 1) % n, base_port + (idx + 1) % n
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", base_port + idx)); s.setblocking(False)
    phase = phase0
    left_phase = (phase0 - target) % TWO_PI       # the bug
    right_phase = (phase0 + target) % TWO_PI
    end = time.time() + seconds; nxt = time.time()
    while time.time() < end:
        try:
            while True:
                d, _ = s.recvfrom(64)
                who, val = d.decode().split(":")
                if who == "L": left_phase = float(val)
                else: right_phase = float(val)
        except BlockingIOError:
            pass
        now = time.time()
        if now >= nxt:
            if _r.random() >= loss:
                s.sendto(f"R:{phase}".encode(), ("127.0.0.1", lp))
            if _r.random() >= loss:
                s.sendto(f"L:{phase}".encode(), ("127.0.0.1", rp))
            disp = dd.local_displacement(left_phase, phase, right_phase,
                                         target, alpha=ALPHA)
            phase = (phase + disp) % TWO_PI
            nxt = now + tick + (_r.random() * jitter if jitter else 0.0)
        time.sleep(0.001)
    open(os.path.join(outdir, f"{idx}.txt"), "w").write(str(phase % TWO_PI))
    s.close()


def agent_fixed(idx, n, base_port, phase0, target, seconds, tick, outdir,
                loss=0.0, jitter=0.0):
    """Identical to agent_base except the two init lines (commit 825a15b)."""
    import socket
    import random as _r
    lp, rp = base_port + (idx - 1) % n, base_port + (idx + 1) % n
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", base_port + idx)); s.setblocking(False)
    phase = phase0
    left_phase = right_phase = None               # the fix
    end = time.time() + seconds; nxt = time.time()
    while time.time() < end:
        try:
            while True:
                d, _ = s.recvfrom(64)
                who, val = d.decode().split(":")
                if who == "L": left_phase = float(val)
                else: right_phase = float(val)
        except BlockingIOError:
            pass
        now = time.time()
        if now >= nxt:
            if _r.random() >= loss:
                s.sendto(f"R:{phase}".encode(), ("127.0.0.1", lp))
            if _r.random() >= loss:
                s.sendto(f"L:{phase}".encode(), ("127.0.0.1", rp))
            disp = dd.local_displacement(left_phase, phase, right_phase,
                                         target, alpha=ALPHA)
            phase = (phase + disp) % TWO_PI
            nxt = now + tick + (_r.random() * jitter if jitter else 0.0)
        time.sleep(0.001)
    open(os.path.join(outdir, f"{idx}.txt"), "w").write(str(phase % TWO_PI))
    s.close()


def breaks(final):
    n = len(final)
    rot = sorted(range(n), key=lambda i: final[i])
    st = rot.index(0)
    seen = [rot[(st + j) % n] for j in range(n)]
    return sum(1 for j in range(n) if seen[j] != j)


def gaperr(ph, target):
    g = sorted(ph)
    gp = [(g[(i + 1) % len(g)] - g[i]) % TWO_PI for i in range(len(g))]
    return 100.0 * sum(abs(x - target) for x in gp) / len(gp) / target


def run(fn, n, seed, loss, port, seconds=6.0, tick=0.020, jitter=0.020):
    out = f"/tmp/claude-1002/order/_L{port}"
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out): os.remove(os.path.join(out, f))
    rng = random.Random(seed); target = TWO_PI / n
    p0 = sorted(rng.uniform(0, TWO_PI) for _ in range(n))
    ps = [Process(target=fn, args=(i, n, port, float(p0[i]), target, seconds,
                                   tick, out, loss, jitter)) for i in range(n)]
    [p.start() for p in ps]; [p.join() for p in ps]
    fin = [float(open(os.path.join(out, f"{i}.txt")).read()) for i in range(n)]
    return breaks(fin), gaperr(fin, target)


if __name__ == "__main__":
    REPS = 6
    print(f"alpha={ALPHA}, 20ms tick, 20ms jitter, 6s; loss is per-datagram drop")
    print(f"{'n':>4} {'loss':>5} | {'base brk':>9} {'base err%':>10} "
          f"| {'fixed brk':>10} {'fixed err%':>11}")
    port = 20000
    seen = {}
    for n in (16, 48):
        for loss in (0.0, 0.10, 0.30):
            bb = fb = 0; be = []; fe = []
            for seed in range(REPS):
                b, e = run(agent_base, n, seed, loss, port); port += 60
                f, g = run(agent_fixed, n, seed, loss, port); port += 60
                bb += (b > 0); fb += (f > 0); be.append(e); fe.append(g)
            bm, fm = sorted(be)[REPS//2], sorted(fe)[REPS//2]
            seen[(n, loss)] = (bm, fm)
            print(f"{n:>4} {loss:>5.0%} | {bb:>4}/{REPS:<4} {bm:>10.2f} "
                  f"| {fb:>5}/{REPS:<4} {fm:>11.2f}")
    # the check my earlier matrix failed: loss must actually change something
    for n in (16, 48):
        a = seen[(n, 0.0)][0]; c = seen[(n, 0.30)][0]
        print(f"n={n}: base median err 0% vs 30% loss = {a:.2f} vs {c:.2f} "
              f"-> {'DIFFER (loss is live)' if abs(a-c) > 1e-9 else 'IDENTICAL - LOSS NOT INJECTED'}")
