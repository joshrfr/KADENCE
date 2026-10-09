"""Equivalence harness: the Python UDP agent of experiments/simulation/desync_distributed.py,
copied verbatim except that (a) the update rate is a parameter (the original
hardcodes 0.1*corr, i.e. mult = 0.1 = alpha/2) and (b) each tick's phase is
recorded so convergence time can be compared with the Rust trace.  The clipped
update, including the pessimistic neighbour initialisation, is the committed
desync_distributed.local_displacement, unchanged."""
import json, math, os, random, socket, sys, time
from multiprocessing import Process
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
from experiments.simulation.desync_distributed import local_displacement
TWO_PI = 2 * math.pi

def agent(idx, n, base_port, phase0, target, seconds, tick, outdir, loss, mult, seed):
    r = random.Random(seed * 1000 + idx)
    lp, rp = base_port + (idx - 1) % n, base_port + (idx + 1) % n
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", base_port + idx)); sock.setblocking(False)
    phase = phase0
    lph = rph = None                 # no motion before a neighbour is heard
    deadline = time.time() + seconds; next_tick = time.time(); trace = []; nrecv = 0
    while time.time() < deadline:
        try:
            while True:
                data, _ = sock.recvfrom(64); nrecv += 1
                who, val = data.decode().split(":")
                if who == "L": lph = float(val)
                else: rph = float(val)
        except BlockingIOError:
            pass
        now = time.time()
        if now >= next_tick:
            if r.random() >= loss: sock.sendto(f"R:{phase}".encode(), ("127.0.0.1", lp))
            if r.random() >= loss: sock.sendto(f"L:{phase}".encode(), ("127.0.0.1", rp))
            disp = local_displacement(lph, phase, rph, target, alpha=2.0 * mult)
            phase = (phase + disp) % TWO_PI
            trace.append(phase)
            next_tick = now + tick
        time.sleep(0.001)
    json.dump({"phase": phase, "trace": trace, "recv": nrecv}, open(os.path.join(outdir, f"{idx}.json"), "w"))

def gap_error(ph, target):
    s = sorted(p % TWO_PI for p in ph)
    g = [s[i+1]-s[i] for i in range(len(s)-1)] + [s[0]+TWO_PI-s[-1]]
    return max(abs(x-target) for x in g)

def run(n, seconds, tick, base_port, phases, loss, mult, seed, outdir):
    os.makedirs(outdir, exist_ok=True)
    target = TWO_PI / n
    ps = [Process(target=agent, args=(i, n, base_port, phases[i], target, seconds, tick, outdir, loss, mult, seed)) for i in range(n)]
    for p in ps: p.start()
    for p in ps: p.join()
    D = [json.load(open(os.path.join(outdir, f"{i}.json"))) for i in range(n)]
    final = gap_error([d["phase"] for d in D], target)
    mt = min(len(d["trace"]) for d in D)
    first = None
    for k in range(mt):
        if gap_error([d["trace"][k] for d in D], target) < 0.1 * target:
            first = k; break
    return {"final_pct": 100*final/target, "loose": final < 0.1*target, "first_loose_tick": first,
            "ticks_min": mt, "recv": sum(d["recv"] for d in D)}

if __name__ == "__main__":
    import argparse
    a = argparse.ArgumentParser()
    a.add_argument("--n", type=int); a.add_argument("--seconds", type=float); a.add_argument("--tick", type=float, default=0.02)
    a.add_argument("--base-port", type=int, default=47000); a.add_argument("--phases-file"); a.add_argument("--loss", type=float, default=0.0)
    a.add_argument("--mult", type=float, help="displacement = mult*corr (alpha/2)"); a.add_argument("--seed", type=int, default=0)
    a.add_argument("--outdir", default="/tmp/kad_py")
    x = a.parse_args()
    ph = [float(t) for t in open(x.phases_file).read().split()]
    print(json.dumps(run(x.n, x.seconds, x.tick, x.base_port, ph, x.loss, x.mult, x.seed, x.outdir)))
