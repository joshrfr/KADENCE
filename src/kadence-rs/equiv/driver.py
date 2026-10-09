import json, os, random, subprocess, sys, math
HERE = os.path.dirname(os.path.abspath(__file__))
CRATE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import py_agent
TWO_PI = 2*math.pi
RS = os.path.join(CRATE, "target", "release", "kadence")
# Binary built from the pre-change revision, supplied by the user (not shipped).
OLD = os.environ.get("KADENCE_OLD_BIN", os.path.join(CRATE, "target", "kadence_old"))
out = os.path.join(HERE, "results.jsonl")
def rust(binp, n, sec, loss, alpha, seed, pf, port, new=True):
    cmd = [binp, "--jobs", str(n), "--seconds", str(sec), "--reps", "1", "--loss", str(loss), "--seed", str(seed)]
    if new: cmd += ["--base-port", str(port)]
    if new: cmd += ["--alpha", str(alpha), "--phases-file", pf]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    j = json.loads(r.stdout)["reps"][0]
    if not new:
        return {"final_pct": j["final_gap_error_pct"], "loose": j["converged"]}
    return {"final_pct": j["final_pct_of_fair"], "loose": j["converged_loose_0p1_target"],
            "first_loose_tick": j.get("tick_of_first_loose"), "ticks_min": j["ticks_min_per_job"],
            "recv": j["datagrams_received"], "sent": j["datagrams_sent"]}
cfgs = [(16, 6, 0.0), (16, 6, 0.1), (48, 10, 0.0)]
seeds = [0, 1, 2, 3]
with open(out, "a") as f:
    for (n, sec, loss) in cfgs:
        for alpha in (0.2, 1.0):
            for seed in seeds:
                rng = random.Random(seed)
                ph = sorted(rng.uniform(0, TWO_PI) for _ in range(n))
                pf = f"/tmp/kad_ph_{n}_{seed}.txt"; open(pf, "w").write(" ".join(repr(x) for x in ph))
                rows = {"py": py_agent.run(n, sec, 0.02, 47000, ph, loss, alpha/2, seed, "/tmp/kad_py"),
                        "rust": rust(RS, n, sec, loss, alpha, seed, pf, 53000)}
                rec = {"n": n, "seconds": sec, "loss": loss, "alpha": alpha, "seed": seed, **rows}
                f.write(json.dumps(rec) + "\n"); f.flush(); print(json.dumps(rec), flush=True)
        for seed in seeds:   # pre-fix binary (own seeded phases, hardcoded 0.1*corr == alpha 0.2)
            r = rust(OLD, n, sec, loss, None, seed, None, 53000, new=False)
            rec = {"n": n, "seconds": sec, "loss": loss, "alpha": 0.2, "seed": seed, "rust_old": r}
            f.write(json.dumps(rec) + "\n"); f.flush(); print(json.dumps(rec), flush=True)
