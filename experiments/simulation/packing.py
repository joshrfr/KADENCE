"""Packing density at a fixed peak ceiling — the metric RR cannot win.

Question: at capacity=1.0, how many heterogeneous jobs fit before peak
demand on either resource exceeds the ceiling? Resource-aware valley-
fill should pack MORE than resource-blind round-robin, because it puts
IO-heavy jobs into CPU valleys (and vice versa) instead of spreading
everyone thin. Same job stream, same ceiling — only the arrangement
differs.
"""
import json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from kadence.oscillator import desync_arrange, peak_contention, TWO_PI
from experiments.simulation.simulate import make_jobs, RESOURCES, PERIOD

CEIL = 1.0

def rr(jobs):
    n=len(jobs)
    for i,j in enumerate(jobs): j.phase={r:TWO_PI*i/max(1,n) for r in RESOURCES}

def coupled(jobs):
    rng=random.Random(1)
    for j in jobs: j.phase={r:rng.uniform(0,TWO_PI) for r in RESOURCES}
    desync_arrange(jobs, RESOURCES)

def fits(jobs):
    p=peak_contention(jobs, RESOURCES, PERIOD)
    return max(p.values()) <= CEIL

def capacity(seed, place):
    # grow the job set until it no longer fits; return count that fit
    n=2
    while n<=80:
        jobs=make_jobs(n, seed); place(jobs)
        if not fits(jobs):
            return n-1
        n+=1
    return 80

out={"ceiling":CEIL,"seeds":30,"generated_at":time.time(),"per_scheme":{}}
for name, place in (("linear-rr",rr),("phase-coupled",coupled)):
    caps=[capacity(s, place) for s in range(30)]
    out["per_scheme"][name]={"mean_jobs_fit":round(sum(caps)/len(caps),2),
                             "min":min(caps),"max":max(caps)}
rr_m=out["per_scheme"]["linear-rr"]["mean_jobs_fit"]
ph_m=out["per_scheme"]["phase-coupled"]["mean_jobs_fit"]
out["phase_packing_gain_pct"]=round(100*(ph_m-rr_m)/rr_m,1)
print(json.dumps(out,indent=2))
