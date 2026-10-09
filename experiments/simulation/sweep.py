"""Duty-cycle sweep: find the regime where phase-coupled beats round-robin.

Instead of claiming a universal win, characterise WHERE
the bubble/oscillator model helps. Hypothesis: it wins when jobs are
bursty (low duty cycle) because competitors can be truly separated
across the period, and RR's resource-blind even spread wastes room on
non-competing jobs.
"""
import json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from kadence.oscillator import Job, desync_arrange, peak_contention, dominant_resource, TWO_PI
from experiments.simulation.simulate import make_jobs, RESOURCES, PERIOD

def scale_duty(jobs, s):
    for j in jobs:
        j.duty = {r: d*s for r, d in j.duty.items()}

def rr(jobs):
    n=len(jobs)
    for i,j in enumerate(jobs): j.phase={r:TWO_PI*i/n for r in RESOURCES}

def coupled(jobs):
    rng=random.Random(1)
    for j in jobs: j.phase={r:rng.uniform(0,TWO_PI) for r in RESOURCES}
    desync_arrange(jobs, RESOURCES)

def peak(jobs):
    p=peak_contention(jobs, RESOURCES, PERIOD)
    return (p["cpu"]+p["io"])/2

out={"sweep":[], "generated_at":time.time()}
for duty_scale in [0.25,0.4,0.55,0.7,0.85,1.0,1.2]:
    rr_p=[]; ph_p=[]
    for seed in range(24):
        a=make_jobs(24,seed); scale_duty(a,duty_scale); rr(a); rr_p.append(peak(a))
        b=make_jobs(24,seed); scale_duty(b,duty_scale); coupled(b); ph_p.append(peak(b))
    rrm=sum(rr_p)/len(rr_p); phm=sum(ph_p)/len(ph_p)
    out["sweep"].append({"duty_scale":duty_scale,
        "rr_peak":round(rrm,3),"phase_peak":round(phm,3),
        "phase_win_pct":round(100*(rrm-phm)/rrm,1)})
print(json.dumps(out,indent=2))
