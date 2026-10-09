import json,glob,statistics as st,collections
import os; os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "s1nested"))  # inputs are read from results/s1nested
rows=[];K4=[];kdist=collections.Counter();pool=collections.defaultdict(list)
for f in sorted(glob.glob("ld*_s*_h3.json")):
    d=json.load(open(f));m=d["meta"];r={"file":f,"last_day":m["last_day"],"seed":m["seed"],"held_day":m["held_day"],"val_day":m["val_day"],"status":d["status"]}
    if d["status"]=="ok":
        R=d["result"];r.update(z=R["chosen_z"],K=R["chosen_K"],peak_nodes=R["peak_heldout"]["nodes"],peak_overload=R["peak_heldout"]["overload"],nodes=R["harmonic_heldout"]["nodes"],overload=R["harmonic_heldout"]["overload"],reduction_pct=R["reduction_pct"])
        g=d["heldout_diagnostic"]
        # fixed K=4: z chosen on validation only
        f4=[(x["val_nodes"],x["z"]) for x in g if x["K"]==4 and x["val_overload"]<=0]
        z4=min(f4)[1];h4=[x for x in g if x["K"]==4 and x["z"]==z4][0]
        r["K4"]={"z":z4,"nodes":h4["heldout_nodes"],"overload":h4["heldout_overload"],"reduction_pct":100*(R["peak_heldout"]["nodes"]-h4["heldout_nodes"])/R["peak_heldout"]["nodes"]}
        # fixed z=2,K=4 (the paper rule)
        p=[x for x in g if x["K"]==4 and x["z"]==2.0][0]
        r["paper_z2K4"]={"nodes":p["heldout_nodes"],"overload":p["heldout_overload"],"reduction_pct":100*(R["peak_heldout"]["nodes"]-p["heldout_nodes"])/R["peak_heldout"]["nodes"]}
        # oracle best K on held-out (diagnostic only): per K best z on val
        for K in sorted({x["K"] for x in g}):
            fk=[(x["val_nodes"],x["z"]) for x in g if x["K"]==K and x["val_overload"]<=0]
            if fk:
                zk=min(fk)[1];hk=[x for x in g if x["K"]==K and x["z"]==zk][0]
                pool[K].append((100*(R["peak_heldout"]["nodes"]-hk["heldout_nodes"])/R["peak_heldout"]["nodes"],hk["heldout_overload"]))
        kdist[R["chosen_K"]]+=1
    rows.append(r)
ok=[r for r in rows if r["status"]=="ok"]
red=[r["reduction_pct"] for r in ok]
def S(v):return {"mean":st.mean(v),"sd":st.stdev(v),"min":min(v),"max":max(v),"n":len(v)}
byday={}
for r in ok: byday.setdefault(r["held_day"],[]).append(r["reduction_pct"])
summary={"n_configs_requested":len(rows),"n_infeasible":len(rows)-len(ok),"n_evaluated":len(ok),
 "reduction_pct":S(red),"zero_overload_heldout":sum(r["overload"]<=0 for r in ok),
 "peak_zero_overload_heldout":sum(r["peak_overload"]<=0 for r in ok),
 "all_beat_peak_nodes":all(r["nodes"]<r["peak_nodes"] for r in ok),
 "reduction_by_held_day":{str(k):st.mean(v) for k,v in byday.items()},
 "chosen_z_counts":dict(collections.Counter(r["z"] for r in ok)),"chosen_K_counts":dict(kdist),
 "fixed_K4_val_selected_z":S([r["K4"]["reduction_pct"] for r in ok])|{"zero_overload":sum(r["K4"]["overload"]<=0 for r in ok)},
 "paper_fixed_z2_K4":S([r["paper_z2K4"]["reduction_pct"] for r in ok])|{"zero_overload":sum(r["paper_z2K4"]["overload"]<=0 for r in ok)},
 "per_K_val_selected_z":{str(K):{"mean_reduction":st.mean(a for a,_ in v),"zero_overload":sum(o<=0 for _,o in v),"n":len(v)} for K,v in sorted(pool.items())},
 "biased_reference_pct":23.6,"rows":rows}
json.dump(summary,open("summary.json","w"),indent=1)
L=["| cfg | held | val | z | K | peak | nodes | overload | red% |","|--|--|--|--|--|--|--|--|--|"]
for r in rows:
    n=f"ld{r[chr(108)+chr(97)+chr(115)+chr(116)+chr(95)+chr(100)+chr(97)+chr(121)]}_s{r[chr(115)+chr(101)+chr(101)+chr(100)]}"
    L.append(f"| {n} | {r[chr(104)+chr(101)+chr(108)+chr(100)+chr(95)+chr(100)+chr(97)+chr(121)]} | {r[chr(118)+chr(97)+chr(108)+chr(95)+chr(100)+chr(97)+chr(121)]} | "+(f"{r[chr(122)]} | {r[chr(75)]} | {r[chr(112)+chr(101)+chr(97)+chr(107)+chr(95)+chr(110)+chr(111)+chr(100)+chr(101)+chr(115)]} | {r[chr(110)+chr(111)+chr(100)+chr(101)+chr(115)]} | {r[chr(111)+chr(118)+chr(101)+chr(114)+chr(108)+chr(111)+chr(97)+chr(100)]:.2e} | {r[chr(114)+chr(101)+chr(100)+chr(117)+chr(99)+chr(116)+chr(105)+chr(111)+chr(110)+chr(95)+chr(112)+chr(99)+chr(116)]:.1f} |" if r["status"]=="ok" else "- | - | - | - | - | infeasible (no validation day) |"))
open("summary.md","w").write("\n".join(L)+"\n")
print("\n".join(L));print(json.dumps({k:v for k,v in summary.items() if k!="rows"},indent=1))
