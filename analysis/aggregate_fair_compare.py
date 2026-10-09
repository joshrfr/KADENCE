"""Aggregate results/fair_compare/google_ld*_s*_h3.json -> summary.json + summary.md"""
import glob, json, os, re, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "fair_compare")
ARMS = ["peak", "mean_z", "harmonic_z", "neural"]
rows = []
for f in sorted(glob.glob(os.path.join(RES, "google_ld*_s*_h3.json"))):
    d = json.load(open(f)); m = d["meta"]; r = {"file": os.path.basename(f), "last_day": m["last_day"],
        "seed": m["seed"], "status": d["status"], "device": m.get("device"), "gpu": m.get("gpu"),
        "held_day": m.get("held_day"), "val_day": m.get("val_day"), "train_days": m.get("train_days"),
        "n_train": None, "arms": {}}
    for a in ARMS:
        x = d["arms"].get(a)
        if not x or "heldout" not in x: continue
        e = {"nodes": x["heldout"]["nodes"], "overload": x["heldout"]["overload"]}
        for k in ("selected_z", "selected_q", "fallback"):
            if k in x: e[k] = x[k]
        if a == "neural":
            tb = x["training_budget"]; r["n_train"] = tb["n_train_windows"]
            e["epochs_run"] = tb["epochs_run"]; e["device"] = tb["device"]
        r["arms"][a] = e
    rows.append(r)
rows.sort(key=lambda r: (-r["last_day"], r["seed"]))

def zero(e): return e["overload"] <= 1e-12
def winner(r):
    a = r["arms"]; ok = {k: v for k, v in a.items() if k != "peak" and v["overload"] <= a["peak"]["overload"] + 1e-12}
    ok["peak"] = a["peak"]
    best = min(v["nodes"] for v in ok.values())
    w = [k for k, v in ok.items() if v["nodes"] == best]
    return w, best
for r in rows:
    r["winner_zero_overload"], r["winner_nodes"] = winner(r)
    p = r["arms"]["peak"]["nodes"]
    for k, e in r["arms"].items(): e["pct_vs_peak"] = round(100.0 * (e["nodes"] - p) / p, 2)

def agg(rs, label):
    out = {"label": label, "n_configs": len(rs), "arms": {}}
    for a in ARMS:
        es = [(r, r["arms"][a]) for r in rs if a in r["arms"]]
        if not es: continue
        n = [e["nodes"] for _, e in es]
        pc = [e["pct_vs_peak"] for _, e in es]
        pk = [r["arms"]["peak"]["nodes"] for r, _ in es]
        out["arms"][a] = {"n_configs": len(es), "mean_nodes": round(st.mean(n), 2), "min": min(n), "max": max(n),
            "mean_pct_vs_peak": round(st.mean(pc), 2), "pooled_pct_vs_peak": round(100.0 * (sum(n) - sum(pk)) / sum(pk), 2),
            "n_zero_overload": sum(zero(e) for _, e in es),
            "mean_overload": round(st.mean(e["overload"] for _, e in es), 6),
            "wins": sum(a in r["winner_zero_overload"] and len(r["winner_zero_overload"]) == 1 for r, _ in es),
            "ties_for_best": sum(a in r["winner_zero_overload"] and len(r["winner_zero_overload"]) > 1 for r, _ in es)}
    both = [r for r in rs if "mean_z" in r["arms"] and "harmonic_z" in r["arms"]]
    mh = {"n_paired": len(both)}
    mh["mean_fewer_nodes"] = sum(r["arms"]["mean_z"]["nodes"] < r["arms"]["harmonic_z"]["nodes"] for r in both)
    mh["harmonic_fewer_nodes"] = sum(r["arms"]["harmonic_z"]["nodes"] < r["arms"]["mean_z"]["nodes"] for r in both)
    mh["tied"] = len(both) - mh["mean_fewer_nodes"] - mh["harmonic_fewer_nodes"]
    out["mean_vs_harmonic"] = mh
    nb = [r for r in rs if "neural" in r["arms"]]
    nv = {"n_paired": len(nb)}
    for c in ("mean_z", "harmonic_z"):
        nv[c + "_fewer_than_neural"] = sum(r["arms"][c]["nodes"] < r["arms"]["neural"]["nodes"] for r in nb)
        nv[c + "_equal_neural"] = sum(r["arms"][c]["nodes"] == r["arms"]["neural"]["nodes"] for r in nb)
        nv[c + "_more_than_neural"] = sum(r["arms"][c]["nodes"] > r["arms"]["neural"]["nodes"] for r in nb)
    nv["neural_fewer_than_peak"] = sum(r["arms"]["neural"]["nodes"] < r["arms"]["peak"]["nodes"] for r in nb)
    nv["neural_zero_overload_and_fewer_than_peak"] = sum(r["arms"]["neural"]["nodes"] < r["arms"]["peak"]["nodes"] and zero(r["arms"]["neural"]) for r in nb)
    out["neural_comparisons"] = nv
    return out

full = [r for r in rows if "neural" in r["arms"]]
nomod = [r for r in rows if "neural" not in r["arms"] and "mean_z" in r["arms"]]
summary = {"configs": rows,
           "aggregate_all": agg(rows, "all configurations (arms averaged over the configs where they exist)"),
           "aggregate_4arm_configs": agg(full, "configs where all four arms are defined (last_day 6, 7)"),
           "aggregate_closed_form_configs": agg([r for r in rows if "mean_z" in r["arms"]], "all configs with a validation day (last_day 5, 6, 7)"),
           "notes": ["winner = fewest held-out nodes among arms whose held-out overload is no worse than peak's own held-out overload (peak always included); zero-overload counts are reported separately",
                     "last_day 5: usable days [3,4] -> val 3, held 4, no training day, neural infeasible",
                     "last_day 4: usable days [3] -> no validation day, only the peak anchor can be scored",
                     "seed changes only the sampled task order and net init; the data day split is fixed by last_day"]}
json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=1)

L = ["# Fair comparison, Google cluster, history 3, 12 configurations", "",
     "| ld | seed | held | train win | peak | mean+z | harm+z | neural (q) | neural ovl | winner |", "|---|---|---|---|---|---|---|---|---|---|"]
def c(r, a):
    e = r["arms"].get(a); return "n/a" if not e else str(e["nodes"]) + ("" if zero(e) else "*")
for r in rows:
    nq = r["arms"].get("neural"); 
    L.append(f"| {r['last_day']} | {r['seed']} | {r['held_day']} | {r['n_train'] or '-'} | {c(r,'peak')} | {c(r,'mean_z')} | {c(r,'harmonic_z')} | "
             f"{c(r,'neural')}{' (q=%g)' % nq['selected_q'] if nq else ''} | {('%.5f' % nq['overload']) if nq else '-'} | {'/'.join(r['winner_zero_overload'])} |")
L += ["", "`*` = held-out overload above zero.", ""]
for key in ("aggregate_all", "aggregate_4arm_configs"):
    g = summary[key]; L += [f"## {g['label']} (n={g['n_configs']})", "",
        "| arm | n | mean nodes | range | mean % vs peak | pooled % vs peak | zero-overload | wins | ties |", "|---|---|---|---|---|---|---|---|---|"]
    for a, v in g["arms"].items():
        L.append(f"| {a} | {v['n_configs']} | {v['mean_nodes']} | {v['min']}-{v['max']} | {v['mean_pct_vs_peak']} | {v['pooled_pct_vs_peak']} | {v['n_zero_overload']}/{v['n_configs']} | {v['wins']} | {v['ties_for_best']} |")
    L += ["", "mean vs harmonic: " + json.dumps(g["mean_vs_harmonic"]), "neural comparisons: " + json.dumps(g["neural_comparisons"]), ""]
L += ["Notes:"] + ["- " + n for n in summary["notes"]]
open(os.path.join(RES, "summary.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
