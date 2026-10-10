import json, glob, collections, statistics as st
BASE = "results/cloudlab_initfix/tablex_085651/raw"
cells = collections.defaultdict(lambda: {"cls": collections.Counter(), "gap": [], "loose": [], "nodes": []})
for f in sorted(glob.glob(BASE + "/*.json")):
    d = json.load(open(f)); p = d["provenance"]
    key = (p["jobs_per_ring"], p["loss"])
    for r in d["reps"]:
        c = cells[key]
        c["cls"].update(r.get("classification_counts") or {})
        v = r.get("across_node_final_pct_of_fair") or {}
        c["gap"] += [x for x in v.values() if isinstance(x, (int, float))]
        if r.get("fraction_nodes_converged_loose_0p1_target") is not None:
            c["loose"].append(r["fraction_nodes_converged_loose_0p1_target"])
        c["nodes"].append(r.get("nodes_reporting", 0))

print(f"{'jobs':>5}{'loss%':>7}{'rings':>7}{'order-broken':>14}{'med gap%':>10}{'loose conv':>12}")
for key in sorted(cells):
    c = cells[key]; tot = sum(c["cls"].values()); br = c["cls"].get("order-broken", 0)
    print(f"{key[0]:>5}{int(key[1]*100):>7}{tot:>7}{f'{br}/{tot}':>14}"
          f"{(st.median(c['gap']) if c['gap'] else float('nan')):>10.2f}"
          f"{(sum(c['loose'])/len(c['loose']) if c['loose'] else 0):>12.2f}")
print("\nclassification labels seen:", dict(sum((c["cls"] for c in cells.values()), collections.Counter())))
print("min nodes_reporting across all reps:", min(min(c["nodes"]) for c in cells.values()))
