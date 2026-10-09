"""Stage 1 of the inference bake-off: can a closed form beat peak-requests?

No training, no GPU. Every arm here is either parameter-free or a least-squares
fit, and all of them are a handful of lines of C once chosen. The point is to
find out whether a neural network is needed at all before any compilation work
starts, because compilation changes cost and not accuracy.

Arms, all scored on the same held-out day, the same sampled task set and the
same first-fit join order as sim.forecast_models.train_eval:

  peak          per-task historical peak, flat across slots (the arm to beat)
  mean_z        cross-day per-slot mean + z * per-slot std
  harmonic_z    K=4 harmonic reconstruction of the mean + z * per-slot std
  lsq_z         least squares on [1, mean, std, harmonic, peak] per element,
                fit on the training days, + z * training residual std

mean_z and harmonic_z differ only in whether the rhythm is smoothed to its first
four harmonics, which isolates whether the harmonic view earns anything.
z is the safety knob and plays the role the quantile plays for the network.
"""
import argparse, hashlib, json, os, resource, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from experiments.simulation.forecast_models import CAP, build_examples, harmonic_recon, nodes_overload, pack  # noqa: E402


def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576


def score(prof, real, n_tasks, seed=0):
    """Pack `prof` as the reservation, then measure against `real` demand."""
    prof = np.ascontiguousarray(np.maximum(prof, 0.0))
    nds, order = pack(prof, n_tasks, seed=seed)
    n_nodes, over = nodes_overload(nds, real)
    return int(n_nodes), float(over)


def day_parts(series, full, t, history, seed):
    """Per-task (mean, std, peak, real) for test day t, built from only the
    `history` days before t plus day t itself (one day of windows at a time)."""
    X, _, Y, meta = build_examples(series[t - history:t + 1],
                                   full[t - history:t + 1], history, True,
                                   max_tasks=None, seed=seed)
    mean = X[:, :, 2:4].mean(axis=1)
    std = X[:, :, 4:6].mean(axis=1)
    peak = X[:, :, 0:2].max(axis=(1, 3))
    del X
    return mean, std, peak, Y


def nested(a):
    """z and K are chosen on the validation day only (smallest nodes subject to
    zero overload there; ties go to smaller z, then smaller K). The single
    chosen (z, K) is then scored on the held-out day. The held-out grid is also
    scored for diagnosis, but it is written under `heldout_diagnostic` and is
    never read by the selection."""
    d = np.load(a.data, allow_pickle=True)
    series_all, full_all = d["series"], d["full"]
    md5 = hashlib.md5(open(a.data, "rb").read()).hexdigest()
    for ld in a.last_days:
        for sd in a.seeds:
            nested_one(a, series_all, full_all, md5, ld, sd,
                       os.path.join(a.out, f"ld{ld}_s{sd}_h{a.history}.json"))


def nested_one(a, series_all, full_all, md5, last_day, seed, out_path):
    t0 = time.time()
    series, full = series_all[:last_day], full_all[:last_day]
    a = argparse.Namespace(**{**vars(a), "seed": seed, "last_day": last_day,
                              "out": out_path})
    D = series.shape[0]
    usable = [t for t in range(a.history, D)
              if full[t - a.history:t + 1].all(axis=0).any()]
    held = usable[-1]
    val = usable[-2] if len(usable) >= 2 else None
    meta = {"data": os.path.basename(a.data),
            "data_md5": md5,
            "history": a.history, "n_tasks": a.n_tasks, "seed": a.seed,
            "cap": CAP, "last_day": a.last_day, "held_day": int(held),
            "val_day": None if val is None else int(val),
            "zs": a.zs, "ks": a.ks,
            "rule": "min nodes s.t. zero overload on validation; ties: smaller z, then smaller K",
            "script": "experiments/forecasting/stage1_cheap.py --nested"}
    if val is None:
        out = {"meta": meta, "status": "infeasible",
               "reason": "no validation day distinct from the held-out day at this history"}
    else:
        def grid(day):
            mean, std, peak, real = day_parts(series, full, day, a.history, a.seed)
            pk = score(np.broadcast_to(peak[:, :, None], real.shape), real,
                       a.n_tasks, a.seed)
            res = {}
            for K in a.ks:
                h = harmonic_recon(mean, K)
                for z in a.zs:
                    res[(K, z)] = score(h + z * std, real, a.n_tasks, a.seed)
            print(f"[nested] day {day} windows {len(real)} RSS {rss_gib():.1f} GiB",
                  flush=True)
            return pk, res
        pk_v, g_v = grid(val)
        feas = [(n, z, K) for (K, z), (n, o) in g_v.items() if o <= 0.0]
        if feas:
            n_v, z_c, K_c = min(feas)
            basis = {"val_nodes": n_v, "val_overload": g_v[(K_c, z_c)][1],
                     "val_peak_nodes": pk_v[0], "val_peak_overload": pk_v[1],
                     "n_feasible_on_val": len(feas)}
        else:
            z_c = K_c = None
            basis = {"val_peak_nodes": pk_v[0], "val_peak_overload": pk_v[1],
                     "n_feasible_on_val": 0}
        pk_h, g_h = grid(held)
        row = {"chosen_z": z_c, "chosen_K": K_c, "validation_basis": basis,
               "peak_heldout": {"nodes": pk_h[0], "overload": pk_h[1]}}
        if z_c is not None:
            n_h, o_h = g_h[(K_c, z_c)]
            row["harmonic_heldout"] = {"nodes": n_h, "overload": o_h}
            row["reduction_pct"] = 100.0 * (pk_h[0] - n_h) / pk_h[0]
        diag = [{"K": K, "z": z, "val_nodes": g_v[(K, z)][0],
                 "val_overload": g_v[(K, z)][1],
                 "heldout_nodes": g_h[(K, z)][0],
                 "heldout_overload": g_h[(K, z)][1]} for (K, z) in g_v]
        out = {"meta": meta, "status": "ok" if z_c is not None else "no_feasible_on_val",
               "result": row, "heldout_diagnostic": diag}
    meta["wall_s"] = round(time.time() - t0, 1)
    meta["peak_rss_gib"] = round(rss_gib(), 2)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"[nested] wrote {a.out} status {out['status']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--history", type=int, default=3)
    ap.add_argument("--n-tasks", type=int, default=1000)
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--zs", nargs="+", type=float,
                    default=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ap.add_argument("--lsq-sample", type=int, default=2_000_000,
                    help="elements sampled for the least-squares fit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--last-day", type=int, default=None,
                    help="truncate the day axis here, which moves the held-out "
                         "day; lets one dataset yield several held-out days")
    ap.add_argument("--nested", action="store_true",
                    help="select z and K on the validation day, score on held-out")
    ap.add_argument("--last-days", nargs="+", type=int, default=[7, 6, 5, 4])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 3, 4, 6, 8, 12])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.nested:
        if a.zs == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
            a.zs = [0.5 * i for i in range(13)]
        return nested(a)

    t0 = time.time()
    d = np.load(a.data, allow_pickle=True)
    series, full = d["series"], d["full"]
    if a.last_day is not None:
        series, full = series[:a.last_day], full[:a.last_day]
    print(f"[stage1] series {series.shape} full {full.shape}", flush=True)
    X, AUX, Y, meta = build_examples(series, full, a.history, True,
                                     max_tasks=a.max_tasks, seed=a.seed)
    print(f"[stage1] windows {X.shape} built in {time.time()-t0:.0f}s, "
          f"peak RSS {rss_gib():.1f} GiB", flush=True)

    days = sorted(set(t for _, t in meta))
    held, val = days[-1], days[-2] if len(days) >= 2 else days[-1]
    tr = np.array([j for j, (_, t) in enumerate(meta) if t not in (held, val)])
    te = np.array([j for j, (_, t) in enumerate(meta) if t == held])
    print(f"[stage1] train windows {len(tr)}  test windows {len(te)}", flush=True)

    # Channel layout from build_examples: 0:2 raw, 2:4 cross-day mean,
    # 4:6 cross-day std, 6:8 K=4 harmonic. The derived groups are broadcast
    # across the history axis, so a mean over that axis recovers the (2,H) map.
    def parts(ix):
        Xi = X[ix]
        return (Xi[:, :, 2:4].mean(axis=1),      # per-slot mean
                Xi[:, :, 4:6].mean(axis=1),      # per-slot std
                Xi[:, :, 6:8].mean(axis=1),      # harmonic of the mean
                Xi[:, :, 0:2].max(axis=(1, 3)))  # per-task peak

    mean_te, std_te, harm_te, peak_te = parts(te)
    real = Y[te]
    rows = []

    n, over = score(np.broadcast_to(peak_te[:, :, None], real.shape),
                    real, a.n_tasks, a.seed)
    rows.append({"arm": "peak", "z": None, "nodes": n, "overload": over,
                 "params": 0, "trained": False})
    print(f"[stage1] peak -> {n} nodes, overload {over:.5f}", flush=True)

    for z in a.zs:
        for name, base in (("mean_z", mean_te), ("harmonic_z", harm_te)):
            n, over = score(base + z * std_te, real, a.n_tasks, a.seed)
            rows.append({"arm": name, "z": z, "nodes": n, "overload": over,
                         "params": 0, "trained": False})
            print(f"[stage1] {name} z={z} -> {n} nodes, overload {over:.5f}",
                  flush=True)

    # Least squares on the training days. Fit per element, not per task, so the
    # design is [1, mean, std, harmonic, peak] against the realised value.
    # Day index 7 of gct_days.npz yields no valid windows, so a day axis
    # truncated to 8 and to 7 both resolve to held-out day 6. Usable distinct
    # held-out days therefore come from last_day 7, 6, 5 and 4, and the shorter
    # ones leave too little to fit, so the fit is skipped rather than fatal.
    if len(tr) < 64:
        print(f'[stage1] lsq skipped: {len(tr)} train windows', flush=True)
        coef = resid_std = None
    else:
        mean_tr, std_tr, harm_tr, peak_tr = parts(tr)
        pk_tr = np.broadcast_to(peak_tr[:, :, None], mean_tr.shape)
        A = np.stack([np.ones(mean_tr.size, np.float32), mean_tr.ravel(),
                      std_tr.ravel(), harm_tr.ravel(), pk_tr.ravel()], axis=1)
        b = Y[tr].ravel()
        if len(b) > a.lsq_sample:
            sel = np.random.default_rng(a.seed).choice(len(b), a.lsq_sample,
                                                       replace=False)
            A, b = A[sel], b[sel]
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        resid_std = float(np.std(b - A @ coef))
        print(f"[stage1] lsq coef {np.round(coef, 5).tolist()} "
              f"resid_std {resid_std:.5f}", flush=True)

        pk_te = np.broadcast_to(peak_te[:, :, None], mean_te.shape)
        fit_te = (coef[0] + coef[1] * mean_te + coef[2] * std_te
                  + coef[3] * harm_te + coef[4] * pk_te)
        for z in a.zs:
            n, over = score(fit_te + z * resid_std, real, a.n_tasks, a.seed)
            rows.append({"arm": "lsq_z", "z": z, "nodes": n, "overload": over,
                         "params": 5, "trained": True})
            print(f"[stage1] lsq_z z={z} -> {n} nodes, overload {over:.5f}",
                  flush=True)

    peak_row = next(r for r in rows if r["arm"] == "peak")
    # The paper's rule: an arm only beats peak if it uses no more nodes at no
    # worse overload.
    beats = [r for r in rows if r["arm"] != "peak"
             and r["nodes"] <= peak_row["nodes"]
             and r["overload"] <= peak_row["overload"] + 1e-12]
    out = {"meta": {"data": os.path.basename(a.data), "history": a.history,
                    "n_tasks": a.n_tasks, "max_tasks": a.max_tasks,
                    "seed": a.seed, "cap": CAP, "last_day": a.last_day,
                    "train_windows": int(len(tr)), "test_windows": int(len(te)),
                    "held_day": int(held), "wall_s": round(time.time() - t0, 1),
                    "peak_rss_gib": round(rss_gib(), 2)},
           "peak_baseline": peak_row, "rows": rows,
           "beats_peak": sorted(beats, key=lambda r: (r["nodes"], r["overload"])),
           "lsq": ({"coef": [float(c) for c in coef],
                    "resid_std": resid_std} if coef is not None else None)}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"[stage1] wrote {a.out}; {len(beats)} arm(s) beat peak", flush=True)
    if beats:
        b0 = out["beats_peak"][0]
        print(f"[stage1] best: {b0['arm']} z={b0['z']} -> {b0['nodes']} nodes "
              f"vs peak {peak_row['nodes']}", flush=True)


if __name__ == "__main__":
    main()
