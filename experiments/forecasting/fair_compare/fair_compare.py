"""Fair re-run of the paper's central negative comparison.

Every arm is scored on exactly the same data: the same held-out day, the same
sampled task set (pack() draws the task order from the seed alone), and the same
first-fit join order. One invocation = one (history, last_day, seed)
configuration and writes one JSON.

Arms
  1 peak        each task's historical peak, flat across slots (the anchor)
  2 mean_z      cross-day per-slot mean + z * per-slot std
  3 harmonic_z  K=4 harmonic reconstruction of the mean + z * per-slot std
  4 neural      multi-quantile Forecaster (experiments/simulation/forecast_models.py), trained on
                every training window of gct_days.npz with early stopping on the
                validation day

Every hyperparameter (z for arms 2 and 3, the reservation quantile for arm 4)
is chosen on the VALIDATION day only: the smallest node count whose validation
overload is no worse than peak-requests' validation overload (ties go to the
smaller z or q). If nothing is feasible the lowest-overload candidate is taken
and flagged `fallback`. Held-out grids are computed and stored under
`heldout_diagnostic` but never read by any selection.

Ablation cells (--ablate) re-create the original paper recipe step by step so
the contribution of data size and training budget can be measured.
"""
import argparse, hashlib, json, os, resource, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))
from experiments.simulation.forecast_models import (CAP, HAVE_TORCH, build_examples,  # noqa: E402
                                 harmonic_recon, nodes_overload, pack)
if HAVE_TORCH:
    import torch
    from experiments.simulation.forecast_models import (Forecaster, infer_chunked, multi_pinball,
                                     pinball_chunked)

QUANTILES = (0.5, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995)
FEAT_EPS = 1e-12


def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576


def log(*a):
    print(*a, flush=True)


def score(prof, real, n_tasks, seed):
    prof = np.ascontiguousarray(np.maximum(prof, 0.0))
    nds, _ = pack(prof, n_tasks, seed=seed)
    n, over = nodes_overload(nds, real)
    return int(n), float(over)


def parts(X):
    mean = X[:, :, 2:4].mean(axis=1)
    std = X[:, :, 4:6].mean(axis=1)
    peak = X[:, :, 0:2].max(axis=(1, 3))
    return mean, std, peak


def select(cands, peak_val_over):
    """cands: list of (param, nodes, over). Validation-only selection."""
    feas = [c for c in cands if c[2] <= peak_val_over + FEAT_EPS]
    if feas:
        p, n, o = min(feas, key=lambda c: (c[1], c[0]))
        return p, n, o, False, len(feas)
    p, n, o = min(cands, key=lambda c: (c[2], c[1], c[0]))
    return p, n, o, True, 0


def build_day(series, full, t, h):
    X, AUX, Y, meta = build_examples(series[t - h:t + 1], full[t - h:t + 1], h,
                                     True, max_tasks=None, seed=0)
    return X, AUX, Y


def build_train(series, full, days, h, max_tasks, seed):
    Xs, As, Ys = [], [], []
    for t in days:
        X, A, Y, _ = build_examples(series[t - h:t + 1], full[t - h:t + 1], h,
                                    True, max_tasks=max_tasks, seed=seed)
        Xs.append(X); As.append(A); Ys.append(Y)
    n = sum(len(x) for x in Xs)
    X = np.empty((n,) + Xs[0].shape[1:], np.float32)
    A = np.empty((n,) + As[0].shape[1:], np.float32)
    Y = np.empty((n,) + Ys[0].shape[1:], np.float32)
    k = 0
    for x, a, y in zip(Xs, As, Ys):
        X[k:k + len(x)] = x; A[k:k + len(x)] = a; Y[k:k + len(x)] = y
        k += len(x)
    del Xs, As, Ys
    return X, A, Y


# --------------------------------------------------------------------------
def train_forecaster(Xtr, Atr, Ytr, Xva, Ava, Yva, seed, dev, max_epochs,
                     patience, fixed_epochs, batch=256, lr=2e-3, emb=48):
    """The project's Forecaster, minibatch AdamW + cosine. Early stopping on the
    validation pinball unless fixed_epochs is set, in which case exactly that
    many epochs run and the final weights are used (the paper's flawed budget)."""
    torch.manual_seed(seed)
    E = fixed_epochs if fixed_epochs else max_epochs
    net = Forecaster(Xtr.shape[2], H=Xtr.shape[-1], emb=emb, backbone="tcn",
                     aux_dim=Atr.shape[1], quantiles=QUANTILES).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, E)
    qs = net.qs
    Xt = torch.from_numpy(Xtr).to(dev)
    At = torch.from_numpy(Atr).to(dev)
    Yt = torch.from_numpy(Ytr).to(dev)
    gen = torch.Generator().manual_seed(seed)
    vix = np.arange(len(Xva))
    hist, best, best_state, bad, best_ep = [], float("inf"), None, 0, -1
    stopped = "max_epochs" if not fixed_epochs else "fixed_budget"
    t0 = time.time()
    for ep in range(E):
        net.train()
        perm = torch.randperm(len(Xt), generator=gen).to(dev)
        tl, nb = 0.0, 0
        for k in range(0, len(perm), batch):
            b = perm[k:k + batch]
            opt.zero_grad()
            loss = multi_pinball(net(Xt[b], At[b]), Yt[b], qs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
            tl += loss.item(); nb += 1
        cur_lr = sched.get_last_lr()[0]
        sched.step()
        net.eval()
        with torch.no_grad():
            v = pinball_chunked(net, Xva, Ava, Yva, vix, qs, dev)
        hist.append({"epoch": ep, "train_pinball": round(tl / nb, 6),
                     "val_pinball": round(v, 6), "lr": cur_lr,
                     "t_s": round(time.time() - t0, 1)})
        if ep % 5 == 0 or ep == E - 1:
            log(f"    ep {ep} train {tl/nb:.5f} val {v:.5f} lr {cur_lr:.2e} "
                f"{time.time()-t0:.0f}s")
        if not fixed_epochs:
            if v < best - 1e-5:
                best, bad, best_ep = v, 0, ep
                best_state = {k: x.detach().cpu().clone()
                              for k, x in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    stopped = "patience"
                    log(f"    early stop at ep {ep}, best {best:.5f} @ {best_ep}")
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    del Xt, At, Yt
    if dev == "cuda":
        torch.cuda.empty_cache()
    last = hist[-1]["epoch"]
    tail = [h["val_pinball"] for h in hist[-10:]]
    budget = {"backbone": "tcn", "emb": emb, "batch": batch, "lr": lr,
              "optimizer": "AdamW wd1e-4, cosine over max/fixed epochs, clip 2.0",
              "quantiles": list(QUANTILES), "max_epochs": E,
              "patience": None if fixed_epochs else patience,
              "fixed_epochs": fixed_epochs, "epochs_run": last + 1,
              "stopped_by": stopped,
              "best_epoch": best_ep if not fixed_epochs else last,
              "best_val_pinball": round(best, 6) if not fixed_epochs else hist[-1]["val_pinball"],
              "n_train_windows": int(len(Xtr)), "device": dev,
              "val_pinball_last10": tail,
              "val_drop_last10": round(tail[0] - tail[-1], 6),
              "weights_used": "best-val checkpoint" if not fixed_epochs else "final epoch"}
    return net, hist, budget


def train_orig(Xtr, Ytr, seed, dev, steps=15, lr=1e-3, chunk=4096):
    """The ORIGINAL rapid_forecast recipe (experiments/simulation/forecast_gpu.py): small conv+GRU
    net, raw 2 channels only, single 0.95 quantile, `steps` full-batch Adam
    steps (what the paper calls 15 epochs). Full-batch gradient is exact via
    accumulation over chunks."""
    from experiments.simulation.forecast_gpu import RhythmQuantileNet, pinball
    torch.manual_seed(seed)
    net = RhythmQuantileNet(Xtr.shape[-1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    N = len(Xtr)
    for _ in range(steps):
        net.train(); opt.zero_grad()
        for k in range(0, N, chunk):
            xb = torch.from_numpy(np.ascontiguousarray(Xtr[k:k + chunk, :, 0:2])).to(dev)
            yb = torch.from_numpy(Ytr[k:k + chunk]).to(dev)
            (pinball(net(xb), yb, 0.95) * (len(xb) / N)).backward()
        opt.step()
    net.eval()
    return net


def predict_orig(net, X, dev, chunk=2048):
    outs = []
    with torch.no_grad():
        for k in range(0, len(X), chunk):
            xb = torch.from_numpy(np.ascontiguousarray(X[k:k + chunk, :, 0:2])).to(dev)
            outs.append(net(xb).cpu())
    return torch.cat(outs, 0).numpy()


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--history", type=int, default=3)
    ap.add_argument("--last-day", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1000)
    ap.add_argument("--zs", nargs="+", type=float, default=[0.25 * i for i in range(25)])
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--no-neural", action="store_true")
    ap.add_argument("--train-max-tasks", type=int, default=None,
                    help="CPU fallback only: cap training tasks per day")
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    h, S = a.history, a.seed
    dev = "cuda" if (HAVE_TORCH and torch.cuda.is_available()) else "cpu"
    if a.require_gpu and dev != "cuda" and not a.no_neural:
        sys.exit("no GPU")

    d = np.load(a.data, allow_pickle=True)
    series, full = d["series"][:a.last_day], d["full"][:a.last_day]
    D = series.shape[0]
    usable = [t for t in range(h, D) if full[t - h:t + 1].all(axis=0).any()]
    meta = {"data": os.path.basename(a.data),
            "data_md5": hashlib.md5(open(a.data, "rb").read()).hexdigest(),
            "data_shape_full": list(d["series"].shape),
            "history": h, "last_day": a.last_day, "seed": S,
            "n_tasks": a.n_tasks, "cap": CAP, "usable_days": usable,
            "zs": a.zs, "quantiles": list(QUANTILES), "K_harmonic": 4,
            "selection_rule": ("min nodes subject to validation overload <= "
                               "peak-requests validation overload; ties -> smaller "
                               "z/q; infeasible -> min overload (flag fallback)"),
            "script": "experiments/forecasting/fair_compare/fair_compare.py",
            "device": dev, "torch": torch.__version__ if HAVE_TORCH else None,
            "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None}
    out = {"meta": meta, "arms": {}}

    def finish(status, **kw):
        meta["wall_s"] = round(time.time() - t0, 1)
        meta["peak_rss_gib"] = round(rss_gib(), 2)
        out["status"] = status
        out.update(kw)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)
        print("@@RESULT@@" + json.dumps(out), flush=True)

    if len(usable) < 2:
        meta["held_day"] = usable[-1] if usable else None
        meta["val_day"] = None
        # Anchor only: there is no validation day to select anything on.
        if usable:
            X, A, Y = build_day(series, full, usable[-1], h)
            _, _, pk = parts(X)
            n, o = score(np.broadcast_to(pk[:, :, None], Y.shape), Y, a.n_tasks, S)
            out["arms"]["peak"] = {"heldout": {"nodes": n, "overload": o}}
        return finish("no_validation_day",
                      reason="fewer than two usable days; nothing can be selected on a validation day")

    held, val = usable[-1], usable[-2]
    train_days = [t for t in usable if t < val]
    meta.update(held_day=int(held), val_day=int(val), train_days=[int(t) for t in train_days])

    Xv, Av, Yv = build_day(series, full, val, h)
    Xh, Ah, Yh = build_day(series, full, held, h)
    meta.update(n_val=int(len(Xv)), n_test=int(len(Xh)))
    log(f"[fc] h={h} ld={a.last_day} seed={S} val={val}({len(Xv)}) held={held}({len(Xh)}) "
        f"train_days={train_days} dev={dev} RSS {rss_gib():.1f}")

    _, order_v = pack(np.zeros((len(Xv), 2, Yv.shape[-1]), np.float32) + 1e-9, a.n_tasks, S)
    _, order_h = pack(np.zeros((len(Xh), 2, Yh.shape[-1]), np.float32) + 1e-9, a.n_tasks, S)
    meta["sampled_task_order_md5"] = {
        "val": hashlib.md5(np.asarray(order_v).tobytes()).hexdigest(),
        "heldout": hashlib.md5(np.asarray(order_h).tobytes()).hexdigest()}

    mv, sv, pv = parts(Xv)
    mh, sh, ph = parts(Xh)
    # Arm 1
    pkv = score(np.broadcast_to(pv[:, :, None], Yv.shape), Yv, a.n_tasks, S)
    pkh = score(np.broadcast_to(ph[:, :, None], Yh.shape), Yh, a.n_tasks, S)
    out["arms"]["peak"] = {"val": {"nodes": pkv[0], "overload": pkv[1]},
                           "heldout": {"nodes": pkh[0], "overload": pkh[1]}}
    log(f"[fc] peak val {pkv} held {pkh}")

    # Arms 2, 3
    for name, bv, bh in (("mean_z", mv, mh),
                         ("harmonic_z", harmonic_recon(mv, 4), harmonic_recon(mh, 4))):
        gv, gh = [], []
        for z in a.zs:
            n, o = score(bv + z * sv, Yv, a.n_tasks, S); gv.append((z, n, o))
            n, o = score(bh + z * sh, Yh, a.n_tasks, S); gh.append((z, n, o))
        z, n, o, fb, nf = select(gv, pkv[1])
        hn, ho = next((c[1], c[2]) for c in gh if c[0] == z)
        out["arms"][name] = {
            "selected_z": z, "fallback": fb, "n_feasible_on_val": nf,
            "validation_basis": {"day": int(val), "val_nodes": n, "val_overload": o,
                                 "val_peak_nodes": pkv[0], "val_peak_overload": pkv[1]},
            "heldout": {"nodes": hn, "overload": ho},
            "heldout_diagnostic": [{"z": c[0], "nodes": c[1], "overload": c[2]} for c in gh],
            "val_grid": [{"z": c[0], "nodes": c[1], "overload": c[2]} for c in gv]}
        log(f"[fc] {name} z={z} val {n}/{o:.5f} held {hn}/{ho:.5f}")
    del mv, sv, mh, sh

    # Arm 4
    can_train = HAVE_TORCH and not a.no_neural and len(train_days) > 0
    if not can_train:
        reason = ("no_neural flag" if a.no_neural else
                  "no training windows: every usable day before the validation day is missing at this "
                  "history/last_day (the same 'insufficient split' as train_eval)")
        out["arms"]["neural"] = {"status": "infeasible", "reason": reason}
        return finish("ok_no_neural")

    Xtr, Atr, Ytr = build_train(series, full, train_days, h, a.train_max_tasks, S)
    meta["train_max_tasks_per_day"] = a.train_max_tasks
    log(f"[fc] train windows {len(Xtr)} RSS {rss_gib():.1f}")
    if len(Xtr) < 64:
        out["arms"]["neural"] = {"status": "infeasible", "reason": "fewer than 64 training windows"}
        return finish("ok_no_neural")

    def neural_arm(net_pred_v, net_pred_h, budget, hist):
        gv, gh = [], []
        for qi, q in enumerate(QUANTILES):
            n, o = score(net_pred_v[:, qi], Yv, a.n_tasks, S); gv.append((q, n, o))
            n, o = score(net_pred_h[:, qi], Yh, a.n_tasks, S); gh.append((q, n, o))
        q, n, o, fb, nf = select(gv, pkv[1])
        hn, ho = next((c[1], c[2]) for c in gh if c[0] == q)
        return {"selected_q": q, "fallback": fb, "n_feasible_on_val": nf,
                "validation_basis": {"day": int(val), "val_nodes": n, "val_overload": o,
                                     "val_peak_nodes": pkv[0], "val_peak_overload": pkv[1]},
                "heldout": {"nodes": hn, "overload": ho},
                "heldout_q95": next({"nodes": c[1], "overload": c[2]} for c in gh if c[0] == 0.95),
                "heldout_diagnostic": [{"q": c[0], "nodes": c[1], "overload": c[2]} for c in gh],
                "val_grid": [{"q": c[0], "nodes": c[1], "overload": c[2]} for c in gv],
                "training_budget": budget, "history": hist}

    def run_forecaster(Xs, As, Ys, fixed, checkpoint_path=None):
        net, hist, budget = train_forecaster(Xs, As, Ys, Xv, Av, Yv, S, dev,
                                             a.max_epochs, a.patience, fixed)
        with torch.no_grad():
            pv_ = infer_chunked(net, Xv, Av, np.arange(len(Xv)), dev).numpy()
            ph_ = infer_chunked(net, Xh, Ah, np.arange(len(Xh)), dev).numpy()
        r = neural_arm(pv_, ph_, budget, hist)
        if checkpoint_path is not None:
            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
            tmp = checkpoint_path + f".{os.getpid()}.tmp"
            torch.save({
                "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                "architecture": {"class": "Forecaster", "backbone": "tcn", "emb": 48,
                                 "history": h, "slots": int(Xtr.shape[-1]),
                                 "n_features": int(Xtr.shape[2]),
                                 "aux_dim": int(Atr.shape[1]),
                                 "quantiles": list(QUANTILES)},
                "data_md5": meta["data_md5"], "seed": S,
                "model_code_sha256": hashlib.sha256(open(
                    sys.modules["sim.forecast_models"].__file__, "rb").read()).hexdigest(),
                "train_days": meta["train_days"], "val_day": int(val),
                "held_day": int(held), "selected_q_on_validation": r["selected_q"],
                "training_budget": budget,
            }, tmp)
            os.replace(tmp, checkpoint_path)
            r["checkpoint"] = {"file": os.path.basename(checkpoint_path),
                               "sha256": hashlib.sha256(open(checkpoint_path, "rb").read()).hexdigest(),
                               "bytes": os.path.getsize(checkpoint_path)}
        del net, pv_, ph_
        return r

    out["arms"]["neural"] = run_forecaster(
        Xtr, Atr, Ytr, None, checkpoint_path=a.out + ".model.pt")
    nr = out["arms"]["neural"]
    log(f"[fc] neural q={nr['selected_q']} held {nr['heldout']} budget "
        f"{nr['training_budget']['stopped_by']} ep {nr['training_budget']['epochs_run']}")

    if a.ablate:
        ab = {}
        rng = np.random.default_rng(1000 + S)
        sub = np.sort(rng.choice(len(Xtr), min(6118, len(Xtr)), replace=False))
        Xs, As, Ys = Xtr[sub], Atr[sub], Ytr[sub]

        def orig_cell(Xt_, Yt_, label):
            net = train_orig(Xt_, Yt_, S, dev)
            pv_, ph_ = predict_orig(net, Xv, dev), predict_orig(net, Xh, dev)
            rv = score(pv_, Yv, a.n_tasks, S); rh = score(ph_, Yh, a.n_tasks, S)
            return {"recipe": "forecast_gpu.RhythmQuantileNet, 15 full-batch Adam steps, q=0.95, raw channels",
                    "n_train_windows": int(len(Xt_)), "val": {"nodes": rv[0], "overload": rv[1]},
                    "heldout_q95": {"nodes": rh[0], "overload": rh[1]}}
        ab["A0_orig_net_15steps_subset6118"] = orig_cell(Xs, Ys, "A0")
        ab["A1_orig_net_15steps_full"] = orig_cell(Xtr, Ytr, "A1")
        for tag, (X_, A_, Y_, fx) in {
                "A2_forecaster_15epochs_subset6118": (Xs, As, Ys, 15),
                "A3_forecaster_15epochs_full": (Xtr, Atr, Ytr, 15),
                "A4_forecaster_converged_subset6118": (Xs, As, Ys, None)}.items():
            log(f"[fc] ablation {tag}")
            r = run_forecaster(X_, A_, Y_, fx)
            r.pop("history"); r.pop("heldout_diagnostic"); r.pop("val_grid")
            ab[tag] = r
        out["ablation"] = ab
    return finish("ok")


if __name__ == "__main__":
    main()
