"""Improved demand forecaster for KADENCE out-of-sample admission.

This is the swappable predictor from the architecture, upgraded in three ways
that are independent of each other so an ablation can attribute the gain:

  1. BACKBONE  -- selectable per-day encoder: cnn (dilated TCN), lstm, gru,
                  attn (transformer encoder). The history days are then
                  aggregated in order by a GRU. One flag swaps the backbone.
  2. FEATURES  -- the raw 2xH day profile is augmented with derived channels
                  computed from the data it already has: cross-day per-slot
                  mean and std, the K=4 harmonic reconstruction (rhythm), and
                  a scalar burstiness/recent-peak summary fed to the decoder.
                  This gives the backbone rich inference signal without any new
                  data source.
  3. RECIPE    -- real training: minibatches, AdamW, cosine LR, gradient
                  clipping, multi-quantile pinball (several quantiles learned
                  jointly), and EARLY STOPPING on a held-out validation day.

Evaluation is unchanged: the predicted per-slot profile at the
chosen reservation quantile is packed greedily onto unit-capacity nodes and
scored on the STRICTLY held-out last day against peak-requests and mean+z-sigma.
If the learned model does not beat peak-requests at equal overload, that is the
reported result ("history beats ML"), not a hidden failure.

CPU-safe (falls back automatically); designed to run on the NRP CUDA image with
an A100/L40. Import from experiments/forecasting/train_forecaster.py or a notebook.
"""
from __future__ import annotations

import json
import os

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAP = 1.0
BACKBONES = ("cnn", "tcn", "lstm", "gru", "attn")


# ----------------------------------------------------------------------------
# Features: derive rich per-task channels from the raw history window.
# ----------------------------------------------------------------------------
def harmonic_recon(profile, K=4):
    """K-harmonic reconstruction of a (2, H) daily profile via rFFT (the rhythm
    the oscillator model uses). Returns (2, H)."""
    f = np.fft.rfft(profile, axis=-1)
    if f.shape[-1] > K + 1:
        f[..., K + 1:] = 0
    return np.fft.irfft(f, n=profile.shape[-1], axis=-1).astype(np.float32)


def build_examples(series, full, history, with_features=True, harmonics=4,
                   max_tasks=None, seed=0):
    """Return (X, aux, Y, meta).

    X    : (n, history, Cin, H) per-day stacked channels (raw + derived)
    aux  : (n, A) per-task scalar summary (burstiness, recent peak, mean, std)
    Y    : (n, 2, H) next-day real profile (prediction target)
    meta : list of (task_idx, test_day)

    Cin = 2 (raw cpu,mem) [+ 2 cross-day mean, +2 cross-day std, +2 harmonic]
    when with_features; else Cin = 2.

    `max_tasks` caps tasks sampled per window (keeps CPU smoke tests tractable;
    leave None on a GPU to use every task).
    """
    D = series.shape[0]
    rng = np.random.default_rng(seed)
    # Two passes. Resolving the (day, task) pairs first lets the outputs be
    # allocated once and filled in place. Appending per-example arrays to a
    # Python list and stacking afterwards holds the list and the stacked copy at
    # the same time, so peak memory is twice the final size; that is what
    # OOM-killed a 24Gi pod at history=5. The rng is advanced in the same order
    # as before, so a given seed still selects the same tasks.
    sel = []
    for t in range(history, D):
        cand = np.all(full[t - history:t + 1], axis=0)
        idx = np.nonzero(cand)[0]
        if max_tasks is not None and len(idx) > max_tasks:
            idx = rng.choice(idx, max_tasks, replace=False)
        sel.append((t, idx))
    n_ex = int(sum(len(idx) for _, idx in sel))
    H_ = series.shape[-1]
    Cin = 8 if with_features else 2
    Adim = 8 if with_features else 1
    X = np.empty((n_ex, history, Cin, H_), np.float32)
    AUX = np.empty((n_ex, Adim), np.float32)
    Y = np.empty((n_ex, 2, H_), np.float32)
    meta = []
    k = 0
    for t, idx in sel:
        for i in idx:
            win = series[t - history:t, i].astype(np.float32)   # (history,2,H)
            if with_features:
                m = win.mean(axis=0)                            # (2,H)
                s = win.std(axis=0)                             # (2,H)
                hb = harmonic_recon(m, harmonics)               # (2,H)
                mb = np.broadcast_to(m, win.shape)
                sb = np.broadcast_to(s, win.shape)
                hbb = np.broadcast_to(hb, win.shape)
                feat = np.concatenate([win, mb, sb, hbb], axis=1)  # (history,8,H)
                peak = win.max(axis=(0, 2))                     # (2,)
                mean = m.mean(axis=1)                           # (2,)
                std = s.mean(axis=1)                            # (2,)
                burst = peak / np.maximum(mean, 1e-6)           # (2,)
                aux = np.concatenate([peak, mean, std, burst]).astype(np.float32)
            else:
                feat = win
                aux = np.zeros(1, np.float32)
            X[k] = feat
            AUX[k] = aux
            Y[k] = series[t, i]
            meta.append((int(i), int(t)))
            k += 1
    assert k == n_ex, f"filled {k} of {n_ex} examples"
    return X, AUX, Y, meta


# ----------------------------------------------------------------------------
# Model: per-day backbone -> ordered aggregation -> aux fusion -> multi-quantile.
# ----------------------------------------------------------------------------
if HAVE_TORCH:
    class DayEncoder(nn.Module):
        """Encode one day's (Cin, H) profile to an embedding of size `emb`."""
        def __init__(self, cin, H, emb, backbone):
            super().__init__()
            self.backbone = backbone
            self.H = H
            if backbone in ("cnn", "tcn"):
                dils = (1, 2, 4, 8) if backbone == "tcn" else (1, 1, 1)
                layers, c = [], cin
                for d in dils:
                    layers += [nn.Conv1d(c, emb, 5, padding=2 * d, dilation=d),
                               nn.GELU()]
                    c = emb
                layers += [nn.AdaptiveAvgPool1d(1)]
                self.net = nn.Sequential(*layers)
            elif backbone in ("lstm", "gru"):
                rnn = nn.LSTM if backbone == "lstm" else nn.GRU
                self.proj = nn.Linear(cin, emb)
                self.net = rnn(emb, emb, batch_first=True, bidirectional=True)
                self.out = nn.Linear(2 * emb, emb)
            elif backbone == "attn":
                self.proj = nn.Linear(cin, emb)
                layer = nn.TransformerEncoderLayer(emb, 4, emb * 2,
                                                   batch_first=True, dropout=0.1)
                self.net = nn.TransformerEncoder(layer, 2)
            else:
                raise ValueError(f"unknown backbone {backbone}")

        def forward(self, x):                                   # x: (M, Cin, H)
            if self.backbone in ("cnn", "tcn"):
                return self.net(x).squeeze(-1)                  # (M, emb)
            seq = self.proj(x.transpose(1, 2))                  # (M, H, emb)
            if self.backbone in ("lstm", "gru"):
                o, _ = self.net(seq)
                return self.out(o.mean(dim=1))
            return self.net(seq).mean(dim=1)                    # attn pool

    class Forecaster(nn.Module):
        """History days -> ordered GRU aggregation -> fuse aux -> Q quantiles."""
        def __init__(self, cin, H=288, emb=48, backbone="tcn", aux_dim=8,
                     quantiles=(0.5, 0.9, 0.95, 0.99)):
            super().__init__()
            self.H, self.Q = H, len(quantiles)
            self.register_buffer("qs", torch.tensor(quantiles).float())
            self.day = DayEncoder(cin, H, emb, backbone)
            self.agg = nn.GRU(emb, emb, batch_first=True)
            self.aux = nn.Sequential(nn.Linear(aux_dim, emb), nn.GELU())
            self.dec = nn.Sequential(
                nn.Linear(2 * emb, 256), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(256, self.Q * 2 * H))

        def forward(self, x, aux):                              # x:(B,D,Cin,H)
            B, D, C, H = x.shape
            e = self.day(x.reshape(B * D, C, H)).reshape(B, D, -1)
            _, h = self.agg(e)                                  # (1,B,emb)
            z = torch.cat([h[-1], self.aux(aux)], dim=1)
            out = self.dec(z).reshape(B, self.Q, 2, H)
            return torch.relu(out)                              # (B,Q,2,H) >=0

    def multi_pinball(pred, target, qs):
        """pred (B,Q,2,H), target (B,2,H), qs (Q,). Mean over all quantiles."""
        t = target.unsqueeze(1)                                 # (B,1,2,H)
        e = t - pred
        q = qs.view(1, -1, 1, 1)
        return torch.mean(torch.maximum(q * e, (q - 1) * e))

    def infer_chunked(net, X, AUX, ix, dev, chunk=512, zero=None):
        """Forward `ix` through `net` in chunks and return (len(ix),Q,2,H) on CPU.

        Training already minibatches, but validation and test did one forward
        over every window. At the full 82,440-task count that asks a
        bidirectional GRU over 288 slots for a single ~15 GiB activation, which
        is what exhausted a 22 GiB card even though the model itself is small.
        Chunking here is what lets the full-scale run fit on one GPU.
        """
        outs = []
        for k in range(0, len(ix), chunk):
            b = ix[k:k + chunk]
            xb = torch.tensor(X[b]).to(dev)
            if zero is not None:
                lo, hi = zero
                xb[:, :, lo:hi, :] = 0.0
            outs.append(net(xb, torch.tensor(AUX[b]).to(dev)).detach().cpu())
        return torch.cat(outs, 0)

    def pinball_chunked(net, X, AUX, Y, ix, qs, dev, chunk=512, zero=None):
        """Chunked multi-quantile pinball over `ix`.

        multi_pinball is a plain mean over all elements and every chunk holds
        the same per-sample element count, so weighting each chunk mean by its
        length reproduces the single-pass value exactly.
        """
        tot, n = 0.0, 0
        for k in range(0, len(ix), chunk):
            b = ix[k:k + chunk]
            xb = torch.tensor(X[b]).to(dev)
            if zero is not None:
                lo, hi = zero
                xb[:, :, lo:hi, :] = 0.0
            out = net(xb, torch.tensor(AUX[b]).to(dev))
            tot += multi_pinball(out, torch.tensor(Y[b]).to(dev),
                                 qs).item() * len(b)
            n += len(b)
        return tot / max(1, n)


# ----------------------------------------------------------------------------
# Packing evaluation (same as forecast_gpu.py).
# ----------------------------------------------------------------------------
def pack(profiles, n_tasks, seed):
    rng = np.random.default_rng(seed)
    N = len(profiles)
    order = rng.choice(N, min(n_tasks, N), replace=False)
    nodes = []
    for i in order:
        p = profiles[i]
        for nd in nodes:
            if np.all(nd["load"] + p <= CAP + 1e-9):
                nd["load"] += p; nd["members"].append(i); break
        else:
            nodes.append({"load": p.copy(), "members": [i]})
    return nodes, order


def nodes_overload(nodes, real):
    over = tot = 0
    for nd in nodes:
        r = real[nd["members"]].sum(axis=0)
        over += int((r > CAP + 1e-9).sum()); tot += r.size
    return len(nodes), over / max(1, tot)


# ----------------------------------------------------------------------------
# Train + evaluate one configuration. Returns a result dict (JSON-serializable).
# ----------------------------------------------------------------------------
def train_eval(data_path, backbone="tcn", history=3, quantile=0.95,
               epochs=120, batch=256, lr=2e-3, emb=48, n_tasks=1000,
               with_features=True, patience=12, seed=0, max_tasks=None,
               quantize=False, diagnose=True, log=print):
    if not HAVE_TORCH:
        return {"status": "torch unavailable", "backbone": backbone}
    torch.manual_seed(seed); np.random.seed(seed)
    d = np.load(data_path, allow_pickle=True)
    series, full = d["series"], d["full"]
    X, AUX, Y, meta = build_examples(series, full, history, with_features,
                                     max_tasks=max_tasks, seed=seed)
    if len(X) < 64:
        return {"status": "too few windows", "backbone": backbone,
                "n": int(len(X))}
    days = sorted(set(t for _, t in meta))
    held = days[-1]
    val = days[-2] if len(days) >= 2 else held
    tr = np.array([j for j, (_, t) in enumerate(meta) if t not in (held, val)])
    va = np.array([j for j, (_, t) in enumerate(meta) if t == val])
    te = np.array([j for j, (_, t) in enumerate(meta) if t == held])
    if len(tr) < 64 or len(te) < 10:
        return {"status": "insufficient split", "backbone": backbone,
                "n_train": int(len(tr)), "n_test": int(len(te))}

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    quantiles = (0.5, 0.9, 0.95, 0.99)
    qi = quantiles.index(quantile) if quantile in quantiles else 2
    net = Forecaster(X.shape[2], H=series.shape[-1], emb=emb, backbone=backbone,
                     aux_dim=AUX.shape[1], quantiles=quantiles).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    qs = net.qs

    def batches(ix):
        rng = np.random.default_rng(seed)
        ix = ix.copy(); rng.shuffle(ix)
        for k in range(0, len(ix), batch):
            b = ix[k:k + batch]
            yield (torch.tensor(X[b]).to(dev), torch.tensor(AUX[b]).to(dev),
                   torch.tensor(Y[b]).to(dev))

    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        net.train()
        for xb, ab, yb in batches(tr):
            opt.zero_grad()
            loss = multi_pinball(net(xb, ab), yb, qs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            vloss = pinball_chunked(net, X, AUX, Y, va, qs, dev)
        if vloss < best_val - 1e-5:
            best_val, bad = vloss, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                log(f"  [{backbone}] early stop @ep{ep} val={best_val:.5f}")
                break
    if best_state is not None:
        net.load_state_dict(best_state)

    net.eval()
    with torch.no_grad():
        pred = infer_chunked(net, X, AUX, te, dev).numpy()   # (nte,Q,2,H)
    learned = pred[:, qi]                                        # chosen quantile
    real = Y[te]
    hist_mean = X[te][:, :, 0:2].mean(axis=1)                    # raw ch mean
    hist_peak = X[te][:, :, 0:2].max(axis=(1, 3))
    peakprof = np.broadcast_to(hist_peak[:, :, None], real.shape)
    zsig = hist_mean + 3.0 * X[te][:, :, 0:2].std(axis=1)

    res = {}
    for name, prof in (("learned_q", learned), ("peak", peakprof),
                       ("blind_z3", zsig)):
        nds, _ = pack(prof, n_tasks, seed=0)
        n, o = nodes_overload(nds, real)
        res[name] = {"nodes": int(n), "overload_pct": round(100 * o, 3)}
    beats_peak = (res["learned_q"]["nodes"] <= res["peak"]["nodes"]
                  and res["learned_q"]["overload_pct"]
                  <= res["peak"]["overload_pct"] + 1e-9)

    out = {
        "status": "ok", "device": dev, "backbone": backbone,
        "data": os.path.basename(data_path), "history": history,
        "quantile": quantile, "epochs": epochs, "with_features": with_features,
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "held_out_day": int(held), "best_val_pinball": round(best_val, 6),
        "in_channels": int(X.shape[2]), "aux_dim": int(AUX.shape[1]),
        "results": res, "beats_peak": bool(beats_peak),
    }

    # ---- iterate-and-improve diagnostics ----------------------------------
    # (1) DATA WEAKNESS: per-test-task pinball error at the reservation
    #     quantile. The weakest decile is what drags accuracy; it is a
    #     candidate to down-weight or drop next iteration. The strongest decile
    #     is where the model already pays off.
    # (2) FEATURE IMPORTANCE: zero each derived-channel group and measure the
    #     rise in validation pinball. A near-zero rise means that feature is
    #     dead weight (safe to remove -> lighter model); a large rise means it
    #     carries signal (keep / invest).
    if diagnose and with_features:
        with torch.no_grad():
            te_err = np.maximum(quantile * (real - learned),
                                (quantile - 1) * (real - learned))
            per_task = te_err.reshape(len(te), -1).mean(axis=1)
            order = np.argsort(per_task)
            dec = max(1, len(order) // 10)
            # channel groups: 0-1 raw, 2-3 mean, 4-5 std, 6-7 harmonic
            groups = {"raw": (0, 2), "xday_mean": (2, 4),
                      "xday_std": (4, 6), "harmonic": (6, 8)}
            base = pinball_chunked(net, X, AUX, Y, va, qs, dev)
            feat_imp = {}
            for g, (a, b) in groups.items():
                if b <= X.shape[2]:
                    feat_imp[g] = round(
                        pinball_chunked(net, X, AUX, Y, va, qs, dev,
                                        zero=(a, b)) - base, 6)
        out["diagnostics"] = {
            "weakest_decile_mean_err": round(float(per_task[order[-dec:]].mean()), 6),
            "strongest_decile_mean_err": round(float(per_task[order[:dec]].mean()), 6),
            "median_task_err": round(float(np.median(per_task)), 6),
            "feature_importance_val_pinball_rise": feat_imp,
            "removable_features": [g for g, v in feat_imp.items() if v < 1e-4],
            "note": ("weakest decile = down-weight/drop candidates; a feature "
                     "with ~0 pinball rise is dead weight and can be dropped to "
                     "shrink the model."),
        }

    # ---- quantized model (deployment / edge RMM) --------------------------
    # Post-training dynamic int8 quantization of the Linear layers, then the
    # Same packing eval. Reports accuracy delta + size so we can trade
    # footprint for a tolerable overload cost.
    if quantize:
        try:
            qnet = torch.quantization.quantize_dynamic(
                net.cpu(), {nn.Linear, nn.GRU}, dtype=torch.qint8).eval()
            with torch.no_grad():
                qpred = qnet(torch.tensor(X[te]),
                             torch.tensor(AUX[te])).numpy()[:, qi]
            qnds, _ = pack(qpred, n_tasks, seed=0)
            qn, qo = nodes_overload(qnds, real)

            def _nbytes(m):
                return sum(p.numel() * p.element_size()
                           for p in m.parameters()) or 1
            out["quantized"] = {
                "dtype": "int8-dynamic(Linear,GRU)",
                "nodes": int(qn), "overload_pct": round(100 * qo, 3),
                "nodes_delta_vs_fp32": int(qn - res["learned_q"]["nodes"]),
                "fp32_param_bytes": int(_nbytes(net)),
                "note": ("dynamic PTQ; for larger savings move to QAT on the "
                         "NRP GPU. Reservation uses the quantized forecast."),
            }
        except Exception as e:  # record the failure, do not crash a sweep
            out["quantized"] = {"status": f"quantization failed: {e}"}
        net.to(dev)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "gct_days.npz"))
    ap.add_argument("--backbone", default="tcn", choices=BACKBONES)
    ap.add_argument("--history", type=int, default=3)
    ap.add_argument("--quantile", type=float, default=0.95)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--no-features", action="store_true")
    ap.add_argument("--max-tasks", type=int, default=None,
                    help="cap tasks/window (use on CPU; None = all, for GPU)")
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "forecast_model.json"))
    a = ap.parse_args()
    r = train_eval(a.data, a.backbone, a.history, a.quantile, a.epochs,
                   with_features=not a.no_features, max_tasks=a.max_tasks,
                   quantize=a.quantize)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(r, open(a.out, "w"), indent=2)
    print(json.dumps(r, indent=2))
