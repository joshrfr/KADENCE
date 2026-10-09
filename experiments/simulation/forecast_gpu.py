"""GPU-trained per-task demand forecaster for out-of-sample admission.

The swappable predictor in the scheduler architecture, upgraded to a proper
learned model trained the way large-scale labs do: a temporal train / test
split, a per-task neural inference structure, a calibrated (quantile) objective,
and a strictly held-out evaluation. Runs on the NRP CUDA image
(python:cuda-v1.5.1); falls back to CPU.

Novel inference structure (per task, weights shared across tasks):
  * each history day's 2-channel (CPU, mem) 288-slot profile is encoded by a
    shared 1-D convolution over the day (a learned rhythm filter bank);
  * a small GRU aggregates the history-day embeddings in order (recency);
  * a decoder emits the next-day per-slot demand at a chosen QUANTILE via the
    pinball loss, so the prediction is a calibrated safe reservation, not a mean.

Then the predicted per-slot profile is used as each task's reservation, tasks
are greedily packed onto unit-capacity nodes, and nodes-at-overload is compared
against the simple baselines (peak, mean+z-sigma) on the real held-out day.

    python3 experiments/simulation/forecast_gpu.py --data data/gct_days.npz --history 3 \
        --quantile 0.95 --epochs 40 --out results/forecast_gpu.json

If the learned model does not beat the simple baselines, that
is itself the "history beats ML" result (prior work, anonymized for review).
"""
from __future__ import annotations

import argparse
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


# ---------- model ----------
if HAVE_TORCH:
    class RhythmQuantileNet(nn.Module):
        """Per-task next-day per-slot quantile forecaster."""
        def __init__(self, H=288, emb=32):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv1d(2, 16, 7, padding=3), nn.ReLU(),
                nn.Conv1d(16, emb, 7, padding=3, stride=2), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1))                     # (B, emb, 1)
            self.gru = nn.GRU(emb, emb, batch_first=True)
            self.dec = nn.Sequential(
                nn.Linear(emb, 128), nn.ReLU(), nn.Linear(128, 2 * H))
            self.H = H

        def forward(self, x):                                # x: (B, D, 2, H)
            B, D, C, H = x.shape
            e = self.enc(x.reshape(B * D, C, H)).reshape(B, D, -1)
            _, h = self.gru(e)                               # (1,B,emb)
            out = self.dec(h[-1]).reshape(B, 2, H)
            return torch.relu(out)                           # demand >= 0


def pinball(pred, target, q):
    e = target - pred
    return torch.mean(torch.maximum(q * e, (q - 1) * e))


# ---------- data ----------
def build_examples(series, full, history):
    """Return list of (X (D,2,H), y (2,H), task_idx, test_day) for tasks present
    across each [t-history, t] window."""
    D = series.shape[0]
    X, Y, meta = [], [], []
    for t in range(history, D):
        cand = np.all(full[t - history:t + 1], axis=0)
        for i in np.nonzero(cand)[0]:
            X.append(series[t - history:t, i])               # (history,2,H)
            Y.append(series[t, i])                            # (2,H)
            meta.append((int(i), int(t)))
    return np.array(X, np.float32), np.array(Y, np.float32), meta


# ---------- packing eval ----------
def pack(profiles, n_tasks, seed):
    """Greedy first-fit: node admits if summed per-slot reservation <= CAP; return
    (nodes, overload) where overload uses the SAME profiles as the real load
    proxy passed in (caller controls what 'real' means)."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "gct_days.npz"))
    ap.add_argument("--history", type=int, default=3)
    ap.add_argument("--quantile", type=float, default=0.95)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n-tasks", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "forecast_gpu.json"))
    args = ap.parse_args()
    d = np.load(args.data, allow_pickle=True)
    series, full = d["series"], d["full"]
    D, N, _, H = series.shape
    X, Y, meta = build_examples(series, full, args.history)
    if len(X) == 0:
        print("no windows; need more days"); return
    test_days = sorted(set(t for _, t in meta))
    held = test_days[-1]                                     # strictly held-out latest day
    tr = np.array([j for j, (_, t) in enumerate(meta) if t != held])
    te = np.array([j for j, (_, t) in enumerate(meta) if t == held])
    prov = {"data": os.path.basename(args.data), "history": args.history,
            "quantile": args.quantile, "epochs": args.epochs,
            "n_train": int(len(tr)), "n_test": int(len(te)), "held_out_day": held,
            "torch": HAVE_TORCH}

    if not HAVE_TORCH or len(tr) < 32 or len(te) < 10:
        prov["status"] = ("torch unavailable or too few examples; run on the NRP "
                          "CUDA image with a multi-day trace")
        json.dump({"provenance": prov}, open(args.out, "w"), indent=2)
        print(json.dumps(prov, indent=2)); return

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    prov["device"] = dev
    net = RhythmQuantileNet(H).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    Xt = torch.tensor(X[tr]).to(dev); Yt = torch.tensor(Y[tr]).to(dev)
    q = args.quantile
    for ep in range(args.epochs):
        net.train(); opt.zero_grad()
        loss = pinball(net(Xt), Yt, q)
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(torch.tensor(X[te]).to(dev)).cpu().numpy()   # (nte,2,H)
    real = Y[te]                                                 # (nte,2,H) real held-out
    # baselines on the same held-out tasks
    hist_mean = X[te].mean(axis=1)                              # (nte,2,H) mean of history
    hist_peak = X[te].max(axis=(1, 3), keepdims=False)          # per-channel peak over history
    peakprof = np.broadcast_to(hist_peak[:, :, None], real.shape)
    zsig = hist_mean + 3.0 * X[te].std(axis=1)                  # mean + 3sigma per slot
    results = {}
    for name, profiles in (("learned_q", pred), ("peak", peakprof), ("blind_z3", zsig)):
        nds, _ = pack(profiles, args.n_tasks, seed=0)
        n, o = nodes_overload(nds, real)
        results[name] = {"nodes": n, "overload_pct": round(100 * o, 3)}
    prov["status"] = "ok"
    out = {"provenance": prov, "results": results}
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
