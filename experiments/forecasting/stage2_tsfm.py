#!/usr/bin/env python3
"""Stage 2 of the forecaster bake-off: pretrained time-series foundation
models (TSFMs), zero-shot, scored on the project's packing metric.

Stage 1 asks whether any closed form or trained arm beats peak-requests.
Stage 2 asks the same of models somebody else trained, with no training here:

  flowstate  IBM FlowState (arXiv 2508.05287). SSM encoder + Functional Basis
             Decoder. Emits 9 quantiles (0.1 .. 0.9). Causal RevIN per series.
  ttm_r2     IBM TinyTimeMixer r2. MLP-Mixer, point forecast only, per-series
             std scaling inside the model.
  ttm_r3     IBM TinyTimeMixer r3. Same family, multi-quantile head.
  reverso_*  Reverso (arXiv 2602.17634), nano / small / base. Long conv +
             DeltaNet, point forecast, per-series min-max scaling inside the
             model. The official code needs CUDA (FlashFFTConv, fla's Triton
             kernels); a pure-torch CPU port of the same graph lives below.

The evaluation is the one in experiments/simulation/forecast_models.py, reproduced exactly: same
build_examples window selection (so the same held-out day and the same task
order), same pack() sample and seed, same nodes_overload(). A forecast becomes
a reservation, the reservations are packed first-fit onto unit-capacity nodes,
and the arm is scored on the REAL held-out day. An arm only beats the baseline
if it uses no more nodes than peak-requests at no worse overload.

How a forecast becomes a reservation (the TSFMs never see the packing metric):
  point forecaster     reservation = max(forecast, 0) + z * sigma, z swept.
                       sigma is a per-series residual std, either from a
                       backtest (forecast the last history day from the days
                       before it) or the per-slot cross-day std that the
                       closed-form blind_z3 arm uses.
  quantile forecaster  reservation = the model's quantile, level swept; plus a
                       "qx" arm that extends past the model's top quantile with
                       q50 + z * (q90 - q50) / 1.2816 (a Gaussian tail).

Univariate models see each of the 2 channels (cpu, mem) as its own series.

Only the sampled packing tasks are forecast (pack() uses nothing else), which is
what makes CPU inference at full scale tractable.

  # prove the pipeline end to end on a tiny synthetic npz (no real data)
  python3 experiments/forecasting/stage2_tsfm.py --make-smoke /tmp/smoke_days.npz
  python3 experiments/forecasting/stage2_tsfm.py --data /tmp/smoke_days.npz --n-tasks 200

  # real run (CPU is fine; Reverso on CPU is the slow one)
  python3 experiments/forecasting/stage2_tsfm.py --models flowstate ttm_r2 ttm_r3 reverso_small

Dependencies are NOT installed by this script. FlowState/TTM need granite-tsfm
(`tsfm_public`) and einops; Reverso needs safetensors + huggingface_hub. Put
them on PYTHONPATH (e.g. pip install --target DIR) rather than globally.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
from experiments.simulation.forecast_models import pack, nodes_overload, CAP  # noqa: E402

import torch  # noqa: E402  (forecast_models already requires it)
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
DEFAULT_DATA = os.path.join(ROOT, "data", "gct_days.npz")
H_DAY = 288                                     # 5-minute slots per day
MODELS = ("flowstate", "ttm_r2", "ttm_r3", "reverso_nano", "reverso_small",
          "reverso_base")
Z90 = 1.2815515655446004                        # N(0,1) 0.9 quantile


# ----------------------------------------------------------------------------
# Data: the same held-out windows and the same sampled task set as the
# reference, without materializing the 8-channel feature tensor.
# ----------------------------------------------------------------------------
def make_smoke(path, days=6, tasks=300, seed=0):
    """Tiny synthetic npz with the real keys/shapes: series (D,N,2,288) float64
    and full (D,N) bool. Heavy-tailed per-task scale (lognormal, a few orders of
    magnitude) times a noisy daily rhythm, so scale handling actually matters.
    Every task is present on every day, so a held-out day exists."""
    rng = np.random.default_rng(seed)
    t = np.arange(H_DAY) / H_DAY
    scale = np.exp(rng.normal(-3.0, 1.2, size=(tasks, 2)))            # heavy tail
    phase = rng.uniform(0, 1, size=(tasks, 1))
    amp = rng.uniform(0.0, 0.8, size=(tasks, 1))
    shape = 1.0 + amp * np.sin(2 * np.pi * (t[None] - phase))          # (N,H)
    series = np.empty((days, tasks, 2, H_DAY))
    for d in range(days):
        noise = np.exp(rng.normal(0, 0.25, size=(tasks, 2, H_DAY)))
        series[d] = scale[:, :, None] * shape[:, None, :] * noise
    full = np.ones((days, tasks), bool)
    full[0, : tasks // 10] = False                                     # a few gaps
    np.savez(path, series=series, full=full)
    return path


def select_windows(full, history, max_tasks, seed):
    """Mirror build_examples' selection loop (including its rng consumption) and
    return (held_day, task_idx) for the strictly held-out day, i.e. the last day
    that has at least one window. task_idx is in the same order as the rows of
    `te` in train_eval, so pack()'s sample lands on the same tasks."""
    D = full.shape[0]
    rng = np.random.default_rng(seed)
    per_day = {}
    for t in range(history, D):
        cand = np.all(full[t - history:t + 1], axis=0)
        idx = np.nonzero(cand)[0]
        if max_tasks is not None and len(idx) > max_tasks:
            idx = rng.choice(idx, max_tasks, replace=False)
        if len(idx):
            per_day[t] = idx
    if not per_day:
        raise SystemExit("no task is present on every day of any window")
    held = max(per_day)
    return held, per_day[held]


def read_days(path, key, days, tasks):
    """series[days][:, tasks] from an .npz without decompressing the whole
    (D,N,2,H) array: stream the zip member and keep one day in memory at a time.
    Falls back to np.load for anything but a C-ordered numeric array."""
    import zipfile
    from numpy.lib import format as npf
    with zipfile.ZipFile(path) as z, z.open(key + ".npy") as f:
        ver = npf.read_magic(f)
        shape, fortran, dtype = (npf.read_array_header_1_0(f) if ver == (1, 0)
                                 else npf.read_array_header_2_0(f))
        if fortran or dtype.hasobject:
            arr = np.load(path, allow_pickle=True)[key]
            return arr[days][:, tasks]
        day_bytes = int(np.prod(shape[1:])) * dtype.itemsize
        out, last = [], max(days)
        for t in range(last + 1):
            if t not in days:                      # discard in 64 MB pieces
                left = day_bytes
                while left:
                    left -= len(f.read(min(left, 1 << 26)))
                continue
            buf = bytearray()
            while len(buf) < day_bytes:
                buf += f.read(min(day_bytes - len(buf), 1 << 26))
            day = np.frombuffer(buf, dtype).reshape(shape[1:])
            out.append(np.array(day[tasks]))
            del day, buf
    return np.stack(out)


def load_eval_set(path, history, n_tasks, max_tasks, seed):
    """-> dict with hist (n,history,2,H) float32 of the sampled tasks, real
    (n,2,H), order (positions of the sample inside the N held-out windows) and
    N. Only the sampled tasks are kept."""
    d = np.load(path, allow_pickle=True)
    full = d["full"]
    held, idx = select_windows(full, history, max_tasks, seed)
    N = len(idx)
    # same draw pack() makes, so the sample is identical by construction
    order = np.random.default_rng(0).choice(N, min(n_tasks, N), replace=False)
    sel = idx[order]
    # only days held-history .. held are needed, and only the sampled tasks
    # within them; stream them so the 3 GB float64 array is never resident
    need = list(range(held - history, held + 1))
    rows = read_days(path, "series", need, sel)                     # (h+1,n,2,H)
    hist = rows[:history].astype(np.float32)                        # (h,n,2,H)
    real = rows[history].astype(np.float32)                         # (n,2,H)
    hist = np.ascontiguousarray(hist.transpose(1, 0, 2, 3))         # (n,h,2,H)
    return {"hist": hist, "real": real, "order": order, "N": N,
            "held": int(held), "n_sampled": int(len(order)),
            "task_ids": sel}


# ----------------------------------------------------------------------------
# Scoring: scatter the sampled reservations into the N-row table pack() expects.
# ----------------------------------------------------------------------------
class Scorer:
    def __init__(self, ev, n_tasks, seed):
        self.ev, self.n_tasks, self.seed = ev, n_tasks, seed
        N = ev["N"]
        self.real_full = np.zeros((N, 2, H_DAY), np.float32)
        self.real_full[ev["order"]] = ev["real"]
        self.shell = np.zeros((N, 2, H_DAY), np.float32)

    def __call__(self, prof):
        """prof (n,2,H): reservation for the sampled tasks, in sample order."""
        full = self.shell.copy()
        full[self.ev["order"]] = prof
        nds, order = pack(full, self.n_tasks, seed=self.seed)
        assert np.array_equal(order, self.ev["order"]), "sample drifted"
        n, o = nodes_overload(nds, self.real_full)
        return {"nodes": int(n), "overload_pct": round(100 * o, 3)}


def closed_form_arms(hist):
    """The two reference arms exactly as train_eval builds them."""
    hist_mean = hist.mean(axis=1)                                    # (n,2,H)
    hist_peak = hist.max(axis=(1, 3))                                # (n,2)
    peak = np.broadcast_to(hist_peak[:, :, None], hist_mean.shape)
    zsig = hist_mean + 3.0 * hist.std(axis=1)
    return {"peak": peak, "blind_z3": zsig}


# ----------------------------------------------------------------------------
# Context handling. The models want a fixed context length; we have
# history * 288 slots. Longer: keep the most recent slots. Shorter: tile the
# history backwards (period = whole days, so daily phase and the series min /
# max are preserved, which matters for min-max and std scaling).
# ----------------------------------------------------------------------------
def fit_context(ctx, length, pad="tile"):
    M, L = ctx.shape
    if L >= length:
        return ctx[:, -length:]
    need = length - L
    if pad == "zero":
        left = np.zeros((M, need), ctx.dtype)
    elif pad == "edge":
        left = np.repeat(ctx[:, :1], need, axis=1)
    else:
        reps = int(math.ceil(need / L))
        left = np.tile(ctx, (1, reps))[:, -need:]
    return np.concatenate([left, ctx], axis=1)


def to_series(hist):
    """(n,h,2,H) -> (n*2, h*H): each channel of each task is one series."""
    n, h, c, H = hist.shape
    return np.ascontiguousarray(
        hist.transpose(0, 2, 1, 3).reshape(n * c, h * H))


def batched(fn, x, bs, log=None, tag=""):
    outs = []
    for k in range(0, len(x), bs):
        outs.append(fn(x[k:k + bs]))
        if log and (k // bs) % 10 == 0:
            log(f"    [{tag}] {min(k + bs, len(x))}/{len(x)}")
    return np.concatenate(outs, 0)


# ----------------------------------------------------------------------------
# Adapters. Each exposes predict(ctx (M,L)) -> (point (M,H), quant (M,Q,H)|None)
# and a `meta` dict used by the report.
# ----------------------------------------------------------------------------
class FlowStateAdapter:
    def __init__(self, repo, revision, scale_factor, bs, pad):
        from tsfm_public import FlowStateForPrediction   # needs granite-tsfm
        self.net = FlowStateForPrediction.from_pretrained(
            repo, revision=revision).eval()
        self.scale_factor, self.bs = scale_factor, bs
        self.levels = [float(q) for q in self.net.config.quantiles]
        self.meta = {"repo": repo, "revision": revision,
                     "params": sum(p.numel() for p in self.net.parameters()),
                     "scale_factor": scale_factor,
                     "context_native": int(self.net.config.context_length)}

    @torch.no_grad()
    def _run(self, x):
        t = torch.from_numpy(x.T.copy()).unsqueeze(-1)            # (L,B,1)
        out = self.net(t, scale_factor=self.scale_factor,
                       prediction_length=H_DAY, batch_first=False)
        q = out.quantile_outputs                                   # (B,Q,T,1)
        if q is None:
            q = out.prediction_outputs
        q = q.squeeze(-1).float().numpy()
        return q

    def predict(self, ctx, log=None):
        q = batched(self._run, ctx, self.bs, log, "flowstate")[..., :H_DAY]
        mid = self.levels.index(0.5)
        return q[:, mid], q


class TTMAdapter:
    def __init__(self, repo, revision, bs, pad):
        from tsfm_public.models.tinytimemixer import (
            TinyTimeMixerConfig, TinyTimeMixerForDecomposedPrediction,
            TinyTimeMixerForPrediction)
        # r3 is a trend + residual pair of mixers and needs the decomposed class;
        # the plain class loads it with most weights unexpected and the rest
        # randomly initialized, so a clean load is enforced below.
        cfg = TinyTimeMixerConfig.from_pretrained(repo, revision=revision)
        cls = (TinyTimeMixerForDecomposedPrediction
               if getattr(cfg, "decompose", False) else TinyTimeMixerForPrediction)
        self.net, info = cls.from_pretrained(repo, revision=revision,
                                             output_loading_info=True)
        self.net.eval()
        if info.get("missing_keys") or info.get("unexpected_keys"):
            raise RuntimeError(
                f"{repo}@{revision}: unclean load via {cls.__name__} "
                f"(missing {len(info.get('missing_keys', []))}, unexpected "
                f"{len(info.get('unexpected_keys', []))})")
        c = self.net.config
        self.ctx_len, self.pred_len = int(c.context_length), int(c.prediction_length)
        self.bs, self.pad = bs, pad
        self.levels = ([float(q) for q in getattr(c, "quantile_levels", [])]
                       if getattr(c, "multi_quantile_head", False) else None)
        self.meta = {"repo": repo, "revision": revision,
                     "params": sum(p.numel() for p in self.net.parameters()),
                     "context_native": self.ctx_len,
                     "scaling": getattr(c, "scaling", None),
                     "pred_len_native": self.pred_len}
        if self.pred_len < H_DAY:
            raise RuntimeError(f"{repo}@{revision} predicts {self.pred_len} < "
                             f"{H_DAY} slots; use a longer-horizon branch")

    @torch.no_grad()
    def _run(self, x):
        t = torch.from_numpy(x).unsqueeze(-1)                      # (B,L,1)
        out = self.net(past_values=t, return_loss=False)
        pt = out.prediction_outputs.squeeze(-1).float().numpy()[:, :H_DAY]
        q = None
        if self.levels and out.quantile_outputs is not None:
            q = out.quantile_outputs.float().numpy()
            # tolerate (B,Q,T,1) / (B,T,1,Q) / (B,T,Q)
            q = np.squeeze(q)
            if q.shape[-1] == len(self.levels) and q.ndim == 3:
                q = q.transpose(0, 2, 1)
            q = q[..., :H_DAY]
            pt = pt  # point head (mean or median per mq_q50_type)
        return np.concatenate(
            [pt[:, None], q] if q is not None else [pt[:, None]], 1)

    def predict(self, ctx, log=None):
        x = fit_context(ctx, self.ctx_len, self.pad)
        o = batched(self._run, x, self.bs, log, "ttm")
        return o[:, 0], (o[:, 1:] if o.shape[1] > 1 else None)


# ---- Reverso: pure-torch CPU port of github.com/shinfxh/reverso (MIT) -------
# The official model hard-imports flashfftconv and fla (CUDA/Triton). This port
# keeps every parameter name and the graph identical so the published
# safetensors load with strict=True, and replaces only the two kernels:
#   FlashFFTConv  -> causal FFT convolution (zero-padded to 2L), float32
#   fla DeltaNet  -> the delta-rule recurrence of fla.ops.delta_rule.naive
# It is float32 where the GPU path is bf16 autocast, and it has NOT been checked
# bit-for-bit against the CUDA kernels (no GPU here).
class _PosEmb(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = (torch.arange(0, d_model, 2).float()
               * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class _Gating(nn.Module):
    def __init__(self, ch, k=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, k, padding=k // 2, groups=ch), nn.SiLU(),
            nn.Conv1d(ch, ch, 1))

    def forward(self, x):
        return torch.sigmoid(self.net(x))


class _MLPBlock(nn.Module):
    def __init__(self, d_in, d_out, d_inter=0):
        super().__init__()
        self.norm = nn.LayerNorm(d_out)
        if d_inter and d_inter > 0:
            self.linear = nn.Linear(d_in, d_inter)
            self.linear_final = nn.Linear(d_inter, d_out)
        else:
            self.linear = nn.Linear(d_in, d_out)
            self.linear_final = nn.Identity()
        self.activation = nn.ReLU()
        self.skip_linear = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        y = self.norm(self.linear_final(self.activation(self.linear(x))))
        return (self.skip_linear(x) + y).permute(0, 2, 1)


class _CNNBlock(nn.Module):
    def __init__(self, ch, seq_len, k=3):
        super().__init__()
        self.k = nn.Parameter(torch.randn(ch, seq_len))
        self.pregate = _Gating(ch, k)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(ch)

    def forward(self, x):                                  # x (B,C,L)
        L = x.shape[-1]
        u = x * self.pregate(x)
        n = 2 * L
        y = torch.fft.irfft(torch.fft.rfft(u, n=n) *
                            torch.fft.rfft(self.k, n=n), n=n)[..., :L]
        y = self.norm(self.activation(y).transpose(1, 2)).transpose(1, 2)
        return y + x


class _ShortConv(nn.Conv1d):
    """fla ShortConvolution: causal depthwise conv + SiLU. Input (B,T,D)."""
    def __init__(self, d, k):
        super().__init__(d, d, k, groups=d, bias=False, padding=k - 1)

    def forward(self, x):
        T = x.shape[1]
        y = super().forward(x.transpose(1, 2))[..., :T]
        return F.silu(y).transpose(1, 2)


class _RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight


class _DeltaNet(nn.Module):
    def __init__(self, d, expand_v=1.0, heads=4, conv=4):
        super().__init__()
        self.h = heads
        kd, vd = d, int(d * expand_v)
        self.dk, self.dv = kd // heads, vd // heads
        self.q_proj = nn.Linear(d, kd, bias=False)
        self.k_proj = nn.Linear(d, kd, bias=False)
        self.v_proj = nn.Linear(d, vd, bias=False)
        self.b_proj = nn.Linear(d, heads, bias=False)
        self.q_conv1d, self.k_conv1d = _ShortConv(kd, conv), _ShortConv(kd, conv)
        self.v_conv1d = _ShortConv(vd, conv)
        self.o_norm = _RMSNorm(self.dv)
        self.o_proj = nn.Linear(vd, d, bias=False)

    def forward(self, x):                                  # x (B,T,D)
        B, T, _ = x.shape
        q = self.q_conv1d(self.q_proj(x)).view(B, T, self.h, self.dk)
        k = self.k_conv1d(self.k_proj(x)).view(B, T, self.h, self.dk)
        v = self.v_conv1d(self.v_proj(x)).view(B, T, self.h, self.dv)
        beta = self.b_proj(x).sigmoid()                    # (B,T,h)
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)   # l2 norm
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        q = (q * self.dk ** -0.5).permute(0, 2, 1, 3)      # (B,h,T,dk)
        k, v = k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
        beta = beta.permute(0, 2, 1).unsqueeze(-1)         # (B,h,T,1)
        S = x.new_zeros(B, self.h, self.dk, self.dv)
        o = torch.empty_like(v)
        for t in range(T):                                 # delta rule
            kt = k[:, :, t]                                # (B,h,dk)
            err = (v[:, :, t] - torch.einsum("bhd,bhdv->bhv", kt, S)) * beta[:, :, t]
            S = S + kt.unsqueeze(-1) * err.unsqueeze(-2)
            o[:, :, t] = torch.einsum("bhd,bhdv->bhv", q[:, :, t], S)
        o = self.o_norm(o.permute(0, 2, 1, 3)).reshape(B, T, -1)
        return self.o_proj(o)


class _AttnBlock(nn.Module):
    def __init__(self, d, expand_v, weave, inter):
        super().__init__()
        self.weave, self.inter = weave, inter
        self.attention = _DeltaNet(d, expand_v)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        xt = x.transpose(1, 2)
        res = xt
        if self.weave and self.inter:
            xt = xt.clone()
            xt[:, 0:1] = xt[:, 0:1] + xt[:, -1:]
        return (self.norm(self.attention(xt)) + res).transpose(1, 2)


class ReversoCPU(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.seq_len, self.out_len = c["seq_len"], c["output_token_len"]
        d = c["d_model"]
        self.use_norm = c["use_norm"]
        self.embedding = nn.Linear(1, d, bias=False)
        mods = [m.strip() for m in c["main_module"].split(",")]
        layers = []
        for i, m in enumerate(mods):
            if m == "conv":
                layers.append(_CNNBlock(d, self.seq_len, c.get("gating_kernel_size", 3)))
            else:
                layers.append(_AttnBlock(d, c.get("expand_v", 1.0),
                                         c.get("state_weaving", False),
                                         0 < i < len(mods) - 1))
            layers.append(_MLPBlock(d, d, c["d_intermediate"]))
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(c["input_token_len"],
                              c.get("output_bottleneck_dim", self.out_len),
                              bias=bool(c["learn_bias"]))
        self.simple_q_proj = nn.Linear(d, d)
        self.key_proj, self.value_proj = nn.Linear(d, d), nn.Linear(d, d)
        self.out_proj = nn.Linear(d, 1)
        self.use_output_pe = bool(c.get("use_output_pe", False))
        if self.use_output_pe:
            self.output_position_embedding = _PosEmb(d, self.seq_len + self.out_len)
            self.post_pe_q_proj = nn.Linear(d, d)

    def forward(self, x):                                  # x (B,L,1)
        if self.use_norm:                                  # per-series min-max
            lo, hi = x.min(1, keepdim=True)[0], x.max(1, keepdim=True)[0]
            rng = torch.clamp(hi - lo, min=1e-5)
            x = (x - lo) / rng
        h = self.layers(self.embedding(x).transpose(1, 2))
        q = self.simple_q_proj(self.head(h).permute(0, 2, 1))
        hp = h.permute(0, 2, 1)
        if self.use_output_pe:
            fh = torch.cat([hp, q], 1)
            fh = fh + self.output_position_embedding(fh)
            hp, q = fh[:, :hp.shape[1]], self.post_pe_q_proj(fh[:, hp.shape[1]:])
        out = self.out_proj(F.scaled_dot_product_attention(
            q, self.key_proj(hp), self.value_proj(hp)))
        return out * rng + lo if self.use_norm else out


class ReversoAdapter:
    def __init__(self, size, bs, pad):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        repo = f"shinfxh/reverso-{size}"
        cfg = json.load(open(hf_hub_download(repo, "config.json")))
        self.net = ReversoCPU(cfg).eval()
        sd = load_file(hf_hub_download(repo, "model.safetensors"))
        miss, unexp = self.net.load_state_dict(sd, strict=False)
        # FlashFFTConv's own tables are not part of the graph we port
        unexp = [k for k in unexp if "flashfftconv" not in k]
        if miss or unexp:
            raise RuntimeError(f"{repo}: state_dict mismatch "
                             f"missing={miss[:4]} unexpected={unexp[:4]}")
        self.ctx_len, self.step = cfg["seq_len"], cfg["output_token_len"]
        self.bs, self.pad, self.levels = bs, pad, None
        self.meta = {"repo": repo, "context_native": self.ctx_len,
                     "params": sum(p.numel() for p in self.net.parameters()),
                     "rollout_step": self.step, "impl": "pure-torch CPU port"}

    @torch.no_grad()
    def _run(self, x):                                     # x (B,ctx_len)
        ctx = torch.from_numpy(x).unsqueeze(-1)
        preds = []
        for _ in range(int(math.ceil(H_DAY / self.step))):  # AR rollout
            o = self.net(ctx[:, -self.ctx_len:])[:, -self.step:]
            preds.append(o)
            ctx = torch.cat([ctx, o], 1)
        return torch.cat(preds, 1)[:, :H_DAY, 0].numpy()

    def predict(self, ctx, log=None):
        x = fit_context(ctx, self.ctx_len, self.pad)
        return batched(self._run, x, self.bs, log, "reverso"), None


def build_adapter(name, a):
    pad = a.pad
    if name == "flowstate":
        return FlowStateAdapter(a.fs_repo, a.fs_revision,
                                a.fs_scale or 24.0 / H_DAY, a.fs_batch, pad)
    if name == "ttm_r2":
        return TTMAdapter("ibm-granite/granite-timeseries-ttm-r2",
                          a.ttm_r2_branch, a.batch, pad)
    if name == "ttm_r3":
        return TTMAdapter("ibm-granite/granite-timeseries-ttm-r3",
                          a.ttm_r3_branch, a.batch, pad)
    return ReversoAdapter(name.split("_", 1)[1], a.batch, pad)


# ----------------------------------------------------------------------------
# Forecast -> reservation sweeps.
# ----------------------------------------------------------------------------
def unstack(x, n):
    """(n*2, ...) -> (n, 2, ...)"""
    return x.reshape(n, 2, *x.shape[1:])


def evaluate_model(name, a, ev, score, base, log):
    t0 = time.time()
    ad = build_adapter(name, a)
    n, h = ev["n_sampled"], a.history
    ctx = to_series(ev["hist"])
    log(f"  [{name}] forecasting {len(ctx)} series "
        f"(context {ctx.shape[1]} slots)")
    point, quant = ad.predict(ctx, log)
    point = np.maximum(unstack(point, n), 0.0)
    qmat = None if quant is None else np.maximum(unstack(quant, n), 0.0)
    t_fc = time.time() - t0

    sigmas = {}
    if "xday" in a.sigma:
        sigmas["xday"] = ev["hist"].std(axis=1)                        # (n,2,H)
    if "backtest" in a.sigma:
        if h < 2:
            log(f"  [{name}] backtest sigma needs --history >= 2; skipped")
        else:
            bt_ctx = to_series(ev["hist"][:, :h - 1])
            bt_pt, _ = ad.predict(bt_ctx, log)
            bt_pt = np.maximum(unstack(bt_pt, n), 0.0)
            resid = ev["hist"][:, h - 1] - bt_pt                       # (n,2,H)
            sigmas["backtest"] = np.broadcast_to(
                resid.std(axis=-1, keepdims=True), resid.shape)

    rows = []

    def add(arm, prof, **kw):
        r = score(np.asarray(prof, np.float32))
        r.update(arm=arm, model=name, **kw)
        r["beats_peak"] = bool(r["nodes"] <= base["peak"]["nodes"] and
                               r["overload_pct"] <= base["peak"]["overload_pct"] + 1e-9)
        rows.append(r)

    for sname, sig in sigmas.items():
        for z in a.z:
            add(f"{name}+z*{sname}", point + z * sig, z=z, sigma=sname)
    if qmat is not None:
        levels = ad.levels
        for qi, q in enumerate(levels):
            add(f"{name}@q{q:g}", qmat[:, :, qi], quantile=q)
        mid, top = levels.index(0.5), levels.index(max(levels))
        sq = (qmat[:, :, top] - qmat[:, :, mid]) / (
            Z90 if max(levels) == 0.9 else 1.0)
        for z in a.z:
            add(f"{name}@qx", qmat[:, :, mid] + z * sq, z=z, sigma="q90tail")
    info = dict(ad.meta)
    info.update(status="ok", emits_quantiles=qmat is not None,
                forecast_seconds=round(t_fc, 1),
                total_seconds=round(time.time() - t0, 1),
                point_mae_vs_real=round(float(np.abs(point - ev["real"]).mean()), 6))
    return info, rows


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--make-smoke", metavar="PATH",
                    help="write a tiny synthetic npz (series, full) and exit")
    ap.add_argument("--models", nargs="*", default=["flowstate", "ttm_r2",
                    "ttm_r3", "reverso_small"], choices=MODELS)
    ap.add_argument("--history", type=int, default=3)
    ap.add_argument("--n-tasks", type=int, default=1000,
                    help="tasks packed (the reference n_tasks)")
    ap.add_argument("--max-tasks", type=int, default=None,
                    help="cap tasks/window, as in build_examples (None = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--z", nargs="+", type=float,
                    default=[0, 0.5, 1, 1.5, 2, 3, 4, 6])
    ap.add_argument("--sigma", nargs="+", default=["backtest", "xday"],
                    choices=["backtest", "xday"])
    ap.add_argument("--pad", default="tile", choices=["tile", "edge", "zero"],
                    help="how a context shorter than the model's is filled")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--fs-batch", type=int, default=16,
                    help="FlowState holds a 512-wide complex SSM state per step; "
                         "~2 GB at batch 100 / 864 slots, so keep this small")
    ap.add_argument("--fs-repo", default="ibm-granite/granite-timeseries-flowstate-r1")
    ap.add_argument("--fs-revision", default="r1.1",
                    help="r1.1 = 18.5M params, 4096 ctx; r1.0 = 9.07M, 2048 ctx")
    ap.add_argument("--fs-scale", type=float, default=None,
                    help="FlowState scale_factor; default 24/288 (daily cycle)")
    ap.add_argument("--ttm-r2-branch", default="512-336-r2")
    ap.add_argument("--ttm-r3-branch", default="512-336-dec-512-r3")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(RESULTS, "stage2_tsfm.json"))
    a = ap.parse_args()
    if a.make_smoke:
        print("wrote", make_smoke(a.make_smoke))
        return
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    log = lambda m: print(m, flush=True)               # noqa: E731

    ev = load_eval_set(a.data, a.history, a.n_tasks, a.max_tasks, a.seed)
    score = Scorer(ev, a.n_tasks, seed=0)
    log(f"[stage2] {os.path.basename(a.data)} held-out day {ev['held']}, "
        f"{ev['N']} windows, {ev['n_sampled']} packed, history {a.history}")

    base = {k: score(v) for k, v in closed_form_arms(ev["hist"]).items()}
    for k, v in base.items():
        log(f"[stage2] baseline {k:9s} {v['nodes']:4d} nodes "
            f"{v['overload_pct']:.3f}% overload")

    t0 = time.time()
    infos, rows = {}, []
    for m in a.models:
        log(f"[stage2] >>> {m}")
        try:
            info, r = evaluate_model(m, a, ev, score, base, log)
            infos[m], rows = info, rows + r
            ok = [x for x in r if x["beats_peak"]]
            best = min(r, key=lambda x: (x["overload_pct"] > base["peak"]["overload_pct"] + 1e-9,
                                         x["nodes"], x["overload_pct"]))
            log(f"[stage2]     best {best['arm']} z={best.get('z', best.get('quantile'))}: "
                f"{best['nodes']}n {best['overload_pct']}%  beats_peak arms: {len(ok)}")
        except ImportError as e:
            infos[m] = {"status": f"skipped, missing dependency: {e.name or e}"}
            log(f"[stage2]     skipped: {infos[m]['status']}")
        except Exception as e:                            # record, do not abort
            infos[m] = {"status": f"failed: {type(e).__name__}: {e}"}
            log(f"[stage2]     failed: {infos[m]['status']}")

    out = {"status": "ok", "data": os.path.basename(a.data),
           "held_out_day": ev["held"], "n_windows": ev["N"],
           "n_packed": ev["n_sampled"], "history": a.history,
           "baselines": base, "models": infos, "rows": rows,
           "elapsed_s": round(time.time() - t0, 1),
           "note": ("zero-shot, no training. Reverso runs on a float32 "
                    "pure-torch port, not the CUDA kernels.")}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)

    # human-readable table: best (fewest nodes at no-worse overload) per arm
    md = ["# Time-series foundation-model arms versus reservation\n",
          f"_{out['data']}, held-out day {ev['held']}, {ev['n_sampled']} packed, "
          f"history {a.history}, {out['elapsed_s']}s._\n",
          "An arm beats peak-requests only with no more nodes at no worse overload.\n",
          "| arm | param | nodes | overload % | beats peak |", "|---|---|---|---|---|"]
    for k, v in base.items():
        md.append(f"| {k} | - | {v['nodes']} | {v['overload_pct']} | - |")
    for r in sorted(rows, key=lambda r: (r["model"], r["nodes"], r["overload_pct"])):
        p = r.get("z", r.get("quantile"))
        md.append(f"| {r['arm']} | {p} | {r['nodes']} | {r['overload_pct']} | "
                  f"{r['beats_peak']} |")
    for m, i in infos.items():
        if i.get("status") != "ok":
            md.append(f"\n- {m}: {i['status']}")
    open(os.path.splitext(a.out)[0] + ".md", "w").write("\n".join(md) + "\n")
    log(f"[stage2] wrote {a.out} and {os.path.splitext(a.out)[0]}.md")


if __name__ == "__main__":
    main()
