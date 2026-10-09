#!/usr/bin/env python3
"""Automated training sweep for the KADENCE demand forecaster.

Iterates backbone x dataset x history x reservation-quantile, trains each with
the real recipe (experiments/simulation/forecast_models.train_eval), and writes a leaderboard plus
the iterate-and-improve diagnostics so we can see, run over run:
  * which configuration wins (nodes at equal overload vs the peak baseline),
  * whether a quantized int8 model holds up,
  * WHICH DATA IS WEAK (weakest-decile task error) and
  * WHICH FEATURES ARE DEAD WEIGHT (near-zero validation pinball rise when
    zeroed) so the next iteration can drop them and stay light.

Designed to run on a GPU (NRP A100/L40); on CPU use --max-tasks to keep it
tractable. Losses to the peak baseline are reported as losses.

  # fast CPU smoke
  python3 experiments/forecasting/train_forecaster.py --quick --max-tasks 400
  # full GPU sweep (on NRP)
  python3 experiments/forecasting/train_forecaster.py --epochs 400 --quantize
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
from experiments.simulation.forecast_models import train_eval, BACKBONES  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
DATASETS = {"google": os.path.join(ROOT, "data", "gct_days.npz"),
            "alibaba": os.path.join(ROOT, "data", "alibaba_days.npz")}


def _atomic_json(path, value):
    """A retry sees either the previous complete checkpoint or the new one."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(value, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _sweep_identity(a):
    inputs = {}
    for ds in a.datasets:
        path = DATASETS[ds]
        st = os.stat(path)  # missing data is a failure, never a partial success
        inputs[ds] = {"path": os.path.abspath(path), "bytes": st.st_size,
                      "mtime_ns": st.st_mtime_ns}
    sources = {}
    for source in (__file__, os.path.join(ROOT, "experiments", "simulation", "forecast_models.py")):
        with open(source, "rb") as f:
            sources[os.path.relpath(source, ROOT)] = hashlib.sha256(f.read()).hexdigest()
    spec = {"backbones": a.backbones, "datasets": a.datasets,
            "history": a.history, "quantiles": a.quantiles,
            "epochs": a.epochs, "max_tasks": a.max_tasks,
            "quantize": a.quantize, "inputs": inputs, "sources_sha256": sources}
    encoded = json.dumps(spec, sort_keys=True).encode()
    return spec, hashlib.sha256(encoded).hexdigest()


def _tag(ds, bb, hist, q):
    return f"{ds}/{bb}/h{hist}/q{q}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=list(BACKBONES),
                    choices=list(BACKBONES))
    ap.add_argument("--datasets", nargs="+", default=["google"],
                    choices=list(DATASETS))
    ap.add_argument("--history", nargs="+", type=int, default=[3])
    ap.add_argument("--quantiles", nargs="+", type=float, default=[0.95])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="tiny config for a correctness smoke test")
    ap.add_argument("--require-gpu", action="store_true",
                    help="assert CUDA is present (NRP A100/L40); no CPU fallback")
    ap.add_argument("--out", default=os.path.join(RESULTS, "forecaster_sweep.json"))
    a = ap.parse_args()
    runtime = {"gpu_required": bool(a.require_gpu)}
    if a.require_gpu:
        import torch
        assert torch.cuda.is_available(), \
            "CUDA GPU required; CPU training is prohibited (per NRP workflow)."
        runtime.update({
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
        })
        print(f"[sweep] GPU: {runtime['device']} | torch {runtime['torch']} "
              f"| CUDA {runtime['cuda_build']}", flush=True)
    if a.quick:
        a.epochs = min(a.epochs, 8)
        if a.max_tasks is None:
            a.max_tasks = 400

    spec, signature = _sweep_identity(a)
    t0 = time.time()
    partial_path = a.out + ".partial.json"
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            progress = json.load(f)
        if progress.get("schema") != 1 or progress.get("signature") != signature:
            raise RuntimeError(f"checkpoint does not match this sweep: {partial_path}; "
                               "archive it explicitly before starting a different sweep")
        print(f"[sweep] resuming {len(progress['completed'])} completed configs", flush=True)
    else:
        progress = {"schema": 1, "signature": signature, "spec": spec,
                    "started_at": t0, "completed": {}}
    runs = []
    for ds in a.datasets:
        path = DATASETS[ds]
        for bb in a.backbones:
            for hist in a.history:
                for q in a.quantiles:
                    tag = _tag(ds, bb, hist, q)
                    if tag in progress["completed"]:
                        print(f"[sweep] resume skip {tag}", flush=True)
                        runs.append(progress["completed"][tag])
                        continue
                    print(f"[sweep] >>> {tag}", flush=True)
                    r = train_eval(path, backbone=bb, history=hist, quantile=q,
                                   epochs=a.epochs, max_tasks=a.max_tasks,
                                   quantize=a.quantize,
                                   log=lambda m: print(m, flush=True))
                    r["dataset"] = ds
                    r["config"] = tag
                    runs.append(r)
                    progress["completed"][tag] = r
                    _atomic_json(partial_path, progress)
                    print(f"[sweep] checkpoint {len(progress['completed'])} configs", flush=True)
                    if r.get("status") == "ok":
                        res = r["results"]
                        print(f"[sweep]     learned={res['learned_q']['nodes']}n "
                              f"peak={res['peak']['nodes']}n "
                              f"beats_peak={r['beats_peak']}", flush=True)
                    gc.collect()
                    if a.require_gpu:
                        import torch
                        torch.cuda.empty_cache()

    # leaderboard: ok runs ranked by (beats_peak desc, learned nodes asc) at
    # the constraint that overload does not exceed peak's.
    ok = [r for r in runs if r.get("status") == "ok"]
    ok.sort(key=lambda r: (not r["beats_peak"], r["results"]["learned_q"]["nodes"]))
    board = []
    for r in ok:
        res = r["results"]
        row = {
            "config": f"{r['dataset']}/{r['backbone']}/h{r['history']}/q{r['quantile']}",
            "learned_nodes": res["learned_q"]["nodes"],
            "learned_overload_pct": res["learned_q"]["overload_pct"],
            "peak_nodes": res["peak"]["nodes"],
            "beats_peak": r["beats_peak"],
            "best_val_pinball": r.get("best_val_pinball"),
        }
        if "quantized" in r and "nodes" in r["quantized"]:
            row["quant_nodes"] = r["quantized"]["nodes"]
            row["quant_delta"] = r["quantized"]["nodes_delta_vs_fp32"]
        if "diagnostics" in r:
            dg = r["diagnostics"]
            row["weakest_decile_err"] = dg["weakest_decile_mean_err"]
            row["removable_features"] = dg["removable_features"]
        board.append(row)

    best = board[0] if board else None
    summary = {
        "generated_at": time.time(), "elapsed_s": round(time.time() - t0, 1),
        "epochs": a.epochs, "max_tasks": a.max_tasks, "n_runs": len(runs),
        "runtime": runtime, "sweep": spec, "signature": signature,
        "n_expected": len(a.datasets) * len(a.backbones) * len(a.history) * len(a.quantiles),
        "complete": all(r.get("status") == "ok" for r in runs),
        "best": best, "leaderboard": board, "runs": runs,
    }
    _atomic_json(a.out, summary)

    # human-readable leaderboard
    md = ["# Forecaster sweep\n",
          "_A sweep over model configurations. It sizes the search; the "
          "reported comparison against reservation is the matched run under "
          "`results/fair_compare/`._\n",
          f"_epochs={a.epochs}, max_tasks={a.max_tasks}, "
          f"{len(ok)}/{len(runs)} runs ok, {summary['elapsed_s']}s._\n",
          "Ranking rule: a config only 'beats peak' if it uses no more nodes "
          "than peak-requests at no worse overload on the held-out day.\n",
          "| config | learned nodes | overload % | peak nodes | beats peak | "
          "quant nodes | weakest-decile err | removable features |",
          "|---|---|---|---|---|---|---|---|"]
    for row in board:
        md.append(
            f"| {row['config']} | {row['learned_nodes']} | "
            f"{row['learned_overload_pct']} | {row['peak_nodes']} | "
            f"{row['beats_peak']} | {row.get('quant_nodes','-')} | "
            f"{row.get('weakest_decile_err','-')} | "
            f"{','.join(row.get('removable_features', [])) or '-'} |")
    if best:
        md.append(f"\n**Best:** {best['config']} "
                  f"({best['learned_nodes']} nodes, beats_peak={best['beats_peak']}).")
    board_path = os.path.join(os.path.dirname(os.path.abspath(a.out)),
                              "forecaster_leaderboard.md")
    open(board_path, "w").write(
        "\n".join(md) + "\n")
    print(f"\n[sweep] wrote {a.out} and results/forecaster_leaderboard.md "
          f"in {summary['elapsed_s']}s", flush=True)
    if best:
        print(f"[sweep] best: {best['config']} -> {best['learned_nodes']} nodes "
              f"(beats_peak={best['beats_peak']})", flush=True)
    if not summary["complete"]:
        raise RuntimeError("sweep finished with non-ok configurations; inspect runs in JSON")


if __name__ == "__main__":
    main()
