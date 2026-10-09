"""Crash/retry behavior of the NRP forecaster sweep."""

import json
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "forecasting" / "train_forecaster.py"
SPEC = importlib.util.spec_from_file_location("train_forecaster_sweep", SCRIPT)
sweep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sweep)


def test_sweep_resumes_only_matching_completed_configs(tmp_path, monkeypatch):
    data = tmp_path / "input.npz"
    data.write_bytes(b"test trace")
    monkeypatch.setattr(sweep, "DATASETS", {"google": str(data)})
    out = tmp_path / "sweep.json"
    args = ["train_forecaster.py", "--datasets", "google", "--backbones",
            "cnn", "tcn", "--epochs", "1", "--out", str(out)]
    monkeypatch.setattr(sys, "argv", args)
    seen = []

    def fake_train(_path, *, backbone, history, quantile, **_kwargs):
        seen.append(backbone)
        if backbone == "tcn" and seen.count("tcn") == 1:
            raise RuntimeError("simulated pod preemption")
        return {"status": "ok", "backbone": backbone, "history": history,
                "quantile": quantile, "beats_peak": False,
                "results": {"learned_q": {"nodes": 3, "overload_pct": 0},
                            "peak": {"nodes": 2, "overload_pct": 0}}}

    monkeypatch.setattr(sweep, "train_eval", fake_train)
    with pytest.raises(RuntimeError, match="simulated pod"):
        sweep.main()
    assert not out.exists()
    partial = json.loads((tmp_path / "sweep.json.partial.json").read_text())
    assert list(partial["completed"]) == ["google/cnn/h3/q0.95"]

    sweep.main()
    result = json.loads(out.read_text())
    assert seen == ["cnn", "tcn", "tcn"]
    assert result["complete"] is True
    assert result["n_expected"] == result["n_runs"] == 2
    assert len(result["leaderboard"]) == 2

    data.write_bytes(b"changed trace")
    with pytest.raises(RuntimeError, match="checkpoint does not match"):
        sweep.main()
    assert seen == ["cnn", "tcn", "tcn"]
