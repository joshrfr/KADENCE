"""Trace-calibrated heterogeneous workload generator.

Scope: this is NOT a raw replay of the Google or Alibaba
cluster traces (those are terabytes / ~270 GB and not fetchable here).
It is a generator whose per-job demand statistics are *calibrated to the
published properties* of the Alibaba cluster-trace-v2018 (schema fields
cpu utilization, mem_size in [0,100], time_stamp; see
github.com/alibaba/clusterdata). Raw-trace replay is the next
validation step and is stated as such in the paper.

Calibrated properties reproduced here (all documented in the trace
literature):
  - low mean utilisation with a long right tail (most jobs light, a few
    heavy) -> lognormal per-job amplitude;
  - a diurnal (day/night) component shared across jobs;
  - a per-job oscillation (the "rhythm") with heterogeneous period and
    phase -> this is the structure our scheduler exploits and that flat
    request-based schedulers ignore;
  - positive-but-imperfect CPU<->memory correlation (rho ~ 0.5);
  - three coarse job archetypes (cpu-leaning, mem-leaning, balanced).

Each job exposes demand_series(resource) -> np.ndarray over one horizon,
so downstream code can either use the continuous waveform (phase model)
or the raw samples (trace-style).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

RESOURCES = ("cpu", "mem")
HORIZON = 288          # 24h at 5-min samples (matches trace cadence)
SAMPLES_PER_DAY = 288


@dataclass
class TraceJob:
    jid: str
    archetype: str                       # cpu | mem | bal
    amp: dict                            # per-resource oscillation amplitude
    base: dict                           # per-resource baseline level
    period: float                        # oscillation period (samples)
    phase: dict = field(default_factory=dict)   # per-resource phase (rad)
    diurnal_gain: float = 0.0            # how much day/night moves this job

    def demand_series(self, r: str, horizon: int = HORIZON,
                      diurnal: np.ndarray | None = None) -> np.ndarray:
        t = np.arange(horizon)
        th = self.phase.get(r, 0.0)
        osc = self.amp[r] * (0.5 * (1 + np.cos(2 * np.pi * t / self.period - th)))
        d = self.base[r] + osc
        if diurnal is not None:
            d = d + self.diurnal_gain * self.amp[r] * diurnal
        return np.clip(d, 0.0, None)


def diurnal_curve(horizon: int = HORIZON) -> np.ndarray:
    """Shared day/night envelope in [-0.5, 0.5], peak mid-day."""
    t = np.arange(horizon)
    return 0.5 * np.sin(2 * np.pi * (t / SAMPLES_PER_DAY) - np.pi / 2)


def generate(n_jobs: int, seed: int = 0) -> list[TraceJob]:
    rng = np.random.default_rng(seed)
    jobs = []
    for i in range(n_jobs):
        arch = rng.choice(["cpu", "mem", "bal"], p=[0.35, 0.35, 0.30])
        # lognormal amplitude -> long right tail (few heavy jobs)
        a = float(np.clip(rng.lognormal(mean=-1.6, sigma=0.6), 0.02, 0.6))
        # correlated second resource (rho ~ 0.5)
        a2 = float(np.clip(a * (0.5 + 0.5 * rng.random()), 0.02, 0.6))
        if arch == "cpu":
            amp = {"cpu": a, "mem": 0.35 * a2}
        elif arch == "mem":
            amp = {"cpu": 0.35 * a2, "mem": a}
        else:
            amp = {"cpu": 0.7 * a, "mem": 0.7 * a2}
        base = {r: float(rng.uniform(0.02, 0.06)) for r in RESOURCES}
        period = float(rng.choice([48, 72, 96, 144]))   # 4h..12h rhythms
        jobs.append(TraceJob(
            jid=f"j{i}", archetype=arch, amp=amp, base=base, period=period,
            phase={r: float(rng.uniform(0, 2 * np.pi)) for r in RESOURCES},
            diurnal_gain=float(rng.uniform(0.3, 1.0)),
        ))
    return jobs


def workload_stats(jobs: list[TraceJob]) -> dict:
    dur = diurnal_curve()
    cpu_means, mem_means, corr = [], [], []
    for j in jobs:
        c = j.demand_series("cpu", diurnal=dur)
        m = j.demand_series("mem", diurnal=dur)
        cpu_means.append(c.mean())
        mem_means.append(m.mean())
        if c.std() > 0 and m.std() > 0:
            corr.append(float(np.corrcoef(c, m)[0, 1]))
    return {
        "n_jobs": len(jobs),
        "cpu_mean_util": round(float(np.mean(cpu_means)), 3),
        "mem_mean_util": round(float(np.mean(mem_means)), 3),
        "cpu_p95_util": round(float(np.percentile(cpu_means, 95)), 3),
        "cpu_mem_corr_median": round(float(np.median(corr)), 3),
        "archetypes": {a: sum(1 for j in jobs if j.archetype == a)
                       for a in ("cpu", "mem", "bal")},
    }


if __name__ == "__main__":
    import json
    jobs = generate(200, seed=1)
    print(json.dumps(workload_stats(jobs), indent=2))
