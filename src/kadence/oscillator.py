"""Job-as-oscillator model and the phase-coupling update law.

Each job is a "bubble": a demand waveform over a repeating cycle. Its
state on each contended resource r is a phase theta_r in [0, 2pi). The
job's instantaneous demand for r is a pulse centred on its phase; the
width is the job's duty cycle (fraction of the cycle it actually needs
r). Natural frequency omega = required service rate = how often the
pulse must be served to meet the SLO.

The coupling law (per resource, over jobs that share it):

    dtheta_i/dt = omega_i + (K/N) * sum_j  s_ij * sin(theta_j - theta_i)

  s_ij = -1 (REPEL / desync) when i and j COMPETE for r
         +1 (ATTRACT / sync)  when i and j COMPLEMENT on r

REPEL is the DESYNC primitive (Degesys 2007): competitors spread their
demand peaks evenly around the cycle, like TDMA slots on a shared radio
channel. ATTRACT interleaves complementary jobs so one fills the other's
valley. Sign is decided by demand correlation, not by job identity.

Nothing here is linear-time-slice scheduling: phases evolve continuously
and settle into a self-organised arrangement.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


TWO_PI = 2 * math.pi


@dataclass
class Job:
    jid: str
    # per-resource duty cycle (fraction of the period the job needs it, 0..1)
    duty: dict[str, float]
    # per-resource peak demand height (share of the resource at the peak)
    height: dict[str, float]
    omega: float = 1.0                      # required service rate (cycles/period)
    phase: dict[str, float] = field(default_factory=dict)   # current theta_r

    def demand(self, r: str, t: float, period: float) -> float:
        """Instantaneous demand for resource r at wall-time t.

        A raised-cosine pulse of width = duty*period centred on the
        job's phase. Zero outside the pulse. This is the 'bubble' shape.
        """
        if r not in self.duty:
            return 0.0
        theta = self.phase.get(r, 0.0)
        centre = (theta / TWO_PI) * period
        half = 0.5 * self.duty[r] * period
        if half <= 0:
            return 0.0
        # distance from pulse centre on the circular period
        d = abs((t - centre + period / 2) % period - period / 2)
        if d > half:
            return 0.0
        return self.height[r] * 0.5 * (1 + math.cos(math.pi * d / half))


def _load(job: Job, r: str) -> float:
    return job.height.get(r, 0.0) * job.duty.get(r, 0.0)


def dominant_resource(job: Job) -> str:
    return max(job.duty, key=lambda r: _load(job, r))


def _sign(job_i: Job, job_j: Job, r: str) -> float:
    """Coupling sign on resource r: repel competitors, attract complements.

    On resource r, two jobs COMPETE when r is the dominant (max-demand)
    resource for BOTH — they will pile up unless spread apart, so repel
    (-1, the DESYNC primitive). Otherwise their demands on r are
    complementary (one leads with r, the other with a different
    resource), so interleaving helps — attract (+1). Decided by demand
    profile, not job identity."""
    if _load(job_i, r) <= 0 or _load(job_j, r) <= 0:
        return 0.0
    both_dominant = dominant_resource(job_i) == r and dominant_resource(job_j) == r
    return -1.0 if both_dominant else +1.0


def couple_step(jobs: list[Job], resources: list[str],
                K: float = 0.6, dt: float = 0.05) -> float:
    """One phase-update step for all jobs on all resources.

    Returns the mean absolute phase change (a convergence signal:
    it decays to ~0 as the arrangement self-organises)."""
    updates: dict[tuple[str, str], float] = {}
    n = len(jobs)
    for r in resources:
        for i in jobs:
            if r not in i.duty:
                continue
            ti = i.phase.get(r, 0.0)
            acc = 0.0
            for j in jobs:
                if j is i or r not in j.duty:
                    continue
                s = _sign(i, j, r)
                acc += s * math.sin(j.phase.get(r, 0.0) - ti)
            # relative arrangement only: natural frequency rotates all
            # phases together and does not change contention, so omit it.
            updates[(i.jid, r)] = (K / max(1, n)) * acc
    total = 0.0
    for i in jobs:
        for r in resources:
            if (i.jid, r) in updates:
                delta = updates[(i.jid, r)] * dt
                i.phase[r] = (i.phase.get(r, 0.0) + delta) % TWO_PI
                total += abs(delta)
    return total / max(1, len(updates))


def desync_arrange(jobs: list[Job], resources: list[str],
                   alpha: float = 0.95, rounds: int = 400) -> None:
    """Decentralised even-spreading via the DESYNC primitive
    (Degesys et al., IPSN 2007), applied PER RESOURCE to only the jobs
    that compete for it (r is their dominant resource).

    Each competitor repeatedly jumps toward the midpoint of its two
    phase-neighbours on the competitor ring for r. This provably
    converges to even spacing — the optimal peak-minimising layout for a
    shared channel — using only local (neighbour) information, no global
    plan. Non-competitors keep their phase on r free to interleave into
    the valleys, which is what lets this beat resource-blind round-robin.
    """
    for r in resources:
        competitors = [j for j in jobs if dominant_resource(j) == r]
        m = len(competitors)
        if m < 2:
            continue
        for _ in range(rounds):
            ordered = sorted(competitors, key=lambda j: j.phase.get(r, 0.0))
            new = {}
            for k, j in enumerate(ordered):
                prev = ordered[(k - 1) % m].phase.get(r, 0.0)
                nxt = ordered[(k + 1) % m].phase.get(r, 0.0)
                # circular midpoint of the two neighbours
                if nxt <= prev:
                    nxt += TWO_PI
                mid = ((prev + nxt) / 2.0) % TWO_PI
                cur = j.phase.get(r, 0.0)
                # move a fraction alpha toward the midpoint (circular)
                diff = ((mid - cur + math.pi) % TWO_PI) - math.pi
                new[j.jid] = (cur + alpha * diff) % TWO_PI
            for j in competitors:
                j.phase[r] = new[j.jid]
    # complementary jobs: place each on the valley of r's competitor load
    for r in resources:
        competitors = [j for j in jobs if dominant_resource(j) == r]
        if len(competitors) < 2:
            continue
        others = [j for j in jobs if dominant_resource(j) != r and r in j.duty]
        # Greedy valley-fill: place each complementary job where the
        # CURRENT total load on r is lowest, updating as we go so we do
        # not stack them all on the same point (which would create a new
        # spike). This is the packing move RR cannot make.
        placed = list(competitors)
        for j in others:
            best_t, best_load = 0.0, float("inf")
            for k in range(80):
                t = 100.0 * k / 80
                load = sum(c.demand(r, t, 100.0) for c in placed)
                if load < best_load:
                    best_load, best_t = load, t
            j.phase[r] = (best_t / 100.0) * TWO_PI
            placed.append(j)


def peak_contention(jobs: list[Job], resources: list[str],
                    period: float, samples: int = 200) -> dict[str, float]:
    """Peak simultaneous demand on each resource over one period.

    This is the quantity phase-coupling minimises: the height of the
    worst instantaneous pile-up. Lower peak = fewer SLO violations at the
    same total work."""
    peaks = {r: 0.0 for r in resources}
    for k in range(samples):
        t = period * k / samples
        for r in resources:
            load = sum(j.demand(r, t, period) for j in jobs)
            if load > peaks[r]:
                peaks[r] = load
    return peaks
