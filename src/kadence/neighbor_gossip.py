"""Strict-neighbor phase-spacing kernel for KADENCE.

This module is deliberately smaller than a scheduler.  It implements the
fixed-membership control primitive that a scheduler can build on:

* jobs occupy an explicitly ordered cyclic lane;
* every update reads only the job's predecessor and successor snapshots;
* adjacent centre gaps converge toward an admitted gap plan; and
* a local displacement limiter preserves every admitted minimum gap.

The lane directory and admission controller are outside the feedback loop.
They may establish membership and a feasible gap plan, but they do not compute
phase updates or broadcast a global order parameter.

For edge i = (i, i+1), let

    e_i = gap(theta_i, theta_(i+1)) - desired_gap_i.

The unconstrained correction at job i is

    u_i = gain_i * e_i - gain_(i-1) * e_(i-1),

which is negative gradient flow for

    V = 1/2 * sum_i gain_i * e_i**2.

With fixed feasible topology and a sufficiently small integration step, V is
non-increasing and the desired gaps are the unique equilibrium modulo common
rotation.  ``RingController`` adds a strictly local limiter so simultaneous
updates cannot consume more than the available safety slack on an edge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


TWO_PI = 2.0 * math.pi
_TOL = 1e-12


class InfeasibleAdmission(ValueError):
    """The requested bubble widths and guards do not fit in one lane."""


class InvalidTopology(ValueError):
    """The supplied phases do not describe one ordered winding of the lane."""


class SafetyViolation(RuntimeError):
    """An update would violate an admitted hard minimum gap."""


@dataclass(frozen=True)
class BubbleSpec:
    """Admission-time description of one job bubble.

    ``width`` is the hard reserved arc on this resource lane, not observed CPU
    utilization mapped to an angle.  A production admission controller derives
    it from a demand bound or an explicit service reservation.
    """

    jid: str
    width: float


@dataclass(frozen=True)
class NeighborSnapshot:
    """The complete state one job exposes to either immediate neighbor."""

    jid: str
    phase: float
    width: float
    epoch: int
    sequence: int


@dataclass(frozen=True)
class GapPlan:
    """Feasible per-edge hard minima and desired centre gaps.

    Entry i describes the clockwise edge from ``order[i]`` to
    ``order[(i + 1) % n]``.
    """

    order: tuple[str, ...]
    minimum: tuple[float, ...]
    desired: tuple[float, ...]
    circumference: float


@dataclass(frozen=True)
class StepReport:
    energy_before: float
    energy_after: float
    max_gap_error: float
    directed_messages: int
    limited_jobs: int


def _require_finite_positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def forward_gap(left_phase: float, right_phase: float,
                circumference: float = TWO_PI) -> float:
    """Clockwise distance from ``left_phase`` to ``right_phase``."""

    _require_finite_positive(circumference, "circumference")
    return (right_phase - left_phase) % circumference


def admit_ring(
    bubbles: Sequence[BubbleSpec],
    *,
    guard: float = 0.0,
    circumference: float = TWO_PI,
    slack_weights: Sequence[float] | None = None,
) -> GapPlan:
    """Create a conservative feasible gap plan for one cyclic lane.

    The hard minimum on an adjacent edge is half of each endpoint's width plus
    ``guard``.  By default, remaining circumference is placed on the final
    seam edge, so the real jobs form a tightly packed line on the cycle.
    ``slack_weights`` can distribute that idle arc differently.
    This function is admission-plane logic; it is intentionally not called by
    the local feedback update.
    """

    _require_finite_positive(circumference, "circumference")
    if not math.isfinite(guard) or guard < 0.0:
        raise ValueError("guard must be finite and non-negative")
    if len(bubbles) < 3:
        raise ValueError("a strict two-neighbor ring requires at least 3 jobs")

    order = tuple(b.jid for b in bubbles)
    if len(set(order)) != len(order):
        raise ValueError("job ids must be unique")
    for bubble in bubbles:
        _require_finite_positive(bubble.width, f"width for {bubble.jid}")

    minimum = tuple(
        0.5 * (bubble.width + bubbles[(i + 1) % len(bubbles)].width) + guard
        for i, bubble in enumerate(bubbles)
    )
    required = math.fsum(minimum)
    if required > circumference + _TOL:
        raise InfeasibleAdmission(
            f"lane requires {required:.12g}, capacity is {circumference:.12g}"
        )

    if slack_weights is None:
        weights = (0.0,) * (len(bubbles) - 1) + (1.0,)
    else:
        if len(slack_weights) != len(bubbles):
            raise ValueError("slack_weights must have one entry per edge")
        weights = tuple(float(weight) for weight in slack_weights)
        if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
            raise ValueError("slack weights must be finite and non-negative")
        if math.fsum(weights) <= 0.0:
            raise ValueError("at least one slack weight must be positive")

    slack = max(0.0, circumference - required)
    weight_sum = math.fsum(weights)
    desired_list = [
        edge_minimum + slack * weight / weight_sum
        for edge_minimum, weight in zip(minimum, weights)
    ]
    # Make closure exact despite floating-point summation.  Put the tiny
    # correction on an edge that was assigned the largest share of slack.
    closure_edge = max(range(len(weights)), key=weights.__getitem__)
    desired_list[closure_edge] += circumference - math.fsum(desired_list)
    if desired_list[closure_edge] + _TOL < minimum[closure_edge]:
        raise InfeasibleAdmission("floating-point closure consumed hard slack")

    return GapPlan(
        order=order,
        minimum=minimum,
        desired=tuple(desired_list),
        circumference=circumference,
    )


def phases_from_gaps(
    gaps: Sequence[float], *, origin: float = 0.0,
    circumference: float = TWO_PI,
) -> tuple[float, ...]:
    """Construct ordered phases from one complete winding of positive gaps."""

    _require_finite_positive(circumference, "circumference")
    if len(gaps) < 3:
        raise ValueError("a strict two-neighbor ring requires at least 3 gaps")
    if any(not math.isfinite(gap) or gap <= 0.0 for gap in gaps):
        raise ValueError("all gaps must be finite and positive")
    if not math.isclose(math.fsum(gaps), circumference, rel_tol=0.0,
                        abs_tol=1e-9):
        raise InvalidTopology("gaps must sum to exactly one circumference")

    phases = [origin % circumference]
    for gap in gaps[:-1]:
        phases.append((phases[-1] + gap) % circumference)
    return tuple(phases)


def local_correction(
    left: NeighborSnapshot,
    current: NeighborSnapshot,
    right: NeighborSnapshot,
    *,
    left_target: float,
    right_target: float,
    left_gain: float = 1.0,
    right_gain: float = 1.0,
    circumference: float = TWO_PI,
) -> float:
    """Return one job's phase correction using exactly two neighbor frames."""

    left_error = forward_gap(left.phase, current.phase, circumference) - left_target
    right_error = forward_gap(current.phase, right.phase, circumference) - right_target
    return right_gain * right_error - left_gain * left_error


def limit_local_displacement(
    requested: float,
    left: NeighborSnapshot,
    current: NeighborSnapshot,
    right: NeighborSnapshot,
    *,
    left_minimum: float,
    right_minimum: float,
    safety_fraction: float = 0.45,
    circumference: float = TWO_PI,
) -> float:
    """Clip one motion using only the slack on the two adjacent edges."""

    if not math.isfinite(requested):
        raise ValueError("requested displacement must be finite")
    if not math.isfinite(safety_fraction) or not 0.0 < safety_fraction <= 0.5:
        raise ValueError("safety_fraction must be in (0, 0.5]")
    if left_minimum < 0.0 or right_minimum < 0.0:
        raise ValueError("minimum gaps must be non-negative")

    if requested >= 0.0:
        right_gap = forward_gap(current.phase, right.phase, circumference)
        available = max(0.0, right_gap - right_minimum)
        return min(requested, safety_fraction * available)
    left_gap = forward_gap(left.phase, current.phase, circumference)
    available = max(0.0, left_gap - left_minimum)
    return max(requested, -safety_fraction * available)


class RingController:
    """Synchronous harness for the strict-neighbor distributed update.

    The harness snapshots and delivers messages for reproducible experiments.
    It does not expose global state to ``local_correction``.  A deployment can
    run the same pure update independently in each job agent.
    """

    def __init__(
        self,
        bubbles: Sequence[BubbleSpec],
        phases: Sequence[float],
        plan: GapPlan,
        *,
        gains: Sequence[float] | None = None,
        common_frequency: float = 0.0,
        epoch: int = 0,
    ) -> None:
        if tuple(b.jid for b in bubbles) != plan.order:
            raise ValueError("bubble order must match the admitted gap plan")
        if len(bubbles) < 3:
            raise ValueError("a strict two-neighbor ring requires at least 3 jobs")
        for bubble in bubbles:
            _require_finite_positive(bubble.width, f"width for {bubble.jid}")
        if len(phases) != len(bubbles):
            raise ValueError("phases must have one entry per bubble")
        if any(not math.isfinite(phase) for phase in phases):
            raise ValueError("phases must be finite")
        _require_finite_positive(plan.circumference, "plan circumference")
        if len(plan.minimum) != len(bubbles) or len(plan.desired) != len(bubbles):
            raise ValueError("gap plan must have one minimum and target per edge")
        if any(not math.isfinite(gap) or gap <= 0.0 for gap in plan.minimum):
            raise ValueError("minimum gaps must be finite and positive")
        if any(
            not math.isfinite(target) or target + _TOL < minimum
            for target, minimum in zip(plan.desired, plan.minimum)
        ):
            raise ValueError("desired gaps must be finite and meet their minima")
        if not math.isclose(math.fsum(plan.desired), plan.circumference,
                            rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("desired gaps must close exactly one circumference")
        if not math.isfinite(common_frequency):
            raise ValueError("common_frequency must be finite")
        if epoch < 0:
            raise ValueError("epoch must be non-negative")

        if gains is None:
            gains = (1.0,) * len(bubbles)
        if len(gains) != len(bubbles):
            raise ValueError("gains must have one entry per edge")
        if any(not math.isfinite(gain) or gain <= 0.0 for gain in gains):
            raise ValueError("gains must be finite and positive")

        self.bubbles = tuple(bubbles)
        self.plan = plan
        self.gains = tuple(float(gain) for gain in gains)
        self.common_frequency = common_frequency
        self.epoch = epoch
        self.sequence = 0
        self.message_count = 0
        self.phases = [phase % plan.circumference for phase in phases]
        self._validate_topology(require_minimum=True)

    def snapshots(self) -> tuple[NeighborSnapshot, ...]:
        return tuple(
            NeighborSnapshot(
                jid=bubble.jid,
                phase=phase,
                width=bubble.width,
                epoch=self.epoch,
                sequence=self.sequence,
            )
            for bubble, phase in zip(self.bubbles, self.phases)
        )

    def gaps(self) -> tuple[float, ...]:
        n = len(self.phases)
        return tuple(
            forward_gap(
                self.phases[i], self.phases[(i + 1) % n],
                self.plan.circumference,
            )
            for i in range(n)
        )

    def gap_errors(self) -> tuple[float, ...]:
        return tuple(
            gap - target for gap, target in zip(self.gaps(), self.plan.desired)
        )

    def energy(self) -> float:
        return 0.5 * math.fsum(
            gain * error * error
            for gain, error in zip(self.gains, self.gap_errors())
        )

    def max_gap_error(self) -> float:
        return max(abs(error) for error in self.gap_errors())

    def step(self, *, dt: float = 0.1,
             safety_fraction: float = 0.45) -> StepReport:
        """Advance every job once using only its two delivered snapshots.

        ``safety_fraction`` may not exceed 1/2: then the two endpoints of an
        edge cannot jointly consume more than that edge's available slack in a
        simultaneous update.  Common rotation is applied separately, because
        it changes no gap and should not be clipped by the safety limiter.
        """

        _require_finite_positive(dt, "dt")
        if not math.isfinite(safety_fraction) or not 0.0 < safety_fraction <= 0.5:
            raise ValueError("safety_fraction must be in (0, 0.5]")

        frames = self.snapshots()
        n = len(frames)
        before = self.energy()
        displacements: list[float] = []
        limited_jobs = 0

        for i, current in enumerate(frames):
            left_i = (i - 1) % n
            right_i = (i + 1) % n
            correction = local_correction(
                frames[left_i], current, frames[right_i],
                left_target=self.plan.desired[left_i],
                right_target=self.plan.desired[i],
                left_gain=self.gains[left_i],
                right_gain=self.gains[i],
                circumference=self.plan.circumference,
            )
            requested = dt * correction

            # Positive motion consumes right-edge slack; negative motion
            # consumes left-edge slack.  Each endpoint may consume at most half
            # so concurrent neighbors cannot cross a hard reservation boundary.
            displacement = limit_local_displacement(
                requested,
                frames[left_i],
                current,
                frames[right_i],
                left_minimum=self.plan.minimum[left_i],
                right_minimum=self.plan.minimum[i],
                safety_fraction=safety_fraction,
                circumference=self.plan.circumference,
            )
            if not math.isclose(displacement, requested, rel_tol=0.0,
                                abs_tol=_TOL):
                limited_jobs += 1
            displacements.append(displacement)

        rotation = dt * self.common_frequency
        self.phases = [
            (phase + displacement + rotation) % self.plan.circumference
            for phase, displacement in zip(self.phases, displacements)
        ]
        self.sequence += 1
        directed_messages = 2 * n
        self.message_count += directed_messages
        self._validate_topology(require_minimum=True)

        return StepReport(
            energy_before=before,
            energy_after=self.energy(),
            max_gap_error=self.max_gap_error(),
            directed_messages=directed_messages,
            limited_jobs=limited_jobs,
        )

    def run(self, *, tolerance: float = 1e-7, max_steps: int = 20_000,
            dt: float = 0.1, safety_fraction: float = 0.45) -> int:
        """Run until all desired gap errors are within ``tolerance``."""

        _require_finite_positive(tolerance, "tolerance")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        for step_number in range(1, max_steps + 1):
            self.step(dt=dt, safety_fraction=safety_fraction)
            if self.max_gap_error() <= tolerance:
                return step_number
        raise RuntimeError(
            f"did not converge below {tolerance:g} in {max_steps} steps; "
            f"error={self.max_gap_error():.6g}"
        )

    def _validate_topology(self, *, require_minimum: bool) -> None:
        gaps = self.gaps()
        if any(gap <= _TOL for gap in gaps):
            raise InvalidTopology("phases must preserve a strict cyclic order")
        if not math.isclose(math.fsum(gaps), self.plan.circumference,
                            rel_tol=0.0, abs_tol=1e-9):
            raise InvalidTopology("phases must make exactly one lane winding")
        if require_minimum:
            for i, (gap, minimum) in enumerate(zip(gaps, self.plan.minimum)):
                if gap + 1e-10 < minimum:
                    raise SafetyViolation(
                        f"edge {i} gap {gap:.12g} is below minimum {minimum:.12g}"
                    )


def perturb_gaps(
    desired: Sequence[float], transfers: Iterable[tuple[int, int, float]],
) -> tuple[float, ...]:
    """Move slack between edges without changing total circumference.

    This is a deterministic experiment helper, not part of the controller.
    Each transfer ``(source, destination, amount)`` subtracts from one gap and
    adds to the other.
    """

    gaps = list(desired)
    for source, destination, amount in transfers:
        if not math.isfinite(amount) or amount < 0.0:
            raise ValueError("transfer amounts must be finite and non-negative")
        gaps[source] -= amount
        gaps[destination] += amount
    if any(gap <= 0.0 for gap in gaps):
        raise ValueError("transfers must preserve positive gaps")
    return tuple(gaps)
