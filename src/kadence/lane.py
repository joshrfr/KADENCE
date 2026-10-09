"""Integrated strict-neighbor KADENCE lane research harness.

The component models one resource lane and composes the independently tested
contracts in this repository:

* :mod:`core.topology` owns epoch-fenced membership transactions;
* :mod:`core.neighbor_gossip` owns fixed-membership gap convergence/safety;
* :mod:`core.pressure_fire` owns conserved backlog, local FIRE pulses, and
  receipt-gated service completion.

This is a deterministic evaluation harness, not a production scheduler.  It
uses an in-process token walk to emulate access to one shared resource and an
immediate reference actuator.  The token order is a neighbor-to-neighbor ring
walk; it is not a global optimization or all-job phase calculation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from kadence.neighbor_gossip import (
    TWO_PI,
    BubbleSpec,
    GapPlan,
    InfeasibleAdmission,
    NeighborSnapshot,
    RingController,
    admit_ring,
    forward_gap,
    phases_from_gaps,
)
from kadence.pressure_fire import (
    Disposition,
    FencedActuator,
    PressureBubble,
    PressureConfig,
    PressurePulse,
)
from kadence.topology import NeighborLease, TopologyAuthority


_TOL = 1e-9


class LaneRuntimeError(RuntimeError):
    """Base error for an invalid integrated lane transition."""


class UnsafeTopologyChange(LaneRuntimeError):
    """A locally proposed topology would violate admitted bubble geometry."""


class UndrainedWork(LaneRuntimeError):
    """A graceful leave attempted to discard outstanding admitted work."""


@dataclass(frozen=True)
class Arrival:
    job_id: str
    event_id: str
    work: float


@dataclass(frozen=True)
class LaneStep:
    started_at: float
    completed_at: float
    fired_jobs: tuple[str, ...]
    pulses_sent: int
    continuous_neighbor_messages: int
    token_hops: int
    requested_capacity: float
    acknowledged_service: float
    idle_capacity: float
    total_backlog: float
    conservation_residual: float
    max_gap_error: float
    minimum_safety_margin: float


@dataclass(frozen=True)
class ChurnResult:
    transaction_id: str
    operation: str
    epoch_before: int
    epoch_after: int
    order_before: tuple[str, ...]
    order_after: tuple[str, ...]
    stranded_work: float


@dataclass
class _RetiredJob:
    bubble_state: str
    actuator_state: str
    spec: BubbleSpec
    failed: bool


class Lane:
    """One admitted resource lane with strict-neighbor control and churn."""

    def __init__(
        self,
        bubbles: Sequence[BubbleSpec],
        *,
        pressure_config: PressureConfig,
        guard: float = 0.0,
        phases: Sequence[float] | None = None,
        common_frequency: float = 0.0,
        lease_duration: float = 30.0,
        start_time: float = 0.0,
    ) -> None:
        if not math.isfinite(start_time) or start_time < 0.0:
            raise ValueError("start_time must be finite and non-negative")
        if not math.isfinite(guard) or guard < 0.0:
            raise ValueError("guard must be finite and non-negative")
        if not math.isclose(
            pressure_config.circumference, TWO_PI, rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError("integrated reference lane currently requires 2*pi")

        self.guard = guard
        self.pressure_config = pressure_config
        self.common_frequency = common_frequency
        self.now = float(start_time)
        self._specs = {bubble.jid: bubble for bubble in bubbles}
        if len(self._specs) != len(bubbles):
            raise ValueError("job ids must be unique")

        self.topology = TopologyAuthority(
            tuple(bubble.jid for bubble in bubbles),
            lease_duration=lease_duration,
            start_time=start_time,
        )
        self.plan = admit_ring(bubbles, guard=guard)
        if phases is None:
            phases = phases_from_gaps(self.plan.desired)
        self.controller = RingController(
            bubbles,
            phases,
            self.plan,
            common_frequency=common_frequency,
            epoch=self.topology.epoch,
        )
        self.bubbles: dict[str, PressureBubble] = {}
        self.actuators: dict[str, FencedActuator] = {}
        for bubble, phase in zip(bubbles, self.controller.phases):
            lease = self.topology.neighbor_view(bubble.jid)
            self.bubbles[bubble.jid] = PressureBubble(
                bubble.jid,
                width=bubble.width,
                phase=phase,
                predecessor=lease.predecessor,
                successor=lease.successor,
                config=pressure_config,
                epoch=lease.epoch,
                now=start_time,
            )
            self.actuators[bubble.jid] = FencedActuator(
                bubble.jid, epoch=lease.epoch,
            )
        self._retired: dict[str, _RetiredJob] = {}
        self._token_cursor = 0

    @property
    def order(self) -> tuple[str, ...]:
        """Operator/evaluation view; jobs receive only ``NeighborLease``."""

        return self.topology.order

    @property
    def stranded_work(self) -> float:
        return math.fsum(
            PressureBubble.load_state(record.bubble_state).backlog
            for record in self._retired.values()
            if record.failed
        )

    @property
    def total_admitted_work(self) -> float:
        active = math.fsum(bubble.admitted_work for bubble in self.bubbles.values())
        retired = math.fsum(
            PressureBubble.load_state(record.bubble_state).admitted_work
            for record in self._retired.values()
        )
        return active + retired

    @property
    def total_acknowledged_work(self) -> float:
        active = math.fsum(
            bubble.acknowledged_work for bubble in self.bubbles.values()
        )
        retired = math.fsum(
            PressureBubble.load_state(record.bubble_state).acknowledged_work
            for record in self._retired.values()
        )
        return active + retired

    @property
    def total_backlog(self) -> float:
        active = math.fsum(bubble.backlog for bubble in self.bubbles.values())
        retired = math.fsum(
            PressureBubble.load_state(record.bubble_state).backlog
            for record in self._retired.values()
        )
        return active + retired

    @property
    def conservation_residual(self) -> float:
        return (
            self.total_admitted_work
            - self.total_acknowledged_work
            - self.total_backlog
        )

    def neighbor_lease(self, job_id: str) -> NeighborLease:
        return self.topology.neighbor_view(job_id)

    def heartbeat(self, job_id: str, *, now: float) -> NeighborLease:
        self._advance_time(now)
        return self.topology.renew_member(
            job_id, expected_epoch=self.topology.epoch, now=now,
        )

    def admit_arrival(self, arrival: Arrival, *, now: float | None = None) -> Disposition:
        when = self.now if now is None else now
        self._advance_time(when)
        try:
            bubble = self.bubbles[arrival.job_id]
        except KeyError as exc:
            raise LaneRuntimeError(
                f"arrival targets inactive job {arrival.job_id!r}"
            ) from exc
        return bubble.admit_arrival(arrival.event_id, arrival.work, now=when)

    def step(
        self,
        *,
        dt: float,
        service_capacity: float,
        service_quantum: float,
        arrivals: Iterable[Arrival] = (),
    ) -> LaneStep:
        """Advance control, deliver local pulses, and acknowledge token service."""

        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if not math.isfinite(service_capacity) or service_capacity < 0.0:
            raise ValueError("service_capacity must be finite and non-negative")
        if not math.isfinite(service_quantum) or service_quantum <= 0.0:
            raise ValueError("service_quantum must be finite and positive")
        started = self.now
        for arrival in arrivals:
            self.admit_arrival(arrival, now=started)

        completed = started + dt
        self._sync_controller_from_bubbles()
        control_report = self.controller.step(dt=dt)
        self._sync_bubbles_from_controller()

        pulses: list[PressurePulse] = []
        fired: list[str] = []
        for job_id in self.order:
            emitted = self.bubbles[job_id].maybe_fire(now=completed)
            if emitted:
                fired.append(job_id)
                pulses.extend(emitted)

        for pulse in sorted(
            pulses, key=lambda item: (item.recipient, item.sender, item.sequence),
        ):
            self.topology.accept_neighbor_message(
                receiver=pulse.recipient,
                sender=pulse.sender,
                epoch=pulse.epoch,
                sequence=pulse.sequence,
            )
            self._deliver_pulse(pulse)

        remaining = service_capacity
        acknowledged = 0.0
        order = self.order
        fired_set = set(fired)
        start = self._token_cursor % len(order)
        token_order = tuple(
            order[(start + offset) % len(order)] for offset in range(len(order))
        )
        for job_id in token_order:
            if remaining <= _TOL or job_id not in fired_set:
                continue
            bubble = self.bubbles[job_id]
            grant = min(service_quantum, remaining, bubble.available_work)
            if grant <= _TOL:
                continue
            action = bubble.issue_service(grant, now=completed)
            result = self.actuators[job_id].apply(
                action, completed_work=grant, completed_at=completed,
            )
            if result.receipt is None:
                raise LaneRuntimeError(
                    f"reference actuator rejected fresh action {action.action_id}"
                )
            disposition = bubble.acknowledge_service(
                result.receipt, now=completed,
            )
            if disposition is not Disposition.APPLIED:
                raise LaneRuntimeError("fresh service receipt was not applied")
            remaining -= grant
            acknowledged += grant
        self._token_cursor = (start + 1) % len(order)
        self.now = completed

        gaps = self._current_gaps()
        safety_margin = min(
            gap - minimum for gap, minimum in zip(gaps, self.plan.minimum)
        )
        if safety_margin < -1e-8:
            raise LaneRuntimeError("integrated step violated an admitted hard gap")
        if abs(self.conservation_residual) > 1e-7:
            raise LaneRuntimeError("integrated step violated work conservation")
        return LaneStep(
            started_at=started,
            completed_at=completed,
            fired_jobs=tuple(fired),
            pulses_sent=len(pulses),
            continuous_neighbor_messages=control_report.directed_messages,
            token_hops=len(order),
            requested_capacity=service_capacity,
            acknowledged_service=acknowledged,
            idle_capacity=max(0.0, remaining),
            total_backlog=self.total_backlog,
            conservation_residual=self.conservation_residual,
            max_gap_error=max(
                abs(gap - desired)
                for gap, desired in zip(gaps, self.plan.desired)
            ),
            minimum_safety_margin=safety_margin,
        )

    def join(
        self,
        transaction_id: str,
        *,
        job_id: str,
        left_job: str,
        right_job: str,
        width: float | None = None,
        now: float,
        timeout: float = 5.0,
    ) -> ChurnResult:
        """Synchronously execute a locally feasible edge-split transaction."""

        self._advance_time(now)
        before = self.order
        old_epoch = self.topology.epoch
        restoring = self._retired.get(job_id)
        if restoring is None:
            if width is None:
                raise ValueError("a new job requires a width")
            candidate_spec = BubbleSpec(job_id, width)
        else:
            if width is not None and not math.isclose(
                width, restoring.spec.width, rel_tol=0.0, abs_tol=1e-12,
            ):
                raise ValueError("restored job width must match its durable state")
            candidate_spec = restoring.spec

        try:
            left_index = before.index(left_job)
        except ValueError as exc:
            raise LaneRuntimeError("join edge contains an unknown member") from exc
        if before[(left_index + 1) % len(before)] != right_job:
            raise UnsafeTopologyChange("join requires a live adjacent edge")
        after = before[:left_index + 1] + (job_id,) + before[left_index + 1:]
        proposed_specs = tuple(
            candidate_spec if member == job_id else self._specs[member]
            for member in after
        )
        proposed_plan = admit_ring(proposed_specs, guard=self.guard)

        live_gap = forward_gap(
            self.bubbles[left_job].phase,
            self.bubbles[right_job].phase,
            proposed_plan.circumference,
        )
        new_index = after.index(job_id)
        left_edge = after.index(left_job)
        required = (
            proposed_plan.minimum[left_edge]
            + proposed_plan.minimum[new_index]
        )
        if live_gap + _TOL < required:
            raise UnsafeTopologyChange(
                f"join edge has gap {live_gap:.6g}, requires {required:.6g}"
            )
        new_phase = (
            self.bubbles[left_job].phase + proposed_plan.minimum[left_edge]
        ) % proposed_plan.circumference

        receipt = self.topology.prepare_join(
            transaction_id,
            new_jid=job_id,
            left_jid=left_job,
            right_jid=right_job,
            expected_epoch=old_epoch,
            now=now,
            timeout=timeout,
        )
        self._acknowledge_all(receipt.required, transaction_id, now)
        self.topology.commit(transaction_id, expected_epoch=old_epoch, now=now)

        self._specs[job_id] = candidate_spec
        if restoring is None:
            self.bubbles[job_id] = PressureBubble(
                job_id,
                width=candidate_spec.width,
                phase=new_phase,
                predecessor=left_job,
                successor=right_job,
                config=self.pressure_config,
                epoch=self.topology.epoch,
                now=now,
            )
            self.actuators[job_id] = FencedActuator(
                job_id, epoch=self.topology.epoch,
            )
        else:
            bubble = PressureBubble.load_state(restoring.bubble_state)
            actuator = FencedActuator.load_state(restoring.actuator_state)
            bubble.phase = new_phase
            self.bubbles[job_id] = bubble
            self.actuators[job_id] = actuator
            del self._retired[job_id]
        self._install_epoch_and_rebuild(proposed_plan, now=now)
        return self._churn_result(
            transaction_id, "restore" if restoring else "join",
            old_epoch, before,
        )

    def leave(
        self,
        transaction_id: str,
        *,
        job_id: str,
        now: float,
        failed: bool = False,
        timeout: float = 5.0,
    ) -> ChurnResult:
        """Remove a drained job or quarantine a failed job's durable state."""

        self._advance_time(now)
        before = self.order
        old_epoch = self.topology.epoch
        bubble = self.bubbles[job_id]
        if not failed and (
            bubble.backlog > _TOL or bubble.pending_work > _TOL
        ):
            raise UndrainedWork(
                f"job {job_id!r} retains backlog or pending service"
            )
        after = tuple(member for member in before if member != job_id)
        proposed_specs = tuple(self._specs[member] for member in after)
        proposed_plan = admit_ring(proposed_specs, guard=self.guard)

        if failed:
            receipt = self.topology.prepare_failed_leave(
                transaction_id,
                jid=job_id,
                expected_epoch=old_epoch,
                now=now,
                timeout=timeout,
            )
        else:
            receipt = self.topology.prepare_leave(
                transaction_id,
                jid=job_id,
                expected_epoch=old_epoch,
                now=now,
                timeout=timeout,
            )
        self._acknowledge_all(receipt.required, transaction_id, now)
        self.topology.commit(transaction_id, expected_epoch=old_epoch, now=now)

        self._retired[job_id] = _RetiredJob(
            bubble_state=bubble.dump_state(),
            actuator_state=self.actuators[job_id].dump_state(),
            spec=self._specs[job_id],
            failed=failed,
        )
        del self.bubbles[job_id]
        del self.actuators[job_id]
        del self._specs[job_id]
        self._install_epoch_and_rebuild(proposed_plan, now=now)
        return self._churn_result(
            transaction_id, "failed_leave" if failed else "leave",
            old_epoch, before,
        )

    def swap(
        self,
        transaction_id: str,
        *,
        left_job: str,
        right_job: str,
        now: float,
        timeout: float = 5.0,
    ) -> ChurnResult:
        """Exchange adjacent jobs after a local geometry safety preflight."""

        self._advance_time(now)
        before = self.order
        old_epoch = self.topology.epoch
        try:
            left_index = before.index(left_job)
        except ValueError as exc:
            raise LaneRuntimeError("swap contains an unknown member") from exc
        right_index = (left_index + 1) % len(before)
        if before[right_index] != right_job:
            raise UnsafeTopologyChange("swap requires a clockwise adjacent pair")
        if right_index == 0:
            raise UnsafeTopologyChange("reference runtime does not swap across seam")

        after_list = list(before)
        after_list[left_index], after_list[right_index] = (
            after_list[right_index], after_list[left_index]
        )
        after = tuple(after_list)
        proposed_specs = tuple(self._specs[member] for member in after)
        proposed_plan = admit_ring(proposed_specs, guard=self.guard)
        candidate_phases = {
            member: self.bubbles[member].phase for member in before
        }
        candidate_phases[left_job], candidate_phases[right_job] = (
            candidate_phases[right_job], candidate_phases[left_job]
        )
        self._validate_candidate(after, candidate_phases, proposed_plan)

        receipt = self.topology.prepare_swap(
            transaction_id,
            left_jid=left_job,
            right_jid=right_job,
            expected_epoch=old_epoch,
            now=now,
            timeout=timeout,
        )
        self._acknowledge_all(receipt.required, transaction_id, now)
        self.topology.commit(transaction_id, expected_epoch=old_epoch, now=now)
        self.bubbles[left_job].phase, self.bubbles[right_job].phase = (
            self.bubbles[right_job].phase, self.bubbles[left_job].phase
        )
        self._install_epoch_and_rebuild(proposed_plan, now=now)
        return self._churn_result(transaction_id, "swap", old_epoch, before)

    def _advance_time(self, now: float) -> None:
        if not math.isfinite(now) or now < self.now - _TOL:
            raise ValueError("lane logical time cannot move backwards")
        self.now = max(self.now, float(now))

    def _ordered_specs(self) -> tuple[BubbleSpec, ...]:
        return tuple(self._specs[job_id] for job_id in self.order)

    def _snapshots(self) -> dict[str, NeighborSnapshot]:
        return {
            job_id: NeighborSnapshot(
                jid=job_id,
                phase=self.bubbles[job_id].phase,
                width=self.bubbles[job_id].width,
                epoch=self.topology.epoch,
                sequence=self.controller.sequence,
            )
            for job_id in self.order
        }

    def _deliver_pulse(self, pulse: PressurePulse) -> None:
        recipient = self.bubbles[pulse.recipient]
        snapshots = self._snapshots()
        index = self.order.index(pulse.recipient)
        left_id = self.order[index - 1]
        right_id = self.order[(index + 1) % len(self.order)]
        recipient.receive_pulse(
            pulse,
            left=snapshots[left_id],
            right=snapshots[right_id],
            left_minimum=self.plan.minimum[index - 1],
            right_minimum=self.plan.minimum[index],
        )

    def _sync_controller_from_bubbles(self) -> None:
        self.controller.phases = [
            self.bubbles[job_id].phase for job_id in self.order
        ]

    def _sync_bubbles_from_controller(self) -> None:
        for job_id, phase in zip(self.order, self.controller.phases):
            self.bubbles[job_id].phase = phase

    def _current_gaps(self) -> tuple[float, ...]:
        phases = tuple(self.bubbles[job_id].phase for job_id in self.order)
        return tuple(
            forward_gap(phases[i], phases[(i + 1) % len(phases)])
            for i in range(len(phases))
        )

    def _acknowledge_all(
        self, required: Iterable[str], transaction_id: str, now: float,
    ) -> None:
        for participant in required:
            # Reading the proposal proves this participant receives only its
            # own old/new links, not either global order.
            self.topology.proposal(transaction_id, participant)
            self.topology.acknowledge(
                transaction_id,
                participant,
                expected_epoch=self.topology.epoch,
                now=now,
            )

    def _install_epoch_and_rebuild(
        self, proposed_plan: GapPlan, *, now: float,
    ) -> None:
        for job_id in self.order:
            lease = self.topology.neighbor_view(job_id)
            bubble = self.bubbles[job_id]
            if bubble.epoch < lease.epoch:
                bubble.install_topology(
                    epoch=lease.epoch,
                    predecessor=lease.predecessor,
                    successor=lease.successor,
                    now=now,
                )
            elif (
                bubble.epoch != lease.epoch
                or bubble.predecessor != lease.predecessor
                or bubble.successor != lease.successor
            ):
                raise LaneRuntimeError("bubble topology diverged from lease")
            actuator = self.actuators[job_id]
            if actuator.epoch < lease.epoch:
                actuator.advance_epoch(lease.epoch)
        self.plan = proposed_plan
        self.controller = RingController(
            self._ordered_specs(),
            tuple(self.bubbles[job_id].phase for job_id in self.order),
            self.plan,
            common_frequency=self.common_frequency,
            epoch=self.topology.epoch,
        )
        self._token_cursor %= len(self.order)

    def _validate_candidate(
        self,
        order: Sequence[str],
        phases: Mapping[str, float],
        plan: GapPlan,
    ) -> None:
        try:
            RingController(
                tuple(self._specs[job_id] for job_id in order),
                tuple(phases[job_id] for job_id in order),
                plan,
                common_frequency=self.common_frequency,
                epoch=self.topology.epoch + 1,
            )
        except (ValueError, InfeasibleAdmission, RuntimeError) as exc:
            raise UnsafeTopologyChange(str(exc)) from exc

    def _churn_result(
        self,
        transaction_id: str,
        operation: str,
        old_epoch: int,
        before: tuple[str, ...],
    ) -> ChurnResult:
        if abs(self.conservation_residual) > 1e-7:
            raise LaneRuntimeError("topology change violated work conservation")
        return ChurnResult(
            transaction_id=transaction_id,
            operation=operation,
            epoch_before=old_epoch,
            epoch_after=self.topology.epoch,
            order_before=before,
            order_after=self.order,
            stranded_work=self.stranded_work,
        )
