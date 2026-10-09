"""Durable pressure-driven integrate-and-fire primitive for KADENCE.

This module extends the strict-neighbor spacing kernel without weakening its
contract.  It keeps four concerns separate:

* admitted arrivals are conserved in ``backlog``;
* backlog pressure charges an integrate-and-fire membrane;
* a fire emits one fenced pulse to each immediate neighbor; and
* only an idempotent actuator receipt removes completed work from backlog.

``PressureBubble`` is a deterministic state machine with a JSON checkpoint.
The caller must durably store the returned checkpoint before delivering a
pulse or service action.  ``FencedActuator`` models the other half of that
write-ahead contract: an exact retry returns the original receipt, while an
unknown action from an old epoch or sequence is rejected.

The pulse phase-response curve implemented here is a bounded *candidate*, not
a convergence proof for the hybrid system.  Its safety claim is narrower: for
fresh predecessor/successor snapshots, every requested phase displacement is
passed through ``neighbor_gossip.limit_local_displacement``.  Consequently it
inherits that function's local hard-gap bound (including its simultaneous
endpoint assumptions).  Delay, message loss, topology churn, and hybrid
convergence remain experiment and proof obligations.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping

from kadence.neighbor_gossip import (
    TWO_PI,
    NeighborSnapshot,
    forward_gap,
    limit_local_displacement,
)


_TOL = 1e-12
_SCHEMA_VERSION = 1


class ProtocolViolation(ValueError):
    """A supposedly idempotent identifier was reused with different data."""


class Disposition(str, Enum):
    """Result of applying a fenced or idempotent protocol message."""

    APPLIED = "applied"
    DUPLICATE = "duplicate"
    FENCED_EPOCH = "fenced_epoch"
    STALE_SEQUENCE = "stale_sequence"
    WRONG_RECIPIENT = "wrong_recipient"
    NOT_NEIGHBOR = "not_neighbor"
    UNKNOWN_ACTION = "unknown_action"
    CANCELLED = "cancelled"


def _finite_nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _finite_positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _valid_epoch_sequence(epoch: int, sequence: int) -> None:
    if not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    if not isinstance(sequence, int) or sequence <= 0:
        raise ValueError("sequence must be a positive integer")


@dataclass(frozen=True)
class PressureConfig:
    """Configuration whose values are part of the durable protocol state.

    ``pressure_scale`` is the work quantity ``Q`` in ``q / Q``.
    ``fire_threshold`` is membrane charge, so a continuously backlogged job
    fires at a rate proportional to pressure.  ``min_inter_fire`` is strictly
    positive and provides the no-Zeno bound.
    """

    pressure_scale: float
    fire_threshold: float = 1.0
    min_inter_fire: float = 0.1
    max_phase_response: float = 0.05
    response_window: float = 0.5
    safety_fraction: float = 0.45
    circumference: float = TWO_PI

    def __post_init__(self) -> None:
        for name in (
            "pressure_scale",
            "fire_threshold",
            "min_inter_fire",
            "max_phase_response",
            "response_window",
            "circumference",
        ):
            object.__setattr__(
                self, name, _finite_positive(getattr(self, name), name),
            )
        object.__setattr__(
            self, "safety_fraction", float(self.safety_fraction),
        )
        if (
            not math.isfinite(self.safety_fraction)
            or not 0.0 < self.safety_fraction <= 0.5
        ):
            raise ValueError("safety_fraction must be in (0, 0.5]")


@dataclass(frozen=True)
class PressurePulse:
    """One directed FIRE message; a fire creates exactly two of these."""

    sender: str
    recipient: str
    epoch: int
    sequence: int
    phase: float
    width: float
    pressure: float
    deadline_class: str
    fired_at: float


@dataclass(frozen=True)
class ServiceAction:
    """Idempotent request to perform at most ``requested_work`` service."""

    action_id: str
    job_id: str
    epoch: int
    sequence: int
    requested_work: float
    issued_at: float


@dataclass(frozen=True)
class ServiceReceipt:
    """Durable binding between an action id and its observed effect."""

    action_id: str
    job_id: str
    epoch: int
    sequence: int
    requested_work: float
    completed_work: float
    completed_at: float


@dataclass(frozen=True)
class ActionResult:
    disposition: Disposition
    receipt: ServiceReceipt | None


@dataclass(frozen=True)
class PulseResult:
    disposition: Disposition
    requested_displacement: float
    applied_displacement: float
    phase_before: float
    phase_after: float


def _validate_pulse(pulse: PressurePulse) -> None:
    if not pulse.sender or not pulse.recipient:
        raise ValueError("pulse endpoints must be non-empty")
    _valid_epoch_sequence(pulse.epoch, pulse.sequence)
    if not math.isfinite(pulse.phase):
        raise ValueError("pulse phase must be finite")
    _finite_positive(pulse.width, "pulse width")
    _finite_nonnegative(pulse.pressure, "pulse pressure")
    _finite_nonnegative(pulse.fired_at, "pulse fired_at")
    if not pulse.deadline_class:
        raise ValueError("deadline_class must be non-empty")


def _validate_snapshot(snapshot: NeighborSnapshot, name: str) -> None:
    if not snapshot.jid:
        raise ValueError(f"{name} jid must be non-empty")
    if not math.isfinite(snapshot.phase):
        raise ValueError(f"{name} phase must be finite")
    _finite_positive(snapshot.width, f"{name} width")
    if not isinstance(snapshot.epoch, int) or snapshot.epoch < 0:
        raise ValueError(f"{name} epoch must be a non-negative integer")
    if not isinstance(snapshot.sequence, int) or snapshot.sequence < 0:
        raise ValueError(f"{name} sequence must be non-negative")


def _validate_action(action: ServiceAction) -> None:
    if not action.action_id or not action.job_id:
        raise ValueError("action identifiers must be non-empty")
    _valid_epoch_sequence(action.epoch, action.sequence)
    _finite_positive(action.requested_work, "requested_work")
    _finite_nonnegative(action.issued_at, "issued_at")


def _validate_receipt(receipt: ServiceReceipt) -> None:
    if not receipt.action_id or not receipt.job_id:
        raise ValueError("receipt identifiers must be non-empty")
    _valid_epoch_sequence(receipt.epoch, receipt.sequence)
    requested = _finite_positive(receipt.requested_work, "requested_work")
    completed = _finite_nonnegative(receipt.completed_work, "completed_work")
    _finite_nonnegative(receipt.completed_at, "completed_at")
    if completed > requested + _TOL:
        raise ValueError("completed_work cannot exceed requested_work")


def bounded_phase_response(
    pulse: PressurePulse,
    left: NeighborSnapshot,
    current: NeighborSnapshot,
    right: NeighborSnapshot,
    *,
    left_minimum: float,
    right_minimum: float,
    config: PressureConfig,
) -> tuple[float, float]:
    """Return candidate and safety-limited response to one neighbor pulse.

    A nearby predecessor pushes ``current`` clockwise; a nearby successor
    pushes it counter-clockwise.  The response fades to zero over
    ``response_window`` and is bounded by ``max_phase_response`` before being
    passed through the fixed-kernel hard-gap limiter.

    This function deliberately has no lane-wide input.  It consumes exactly a
    predecessor frame, the local frame, a successor frame, and one pulse from
    either predecessor or successor.
    """

    _validate_pulse(pulse)
    _validate_snapshot(left, "left snapshot")
    _validate_snapshot(current, "current snapshot")
    _validate_snapshot(right, "right snapshot")
    if len({left.jid, current.jid, right.jid}) != 3:
        raise ValueError("left, current, and right snapshots must be distinct")
    if not (
        left.epoch == current.epoch == right.epoch == pulse.epoch
    ):
        raise ValueError("pulse and snapshots must share one epoch")
    left_minimum = _finite_nonnegative(left_minimum, "left_minimum")
    right_minimum = _finite_nonnegative(right_minimum, "right_minimum")
    if pulse.recipient != current.jid:
        raise ValueError("pulse recipient does not match current job")

    effective_left = left
    effective_right = right
    if pulse.sender == left.jid:
        effective_left = NeighborSnapshot(
            pulse.sender,
            pulse.phase % config.circumference,
            pulse.width,
            pulse.epoch,
            pulse.sequence,
        )
        gap = forward_gap(
            effective_left.phase, current.phase, config.circumference,
        )
        minimum = left_minimum
        direction = 1.0
    elif pulse.sender == right.jid:
        effective_right = NeighborSnapshot(
            pulse.sender,
            pulse.phase % config.circumference,
            pulse.width,
            pulse.epoch,
            pulse.sequence,
        )
        gap = forward_gap(
            current.phase, effective_right.phase, config.circumference,
        )
        minimum = right_minimum
        direction = -1.0
    else:
        raise ValueError("pulse sender must be an immediate neighbor")

    proximity = max(
        0.0,
        min(1.0, (minimum + config.response_window - gap)
            / config.response_window),
    )
    # A pressure that can recharge the membrane within one refractory interval
    # receives full strength.  Slower pressure receives a proportional nudge.
    full_rate_pressure = config.fire_threshold / config.min_inter_fire
    pressure_factor = min(1.0, pulse.pressure / full_rate_pressure)
    requested = (
        direction * config.max_phase_response * proximity * pressure_factor
    )
    applied = limit_local_displacement(
        requested,
        effective_left,
        current,
        effective_right,
        left_minimum=left_minimum,
        right_minimum=right_minimum,
        safety_fraction=config.safety_fraction,
        circumference=config.circumference,
    )
    return requested, applied


class FencedActuator:
    """Reference durable idempotency/fencing ledger for a per-job actuator.

    The class records receipts, not a real side effect.  A deployment must
    atomically bind the same action id to the actual resource operation and
    persist its receipt.  Known exact retries remain valid after an epoch
    change; an unknown old-epoch action is fenced.
    """

    def __init__(self, job_id: str, epoch: int = 0) -> None:
        if not job_id:
            raise ValueError("job_id must be non-empty")
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.job_id = job_id
        self.epoch = epoch
        self._last_sequence = 0
        self._actions: dict[str, ServiceAction] = {}
        self._receipts: dict[str, ServiceReceipt] = {}

    def advance_epoch(self, epoch: int) -> None:
        if not isinstance(epoch, int) or epoch <= self.epoch:
            raise ValueError("new actuator epoch must be strictly greater")
        self.epoch = epoch
        self._last_sequence = 0

    def apply(
        self,
        action: ServiceAction,
        *,
        completed_work: float | None = None,
        completed_at: float,
    ) -> ActionResult:
        """Apply once, return a cached receipt on an exact retry."""

        _validate_action(action)
        completed_at = _finite_nonnegative(completed_at, "completed_at")
        if action.job_id != self.job_id:
            return ActionResult(Disposition.WRONG_RECIPIENT, None)

        known = self._actions.get(action.action_id)
        if known is not None:
            if known != action:
                raise ProtocolViolation(
                    "action_id was reused with different action data"
                )
            return ActionResult(
                Disposition.DUPLICATE, self._receipts[action.action_id],
            )

        if action.epoch != self.epoch:
            return ActionResult(Disposition.FENCED_EPOCH, None)
        if action.sequence <= self._last_sequence:
            return ActionResult(Disposition.STALE_SEQUENCE, None)

        if completed_work is None:
            completed_work = action.requested_work
        completed_work = _finite_nonnegative(
            completed_work, "completed_work",
        )
        if completed_work > action.requested_work + _TOL:
            raise ValueError("completed_work cannot exceed requested_work")
        if completed_at + _TOL < action.issued_at:
            raise ValueError("completion cannot precede action issue")

        receipt = ServiceReceipt(
            action_id=action.action_id,
            job_id=action.job_id,
            epoch=action.epoch,
            sequence=action.sequence,
            requested_work=action.requested_work,
            completed_work=completed_work,
            completed_at=completed_at,
        )
        self._actions[action.action_id] = action
        self._receipts[action.action_id] = receipt
        self._last_sequence = action.sequence
        return ActionResult(Disposition.APPLIED, receipt)

    def dump_state(self) -> str:
        payload = {
            "schema": _SCHEMA_VERSION,
            "job_id": self.job_id,
            "epoch": self.epoch,
            "last_sequence": self._last_sequence,
            "actions": [
                asdict(self._actions[action_id])
                for action_id in sorted(self._actions)
            ],
            "receipts": [
                asdict(self._receipts[action_id])
                for action_id in sorted(self._receipts)
            ],
        }
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
        )

    @classmethod
    def load_state(cls, encoded: str) -> "FencedActuator":
        raw = json.loads(encoded)
        if raw.get("schema") != _SCHEMA_VERSION:
            raise ValueError("unsupported actuator checkpoint schema")
        actuator = cls(str(raw["job_id"]), int(raw["epoch"]))
        actuator._last_sequence = int(raw["last_sequence"])
        if actuator._last_sequence < 0:
            raise ValueError("last_sequence must be non-negative")
        for item in raw["actions"]:
            action = ServiceAction(**item)
            _validate_action(action)
            if action.action_id in actuator._actions:
                raise ValueError("duplicate action in actuator checkpoint")
            if action.job_id != actuator.job_id:
                raise ValueError("actuator checkpoint contains another job")
            actuator._actions[action.action_id] = action
        for item in raw["receipts"]:
            receipt = ServiceReceipt(**item)
            _validate_receipt(receipt)
            if receipt.action_id in actuator._receipts:
                raise ValueError("duplicate receipt in actuator checkpoint")
            if receipt.job_id != actuator.job_id:
                raise ValueError("actuator checkpoint contains another job")
            actuator._receipts[receipt.action_id] = receipt
        if set(actuator._actions) != set(actuator._receipts):
            raise ValueError("actuator checkpoint has an incomplete receipt")
        for action_id, action in actuator._actions.items():
            receipt = actuator._receipts[action_id]
            if (
                receipt.job_id != action.job_id
                or receipt.epoch != action.epoch
                or receipt.sequence != action.sequence
                or not math.isclose(
                    receipt.requested_work,
                    action.requested_work,
                    rel_tol=0.0,
                    abs_tol=_TOL,
                )
                or receipt.completed_at + _TOL < action.issued_at
            ):
                raise ValueError("actuator receipt does not match its action")
        active_sequences = [
            action.sequence for action in actuator._actions.values()
            if action.epoch == actuator.epoch
        ]
        if active_sequences and max(active_sequences) > actuator._last_sequence:
            raise ValueError("actuator sequence fence trails accepted action")
        return actuator


class PressureBubble:
    """Durable, strict-neighbor pressure oscillator for one admitted job."""

    def __init__(
        self,
        job_id: str,
        *,
        width: float,
        phase: float,
        predecessor: str,
        successor: str,
        config: PressureConfig,
        epoch: int = 0,
        deadline_class: str = "best-effort",
        now: float = 0.0,
    ) -> None:
        self._validate_topology_ids(job_id, predecessor, successor)
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        if not math.isfinite(phase):
            raise ValueError("phase must be finite")
        if not deadline_class:
            raise ValueError("deadline_class must be non-empty")

        self.job_id = job_id
        self.width = _finite_positive(width, "width")
        self.phase = phase % config.circumference
        self.predecessor = predecessor
        self.successor = successor
        self.config = config
        self.epoch = epoch
        self.deadline_class = deadline_class

        self.backlog = 0.0
        self.deadline_pressure = 0.0
        self.membrane = 0.0
        self.last_update_at = _finite_nonnegative(now, "now")
        self.last_fire_at: float | None = None
        self._out_sequence = 0
        self._arrivals: dict[str, float] = {}
        self._pending_actions: dict[str, ServiceAction] = {}
        self._acknowledged: dict[str, ServiceReceipt] = {}
        self._seen_pulses: dict[str, PressurePulse] = {}

    @staticmethod
    def _validate_topology_ids(
        job_id: str, predecessor: str, successor: str,
    ) -> None:
        if not job_id or not predecessor or not successor:
            raise ValueError("job and neighbor ids must be non-empty")
        if len({job_id, predecessor, successor}) != 3:
            raise ValueError(
                "strict-neighbor membership requires two distinct neighbors"
            )

    @property
    def pressure(self) -> float:
        return self.backlog / self.config.pressure_scale + self.deadline_pressure

    @property
    def pending_work(self) -> float:
        return math.fsum(
            action.requested_work for action in self._pending_actions.values()
        )

    @property
    def available_work(self) -> float:
        return max(0.0, self.backlog - self.pending_work)

    @property
    def admitted_work(self) -> float:
        return math.fsum(self._arrivals.values())

    @property
    def acknowledged_work(self) -> float:
        return math.fsum(
            receipt.completed_work for receipt in self._acknowledged.values()
        )

    @property
    def conservation_residual(self) -> float:
        """Zero iff admitted = outstanding + acknowledged work."""

        return self.admitted_work - self.backlog - self.acknowledged_work

    def _advance_membrane(self, now: float) -> None:
        now = _finite_nonnegative(now, "now")
        if now + _TOL < self.last_update_at:
            raise ValueError("logical time cannot move backwards")
        elapsed = max(0.0, now - self.last_update_at)
        if self.backlog > _TOL:
            self.membrane += self.pressure * elapsed
        else:
            self.membrane = 0.0
        self.last_update_at = now

    def _next_sequence(self) -> int:
        self._out_sequence += 1
        return self._out_sequence

    def admit_arrival(self, event_id: str, work: float, *, now: float) -> Disposition:
        """Add an admitted arrival exactly once; duplicates are harmless."""

        if not event_id:
            raise ValueError("event_id must be non-empty")
        work = _finite_positive(work, "arrival work")
        self._advance_membrane(now)
        known = self._arrivals.get(event_id)
        if known is not None:
            if not math.isclose(known, work, rel_tol=0.0, abs_tol=_TOL):
                raise ProtocolViolation(
                    "arrival event_id was reused with different work"
                )
            return Disposition.DUPLICATE
        self._arrivals[event_id] = work
        self.backlog += work
        return Disposition.APPLIED

    def set_deadline_pressure(self, pressure: float, *, now: float) -> None:
        """Change urgency after integrating the prior pressure up to ``now``."""

        pressure = _finite_nonnegative(pressure, "deadline pressure")
        self._advance_membrane(now)
        self.deadline_pressure = pressure

    def maybe_fire(self, *, now: float) -> tuple[PressurePulse, ...]:
        """Emit at most one pulse pair, respecting a strict refractory time."""

        self._advance_membrane(now)
        if self.backlog <= _TOL or self.membrane + _TOL < self.config.fire_threshold:
            return ()
        if (
            self.last_fire_at is not None
            and now - self.last_fire_at + _TOL < self.config.min_inter_fire
        ):
            return ()

        sequence = self._next_sequence()
        self.membrane = max(0.0, self.membrane - self.config.fire_threshold)
        self.last_fire_at = float(now)
        common = {
            "sender": self.job_id,
            "epoch": self.epoch,
            "sequence": sequence,
            "phase": self.phase,
            "width": self.width,
            "pressure": self.pressure,
            "deadline_class": self.deadline_class,
            "fired_at": float(now),
        }
        return (
            PressurePulse(recipient=self.predecessor, **common),
            PressurePulse(recipient=self.successor, **common),
        )

    def issue_service(self, work: float, *, now: float) -> ServiceAction:
        """Create a write-ahead action without changing backlog."""

        work = _finite_positive(work, "service work")
        self._advance_membrane(now)
        if work > self.available_work + _TOL:
            raise ValueError(
                f"service request {work:g} exceeds unreserved backlog "
                f"{self.available_work:g}"
            )
        sequence = self._next_sequence()
        action = ServiceAction(
            action_id=f"{self.job_id}:{self.epoch}:{sequence}",
            job_id=self.job_id,
            epoch=self.epoch,
            sequence=sequence,
            requested_work=work,
            issued_at=float(now),
        )
        self._pending_actions[action.action_id] = action
        return action

    def cancel_service(self, action_id: str, *, now: float) -> Disposition:
        """Release an unexecuted/fenced request without removing work."""

        self._advance_membrane(now)
        if action_id in self._acknowledged:
            return Disposition.DUPLICATE
        if self._pending_actions.pop(action_id, None) is None:
            return Disposition.UNKNOWN_ACTION
        return Disposition.CANCELLED

    def acknowledge_service(
        self, receipt: ServiceReceipt, *, now: float,
    ) -> Disposition:
        """Subtract only the effect bound to a known pending action id."""

        _validate_receipt(receipt)
        self._advance_membrane(now)
        known_receipt = self._acknowledged.get(receipt.action_id)
        if known_receipt is not None:
            if known_receipt != receipt:
                raise ProtocolViolation(
                    "action receipt changed across an idempotent retry"
                )
            return Disposition.DUPLICATE

        action = self._pending_actions.get(receipt.action_id)
        if action is None:
            return Disposition.UNKNOWN_ACTION
        if (
            receipt.job_id != action.job_id
            or receipt.epoch != action.epoch
            or receipt.sequence != action.sequence
            or not math.isclose(
                receipt.requested_work,
                action.requested_work,
                rel_tol=0.0,
                abs_tol=_TOL,
            )
        ):
            raise ProtocolViolation("receipt does not match pending action")
        if receipt.completed_at + _TOL < action.issued_at:
            raise ProtocolViolation("receipt predates its pending action")
        if receipt.completed_at > now + _TOL:
            raise ProtocolViolation("receipt completion lies in the future")
        if receipt.completed_work > self.backlog + _TOL:
            raise ProtocolViolation("receipt would make backlog negative")

        self.backlog = max(0.0, self.backlog - receipt.completed_work)
        del self._pending_actions[action.action_id]
        self._acknowledged[action.action_id] = receipt
        if self.backlog <= _TOL:
            self.backlog = 0.0
            self.membrane = 0.0
        return Disposition.APPLIED

    def install_topology(
        self,
        *,
        epoch: int,
        predecessor: str,
        successor: str,
        now: float,
    ) -> None:
        """Install a strictly newer neighbor lease without touching work."""

        self._advance_membrane(now)
        if not isinstance(epoch, int) or epoch <= self.epoch:
            raise ValueError("new topology epoch must be strictly greater")
        self._validate_topology_ids(self.job_id, predecessor, successor)
        self.epoch = epoch
        self.predecessor = predecessor
        self.successor = successor
        self._out_sequence = 0
        self._seen_pulses.clear()

    def receive_pulse(
        self,
        pulse: PressurePulse,
        *,
        left: NeighborSnapshot,
        right: NeighborSnapshot,
        left_minimum: float,
        right_minimum: float,
    ) -> PulseResult:
        """Fence, deduplicate, and apply one bounded strict-neighbor pulse."""

        _validate_pulse(pulse)
        before = self.phase

        def empty(disposition: Disposition) -> PulseResult:
            return PulseResult(disposition, 0.0, 0.0, before, before)

        if pulse.recipient != self.job_id:
            return empty(Disposition.WRONG_RECIPIENT)
        if pulse.epoch != self.epoch:
            return empty(Disposition.FENCED_EPOCH)
        if pulse.sender not in {self.predecessor, self.successor}:
            return empty(Disposition.NOT_NEIGHBOR)

        prior = self._seen_pulses.get(pulse.sender)
        if prior is not None:
            if pulse.sequence < prior.sequence:
                return empty(Disposition.STALE_SEQUENCE)
            if pulse.sequence == prior.sequence:
                if pulse != prior:
                    raise ProtocolViolation(
                        "pulse sequence was reused with different data"
                    )
                return empty(Disposition.DUPLICATE)

        if left.jid != self.predecessor or right.jid != self.successor:
            raise ValueError("snapshots do not match the current neighbor lease")
        if left.epoch != self.epoch or right.epoch != self.epoch:
            raise ValueError("neighbor snapshots must match the current epoch")
        current = NeighborSnapshot(
            jid=self.job_id,
            phase=self.phase,
            width=self.width,
            epoch=self.epoch,
            sequence=self._out_sequence,
        )
        requested, applied = bounded_phase_response(
            pulse,
            left,
            current,
            right,
            left_minimum=left_minimum,
            right_minimum=right_minimum,
            config=self.config,
        )
        self.phase = (self.phase + applied) % self.config.circumference
        self._seen_pulses[pulse.sender] = pulse
        return PulseResult(
            Disposition.APPLIED,
            requested,
            applied,
            before,
            self.phase,
        )

    def dump_state(self) -> str:
        """Return a canonical JSON checkpoint containing every dedupe fence."""

        payload: dict[str, Any] = {
            "schema": _SCHEMA_VERSION,
            "config": asdict(self.config),
            "job_id": self.job_id,
            "width": self.width,
            "phase": self.phase,
            "predecessor": self.predecessor,
            "successor": self.successor,
            "epoch": self.epoch,
            "deadline_class": self.deadline_class,
            "backlog": self.backlog,
            "deadline_pressure": self.deadline_pressure,
            "membrane": self.membrane,
            "last_update_at": self.last_update_at,
            "last_fire_at": self.last_fire_at,
            "out_sequence": self._out_sequence,
            "arrivals": sorted(self._arrivals.items()),
            "pending_actions": [
                asdict(self._pending_actions[action_id])
                for action_id in sorted(self._pending_actions)
            ],
            "acknowledged": [
                asdict(self._acknowledged[action_id])
                for action_id in sorted(self._acknowledged)
            ],
            "seen_pulses": [
                asdict(self._seen_pulses[sender])
                for sender in sorted(self._seen_pulses)
            ],
        }
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
        )

    @classmethod
    def load_state(cls, encoded: str) -> "PressureBubble":
        """Recover a bubble from ``dump_state`` and validate its invariants."""

        raw: Mapping[str, Any] = json.loads(encoded)
        if raw.get("schema") != _SCHEMA_VERSION:
            raise ValueError("unsupported pressure checkpoint schema")
        config = PressureConfig(**raw["config"])
        bubble = cls(
            str(raw["job_id"]),
            width=float(raw["width"]),
            phase=float(raw["phase"]),
            predecessor=str(raw["predecessor"]),
            successor=str(raw["successor"]),
            config=config,
            epoch=int(raw["epoch"]),
            deadline_class=str(raw["deadline_class"]),
            now=float(raw["last_update_at"]),
        )
        bubble.backlog = _finite_nonnegative(raw["backlog"], "backlog")
        bubble.deadline_pressure = _finite_nonnegative(
            raw["deadline_pressure"], "deadline_pressure",
        )
        bubble.membrane = _finite_nonnegative(raw["membrane"], "membrane")
        last_fire = raw["last_fire_at"]
        bubble.last_fire_at = (
            None if last_fire is None
            else _finite_nonnegative(last_fire, "last_fire_at")
        )
        bubble._out_sequence = int(raw["out_sequence"])
        if bubble._out_sequence < 0:
            raise ValueError("out_sequence must be non-negative")

        for event_id, work in raw["arrivals"]:
            if not event_id or event_id in bubble._arrivals:
                raise ValueError("invalid or duplicate arrival id in checkpoint")
            bubble._arrivals[str(event_id)] = _finite_positive(
                work, "arrival work",
            )
        for item in raw["pending_actions"]:
            action = ServiceAction(**item)
            _validate_action(action)
            if action.action_id in bubble._pending_actions:
                raise ValueError("duplicate pending action in checkpoint")
            if action.job_id != bubble.job_id:
                raise ValueError("pending action belongs to another job")
            expected_id = f"{bubble.job_id}:{action.epoch}:{action.sequence}"
            if action.action_id != expected_id:
                raise ValueError("pending action id is not canonical")
            if action.epoch == bubble.epoch:
                if action.sequence > bubble._out_sequence:
                    raise ValueError("pending action exceeds sequence fence")
            bubble._pending_actions[action.action_id] = action
        for item in raw["acknowledged"]:
            receipt = ServiceReceipt(**item)
            _validate_receipt(receipt)
            if receipt.action_id in bubble._acknowledged:
                raise ValueError("duplicate receipt in checkpoint")
            if receipt.job_id != bubble.job_id:
                raise ValueError("receipt belongs to another job")
            expected_id = f"{bubble.job_id}:{receipt.epoch}:{receipt.sequence}"
            if receipt.action_id != expected_id:
                raise ValueError("receipt action id is not canonical")
            if receipt.epoch == bubble.epoch:
                if receipt.sequence > bubble._out_sequence:
                    raise ValueError("receipt exceeds sequence fence")
            bubble._acknowledged[receipt.action_id] = receipt
        if set(bubble._pending_actions) & set(bubble._acknowledged):
            raise ValueError("an action cannot be pending and acknowledged")
        for item in raw["seen_pulses"]:
            pulse = PressurePulse(**item)
            _validate_pulse(pulse)
            if pulse.sender in bubble._seen_pulses:
                raise ValueError("duplicate pulse sender in checkpoint")
            bubble._seen_pulses[pulse.sender] = pulse

        if bubble.pending_work > bubble.backlog + _TOL:
            raise ValueError("checkpoint reserves more work than its backlog")
        if not math.isclose(
            bubble.conservation_residual,
            0.0,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise ValueError("checkpoint violates work conservation")
        if bubble.last_fire_at is not None:
            if bubble.last_fire_at > bubble.last_update_at + _TOL:
                raise ValueError("last fire lies after checkpoint logical time")
        if any(
            pulse.epoch != bubble.epoch
            or pulse.sender not in {bubble.predecessor, bubble.successor}
            for pulse in bubble._seen_pulses.values()
        ):
            raise ValueError("checkpoint contains a pulse outside its lease")
        return bubble
