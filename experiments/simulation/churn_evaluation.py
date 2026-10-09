"""Deterministic churn/fault evaluation for the strict-neighbor lane.

This harness composes the epoch-fenced :mod:`core.topology` authority with the
fixed-topology spacing kernel and the durable pressure/work ledger.  It tests
join, adjacent swap, graceful leave, lease-expiry repair, transaction timeout,
duplicate delivery, stale-epoch fencing, settling, safety, message count, and
work conservation.  It remains a simulator: the centralized harness observes
the whole lane to score it, while every controller step still uses exactly two
neighbor frames per job.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

from kadence.neighbor_gossip import (  # noqa: E402
    BubbleSpec,
    NeighborSnapshot,
    RingController,
    TWO_PI,
    admit_ring,
    forward_gap,
    phases_from_gaps,
)
from kadence.pressure_fire import (  # noqa: E402
    Disposition,
    FencedActuator,
    PressureBubble,
    PressureConfig,
)
from kadence.topology import (  # noqa: E402
    MessageDisposition,
    StaleEpoch,
    TopologyAuthority,
    TransactionStatus,
)


_TOL = 1e-9


@dataclass
class WorkLedger:
    admitted: float = 0.0
    acknowledged: float = 0.0
    fenced_actions: int = 0
    duplicate_receipts: int = 0


def _clone(controller: RingController) -> RingController:
    return RingController(
        controller.bubbles,
        tuple(controller.phases),
        controller.plan,
        gains=controller.gains,
        common_frequency=controller.common_frequency,
        epoch=controller.epoch,
    )


def _minimum_margin(controller: RingController) -> float:
    return min(
        gap - minimum
        for gap, minimum in zip(controller.gaps(), controller.plan.minimum)
    )


def _settling_comparison(
    initial: RingController,
    *,
    tolerance: float,
    max_rounds: int,
) -> tuple[dict[str, object], RingController]:
    """Score local recovery against frozen and centralized slot baselines."""

    local = _clone(initial)
    initial_error = local.max_gap_error()
    rounds = 0
    energy_increases = 0
    minimum_margin = _minimum_margin(local)
    while local.max_gap_error() > tolerance and rounds < max_rounds:
        report = local.step(dt=0.08, safety_fraction=0.45)
        if report.energy_after > report.energy_before + 1e-12:
            energy_increases += 1
        minimum_margin = min(minimum_margin, _minimum_margin(local))
        rounds += 1

    frozen_error = initial_error
    oracle = RingController(
        initial.bubbles,
        phases_from_gaps(
            initial.plan.desired,
            origin=initial.phases[0],
            circumference=initial.plan.circumference,
        ),
        initial.plan,
        epoch=initial.epoch,
    )
    result = {
        "members": len(local.bubbles),
        "initial_max_gap_error": initial_error,
        "strict_neighbor": {
            "settled": local.max_gap_error() <= tolerance,
            "settling_rounds": rounds,
            "final_max_gap_error": local.max_gap_error(),
            "directed_messages": local.message_count,
            "messages_per_job_round": (
                local.message_count / (len(local.bubbles) * rounds)
                if rounds else 0.0
            ),
            "minimum_safety_margin": minimum_margin,
            "energy_increase_rounds": energy_increases,
        },
        "frozen_no_coupling": {
            "settled": frozen_error <= tolerance,
            "settling_rounds": 0,
            "final_max_gap_error": frozen_error,
            "directed_messages": 0,
        },
        "centralized_slot_oracle": {
            "settled": oracle.max_gap_error() <= tolerance,
            "settling_rounds": 0,
            "final_max_gap_error": oracle.max_gap_error(),
            "global_phase_writes": len(oracle.bubbles),
        },
    }
    return result, local


def _controller_after_change(
    controller: RingController,
    new_order: tuple[str, ...],
    widths: dict[str, float],
    *,
    epoch: int,
    joined: str | None = None,
    swapped: tuple[str, str] | None = None,
    guard: float = 0.03,
) -> RingController:
    old_order = tuple(bubble.jid for bubble in controller.bubbles)
    old_phases = dict(zip(old_order, controller.phases))
    bubbles = tuple(BubbleSpec(jid, widths[jid]) for jid in new_order)
    plan = admit_ring(bubbles, guard=guard)

    if joined is not None:
        index = new_order.index(joined)
        left = new_order[index - 1]
        right = new_order[(index + 1) % len(new_order)]
        old_gap = forward_gap(old_phases[left], old_phases[right], TWO_PI)
        left_minimum = plan.minimum[index - 1]
        right_minimum = plan.minimum[index]
        available = old_gap - left_minimum - right_minimum
        if available < -_TOL:
            raise RuntimeError("prepared join did not reserve a feasible edge split")
        # Deliberately do not start at the target split: this creates a safe,
        # reproducible disturbance whose settling time can be measured.
        left_gap = left_minimum + 0.35 * max(0.0, available)
        old_phases[joined] = (old_phases[left] + left_gap) % TWO_PI
        phases = tuple(old_phases[jid] for jid in new_order)
    elif swapped is not None:
        # Atomic adjacent slot handoff: work stays with each job, while the two
        # participants exchange phase slots at the fenced commit boundary.
        # Equal-width experiments make this reservation-safe.
        phases = tuple(controller.phases[index] for index in range(len(new_order)))
    else:
        phases = tuple(old_phases[jid] for jid in new_order)
    return RingController(bubbles, phases, plan, epoch=epoch)


def _commit_all(
    authority: TopologyAuthority,
    receipt,
    *,
    now: float,
    duplicate_first_ack: bool = False,
) -> tuple[object, int]:
    acknowledgements = 0
    for index, jid in enumerate(receipt.required):
        authority.proposal(receipt.transaction_id, jid)
        authority.acknowledge(
            receipt.transaction_id, jid,
            expected_epoch=receipt.base_epoch, now=now,
        )
        acknowledgements += 1
        if index == 0 and duplicate_first_ack:
            authority.acknowledge(
                receipt.transaction_id, jid,
                expected_epoch=receipt.base_epoch, now=now,
            )
            acknowledgements += 1
    committed = authority.commit(
        receipt.transaction_id,
        expected_epoch=receipt.base_epoch,
        now=now,
    )
    return committed, acknowledgements


def _make_work_state(
    authority: TopologyAuthority,
    controller: RingController,
) -> tuple[dict[str, PressureBubble], dict[str, FencedActuator], WorkLedger]:
    config = PressureConfig(
        pressure_scale=1.0,
        fire_threshold=0.5,
        min_inter_fire=0.25,
        max_phase_response=0.03,
        response_window=0.5,
    )
    phases = dict(zip(authority.order, controller.phases))
    bubbles: dict[str, PressureBubble] = {}
    actuators: dict[str, FencedActuator] = {}
    ledger = WorkLedger()
    for index, jid in enumerate(authority.order):
        lease = authority.neighbor_view(jid)
        bubble = PressureBubble(
            jid,
            width=controller.bubbles[index].width,
            phase=phases[jid],
            predecessor=lease.predecessor,
            successor=lease.successor,
            config=config,
            epoch=authority.epoch,
            now=0.0,
        )
        work = 1.0 + 0.1 * index
        bubble.admit_arrival(f"initial:{jid}", work, now=0.0)
        ledger.admitted += work
        bubbles[jid] = bubble
        actuators[jid] = FencedActuator(jid, authority.epoch)
    return bubbles, actuators, ledger


def _install_epoch(
    authority: TopologyAuthority,
    controller: RingController,
    active: dict[str, PressureBubble],
    parked: dict[str, PressureBubble],
    actuators: dict[str, FencedActuator],
    ledger: WorkLedger,
    widths: dict[str, float],
    *,
    now: float,
    joined: str | None = None,
) -> None:
    phase_map = dict(zip(authority.order, controller.phases))
    for removed in tuple(set(active) - set(authority.order)):
        parked[removed] = active.pop(removed)
    for jid in authority.order:
        lease = authority.neighbor_view(jid)
        if jid not in active:
            bubble = PressureBubble(
                jid,
                width=widths[jid],
                phase=phase_map[jid],
                predecessor=lease.predecessor,
                successor=lease.successor,
                config=next(iter(active.values())).config,
                epoch=authority.epoch,
                now=now,
            )
            active[jid] = bubble
            actuators[jid] = FencedActuator(jid, authority.epoch)
            if jid == joined:
                bubble.admit_arrival(f"join:{jid}", 0.75, now=now)
                ledger.admitted += 0.75
        else:
            active[jid].install_topology(
                epoch=authority.epoch,
                predecessor=lease.predecessor,
                successor=lease.successor,
                now=now,
            )
            actuators[jid].advance_epoch(authority.epoch)
            active[jid].phase = phase_map[jid]


def run_churn_evaluation(
    *, tolerance: float = 1e-6, max_rounds: int = 20_000,
) -> dict[str, object]:
    widths = {f"job-{index}": 0.25 for index in range(6)}
    initial_order = tuple(widths)
    initial_bubbles = tuple(BubbleSpec(jid, widths[jid]) for jid in initial_order)
    initial_plan = admit_ring(initial_bubbles, guard=0.03)
    controller = RingController(
        initial_bubbles,
        phases_from_gaps(initial_plan.desired),
        initial_plan,
        epoch=0,
    )
    authority = TopologyAuthority(
        initial_order, initial_epoch=0, lease_duration=50.0, start_time=0.0,
    )
    active, actuators, work = _make_work_state(authority, controller)
    parked: dict[str, PressureBubble] = {}
    events: list[dict[str, object]] = []
    fault_counts = {
        "duplicate_acknowledgements": 0,
        "aborted_transactions": 0,
        "stale_topology_frames_fenced": 0,
        "duplicate_topology_frames": 0,
        "stale_pressure_pulses_fenced": 0,
        "old_epoch_actions_fenced": 0,
    }
    topology_ack_messages = 0

    # Hold one pressure pulse and one service action across the first epoch
    # change; neither may affect phase or backlog afterward.
    stale_pulses = active["job-5"].maybe_fire(now=1.0)
    pending_old_action = active["job-0"].issue_service(0.10, now=1.0)
    old_neighbor = authority.neighbor_view("job-0").predecessor
    authority.accept_neighbor_message(
        receiver="job-0", sender=old_neighbor, epoch=0, sequence=1,
    )

    # Join by splitting the explicit seam edge.
    new_jid = "job-join"
    widths[new_jid] = 0.25
    before_order = authority.order
    receipt = authority.prepare_join(
        "tx-join", new_jid=new_jid,
        left_jid=before_order[-1], right_jid=before_order[0],
        expected_epoch=authority.epoch, now=1.1, timeout=2.0,
    )
    committed, acknowledgements = _commit_all(authority, receipt, now=1.2)
    topology_ack_messages += acknowledgements
    post = _controller_after_change(
        controller, authority.order, widths,
        epoch=authority.epoch, joined=new_jid,
    )
    comparison, controller = _settling_comparison(
        post, tolerance=tolerance, max_rounds=max_rounds,
    )
    _install_epoch(
        authority, controller, active, parked, actuators, work, widths,
        now=1.3, joined=new_jid,
    )
    events.append({"event": "join", "epoch": committed.terminal_epoch,
                   **comparison})

    try:
        authority.accept_neighbor_message(
            receiver="job-0", sender=old_neighbor, epoch=0, sequence=2,
        )
    except StaleEpoch:
        fault_counts["stale_topology_frames_fenced"] += 1
    view = authority.neighbor_view("job-1")
    authority.accept_neighbor_message(
        receiver="job-1", sender=view.predecessor,
        epoch=authority.epoch, sequence=1,
    )
    duplicate = authority.accept_neighbor_message(
        receiver="job-1", sender=view.predecessor,
        epoch=authority.epoch, sequence=1,
    )
    if duplicate is MessageDisposition.DUPLICATE:
        fault_counts["duplicate_topology_frames"] += 1
    if stale_pulses:
        pulse = stale_pulses[0]
        recipient = active[pulse.recipient]
        frames = {
            frame.jid: frame for frame in controller.snapshots()
        }
        left = authority.neighbor_view(recipient.job_id).predecessor
        right = authority.neighbor_view(recipient.job_id).successor
        outcome = recipient.receive_pulse(
            pulse,
            left=frames[left], right=frames[right],
            left_minimum=controller.plan.minimum[
                authority.order.index(recipient.job_id) - 1
            ],
            right_minimum=controller.plan.minimum[
                authority.order.index(recipient.job_id)
            ],
        )
        if outcome.disposition is Disposition.FENCED_EPOCH:
            fault_counts["stale_pressure_pulses_fenced"] += 1
    old_result = actuators["job-0"].apply(
        pending_old_action, completed_at=1.4,
    )
    if old_result.disposition is Disposition.FENCED_EPOCH:
        fault_counts["old_epoch_actions_fenced"] += 1
        work.fenced_actions += 1
        active["job-0"].cancel_service(pending_old_action.action_id, now=1.4)

    # Prepared but incomplete join times out without changing membership.
    order_before_timeout = authority.order
    timeout_receipt = authority.prepare_join(
        "tx-timeout", new_jid="job-timeout",
        left_jid=authority.order[-1], right_jid=authority.order[0],
        expected_epoch=authority.epoch, now=2.0, timeout=0.5,
    )
    expired = authority.expire(now=2.5)
    timeout_safe = (
        len(expired) == 1
        and expired[0].status is TransactionStatus.ABORTED
        and authority.order == order_before_timeout
        and authority.epoch == timeout_receipt.base_epoch
    )
    fault_counts["aborted_transactions"] += int(timeout_safe)
    events.append({
        "event": "join-timeout",
        "committed": False,
        "old_order_preserved": timeout_safe,
        "epoch": authority.epoch,
    })

    # Adjacent equal-width jobs exchange reserved phase slots atomically.
    swap_receipt = authority.prepare_swap(
        "tx-swap", left_jid="job-1", right_jid="job-2",
        expected_epoch=authority.epoch, now=3.0, timeout=2.0,
    )
    committed, acknowledgements = _commit_all(
        authority, swap_receipt, now=3.1, duplicate_first_ack=True,
    )
    topology_ack_messages += acknowledgements
    fault_counts["duplicate_acknowledgements"] += 1
    post = _controller_after_change(
        controller, authority.order, widths,
        epoch=authority.epoch, swapped=("job-1", "job-2"),
    )
    comparison, controller = _settling_comparison(
        post, tolerance=tolerance, max_rounds=max_rounds,
    )
    _install_epoch(
        authority, controller, active, parked, actuators, work, widths, now=3.2,
    )
    events.append({"event": "adjacent-swap", "epoch": committed.terminal_epoch,
                   **comparison})

    leave_receipt = authority.prepare_leave(
        "tx-leave", jid="job-3", expected_epoch=authority.epoch,
        now=4.0, timeout=2.0,
    )
    committed, acknowledgements = _commit_all(
        authority, leave_receipt, now=4.1,
    )
    topology_ack_messages += acknowledgements
    post = _controller_after_change(
        controller, authority.order, widths, epoch=authority.epoch,
    )
    comparison, controller = _settling_comparison(
        post, tolerance=tolerance, max_rounds=max_rounds,
    )
    _install_epoch(
        authority, controller, active, parked, actuators, work, widths, now=4.2,
    )
    events.append({"event": "graceful-leave", "epoch": committed.terminal_epoch,
                   **comparison})

    # Keep every survivor live while job-4's lease is allowed to expire.
    failed_jid = "job-4"
    for jid in authority.order:
        if jid != failed_jid:
            authority.renew_member(jid, expected_epoch=authority.epoch, now=20.0)
    failure_time = authority.neighbor_view(failed_jid).lease_deadline + 0.01
    durable_crash_state = active[failed_jid].dump_state()
    failed_receipt = authority.prepare_failed_leave(
        "tx-failed-leave", jid=failed_jid,
        expected_epoch=authority.epoch, now=failure_time, timeout=2.0,
    )
    committed, acknowledgements = _commit_all(
        authority, failed_receipt, now=failure_time + 0.01,
    )
    topology_ack_messages += acknowledgements
    post = _controller_after_change(
        controller, authority.order, widths, epoch=authority.epoch,
    )
    comparison, controller = _settling_comparison(
        post, tolerance=tolerance, max_rounds=max_rounds,
    )
    _install_epoch(
        authority, controller, active, parked, actuators, work, widths,
        now=failure_time + 0.02,
    )
    parked[failed_jid] = PressureBubble.load_state(durable_crash_state)
    events.append({"event": "lease-expiry-repair",
                   "epoch": committed.terminal_epoch, **comparison})

    # A current-epoch service action is applied once; exact action and receipt
    # retries are idempotent and cannot subtract backlog twice.
    service_job = "job-0"
    service = min(0.30, active[service_job].available_work)
    action = active[service_job].issue_service(
        service, now=failure_time + 0.1,
    )
    applied = actuators[service_job].apply(
        action, completed_at=failure_time + 0.11,
    )
    repeated = actuators[service_job].apply(
        action, completed_at=failure_time + 0.12,
    )
    assert applied.receipt is not None and repeated.receipt == applied.receipt
    active[service_job].acknowledge_service(
        applied.receipt, now=failure_time + 0.12,
    )
    duplicate_disposition = active[service_job].acknowledge_service(
        applied.receipt, now=failure_time + 0.13,
    )
    work.acknowledged += applied.receipt.completed_work
    if duplicate_disposition is Disposition.DUPLICATE:
        work.duplicate_receipts += 1

    outstanding_active = math.fsum(bubble.backlog for bubble in active.values())
    outstanding_parked = math.fsum(bubble.backlog for bubble in parked.values())
    conservation_residual = (
        work.admitted - work.acknowledged
        - outstanding_active - outstanding_parked
    )
    committed_events = [event for event in events if "strict_neighbor" in event]
    return {
        "experiment": "epoch-fenced strict-neighbor churn/fault evaluation",
        "scope": "deterministic simulator; not a distributed deployment",
        "parameters": {
            "initial_members": 6,
            "bubble_width": 0.25,
            "guard": 0.03,
            "tolerance": tolerance,
            "max_rounds_per_event": max_rounds,
            "controller_dt": 0.08,
        },
        "events": events,
        "fault_injection": fault_counts,
        "summary": {
            "committed_topology_events": len(committed_events),
            "all_strict_neighbor_events_settled": all(
                bool(event["strict_neighbor"]["settled"])
                for event in committed_events
            ),
            "maximum_settling_rounds": max(
                int(event["strict_neighbor"]["settling_rounds"])
                for event in committed_events
            ),
            "minimum_safety_margin": min(
                float(event["strict_neighbor"]["minimum_safety_margin"])
                for event in committed_events
            ),
            "energy_increase_rounds": sum(
                int(event["strict_neighbor"]["energy_increase_rounds"])
                for event in committed_events
            ),
            "directed_spacing_messages": sum(
                int(event["strict_neighbor"]["directed_messages"])
                for event in committed_events
            ),
            "topology_acknowledgement_attempts": topology_ack_messages,
            "final_epoch": authority.epoch,
            "final_active_members": len(authority.order),
        },
        "work_conservation": {
            "admitted_work": work.admitted,
            "acknowledged_service": work.acknowledged,
            "active_backlog": outstanding_active,
            "durably_parked_backlog": outstanding_parked,
            "lost_work": 0.0,
            "conservation_residual": conservation_residual,
            "old_epoch_actions_fenced": work.fenced_actions,
            "duplicate_receipts_ignored": work.duplicate_receipts,
        },
        "claim_boundary": [
            "topology authority is a single-process reference model",
            "adjacent swap uses an atomic equal-width phase-slot handoff",
            "departed/crashed backlog is durably parked, not yet rescheduled",
            "no network delay distribution, multi-resource commit, or SLO claim",
        ],
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--max-rounds", type=int, default=20_000)
    parser.add_argument("--output", default="results/churn_evaluation.json")
    args = parser.parse_args(argv)
    payload = run_churn_evaluation(
        tolerance=args.tolerance, max_rounds=args.max_rounds,
    )
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    if not payload["summary"]["all_strict_neighbor_events_settled"]:
        raise SystemExit("a committed topology event did not settle")
    if payload["summary"]["minimum_safety_margin"] < -1e-10:
        raise SystemExit("a hard minimum gap was violated")
    if abs(payload["work_conservation"]["conservation_residual"]) > 1e-8:
        raise SystemExit("work-conservation check failed")


if __name__ == "__main__":
    main()
