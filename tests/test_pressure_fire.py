import json
import unittest

from kadence.neighbor_gossip import NeighborSnapshot, forward_gap
from kadence.pressure_fire import (
    Disposition,
    FencedActuator,
    PressureBubble,
    PressureConfig,
    PressurePulse,
    ProtocolViolation,
    ServiceAction,
    ServiceReceipt,
)


class PressureFixture(unittest.TestCase):
    def make_bubble(
        self,
        *,
        phase: float = 1.0,
        epoch: int = 2,
        config: PressureConfig | None = None,
    ) -> PressureBubble:
        return PressureBubble(
            "job",
            width=0.4,
            phase=phase,
            predecessor="left",
            successor="right",
            config=config or PressureConfig(
                pressure_scale=1.0,
                fire_threshold=1.0,
                min_inter_fire=1.0,
                max_phase_response=0.2,
                response_window=1.0,
            ),
            epoch=epoch,
            now=0.0,
        )


class WorkConservationTests(PressureFixture):
    def test_only_an_idempotent_acknowledgement_removes_work(self) -> None:
        bubble = self.make_bubble()
        self.assertEqual(
            bubble.admit_arrival("arrival-1", 5.0, now=0.0),
            Disposition.APPLIED,
        )
        self.assertEqual(
            bubble.admit_arrival("arrival-1", 5.0, now=0.1),
            Disposition.DUPLICATE,
        )
        self.assertEqual(bubble.backlog, 5.0)

        action = bubble.issue_service(3.0, now=0.1)
        self.assertEqual(bubble.backlog, 5.0)
        self.assertEqual(bubble.pending_work, 3.0)
        self.assertEqual(bubble.available_work, 2.0)
        with self.assertRaises(ValueError):
            bubble.issue_service(2.1, now=0.1)

        actuator = FencedActuator("job", epoch=2)
        result = actuator.apply(
            action, completed_work=2.0, completed_at=0.2,
        )
        self.assertEqual(result.disposition, Disposition.APPLIED)
        self.assertIsNotNone(result.receipt)
        self.assertEqual(bubble.backlog, 5.0)

        self.assertEqual(
            bubble.acknowledge_service(result.receipt, now=0.2),
            Disposition.APPLIED,
        )
        self.assertEqual(bubble.backlog, 3.0)
        self.assertEqual(bubble.pending_work, 0.0)

        self.assertEqual(
            bubble.acknowledge_service(result.receipt, now=0.3),
            Disposition.DUPLICATE,
        )
        self.assertEqual(bubble.backlog, 3.0)
        self.assertAlmostEqual(bubble.admitted_work, 5.0)
        self.assertAlmostEqual(bubble.acknowledged_work, 2.0)
        self.assertAlmostEqual(bubble.conservation_residual, 0.0)
        with self.assertRaises(ProtocolViolation):
            bubble.admit_arrival("arrival-1", 6.0, now=0.3)

    def test_fenced_or_cancelled_action_never_consumes_backlog(self) -> None:
        bubble = self.make_bubble()
        bubble.admit_arrival("a", 2.0, now=0.0)
        action = bubble.issue_service(1.0, now=0.0)
        self.assertEqual(
            bubble.cancel_service(action.action_id, now=0.1),
            Disposition.CANCELLED,
        )
        self.assertEqual(bubble.backlog, 2.0)

        receipt = ServiceReceipt(
            action_id=action.action_id,
            job_id=action.job_id,
            epoch=action.epoch,
            sequence=action.sequence,
            requested_work=action.requested_work,
            completed_work=1.0,
            completed_at=0.2,
        )
        self.assertEqual(
            bubble.acknowledge_service(receipt, now=0.2),
            Disposition.UNKNOWN_ACTION,
        )
        self.assertEqual(bubble.backlog, 2.0)


class FencedActuatorTests(unittest.TestCase):
    def test_exact_retry_is_cached_but_unknown_old_messages_are_fenced(self) -> None:
        actuator = FencedActuator("job", epoch=3)
        action = ServiceAction("a-2", "job", 3, 2, 4.0, 1.0)
        first = actuator.apply(action, completed_work=3.0, completed_at=1.5)
        retry = actuator.apply(action, completed_work=4.0, completed_at=9.0)
        self.assertEqual(first.disposition, Disposition.APPLIED)
        self.assertEqual(retry.disposition, Disposition.DUPLICATE)
        self.assertEqual(retry.receipt, first.receipt)

        stale_sequence = ServiceAction("a-1", "job", 3, 1, 1.0, 1.6)
        self.assertEqual(
            actuator.apply(stale_sequence, completed_at=1.7).disposition,
            Disposition.STALE_SEQUENCE,
        )

        actuator.advance_epoch(4)
        old_unknown = ServiceAction("a-3", "job", 3, 3, 1.0, 1.6)
        self.assertEqual(
            actuator.apply(old_unknown, completed_at=1.7).disposition,
            Disposition.FENCED_EPOCH,
        )
        # A known retry still resolves to its original effect after fencing.
        self.assertEqual(
            actuator.apply(action, completed_at=10.0).receipt,
            first.receipt,
        )

    def test_actuator_checkpoint_preserves_exactly_once_receipt(self) -> None:
        actuator = FencedActuator("job", epoch=7)
        action = ServiceAction("durable", "job", 7, 4, 2.0, 3.0)
        first = actuator.apply(action, completed_work=1.5, completed_at=3.5)

        recovered = FencedActuator.load_state(actuator.dump_state())
        duplicate = recovered.apply(action, completed_at=20.0)
        self.assertEqual(duplicate.disposition, Disposition.DUPLICATE)
        self.assertEqual(duplicate.receipt, first.receipt)

        conflict = ServiceAction("durable", "job", 7, 4, 9.0, 3.0)
        with self.assertRaises(ProtocolViolation):
            recovered.apply(conflict, completed_at=20.0)


class IntegrateAndFireTests(PressureFixture):
    def test_fires_only_to_two_neighbors_and_obeys_refractory_time(self) -> None:
        bubble = self.make_bubble()
        bubble.admit_arrival("burst", 2.0, now=0.0)

        self.assertEqual(bubble.maybe_fire(now=0.49), ())
        pulses = bubble.maybe_fire(now=0.5)
        self.assertEqual(len(pulses), 2)
        self.assertEqual(
            {pulse.recipient for pulse in pulses}, {"left", "right"},
        )
        self.assertTrue(all(pulse.sender == "job" for pulse in pulses))
        self.assertEqual({pulse.sequence for pulse in pulses}, {1})
        self.assertEqual({pulse.epoch for pulse in pulses}, {2})

        # Sufficient membrane charge is present, but no second event can fire
        # at the same instant or within the one-second refractory interval.
        self.assertEqual(bubble.maybe_fire(now=0.5), ())
        self.assertEqual(bubble.maybe_fire(now=1.49), ())
        second = bubble.maybe_fire(now=1.5)
        self.assertEqual(len(second), 2)
        self.assertEqual({pulse.sequence for pulse in second}, {2})
        self.assertGreaterEqual(
            second[0].fired_at - pulses[0].fired_at,
            bubble.config.min_inter_fire,
        )

    def test_pressure_without_work_cannot_fire(self) -> None:
        bubble = self.make_bubble()
        bubble.set_deadline_pressure(100.0, now=0.0)
        self.assertEqual(bubble.maybe_fire(now=100.0), ())
        self.assertEqual(bubble.membrane, 0.0)

    def test_deadline_pressure_changes_future_charge_not_past_charge(self) -> None:
        bubble = self.make_bubble()
        bubble.admit_arrival("small", 0.25, now=0.0)
        bubble.set_deadline_pressure(0.75, now=1.0)
        self.assertAlmostEqual(bubble.membrane, 0.25)
        self.assertEqual(bubble.maybe_fire(now=1.74), ())
        self.assertEqual(len(bubble.maybe_fire(now=1.75)), 2)


class PulseResponseTests(PressureFixture):
    def setUp(self) -> None:
        self.bubble = self.make_bubble()
        self.left = NeighborSnapshot("left", 0.1, 0.4, 2, 3)
        self.right = NeighborSnapshot("right", 1.85, 0.4, 2, 4)

    def pulse(self, **changes: object) -> PressurePulse:
        values = {
            "sender": "left",
            "recipient": "job",
            "epoch": 2,
            "sequence": 5,
            "phase": 0.1,
            "width": 0.4,
            "pressure": 1.0,
            "deadline_class": "urgent",
            "fired_at": 1.0,
        }
        values.update(changes)
        return PressurePulse(**values)

    def test_response_is_bounded_and_preserves_both_hard_gaps(self) -> None:
        result = self.bubble.receive_pulse(
            self.pulse(),
            left=self.left,
            right=self.right,
            left_minimum=0.8,
            right_minimum=0.8,
        )
        self.assertEqual(result.disposition, Disposition.APPLIED)
        self.assertGreater(result.applied_displacement, 0.0)
        self.assertLessEqual(
            abs(result.requested_displacement),
            self.bubble.config.max_phase_response,
        )
        # The right edge had only 0.05 radians of slack, so the inherited
        # local limiter is tighter than the response-amplitude bound.
        self.assertLessEqual(result.applied_displacement, 0.45 * 0.05 + 1e-12)
        self.assertGreaterEqual(
            forward_gap(self.left.phase, self.bubble.phase), 0.8,
        )
        self.assertGreaterEqual(
            forward_gap(self.bubble.phase, self.right.phase), 0.8,
        )

    def test_duplicate_stale_wrong_and_non_neighbor_pulses_do_not_move(self) -> None:
        accepted = self.pulse()
        self.bubble.receive_pulse(
            accepted,
            left=self.left,
            right=self.right,
            left_minimum=0.8,
            right_minimum=0.8,
        )
        phase = self.bubble.phase

        duplicate = self.bubble.receive_pulse(
            accepted,
            left=self.left,
            right=self.right,
            left_minimum=0.8,
            right_minimum=0.8,
        )
        self.assertEqual(duplicate.disposition, Disposition.DUPLICATE)
        self.assertEqual(self.bubble.phase, phase)

        stale = self.bubble.receive_pulse(
            self.pulse(sequence=4),
            left=self.left,
            right=self.right,
            left_minimum=0.8,
            right_minimum=0.8,
        )
        self.assertEqual(stale.disposition, Disposition.STALE_SEQUENCE)
        old_epoch = self.bubble.receive_pulse(
            self.pulse(epoch=1, sequence=9),
            left=self.left,
            right=self.right,
            left_minimum=0.8,
            right_minimum=0.8,
        )
        self.assertEqual(old_epoch.disposition, Disposition.FENCED_EPOCH)
        outsider = self.bubble.receive_pulse(
            self.pulse(sender="outsider", sequence=9),
            left=self.left,
            right=self.right,
            left_minimum=0.8,
            right_minimum=0.8,
        )
        self.assertEqual(outsider.disposition, Disposition.NOT_NEIGHBOR)
        wrong = self.bubble.receive_pulse(
            self.pulse(recipient="someone-else", sequence=9),
            left=self.left,
            right=self.right,
            left_minimum=0.8,
            right_minimum=0.8,
        )
        self.assertEqual(wrong.disposition, Disposition.WRONG_RECIPIENT)
        self.assertEqual(self.bubble.phase, phase)

        with self.assertRaises(ProtocolViolation):
            self.bubble.receive_pulse(
                self.pulse(pressure=2.0),
                left=self.left,
                right=self.right,
                left_minimum=0.8,
                right_minimum=0.8,
            )

    def test_pulse_dedupe_survives_recovery(self) -> None:
        accepted = self.pulse()
        self.bubble.receive_pulse(
            accepted,
            left=self.left,
            right=self.right,
            left_minimum=0.8,
            right_minimum=0.8,
        )
        recovered = PressureBubble.load_state(self.bubble.dump_state())
        phase = recovered.phase
        duplicate = recovered.receive_pulse(
            accepted,
            left=self.left,
            right=self.right,
            left_minimum=0.8,
            right_minimum=0.8,
        )
        self.assertEqual(duplicate.disposition, Disposition.DUPLICATE)
        self.assertEqual(recovered.phase, phase)


class RecoveryAndEpochTests(PressureFixture):
    def test_recovery_preserves_backlog_pending_actions_and_dedupe(self) -> None:
        bubble = self.make_bubble()
        bubble.admit_arrival("arrival", 4.0, now=0.0)
        fired = bubble.maybe_fire(now=0.25)
        self.assertEqual(len(fired), 2)
        action = bubble.issue_service(2.0, now=0.25)

        recovered = PressureBubble.load_state(bubble.dump_state())
        self.assertEqual(recovered.backlog, 4.0)
        self.assertEqual(recovered.pending_work, 2.0)
        self.assertEqual(
            recovered.admit_arrival("arrival", 4.0, now=0.3),
            Disposition.DUPLICATE,
        )
        self.assertEqual(recovered.maybe_fire(now=0.9), ())

        actuator = FencedActuator("job", epoch=2)
        receipt = actuator.apply(
            action, completed_work=2.0, completed_at=1.0,
        ).receipt
        self.assertIsNotNone(receipt)
        self.assertEqual(
            recovered.acknowledge_service(receipt, now=1.0),
            Disposition.APPLIED,
        )
        self.assertEqual(recovered.backlog, 2.0)

        recovered_again = PressureBubble.load_state(recovered.dump_state())
        self.assertEqual(
            recovered_again.acknowledge_service(receipt, now=1.1),
            Disposition.DUPLICATE,
        )
        self.assertEqual(recovered_again.backlog, 2.0)
        self.assertAlmostEqual(recovered_again.conservation_residual, 0.0)

    def test_new_topology_fences_pulses_but_preserves_old_action_receipt(self) -> None:
        bubble = self.make_bubble()
        bubble.admit_arrival("arrival", 2.0, now=0.0)
        action = bubble.issue_service(1.0, now=0.0)
        actuator = FencedActuator("job", epoch=2)
        receipt = actuator.apply(action, completed_at=0.1).receipt
        self.assertIsNotNone(receipt)

        bubble.install_topology(
            epoch=3,
            predecessor="new-left",
            successor="new-right",
            now=0.1,
        )
        self.assertEqual(bubble.backlog, 2.0)
        # A receipt proves an already accepted old-epoch effect and therefore
        # remains consumable by exact action id after the topology changes.
        self.assertEqual(
            bubble.acknowledge_service(receipt, now=0.2),
            Disposition.APPLIED,
        )
        self.assertEqual(bubble.backlog, 1.0)

        old_pulse = PressurePulse(
            "left", "job", 2, 9, 0.1, 0.4, 1.0, "urgent", 0.1,
        )
        result = bubble.receive_pulse(
            old_pulse,
            left=NeighborSnapshot("new-left", 0.0, 0.4, 3, 1),
            right=NeighborSnapshot("new-right", 2.0, 0.4, 3, 1),
            left_minimum=0.5,
            right_minimum=0.5,
        )
        self.assertEqual(result.disposition, Disposition.FENCED_EPOCH)

    def test_corrupt_checkpoint_that_loses_work_fails_closed(self) -> None:
        bubble = self.make_bubble()
        bubble.admit_arrival("arrival", 3.0, now=0.0)
        checkpoint = json.loads(bubble.dump_state())
        checkpoint["backlog"] = 2.0
        with self.assertRaisesRegex(ValueError, "work conservation"):
            PressureBubble.load_state(json.dumps(checkpoint))


if __name__ == "__main__":
    unittest.main()
