import unittest

from kadence.neighbor_gossip import (
    BubbleSpec,
    InfeasibleAdmission,
    NeighborSnapshot,
    RingController,
    TWO_PI,
    admit_ring,
    limit_local_displacement,
    local_correction,
    perturb_gaps,
    phases_from_gaps,
)


class AdmissionTests(unittest.TestCase):
    def test_plan_closes_ring_and_preserves_hard_minima(self) -> None:
        bubbles = (
            BubbleSpec("a", 0.45),
            BubbleSpec("b", 0.70),
            BubbleSpec("c", 0.30),
            BubbleSpec("d", 0.55),
        )
        plan = admit_ring(bubbles, guard=0.04)

        self.assertAlmostEqual(sum(plan.desired), TWO_PI, places=12)
        self.assertTrue(all(
            desired >= minimum
            for desired, minimum in zip(plan.desired, plan.minimum)
        ))
        self.assertEqual(plan.desired[:-1], plan.minimum[:-1])
        self.assertGreater(plan.desired[-1], plan.minimum[-1])
        self.assertEqual(plan.order, ("a", "b", "c", "d"))

        first_edge_plan = admit_ring(
            bubbles, guard=0.04, slack_weights=(1.0, 0.0, 0.0, 0.0),
        )
        self.assertGreater(first_edge_plan.desired[0], first_edge_plan.minimum[0])
        self.assertEqual(first_edge_plan.desired[1:], first_edge_plan.minimum[1:])

    def test_rejects_overloaded_lane_instead_of_silent_loss(self) -> None:
        bubbles = tuple(BubbleSpec(str(i), 2.0) for i in range(4))
        with self.assertRaises(InfeasibleAdmission):
            admit_ring(bubbles, guard=0.1)


class LocalControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bubbles = (
            BubbleSpec("a", 0.40),
            BubbleSpec("b", 0.75),
            BubbleSpec("c", 0.30),
            BubbleSpec("d", 0.60),
            BubbleSpec("e", 0.50),
        )
        self.plan = admit_ring(self.bubbles, guard=0.05)
        self.initial_gaps = perturb_gaps(
            self.plan.desired,
            ((4, 0, 0.20), (4, 2, 0.15), (4, 1, 0.10)),
        )
        self.initial_phases = phases_from_gaps(self.initial_gaps)

    def test_pure_update_uses_the_two_adjacent_gap_errors(self) -> None:
        left = NeighborSnapshot("left", 5.5, 0.2, 2, 10)
        current = NeighborSnapshot("self", 0.2, 0.2, 2, 10)
        right = NeighborSnapshot("right", 1.4, 0.2, 2, 10)

        correction = local_correction(
            left,
            current,
            right,
            left_target=0.8,
            right_target=1.0,
            left_gain=2.0,
            right_gain=3.0,
        )
        left_error = ((0.2 - 5.5) % TWO_PI) - 0.8
        right_error = ((1.4 - 0.2) % TWO_PI) - 1.0
        self.assertAlmostEqual(correction, 3.0 * right_error - 2.0 * left_error)

        limited = limit_local_displacement(
            10.0,
            left,
            current,
            right,
            left_minimum=0.7,
            right_minimum=1.0,
            safety_fraction=0.5,
        )
        self.assertAlmostEqual(limited, 0.5 * (1.2 - 1.0))

    def test_energy_falls_and_gaps_converge(self) -> None:
        controller = RingController(
            self.bubbles, self.initial_phases, self.plan,
        )
        prior = controller.energy()
        for _ in range(250):
            report = controller.step(dt=0.08)
            self.assertLessEqual(report.energy_after, prior + 1e-12)
            prior = report.energy_after

        controller.run(tolerance=1e-8, dt=0.08)
        for actual, desired in zip(controller.gaps(), self.plan.desired):
            self.assertAlmostEqual(actual, desired, places=7)

    def test_local_limiter_preserves_every_admitted_minimum(self) -> None:
        controller = RingController(
            self.bubbles, self.initial_phases, self.plan,
        )
        for _ in range(500):
            controller.step(dt=1.0, safety_fraction=0.45)
            self.assertTrue(all(
                gap + 1e-10 >= minimum
                for gap, minimum in zip(controller.gaps(), self.plan.minimum)
            ))

    def test_common_vibration_rotates_without_changing_relative_motion(self) -> None:
        stationary = RingController(
            self.bubbles, self.initial_phases, self.plan,
            common_frequency=0.0,
        )
        vibrating = RingController(
            self.bubbles, self.initial_phases, self.plan,
            common_frequency=1.7,
        )
        for _ in range(80):
            stationary.step(dt=0.05)
            vibrating.step(dt=0.05)
        for still_gap, moving_gap in zip(stationary.gaps(), vibrating.gaps()):
            self.assertAlmostEqual(still_gap, moving_gap, places=11)

    def test_each_round_sends_exactly_two_frames_per_job(self) -> None:
        controller = RingController(
            self.bubbles, self.initial_phases, self.plan,
        )
        rounds = 17
        for _ in range(rounds):
            report = controller.step(dt=0.05)
            self.assertEqual(report.directed_messages, 2 * len(self.bubbles))
        self.assertEqual(controller.message_count, 2 * len(self.bubbles) * rounds)


if __name__ == "__main__":
    unittest.main()
