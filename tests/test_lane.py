import unittest

from kadence.neighbor_gossip import BubbleSpec, InfeasibleAdmission
from kadence.pressure_fire import PressureConfig
from kadence.lane import Arrival, Lane, UndrainedWork


def make_lane(*, lease_duration: float = 10.0) -> Lane:
    return Lane(
        tuple(BubbleSpec(job_id, 0.3) for job_id in ("a", "b", "c", "d")),
        pressure_config=PressureConfig(
            pressure_scale=1.0,
            fire_threshold=0.5,
            min_inter_fire=0.1,
            max_phase_response=0.02,
            response_window=0.5,
        ),
        lease_duration=lease_duration,
    )


class LaneTests(unittest.TestCase):
    def test_step_composes_local_control_pressure_and_receipted_service(self) -> None:
        lane = make_lane()
        report = lane.step(
            dt=1.0,
            service_capacity=4.0,
            service_quantum=1.0,
            arrivals=(
                Arrival("a", "arrival:a", 1.0),
                Arrival("b", "arrival:b", 1.0),
                Arrival("c", "arrival:c", 1.0),
                Arrival("d", "arrival:d", 1.0),
            ),
        )

        self.assertEqual(set(report.fired_jobs), set(lane.order))
        self.assertEqual(report.pulses_sent, 2 * len(lane.order))
        self.assertEqual(report.continuous_neighbor_messages, 2 * len(lane.order))
        self.assertEqual(report.token_hops, len(lane.order))
        self.assertAlmostEqual(report.acknowledged_service, 4.0)
        self.assertAlmostEqual(report.total_backlog, 0.0)
        self.assertAlmostEqual(report.conservation_residual, 0.0)
        self.assertGreaterEqual(report.minimum_safety_margin, -1e-9)

    def test_join_swap_and_graceful_leave_are_epoch_fenced(self) -> None:
        lane = make_lane()

        joined = lane.join(
            "join-e", job_id="e", left_job="d", right_job="a",
            width=0.3, now=1.0,
        )
        self.assertEqual(joined.epoch_after, joined.epoch_before + 1)
        self.assertEqual(lane.order, ("a", "b", "c", "d", "e"))

        swapped = lane.swap(
            "swap-b-c", left_job="b", right_job="c", now=2.0,
        )
        self.assertEqual(swapped.epoch_after, joined.epoch_after + 1)
        self.assertEqual(lane.order, ("a", "c", "b", "d", "e"))

        left = lane.leave("leave-e", job_id="e", now=3.0)
        self.assertEqual(left.epoch_after, swapped.epoch_after + 1)
        self.assertEqual(lane.order, ("a", "c", "b", "d"))
        for job_id in lane.order:
            lease = lane.neighbor_lease(job_id)
            self.assertEqual(lease.epoch, left.epoch_after)
            self.assertEqual(lane.bubbles[job_id].epoch, left.epoch_after)
            self.assertEqual(lane.actuators[job_id].epoch, left.epoch_after)
        self.assertAlmostEqual(lane.conservation_residual, 0.0)

    def test_graceful_leave_rejects_undrained_work_without_mutation(self) -> None:
        lane = make_lane()
        lane.admit_arrival(Arrival("b", "held", 2.0))
        epoch = lane.topology.epoch
        order = lane.order

        with self.assertRaises(UndrainedWork):
            lane.leave("unsafe-leave", job_id="b", now=0.5)

        self.assertEqual(lane.topology.epoch, epoch)
        self.assertEqual(lane.order, order)
        self.assertAlmostEqual(lane.total_backlog, 2.0)
        self.assertAlmostEqual(lane.conservation_residual, 0.0)

    def test_failed_leave_parks_and_restore_recovers_durable_work(self) -> None:
        lane = make_lane(lease_duration=1.0)
        lane.admit_arrival(Arrival("b", "durable", 2.5))
        for job_id in ("a", "c", "d"):
            lane.heartbeat(job_id, now=0.9)

        failed = lane.leave(
            "fail-b", job_id="b", now=1.0, failed=True,
        )
        self.assertEqual(failed.operation, "failed_leave")
        self.assertAlmostEqual(failed.stranded_work, 2.5)
        self.assertNotIn("b", lane.order)
        self.assertAlmostEqual(lane.total_backlog, 2.5)
        self.assertAlmostEqual(lane.conservation_residual, 0.0)

        restored = lane.join(
            "restore-b", job_id="b", left_job="d", right_job="a", now=1.1,
        )
        self.assertEqual(restored.operation, "restore")
        self.assertIn("b", lane.order)
        self.assertAlmostEqual(restored.stranded_work, 0.0)
        self.assertAlmostEqual(lane.bubbles["b"].backlog, 2.5)
        self.assertAlmostEqual(lane.conservation_residual, 0.0)

    def test_infeasible_join_fails_before_prepare(self) -> None:
        lane = make_lane()
        epoch = lane.topology.epoch
        order = lane.order

        with self.assertRaises(InfeasibleAdmission):
            lane.join(
                "too-wide", job_id="x", left_job="d", right_job="a",
                width=10.0, now=1.0,
            )

        self.assertEqual(lane.topology.epoch, epoch)
        self.assertEqual(lane.order, order)

    def test_repeated_bursts_preserve_work_and_every_hard_gap(self) -> None:
        lane = make_lane()
        admitted = 0.0
        acknowledged = 0.0
        for tick in range(1, 101):
            arrivals = []
            for index, job_id in enumerate(lane.order):
                if (tick + index) % (index + 2) == 0:
                    work = 0.05 * (index + 1)
                    admitted += work
                    arrivals.append(
                        Arrival(job_id, f"burst:{tick}:{job_id}", work)
                    )
            report = lane.step(
                dt=0.1,
                service_capacity=0.3,
                service_quantum=0.1,
                arrivals=arrivals,
            )
            acknowledged += report.acknowledged_service
            self.assertGreaterEqual(report.minimum_safety_margin, -1e-8)
            self.assertAlmostEqual(report.conservation_residual, 0.0, places=8)

        self.assertAlmostEqual(lane.total_admitted_work, admitted, places=8)
        self.assertAlmostEqual(
            lane.total_acknowledged_work, acknowledged, places=8,
        )
        self.assertAlmostEqual(
            admitted, acknowledged + lane.total_backlog, places=8,
        )


if __name__ == "__main__":
    unittest.main()
