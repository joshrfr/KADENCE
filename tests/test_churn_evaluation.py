import unittest

from experiments.simulation.churn_evaluation import run_churn_evaluation


class ChurnEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = run_churn_evaluation(tolerance=1e-6, max_rounds=20_000)

    def test_committed_events_settle_safely_with_exact_message_accounting(self) -> None:
        self.assertTrue(
            self.payload["summary"]["all_strict_neighbor_events_settled"]
        )
        self.assertGreaterEqual(
            self.payload["summary"]["minimum_safety_margin"], -1e-10
        )
        self.assertEqual(self.payload["summary"]["energy_increase_rounds"], 0)
        for event in self.payload["events"]:
            if "strict_neighbor" not in event:
                continue
            result = event["strict_neighbor"]
            expected = 2 * event["members"] * result["settling_rounds"]
            self.assertEqual(result["directed_messages"], expected)

    def test_faults_are_fenced_or_aborted(self) -> None:
        faults = self.payload["fault_injection"]
        self.assertEqual(faults["aborted_transactions"], 1)
        self.assertEqual(faults["stale_topology_frames_fenced"], 1)
        self.assertEqual(faults["duplicate_topology_frames"], 1)
        self.assertEqual(faults["stale_pressure_pulses_fenced"], 1)
        self.assertEqual(faults["old_epoch_actions_fenced"], 1)
        timeout = next(
            event for event in self.payload["events"]
            if event["event"] == "join-timeout"
        )
        self.assertTrue(timeout["old_order_preserved"])

    def test_admitted_work_is_neither_dropped_nor_double_subtracted(self) -> None:
        work = self.payload["work_conservation"]
        self.assertEqual(work["lost_work"], 0.0)
        self.assertAlmostEqual(work["conservation_residual"], 0.0, places=9)
        self.assertGreater(work["durably_parked_backlog"], 0.0)
        self.assertEqual(work["old_epoch_actions_fenced"], 1)
        self.assertEqual(work["duplicate_receipts_ignored"], 1)

    def test_frozen_baseline_exposes_nontrivial_recovery(self) -> None:
        disturbed = [
            event for event in self.payload["events"]
            if "frozen_no_coupling" in event
            and not event["frozen_no_coupling"]["settled"]
        ]
        self.assertGreaterEqual(len(disturbed), 2)
        self.assertTrue(all(
            event["centralized_slot_oracle"]["settled"]
            for event in disturbed
        ))


if __name__ == "__main__":
    unittest.main()
