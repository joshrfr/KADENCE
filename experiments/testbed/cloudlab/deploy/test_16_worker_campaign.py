"""Fast, local campaign validation tests; no CloudLab access required."""
import json
from pathlib import Path
import tempfile
import unittest

from run_16_worker_campaign import JOBS, LOSSES, REPS, atomic_json, cell_name, schedule, valid_result
from analyze_16_worker_campaign import summarize
from experiments.testbed.cloudlab.kadence_node import _order_preserved


class CampaignTests(unittest.TestCase):
    def test_order_classifier_accepts_rotation_but_rejects_overtake(self):
        self.assertTrue(_order_preserved([5.9, 0.2, 1.1, 3.0]))
        self.assertFalse(_order_preserved([0.1, 2.0, 1.0, 3.0]))

    def test_schedule_is_paired_complete_and_deterministic(self):
        rows = schedule(1234)
        self.assertEqual(len(rows), len(JOBS) * len(LOSSES) * REPS)
        self.assertEqual(rows, schedule(1234))
        self.assertEqual(len({cell_name(x) for x in rows}), len(rows))
        for rep in range(REPS):
            subset = [x for x in rows if x["rep"] == rep]
            self.assertEqual({(x["jobs"], x["loss"]) for x in subset},
                             {(j, l) for j in JOBS for l in LOSSES})
            self.assertEqual({x["seed"] for x in subset}, {1234 + rep * 1000})
            self.assertEqual({x["alpha"] for x in subset}, {1.0})

    def test_incomplete_worker_repetition_rejected(self):
        row = {"jobs": 8, "loss": 0.1}
        doc = {"provenance": {"mode": "ssh", "jobs_per_ring": 8,
                              "rings_per_node": 1, "reps": 1, "loss": 0.1,
                              "jitter": 0.02, "alpha": 1.0},
               "reps": [{"nodes_reporting": 15}]}
        self.assertTrue(any("16 reporting" in error for error in valid_result(doc, row)))

    def test_missing_cells_never_imputed(self):
        rows = schedule(1234)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atomic_json(root / "manifest.json", {
                "experiment_id": "synthetic", "kind": "test", "not_measured": [],
                "schedule": rows, "jobs_per_ring": list(JOBS),
                "loss_levels": list(LOSSES), "repetitions_per_condition": REPS})
            row = rows[0]
            name = cell_name(row)
            (root / "raw").mkdir()
            (root / "status").mkdir()
            atomic_json(root / "status" / f"{name}.json", {"returncode": 0, "errors": []})
            atomic_json(root / "raw" / f"{name}.json", {
                "provenance": {"mode": "ssh", "jobs_per_ring": row["jobs"],
                               "rings_per_node": 1, "reps": 1,
                               "loss": row["loss"], "jitter": 0.02, "alpha": 1.0},
                "reps": [{"nodes_reporting": 16, "impl": "python",
                          "across_node_final_pct_of_fair": {"median": 3.0, "p95": 4.0},
                          "fraction_nodes_converged_loose_0p1_target": 0.75,
                          "fraction_nodes_converged_strict_1e6": 0.25,
                          "classification_counts": {"converged": 12, "order-broken": 4},
                          "measured_datagrams_sent": 100,
                          "measured_datagrams_received": 90,
                          "measured_ticks_executed": 50,
                          "cross_node_messages_measured": 0,
                          "wall_s": 30.0}]})
            summary = summarize(root)
            self.assertEqual(summary["valid_cells"], 1)
            self.assertEqual(summary["expected_cells"], 45)
            self.assertFalse(summary["complete"])
            self.assertEqual(len(summary["failures"]), 44)
            self.assertEqual(sum(c["n_valid"] for c in summary["conditions"]), 1)
            valid_condition = next(c for c in summary["conditions"] if c["n_valid"])
            self.assertEqual(valid_condition["classification_counts"]["order-broken"], 4)


if __name__ == "__main__":
    unittest.main()
