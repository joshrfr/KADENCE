import gzip
import os
import tempfile
import unittest

from experiments.simulation.raw_trace_replay import (
    TraceFormatError,
    evaluate_trace,
    extract_google_slice,
    group_intervals,
    parse_google_task_usage,
    replay_policy,
    sha256_file,
)


FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "google_task_usage_synthetic.csv"
)


class GoogleTraceParserTests(unittest.TestCase):
    def test_parse_identity_work_and_gzip(self) -> None:
        samples = list(parse_google_task_usage(FIXTURE))
        self.assertEqual(len(samples), 16)
        self.assertEqual(samples[0].task_key, "101:0")
        self.assertEqual(samples[0].duration_seconds, 300.0)
        self.assertAlmostEqual(samples[0].cpu_work, 30.0)

        with tempfile.TemporaryDirectory() as directory:
            compressed = os.path.join(directory, "fixture.csv.gz")
            with open(FIXTURE, "rb") as source, gzip.open(compressed, "wb") as dest:
                dest.write(source.read())
            zipped = list(parse_google_task_usage(compressed))
        self.assertEqual(zipped, samples)

    def test_filter_and_exact_extraction(self) -> None:
        filtered = list(parse_google_task_usage(
            FIXTURE,
            task_filter=frozenset(("101:0", "103:0")),
            start_us=300_000_000,
            end_us=900_000_000,
        ))
        self.assertEqual(len(filtered), 4)
        self.assertTrue(all(
            sample.task_key in {"101:0", "103:0"} for sample in filtered
        ))

        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "slice.csv")
            report = extract_google_slice(
                FIXTURE,
                output,
                tasks=("101:0", "103:0"),
                start_us=300_000_000,
                end_us=900_000_000,
            )
            extracted = list(parse_google_task_usage(output))
            self.assertEqual(report.rows, 4)
            self.assertEqual(report.sha256, sha256_file(output))
            self.assertEqual(extracted, filtered)

    def test_missing_measurement_is_rejected_not_imputed(self) -> None:
        row = "0,300000000,1,0,1,,0.2,0,0,0,0,0,0,0,0,0,0,0,0,0\n"
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "bad.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(row)
            with self.assertRaises(TraceFormatError):
                list(parse_google_task_usage(path))


class ReplayTests(unittest.TestCase):
    def test_every_policy_conserves_work_and_capacity(self) -> None:
        intervals = group_intervals(parse_google_task_usage(FIXTURE))
        for policy in (
            "max-min-fair", "max-pressure", "neighbor-token-scaffold"
        ):
            result = replay_policy(intervals, policy, capacity=0.20)
            self.assertAlmostEqual(result["conservation_residual"], 0.0)
            self.assertEqual(result["lost_work"], 0.0)
            self.assertLessEqual(
                result["acknowledged_service"],
                0.20 * 300.0 * len(intervals) + 1e-9,
            )
        local = replay_policy(
            intervals, "neighbor-token-scaffold", capacity=0.20
        )
        self.assertEqual(local["directed_neighbor_messages"], 2 * 4 * 4)

    def test_evidence_label_prevents_fixture_raw_claim(self) -> None:
        payload = evaluate_trace(
            FIXTURE,
            data_kind="synthetic-fixture",
            capacity_fraction_of_peak=0.75,
        )
        self.assertIn("not raw-trace evidence", payload["evidence_level"])
        self.assertEqual(payload["data"]["rows"], 16)
        for result in payload["policies"].values():
            self.assertLess(abs(result["conservation_residual"]), 1e-7)

    def test_unknown_provenance_label_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_trace(FIXTURE, data_kind="raw")


if __name__ == "__main__":
    unittest.main()
