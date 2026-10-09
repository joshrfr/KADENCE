"""Executable counterexamples for claims the legacy mechanism cannot support."""

import copy
import math
import unittest

from kadence.oscillator import Job, couple_step, desync_arrange, peak_contention


class LegacyCounterexampleTests(unittest.TestCase):
    def test_desync_arrangement_is_independent_of_natural_frequency(self) -> None:
        jobs = [
            Job(
                jid=f"job-{i}",
                duty={"cpu": 0.10 + i * 0.01, "io": 0.04},
                height={"cpu": 0.4, "io": 0.1},
                omega=1.0,
                phase={"cpu": 0.3 * i, "io": 0.2 * i},
            )
            for i in range(4)
        ]
        changed_frequency = copy.deepcopy(jobs)
        for i, job in enumerate(changed_frequency):
            job.omega = 100.0 + 37.0 * i

        desync_arrange(jobs, ["cpu", "io"])
        desync_arrange(changed_frequency, ["cpu", "io"])

        self.assertEqual(
            [job.phase for job in jobs],
            [job.phase for job in changed_frequency],
        )

    def test_attractive_cpu_coupling_aligns_complementary_cpu_peaks(self) -> None:
        cpu_dominant = Job(
            jid="cpu-dominant",
            duty={"cpu": 0.10, "io": 0.01},
            height={"cpu": 0.70, "io": 0.10},
            phase={"cpu": 0.0},
        )
        io_dominant = Job(
            jid="io-dominant",
            duty={"cpu": 0.10, "io": 0.30},
            height={"cpu": 0.70, "io": 0.70},
            phase={"cpu": math.pi + 1e-3},
        )
        jobs = [cpu_dominant, io_dominant]
        initial_peak = peak_contention(jobs, ["cpu"], period=100.0)["cpu"]

        for _ in range(2_000):
            couple_step(jobs, ["cpu"], K=0.6, dt=0.05)

        final_peak = peak_contention(jobs, ["cpu"], period=100.0)["cpu"]
        final_gap = abs(
            ((cpu_dominant.phase["cpu"] - io_dominant.phase["cpu"] + math.pi)
             % (2.0 * math.pi)) - math.pi
        )

        self.assertLess(final_gap, 1e-6)
        self.assertGreater(final_peak, 1.9 * initial_peak)


if __name__ == "__main__":
    unittest.main()
