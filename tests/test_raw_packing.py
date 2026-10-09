import unittest

import numpy as np

from experiments.simulation import raw_packing as rp
from experiments.simulation.packing_fine import paired_gain


def _pair():
    t = np.arange(48)
    a = 0.5 * (1 + np.cos(2 * np.pi * t / 48))            # peak at t=0
    x = np.zeros((2, 2, 48))
    x[0, 0] = a
    x[1, 0] = a                                          # same rhythm, same timing
    x[:, 1] = 0.1
    return x


class RawPackingTests(unittest.TestCase):
    def test_shifting_complementary_rhythms_lowers_peak(self):
        x = _pair()
        self.assertGreater(rp.peak(x, rp.arrange_fixed(x)), 1.9)
        self.assertLessEqual(rp.peak(x, rp.arrange_coupled(x)), 1.2)
        self.assertLessEqual(rp.peak(x, rp.arrange_greedy(x)), 1.2)

    def test_rr_spreads_by_index(self):
        x = np.zeros((4, 2, 48))
        self.assertEqual(list(rp.arrange_rr(x)), [0, 12, 24, 36])

    def test_common_mode_removal_keeps_shape_and_nonnegativity(self):
        x = _pair()
        y = rp.remove_common_mode(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue((y >= 0).all())

    def test_capacity_counts_until_first_failure(self):
        x = _pair()
        order = np.array([0, 1])
        self.assertEqual(
            rp.capacity(x, order, rp.arrange_fixed, 1.5, 2), 1
        )
        self.assertEqual(
            rp.capacity(x, order, rp.arrange_coupled, 1.5, 2), 2
        )

    def test_paired_gain_counts(self):
        gain = paired_gain([4, 4, 4], [5, 4, 3])
        self.assertEqual(
            (
                gain["seeds_new_better"],
                gain["seeds_tied"],
                gain["seeds_new_worse"],
            ),
            (1, 1, 1),
        )
        self.assertEqual(gain["gain_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
