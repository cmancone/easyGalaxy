import unittest
import ezgal.astro_filter
import tests
import numpy as np
import math


class test_calc_mags(unittest.TestCase):

    v = None

    def setUp(self):

        self.v = ezgal.astro_filter(tests.v_filter)

    def test_bc03_young(self):

        self.assertAlmostEqual(
            self.v.calc_mag(tests.bc03_vs, tests.bc03[:, 0], 1), -0.4223330, 7)

    def test_bc03_older(self):

        self.assertAlmostEqual(
            self.v.calc_mag(tests.bc03_vs, tests.bc03[:, 100], 1), 2.0494348,
            7)

    def test_bc03_low_z(self):

        self.assertAlmostEqual(
            self.v.calc_mag(tests.bc03_vs, tests.bc03[:, 100], 0.025),
            2.5083685, 7)

    def test_bc03_too_high_z(self):

        self.assertTrue(math.isnan(self.v.calc_mag(tests.bc03_vs,
                                                   tests.bc03[:, 100], 100)))


if __name__ == '__main__':
    unittest.main()
