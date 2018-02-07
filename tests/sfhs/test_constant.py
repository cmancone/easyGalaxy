import unittest
import ezgal.sfhs
import numpy as np


class test_constant(unittest.TestCase):

    # ezgal.sfhhs.constant( t, l )
    # returns 0 if t > l, else 1

    def test_constant_00(self):

        self.assertAlmostEqual(ezgal.sfhs.constant(0, 1), 1.0, 7)

    def test_constant_05(self):

        self.assertAlmostEqual(ezgal.sfhs.constant(0.5, 1), 1.0, 7)

    def test_constant_10(self):

        self.assertAlmostEqual(ezgal.sfhs.constant(1, 1), 1.0, 7)

    def test_constant_11(self):

        self.assertAlmostEqual(ezgal.sfhs.constant(1.1, 1), 0, 7)

    def test_constant_array(self):

        self.assertTrue(np.allclose(
            ezgal.sfhs.constant(
                np.asarray([0, 0.5, 1, 1.5]), 1), [1, 1, 1, 0], 1e-7))


if __name__ == '__main__':
    unittest.main()
