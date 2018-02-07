import unittest
import ezgal.sfhs
import numpy as np


class test_exponential(unittest.TestCase):
    def test_exponential_0(self):

        self.assertAlmostEqual(ezgal.sfhs.exponential(0, 1), 1.0, 7)

    def test_exponential_1(self):

        self.assertAlmostEqual(ezgal.sfhs.exponential(1, 1), 0.3678794, 7)

    def test_exponential_2(self):

        self.assertAlmostEqual(ezgal.sfhs.exponential(2, 1), 0.1353353, 7)


if __name__ == '__main__':
    unittest.main()
