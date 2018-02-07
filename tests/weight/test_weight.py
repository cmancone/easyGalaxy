import unittest
import ezgal.weight
import numpy as np


class test_weight(unittest.TestCase):
    def test_init(self):

        weight = ezgal.weight(1)

        self.assertAlmostEqual(weight.weight, 1, 7)

    def test_mult(self):

        weight = ezgal.weight(2) * ezgal.weight(3)

        self.assertAlmostEqual(weight.weight, 6, 7)

    def test_mult_scalar(self):

        weight = ezgal.weight(2) * 3

        self.assertAlmostEqual(weight.weight, 6, 7)

    def test_imult(self):

        weight = ezgal.weight(3)
        weight *= ezgal.weight(4)

        self.assertAlmostEqual(weight.weight, 12, 7)

    def test_imult_scalar(self):

        weight = ezgal.weight(3)
        weight *= 4

        self.assertAlmostEqual(weight.weight, 12, 7)


if __name__ == '__main__':
    unittest.main()
