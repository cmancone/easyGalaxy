import unittest
import ezgal.astro_filter
import tests
import numpy as np


class test_solar_vega(unittest.TestCase):

    astro_filter = None

    def setUp(self):

        self.astro_filter = ezgal.astro_filter(tests.v_filter)

    def test_vega(self):

        self.astro_filter.set_vega_conversion(tests.vega)
        self.assertAlmostEqual(self.astro_filter.to_vega, 0.0069388, 7)

    def test_solar(self):

        self.astro_filter.set_solar_magnitude(tests.solar)
        self.assertAlmostEqual(self.astro_filter.solar, 4.8245616, 7)

    def test_ab_flux(self):

        self.assertAlmostEqual(self.astro_filter.ab_flux * 1e21, 5.6971937, 7)


if __name__ == '__main__':
    unittest.main()
