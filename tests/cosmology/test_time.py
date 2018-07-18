import unittest
import ezgal.cosmology
import tests
import numpy as np


class test_time(unittest.TestCase):

    cosmo = None

    def setUp(self):

        # standard cosmology
        self.cosmo = ezgal.cosmology.Cosmology(
            Om=0.272, Ol=0.728, h=0.704, w=-1)

    def test_Th(self):

        self.assertAlmostEqual(self.cosmo.Th() / 1e17, 4.3830639, 7)

    def test_Th_s(self):

        self.assertAlmostEqual(self.cosmo.Th(s=True) / 1e17, 4.3830639, 7)

    def test_Th_year(self):

        self.assertAlmostEqual(self.cosmo.Th(yr=True) / 1e10, 1.3889091, 7)

    def test_Th_myr(self):

        self.assertAlmostEqual(self.cosmo.Th(myr=True) / 1e4, 1.3889091, 7)

    def test_Th_gyr(self):

        self.assertAlmostEqual(self.cosmo.Th(gyr=True) / 1e1, 1.3889091, 7)

    def test_Tl(self):

        self.assertAlmostEqual(self.cosmo.Tl(1) / 1e17, 2.4578361, 7)

    def test_Tl_s(self):

        self.assertAlmostEqual(self.cosmo.Tl(1, s=True) / 1e17, 2.4578361, 7)

    def test_Tl_year(self):

        self.assertAlmostEqual(self.cosmo.Tl(1, yr=True) / 1e9, 7.7884125, 7)

    def test_Tl_myr(self):

        self.assertAlmostEqual(self.cosmo.Tl(1, myr=True) / 1e3, 7.7884125, 7)

    def test_Tl_gyr(self):

        self.assertAlmostEqual(self.cosmo.Tl(1, gyr=True), 7.7884125, 7)

    def test_Tu(self):

        self.assertAlmostEqual(self.cosmo.Tu() / 1e17, 4.3421816, 7)

    def test_Tu_s(self):

        self.assertAlmostEqual(self.cosmo.Tu(s=True) / 1e17, 4.3421816, 7)

    def test_Tu_year(self):

        self.assertAlmostEqual(self.cosmo.Tu(yr=True) / 1e10, 1.3759543, 7)

    def test_Tu_myr(self):

        self.assertAlmostEqual(self.cosmo.Tu(myr=True) / 1e4, 1.3759543, 7)

    def test_Tu_gyr(self):

        self.assertAlmostEqual(self.cosmo.Tu(gyr=True) / 1e1, 1.3759543, 7)

    def test_timeConversion_s(self):

        self.assertAlmostEqual(self.cosmo.timeConversion(s=True), 1, 0)

    def test_timeConversion_yr(self):

        self.assertAlmostEqual(
            self.cosmo.timeConversion(yr=True) * 1e8,
            3.1688088, 7)

    def test_timeConversion_myrs(self):

        self.assertAlmostEqual(
            self.cosmo.timeConversion(myr=True) * 1e14,
            3.1688088, 7)

    def test_timeConversion_gyrs(self):

        self.assertAlmostEqual(
            self.cosmo.timeConversion(gyr=True) * 1e17,
            3.1688088, 7)


if __name__ == '__main__':
    unittest.main()
