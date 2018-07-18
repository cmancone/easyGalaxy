import unittest
import ezgal.astro_filter
from ezgal.cosmology import Cosmology
import tests
import numpy as np


def build_grid_astro_filter():

    astro_filter = ezgal.astro_filter(tests.v_filter,
                                      cosmology=Cosmology(),
                                      vega=tests.vega,
                                      solar=tests.solar)
    astro_filter.grid(tests.zf, tests.bc03_vs, tests.grid_zs, tests.bc03_ages,
                      tests.bc03)

    return astro_filter


class test_gridded_funcs(unittest.TestCase):

    # the astro_filter works by receiving a grid o
    # SEDs from ezgal. It then uses
    # astro_filter.calc_mag to calculate all the relevant
    # magnitudes, etc, for those SEDs as a function of z.
    # It stores the results in a zf_grid object (tested
    # separately of course) and then when it is asked for
    # magnitudes it pretty much just asks zf_grid, which
    # does some interpolation as necessary.  So these
    # gridded functions I am trying to test here don't
    # do very much themselves: they are thin wrappers
    # around zf_grid.  So some simple tests will suffice,
    # as zf_grid is already thoroughly tested.
    astro_filter = None

    # a lot of computation is needed for the gridding process
    # so if we use the normal setUp method to prepare our
    # astro_filter object, we will waste a lot of time:
    # setUp is called before each individual test,
    # but the grid only has to be built once.  So use
    # setUpClass instead
    @classmethod
    def setUpClass(cls):

        # we need to add on a filter
        cls.astro_filter = build_grid_astro_filter()

        # and then call the grid method with
        # an actual grid of seds.  This will
        # calculate thhe evolution as a function
        # of redshfit
        #cls.astro_filter.grid( tests.zf, tests.bc03_vs, tests.grid_zs, tests.bc03_ages, tests.bc03 )

        # we have to grid the solar spectrum separately
        cls.astro_filter.grid_solar(tests.zf, tests.solar[:, 0],
                                    tests.solar[:, 1])

        # and the masses
        cls.astro_filter.grid_masses(tests.zf, tests.bc03_ages,
                                     tests.bc03_masses)

    ###################
    ## Apparent Mags ##
    ###################
    def test_apparent_mags(self):

        # standard calling sequence
        self.assertTrue(np.allclose(
            self.astro_filter.get_apparent_mags(tests.zf, tests.test_zs),
            np.asarray([37.08456776, 42.39315957, 44.61016611, 47.43454964,
                        48.24942464]), 1e-6))

    def test_apparent_mags_vega(self):

        # make sure it remembers to calculate vega magnitudes
        self.assertTrue(np.allclose(
            self.astro_filter.get_apparent_mags(
                tests.zf, tests.test_zs, vega=True),
            np.asarray([37.09150655, 42.40009836, 44.6171049, 47.44148843,
                        48.25636343]),
            1e-6))

    def test_apparent_mags_invalid_zf(self):

        # it should raise an exception if we request a non-gridd zf
        with self.assertRaises(ValueError):
            self.astro_filter.get_apparent_mags(tests.zf + 1,
                                                tests.test_zs,
                                                vega=True)

    ###################
    ## Absolute Mags ##
    ###################
    def test_absolute_mags(self):

        # standard calling sequence
        self.assertTrue(np.allclose(
            self.astro_filter.get_absolute_mags(tests.zf, tests.test_zs),
            np.asarray([1.40721352, 1.13022703, 1.12120338, 2.50513901,
                        2.55585118]), 1e-6))

    def test_absolute_mags_vega(self):

        # make sure it remembers to calculate vega magnitudes
        self.assertTrue(np.allclose(
            self.astro_filter.get_absolute_mags(
                tests.zf, tests.test_zs, vega=True),
            np.asarray([1.41415231, 1.13716582, 1.12814217, 2.5120778,
                        2.56278996]),
            1e-6))

    def test_absolute_mags_invalid_zf(self):

        # it should raise an exception if we request a non-gridd zf
        with self.assertRaises(ValueError):
            self.astro_filter.get_absolute_mags(tests.zf + 1,
                                                tests.test_zs,
                                                vega=True)

    ############################
    ## Observed Absolute Mags ##
    ############################
    def test_observed_absolute_mags(self):

        # standard calling sequence
        self.assertTrue(np.allclose(
            self.astro_filter.get_observed_absolute_mags(
                tests.zf, tests.test_zs), np.asarray([
                    1.32404124, 0.04532745, -0.09802977, 1.41568863, 1.48096523
                ]), 1e-6))

    def test_observed_absolute_mags_vega(self):

        # make sure it remembers to calculate vega magnitudes
        self.assertTrue(np.allclose(
            self.astro_filter.get_observed_absolute_mags(
                tests.zf, tests.test_zs, vega=True),
            np.asarray([1.33098003, 0.05226624, -0.09109098, 1.42262742,
                        1.48790402]),
            1e-6))

    def test_observed_absolute_mags_invalid_zf(self):

        # it should raise an exception if we request a non-gridd zf
        with self.assertRaises(ValueError):
            self.astro_filter.get_observed_absolute_mags(tests.zf + 1,
                                                         tests.test_zs,
                                                         vega=True)

    ###############
    ## KCorrects ##
    ###############
    def test_kcorrects(self):

        # standard calling sequence
        self.assertTrue(np.allclose(
            self.astro_filter.get_kcorrects(tests.zf, tests.test_zs),
            np.asarray([-0.08317228, -1.08489958, -1.21923315, -1.08945039,
                        -1.07488595]), 1e-6))

    def test_kcorrects_invalid_zf(self):

        # it should raise an exception if we request a non-gridd zf
        with self.assertRaises(ValueError):
            self.astro_filter.get_kcorrects(tests.zf + 1, tests.test_zs)

    ###############
    ## ECorrects ##
    ###############
    def test_ecorrects(self):

        # standard calling sequence
        self.assertTrue(np.allclose(
            self.astro_filter.get_ecorrects(tests.zf, tests.test_zs),
            np.asarray([0., -0.27698649, -0.28601014, 1.0979255, 1.14863766
                        ]), 1e-6))

    def test_ecorrects_invalid_zf(self):

        # it should raise an exception if we request a non-gridd zf
        with self.assertRaises(ValueError):
            self.astro_filter.get_ecorrects(tests.zf + 1, tests.test_zs)

    ################
    ## EKCorrects ##
    ################
    def test_ekcorrects(self):

        # standard calling sequence
        self.assertTrue(np.allclose(
            self.astro_filter.get_ekcorrects(tests.zf, tests.test_zs),
            np.asarray([-0.08317228, -1.36188607, -1.50524329, 0.00847511,
                        0.07375171]), 1e-6))

    def test_ekcorrects_invalid_zf(self):

        # it should raise an exception if we request a non-gridd zf
        with self.assertRaises(ValueError):
            self.astro_filter.get_ekcorrects(tests.zf + 1, tests.test_zs)

    ################
    ## Solar Mags ##
    ################
    def test_solar_mags(self):

        # standard calling sequence
        self.assertTrue(np.allclose(
            self.astro_filter.get_solar_mags(tests.zf, tests.test_zs),
            np.asarray([4.84424744, 5.89246026, 9.15878231, 13.14447493,
                        16.62662314]), 1e-6))

    def test_solar_mags_vega(self):

        # make sure it remembers to calculate vega magnitudes
        self.assertTrue(np.allclose(
            self.astro_filter.get_solar_mags(
                tests.zf, tests.test_zs, vega=True),
            np.asarray([4.85118623, 5.89939905, 9.1657211, 13.15141372,
                        16.63356193]),
            1e-6))

    def test_solar_mags_invalid_zf(self):

        # it should raise an exception if we request a non-gridded zf
        with self.assertRaises(ValueError):
            self.astro_filter.get_solar_mags(tests.zf + 1,
                                             tests.test_zs,
                                             vega=True)

    def test_solar_mags_no_solar(self):

        # it should raise an exception if we request solar mags if not providing
        # the solar spectrum.  We have to build a new astro_filter object to test
        # this, since setUpClass grids it by default
        astro_filter = build_grid_astro_filter()
        with self.assertRaises(ValueError):
            astro_filter.get_solar_mags(tests.zf, tests.test_zs, vega=True)

    ############
    ## Masses ##
    ############
    def test_masses(self):

        # standard calling sequence
        self.assertTrue(np.allclose(
            self.astro_filter.get_masses(tests.zf, tests.test_zs), np.asarray(
                [1., 1., 0.941732, 0.8794115, 0.872226]), 1e-6))

    def test_masses_invalid_zf(self):

        # it should raise an exception if we request a non-gridded zf
        with self.assertRaises(ValueError):
            self.astro_filter.get_masses(tests.zf + 1, tests.test_zs)

    def test_masses_no_solar(self):

        # it should raise an exception if we request solar mags if not providing
        # the solar spectrum.  We have to build a new astro_filter object to test
        # this, since setUpClass grids it by default
        astro_filter = build_grid_astro_filter()
        with self.assertRaises(ValueError):
            astro_filter.get_masses(tests.zf, tests.test_zs)


if __name__ == '__main__':
    unittest.main()
