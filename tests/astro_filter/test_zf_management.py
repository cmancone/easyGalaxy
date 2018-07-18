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


class test_zf_management(unittest.TestCase):

    ##
    ## see test_gridded_funcs for more notes about this process
    ##
    @classmethod
    def setUpClass(cls):

        # we need to add on a filter
        cls.astro_filter = build_grid_astro_filter()

        # we have to grid the solar spectrum separately
        cls.astro_filter.grid_solar(tests.zf, tests.solar[:, 0],
                                    tests.solar[:, 1])

        # and the masses
        cls.astro_filter.grid_masses(tests.zf, tests.bc03_ages,
                                     tests.bc03_masses)

    ###################
    ## Apparent Mags ##
    ###################
    def test_get_zf_grid(self):

        # get_zf_grid returns the zf_grid object if available
        self.assertEqual(
            type(self.astro_filter.get_zf_grid(tests.zf)),
            ezgal.zf_grid.zf_grid)

    def test_get_zf_grid_invalid(self):

        # it returns false if an invalid zf is requested
        self.assertFalse(self.astro_filter.get_zf_grid(tests.zf + 1))

    def test_get_zf_ind(self):

        # returns its own internal index for where the
        # zf grid lives.  We only have one zf so the
        # answer is 0
        self.assertEqual(self.astro_filter.get_zf_ind(tests.zf), 0)

    def test_get_zf_ind(self):

        # and returns -1 for an invalid zf
        self.assertEqual(self.astro_filter.get_zf_ind(tests.zf + 1), -1)

    def test_has_zf(self):

        self.assertTrue(self.astro_filter.has_zf(tests.zf))

    def test_not_has_zf(self):

        self.assertFalse(self.astro_filter.has_zf(tests.zf + 1))

    # has_zf will also check for the presence or absence of
    # solar mags/masses.
    def test_has_zf_solar(self):

        self.assertTrue(self.astro_filter.has_zf(tests.zf, solar=True))

    def test_has_zf_masses(self):

        self.assertTrue(self.astro_filter.has_zf(tests.zf, masses=True))

    def test_not_has_zf_solar(self):

        # we need a new astro_filter object since the normal
        # one has solar pre-gridded
        astro_filter = build_grid_astro_filter()
        self.assertFalse(astro_filter.has_zf(tests.zf, solar=True))

    def test_not_has_zf_masses(self):

        # we need a new astro_filter object since the normal
        # one has solar pre-gridded
        astro_filter = build_grid_astro_filter()
        self.assertFalse(astro_filter.has_zf(tests.zf, masses=True))

    def test_extend_zf_list_empty(self):

        # if we pass nothing we should just get back the zf list
        self.assertTrue(np.array_equal(
            self.astro_filter.extend_zf_list([]), [3]))

    def test_extend_zf_list_mult(self):

        # we should get back all items if we pass stuff in
        zfs = self.astro_filter.extend_zf_list([1, 4])

        # it doesn't come back in any particular order so sort it
        zfs.sort()

        self.assertTrue(np.array_equal(zfs, [1, 3, 4]))

    def test_extend_zf_list_duplicates(self):

        # we shouldn't get back duplicates
        zfs = self.astro_filter.extend_zf_list([1, 3])

        # it doesn't come back in any particular order so sort it
        zfs.sort()

        self.assertTrue(np.array_equal(zfs, [1, 3]))

    # check for proper tolerance
    def test_extend_zf_list_tolerance_good(self):

        # duplicates are detected by being within
        # a certain tolerance.  This should return
        # only one entry
        zfs = self.astro_filter.extend_zf_list(
            [3 + self.astro_filter.tol * 0.9])

        self.assertTrue(np.allclose(zfs, [3]))

    # check for proper tolerance
    def test_extend_zf_list_tolerance_bad(self):

        # this should return two entries, as we are
        # over the allowed tolerance
        zfs = self.astro_filter.extend_zf_list(
            [3 + self.astro_filter.tol * 1.1])

        self.assertTrue(np.allclose(zfs, [3, 3 + self.astro_filter.tol * 1.1]))

    # check that clear cache works
    def test_clear_cache(self):

        astro_filter = build_grid_astro_filter()
        astro_filter.clear_cache()

        self.assertFalse(astro_filter.has_zf(3))


if __name__ == '__main__':
    unittest.main()
