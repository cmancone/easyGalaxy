import unittest
import ezgal.zf_grid
import numpy as np

# the zf_grid needs a lot of data to work.
# however, it is very dumb and really just
# does some simple interpolation.  So rather
# than try to get the fancy data it usually
# works on, we're just going to work with some
# simple data.  It still does the testing
# we need.
zf = 3
zs = [0, 0.5, 1, 1.5, 2, 2.5, 2.95]
ages = [3, 2.5, 2, 1.5, 1, 0.5, 0.05]
rest = [0.05, 0.5, 1, 1.5, 2, 2.5, 3]
obs = [3, 2.5, 2, 1.5, 1, 0.5, 0.05]
dms = [0.05, 0.5, 1, 1.5, 2, 2.5, 3]
solar = [0.05, 0.5, 1, 1.5, 2, 2.5, 3]
masses = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
dms = [0.05, 0.5, 1, 1.5, 2, 2.5, 3]

# offset the primary test-zs from our z grid,
# so some interpolation is actually used
test_zs = [0.25, 0.75, 1.25, 1.75, 2.25, 2.75]


class test_zf_grid(unittest.TestCase):

    zf_grid = None
    zf_grid_empty = None

    def setUp(self):

        # this is used for most of the testing
        self.zf_grid = ezgal.zf_grid.zf_grid(zf,
                                             np.asarray(zs),
                                             np.asarray(ages),
                                             np.asarray(rest),
                                             np.asarray(obs),
                                             dms=np.asarray(dms))

        self.zf_grid.store_solar_mags(np.asarray(solar))
        self.zf_grid.store_masses(np.asarray(masses))

        # this one is used to make sure an exception
        # is thrown when we try to fetch optional
        # data that has not been set
        self.zf_grid_empty = ezgal.zf_grid.zf_grid(
            zf, np.asarray(zs), np.asarray(ages), np.asarray(rest),
            np.asarray(obs))
