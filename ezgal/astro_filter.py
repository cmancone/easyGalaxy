from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import numpy as np
from . import utils, zf_grid
# more modules are loaded in astro_filter.__init__()
# they are split up this way so that astro_filter_light doesn't have to import those modules
__ver__ = '2.0'


class astro_filter(object):
    """ filter = ezgal.astro_filter( filename, units='a', cosmology=None )

    Loads a filter response curve for use with ezgal
    The file should be a two column ascii files with filter response curves.
    First column should be wavelength (or frequency), second column should be
    total response (fraction).
    By default units are assumed to be angstroms.  Use units='value' To specify
    non-default units.
    To specify units of hertz, set units='Hz' (case-insensitive)
    To specify alternate units (for wavelength) see code list in
    ezgal.utils.to_meters() """

    cosmo = None  # cosmology object

    ab_source_flux = 3.631e-20  # flux of a zero mag ab source
    ls = np.array([])  # wavelengths for filter response curve (angstroms)
    vs = np.array([])  # frequencies for filter response curve
    diffs = np.array([])  # frequency width of each bin
    tran = np.array([])  # fractional transmission
    tran_ls = np.array(
        [])  # fractional transmission (corresponding to wavelengths in ls)
    npts = 0  # number of points in the transmission curve
    ab_flux = 0  # flux of a zero mag ab source through the filter
    vega_flux = 0  # flux of vega through the filter
    to_vega = 0  # vega_mag = ab_mag + self.to_vega
    has_vega = False  # whether or not the vega conversion is calculated
    solar = 0  # solar magnitude
    has_solar = False  # whether or not the solar magnitude is calculated

    # filter properties, standard definitions taken from:
    # [1] http://www.stsci.edu/hst/wfpc2/documents/handbook/cycle17/ch6_exposuretime2.html#480221
    # [2] http://www.stsci.edu/hst/wfc3/documents/handbooks/currentIHB/c06_uvis06.html#57
    mean = 0.0  # mean wavelength (angstroms, [1])
    pivot = 0.0  # pivot wavelength (angstroms, [1])
    average = 0.0  # average wavelength (angstroms, [1])
    sig = 0.0  # effective dimensionless gaussian width ([1])
    width = 0.0  # effective width of bandpass (angstroms, [1])
    equivalent_width = 0.0  # equivalent width (angstroms, [2])
    rectangular_width = 0.0  # rectangular width (angstroms, [2])

    nzfs = 0  # number of formation redshifts that have been gridded
    zfs = np.array([])  # list of formation redshifts that have been gridded
    zf_grids = []  # list of zf_grid objects

    # tolerance for determining whether a given zf matches a stored zf
    # the tolerance is typical set by ezgal after creating a new astro filter
    # but it is also defined here to have a default value
    tol = 1e-8

    ##############
    ## __init__ ##
    ##############
    def __init__(self,
                 filename,
                 units='a',
                 cosmology=None,
                 vega=False,
                 solar=False):

        # load additional modules.  Yes, this is strange.  But this way
        # astro_filter_light can inherit astro_filter.
        # this is necessary because astro_filter_light is intended to work
        # without any of these modules
        import scipy.interpolate as interpolate
        import scipy.integrate
        global interpolate
        global scipy

        # check that we were passed a file and it exists
        if type(filename) == type(str('')):
            if not os.path.isfile(filename):
                raise ValueError(
                    'The specified filter transmission file does not exist!')

            # read it in
            file = utils.rascii(filename)
        elif filename is None:
            # very basic load - no filter response curve
            self.npts = 0
            self.to_vega = vega
            self.has_vega = True
            if cosmology is not None: self.cosmo = cosmology
            self.solar = solar
            self.has_solar = not np.isnan(solar)
            return
        else:
            # is this a numpy array?
            if type(filename) == type(np.array([])):
                file = filename
            else:
                raise ValueError(
                    'Please pass the filename for the filter transmission file!')

        ls = file[:, 0]

        # calculate wavelengths in both angstroms and hertz
        units = units.lower()
        if units == 'hz':
            vs = ls
            ls = utils.to_lambda(vs, units='a')
        else:
            vs = utils.to_hertz(ls, units=units)
            ls = utils.convert_length(ls, incoming=units, outgoing='a')

        # store everything sorted
        sind = vs.argsort()
        self.vs = vs[sind]  # frequencies
        self.tran = file[sind, 1]  # corresponding transmission
        self.ls = ls[sind[::-1]]  # wavelengths (angstroms)
        self.tran_ls = self.tran[::-1]  # corresponding transmission
        self.npts = self.vs.size

        # frequency widths of each datapoint
        self.diffs = np.roll(self.vs, -1) - self.vs
        self.diffs[-1] = self.diffs[-2]

        # calculate filter properties and store in the object
        self.calc_filter_properties()

        # normalization for calculating ab mags for this filter
        self.ab_flux = self.ab_source_flux * scipy.integrate.simps(
            self.tran / self.vs, self.vs)

        # store the cosmology object if passed
        if cosmology is not None: self.cosmo = cosmology

        # calculate ab-to-vega conversion if vega spectrum was passed
        if type(vega) == type(np.array([])): self.set_vega_conversion(vega)

        # calculate solar magnitude if solar spectrum was passed
        if type(solar) == type(np.array([])): self.set_solar_magnitude(solar)

        self.zfs = np.array([])
        self.zf_grids = []

    ############################
    ## calc filter properties ##
    ############################
    def calc_filter_properties(self):
        """ ezgal.astro_filter.calc_filter_properties()

        Calculates the following properties of the filter response curve and
        stores them in the astro_filter object:
        Mean wavelength (angstroms)
        pivot wavelength (angstroms)
        average wavelength (angstroms)
        effective dimensionless gaussian width
        effective width of bandpass (angstroms)
        equivalent width (angstroms)
        rectangular width (angstroms)

        Uses standard definitions defined in wfpc2 instrument handbook:
        http://www.stsci.edu/hst/wfpc2/documents/handbook/cycle17/ch6_exposuretime2.html#480221
        Definitions of equivalent width and rectangular width come from here:
        http://www.stsci.edu/hst/wfc3/documents/handbooks/currentIHB/c06_uvis06.html#57

        Currently does not use a very precise method for integration... """

        # dlambda
        diff = self.ls - np.roll(self.ls, 1)
        # useful integral region
        m = (self.tran_ls > 0) & (diff > 0)

        # mean wavelength
        self.mean = np.exp((self.tran_ls[m] * np.log(self.ls[m]) * diff[m] /
                            self.ls[m]).sum() /
                           (self.tran_ls[m] * diff[m] / self.ls[m]).sum())
        # pivot wavelength
        self.pivot = np.sqrt((self.tran_ls[m] * self.ls[m] * diff[m]).sum() /
                             (self.tran_ls[m] * diff[m] / self.ls[m]).sum())
        # average wavelength
        self.average = (self.tran_ls[m] * self.ls[m] * diff[m]).sum() / (
            self.tran_ls[m] * diff[m]).sum()
        # effective dimensionless gaussian width
        self.sig = np.sqrt((self.tran_ls[m] * np.log(self.ls[m] / self.mean)
                            **2.0 * diff[m] / self.ls[m]).sum() /
                           (self.tran_ls[m] * diff[m] / self.ls[m]).sum())
        # effective width
        self.width = 2.0 * np.sqrt(2.0 * np.log(2.0)) * self.sig * self.mean
        # equivalent width
        self.equivalent_width = (self.tran_ls[m] * diff[m]).sum()
        # rectangular width
        self.rectangular_width = self.equivalent_width / self.tran_ls[m].max()

    #########################
    ## set vega conversion ##
    #########################
    def set_vega_conversion(self, vega):
        """ ezgal.astro_filter.set_vega_conversion( vega )

        Pass a numpy array with the vega spectrum.  Array should have two columns:

        frequency (hertz)
        flux (ergs/s/cm^2/Hz) """

        if type(vega) != type(np.array([])) or vega.size == 0:
            raise ValueError(
                'Pass a numpy array with the vega spectrum to calculate vega conversion!')

        # does the vega spectrum extend out to this filter?
        if self.vs.min() < vega[:, 0].min() or self.vs.max() > vega[:, 0].max(
        ):
            return False

        self.to_vega = -1.0 * self.calc_mag(vega[:, 0], vega[:, 1], 0)
        self.vega_flux = self.ab_source_flux / 10.0**(-0.4 * self.to_vega)
        self.has_vega = True

    #########################
    ## set solar magnitude ##
    #########################
    def set_solar_magnitude(self, solar):
        """ ezgal.astro_filter.set_solar_mag( vega )

        Pass a numpy array with the solar spectrum.  Array should have two columns:

        frequency (hertz)
        flux (ergs/s/cm^2/Hz) """

        if type(solar) != type(np.array([])) or solar.size == 0:
            raise ValueError(
                'Pass a numpy array with the solar spectrum to calculate the solar magnitude!')

        # does the solar spectrum extend out to this filter?
        if self.vs.min() < solar[:, 0].min() or self.vs.max() > solar[:,
                                                                      0].max():
            self.solar = np.nan
            return False

        self.solar = self.calc_mag(solar[:, 0], solar[:, 1], 0)
        self.has_solar = True

    #######################
    ## get apparent mags ##
    #######################
    def get_apparent_mags(self, zf, zs, vega=False):
        """ mag = ezgal.astro_filter.get_apparent_mags( zf, zs, vega=False )

        Returns the apparent magnitude of the model at the given redshifts,
        given the formation redshift.
        Uses the zf_grid object to speed up calculations.  Can only be used for
        formation redshifts that have been gridded.
        Outputs vega mags if vega=True
        """

        zf_grid = self.get_zf_grid(zf)
        if zf_grid == False:
            raise ValueError(
                'Cannot fetch mag for given formation redshift because it has not been gridded!')

        to_vega = self.to_vega if vega else 0.0

        return zf_grid.get_obs_mags(zs) + self.calc_dm(zs) + to_vega

    #######################
    ## get absolute mags ##
    #######################
    def get_absolute_mags(self, zf, zs, vega=False):
        """ mag = ezgal.astro_filter.get_absolute_mags( zf, zs, vega=False )

        Returns the absolute magnitude of the model at the given redshifts,
        given the formation redshift.
        Uses the zf_grid object to speed up calculations.  Can only be used for
        formation redshifts that have been gridded.
        Outputs vega mags if vega=True
        """

        zf_grid = self.get_zf_grid(zf)
        if zf_grid == False:
            raise ValueError(
                'Cannot fetch mag for given formation redshift because it has not been gridded!')

        to_vega = self.to_vega if vega else 0.0

        return zf_grid.get_rest_mags(zs) + to_vega

    ################################
    ## get observed absolute mags ##
    ################################
    def get_observed_absolute_mags(self, zf, zs, vega=False):
        """ mag = ezgal.astro_filter.get_observed_absolute_mags( zf, zs, vega=False )

        Returns the observed-frame absolute magnitude of the model at the given
        redshifts, given the formation redshift.
        Uses the zf_grid object to speed up calculations.  Can only be used for
        formation redshifts that have been gridded.
        Outputs vega mags if vega=True
        """

        zf_grid = self.get_zf_grid(zf)
        if zf_grid == False:
            raise ValueError(
                'Cannot fetch mag for given formation redshift because it has not been gridded!')

        to_vega = self.to_vega if vega else 0.0

        return zf_grid.get_obs_mags(zs) + to_vega

    ###################
    ## get kcorrects ##
    ###################
    def get_kcorrects(self, zf, zs):
        """ kcorrect = ezgal.astro_filter.get_kcorrects( zf, z )

        Returns the k-correction of the model at the given redshift, given the
        formation redshift.
        Uses the zf_grid object to speed up calculations.  Can only be used for
        formation redshifts that have been gridded.
        """

        zf_grid = self.get_zf_grid(zf)
        if zf_grid == False:
            raise ValueError(
                'Cannot fetch k-correction for given formation redshift because it has not been gridded!')

        return zf_grid.get_obs_mags(zs) - zf_grid.get_rest_mags(zs)

    ###################
    ## get ecorrects ##
    ###################
    def get_ecorrects(self, zf, zs):
        """ kcorrect = ezgal.astro_filter.get_ecorrects( zf, z )

        Returns the e-correction of the model at the given redshift, given the
        formation redshift.
        Uses the zf_grid object to speed up calculations.  Can only be used for
        formation redshifts that have been gridded.
        """

        zf_grid = self.get_zf_grid(zf)
        if zf_grid == False:
            raise ValueError(
                'Cannot fetch e-correction for given formation redshift because it has not been gridded!')

        return zf_grid.get_rest_mags(zs) - zf_grid.rest[0]

    ####################
    ## get ekcorrects ##
    ####################
    def get_ekcorrects(self, zf, zs):
        """ kcorrect = ezgal.astro_filter.get_ekcorrects( zf, z )

        Returns the e+k correction of the model at the given redshift, given
        the formation redshift.
        Uses the zf_grid object to speed up calculations.  Can only be used for
        formation redshifts that have been gridded.
        """

        zf_grid = self.get_zf_grid(zf)
        if zf_grid == False:
            raise ValueError(
                'Cannot fetch e+k correction for given formation redshift because it has not been gridded!')

        return zf_grid.get_obs_mags(zs) - zf_grid.rest[0]

    ####################
    ## get solar mags ##
    ####################
    def get_solar_mags(self, zf, zs, vega=False):
        """ solar = ezgal.astro_filter.get_solar_mags( zf, z, vega=False )

        Returns the observed-frame absolute solar magnitude at the given
        redshift, given the formation redshift.
        Uses the zf_grid object to speed up calculations.  Can only be used for
        formation redshifts that have been gridded.
        """

        zf_grid = self.get_zf_grid(zf)
        if zf_grid == False or not zf_grid.has_solar:
            raise ValueError(
                'Cannot fetch solar magnitudes for given formation redshift because it has not been gridded!')

        to_vega = self.to_vega if vega else 0.0

        return zf_grid.get_solar_mags(zs) + to_vega

    ################
    ## get masses ##
    ################
    def get_masses(self, zf, zs):
        """ masses = ezgal.astro_filter.get_masses( zf, z )

        Returns the stellar mass (in solar masses) at the given redshift, given
        the formation redshift.
        Uses the zf_grid object for consistency.  Can only be used for
        formation redshifts that have been gridded.
        """

        zf_grid = self.get_zf_grid(zf)
        if zf_grid == False or not zf_grid.has_masses:
            raise ValueError(
                'Cannot fetch masses for given formation redshift because it has not been gridded!')

        return zf_grid.get_masses(zs)

    #############
    ## calc dm ##
    #############
    def calc_dm(self, zs):
        """ dm = ezgal.astro_filter.get_dm( zs )

        Returns the distance modulus for given redshifts
        """

        zs = np.asarray(zs)
        if len(zs.shape) == 0: zs = np.array([zs])

        dms = np.empty(zs.size)
        for i in range(zs.size):
            dms[i] = self.cosmo.DistMod(zs[i])

        if zs.size == 1: return dms[0]
        return dms

    ###################
    ## calc rest mag ##
    ###################
    def calc_rest_mag(self, vs, sed):
        """ mag = ezgal.astro_filter.get_mag( vs, sed )

        Calculate the rest-frame absolute AB magnitude of an SED through the filter """

        return self.calc_mag(vs, sed, 0)

    ##################
    ## calc obs mag ##
    ##################
    def calc_obs_mag(self, vs, sed, z):
        """ mag = ezgal.astro_filter.get_mag( vs, sed, z )

        Calculate the observed-frame absolute AB magnitude of an SED through the filter """

        return self.calc_mag(vs, sed, z)

    ##############
    ## calc mag ##
    ##############
    def calc_mag(self, vs, sed, z):
        """ mag = ezgal.astro_filter.calc_mag( vs, sed, z )

        :param vs: List of sed frequencies.
        :param sed: The SED, with units of ergs/s/cm^2/Hz
        :param z: The redshift to redshift the SED to.
        :type vs: list, array
        :type sed: list, array
        :type z: int, float
        :returns: Absolute AB magnitude
        :rtype: float

        :Example:
            >>> import ezgal
            >>> model = ezgal.model( 'bc03_ssp_z_0.02_chab.model' )
            >>> model.add_filter( 'ch1' )
            >>> zf = 3
            >>> z = 1
            >>> # Use ``EzGal`` to calculate the rest-frame ch1 mag given zf & z
            >>> model.get_absolute_mags( zf, filters='ch1', zs=z )
            5.6135203220610741
            >>> # now calculate it directly
            >>> sed = model.get_sed_z( zf, z )
            >>> model.filters['ch1'].calc_mag( model.vs, sed, 0 )
            5.6135203220610741
            >>> # same for apparent mag
            >>> model.get_apparent_mags( zf, filters='ch1', zs=z )
            47.969095009830326
            >>> model.filters['ch1'].calc_mag( model.vs, sed, z ) + model.get_distance_moduli( z, nfilters=1 )
            47.969095009830326

        Calculate the absolute AB magnitude of the given sed at the given redshift. Set ``z=0`` for rest-frame magnitudes.  ``vs`` should give the frequency (in Hz) of every point in the SED, and the sed should have units of ergs/s/cm^2/Hz."""

        # make sure an acceptable number of sed points actually go through the filter...
        shifted = vs / (1 + z)
        c = ((shifted > self.vs.min()) & (shifted < self.vs.max())).sum()
        if c < 5: return np.nan
        # and that the SED actually covers the whole filter
        if shifted.min() > self.vs.min() or shifted.max() < self.vs.max():
            return np.nan

        interp = interpolate.interp1d(vs, sed)
        sed_flux = (1 + z) * scipy.integrate.simps(
            interp(self.vs * (1 + z)) * self.tran / self.vs, self.vs)

        return -2.5 * np.log10(sed_flux / self.ab_flux)

    ##########
    ## grid ##
    ##########
    def grid(self, zf, vs, zs, ages, seds, force=False):
        """ astro_filter.grid( zf, vs, ages, seds, force=False )

        Calculate rest and observed frame magnitude for the given seds given age and formation redshift.
        If this formation redshift is already gridded then no additional calculations will be made
        unless force=True """

        # don't bother doing anything if this formation redshift is already gridded
        if self.has_zf(zf) and not (force): return True

        # now we want to calculate rest frame and observed frame absolute mags
        w = np.where(zs < zf)[0]
        nseds = len(w)

        rest = np.empty(nseds)
        obs = np.empty(nseds)
        dms = self.calc_dm(zs)
        for i in range(nseds):
            rest[i] = self.calc_rest_mag(vs, seds[:, w[i]])
            obs[i] = self.calc_obs_mag(vs, seds[:, w[i]], zs[i])

        # finally store the grid information
        self.store_grid(zf, zs[w], ages[w], rest, obs, dms)

    ################
    ## grid solar ##
    ################
    def grid_solar(self, zf, solar_vs, solar_sed):
        """ astro_filter.grid_solar()

        Grid up the observed magnitude of the sun as a function of redshift from the given solar SED and store """

        # This assumes that the given zf is already gridded - otherwise there is nothing to do
        if not self.has_zf(zf):
            raise ValueError(
                'Cannot grid solar observed mags because the formation redshift has not been gridded yet!')

        # get the zf grid
        zf_grid = self.get_zf_grid(zf)

        # fetch redshifts
        zs = zf_grid.zs

        # and calculate observed-frame solar mag at those redshifts
        mags = np.zeros(zs.size)
        for (i, z) in enumerate(zs):
            mags[i] = self.calc_mag(solar_vs, solar_sed, z)

        # and store
        zf_grid.store_solar_mags(mags)

    #################
    ## grid masses ##
    #################
    def grid_masses(self, zf, ages, masses):
        """ astro_filter.grid_masses()

        Grid up the stellar mass as a function of redshift and store """

        # This assumes that the given zf is already gridded - otherwise there is nothing to do
        if not self.has_zf(zf):
            raise ValueError(
                'Cannot grid masses because the formation redshift has not been gridded yet!')

        # get the zf grid
        zf_grid = self.get_zf_grid(zf)

        # interpolate onto age grid and store
        zf_grid.store_masses(np.interp(zf_grid.ages, ages, masses))

    ################
    ## store grid ##
    ################
    def store_grid(self,
                   zf,
                   zs,
                   ages,
                   rest,
                   obs,
                   dms=None,
                   solar=None,
                   masses=None):
        """ generate and store a zf_grid object for the given formation redshift and grid data """

        grid = zf_grid.zf_grid(zf, zs, ages, rest, obs, dms)
        if solar is not None: grid.store_solar_mags(solar)
        if masses is not None: grid.store_masses(masses)

        # start a new list if we don't have any formation redshifts yet
        if self.nzfs == 0:
            self.zf_grids = []
            zfs = np.array([])

        if self.has_zf(zf):
            # overwrite old grid
            self.zf_grids[self.get_zf_ind(zf)] = grid
        else:
            # save new one
            self.zf_grids.append(grid)
            zf = np.array([zf])
            if self.nzfs == 0:
                self.zfs = zf
            else:
                self.zfs = np.concatenate((self.zfs, zf))
            self.nzfs += 1

    #################
    ## get zf grid ##
    #################
    def get_zf_grid(self, zf):
        """ zf_grid = ezgal.astro_filter.get_zf_grid( zf )

        returns the zf_grid object for calculating model properties for a given zf.
        returns False if the formation redshift has not been gridded """

        ind = self.get_zf_ind(zf)
        if ind < 0: return False
        return self.zf_grids[ind]

    ################
    ## get zf ind ##
    ################
    def get_zf_ind(self, zf):
        """ ind = ezgal.astro_filter.get_zf_ind( zf )

        returns the index to ezgal.astro_filter.zf_grids corresponding to the given formation redshift.
        returns false if the given formation redshift has not been gridded. """

        if self.nzfs == 0: return -1

        dists = np.abs(self.zfs - zf)
        ind = dists.argmin()
        if dists[ind] > self.tol: return -1
        return ind

    ############
    ## has zf ##
    ############
    def has_zf(self, zf, solar=False, masses=False):
        """ bool = ezgal.astro_filter.has_zf( zf, solar=False, masses=False )

        returns True or False depending on whether a particular formation redshift has been gridded.
        A tolerance of 1e-8 is used to decide if the passed formation redshift matches a gridded one. """

        if self.nzfs == 0: return False

        # try to fetch zf grid
        zf_grid = self.get_zf_grid(zf)

        # if nothing is found, then it isn't gridded
        if not zf_grid: return False

        # are we checking for having solar data gridded as well?
        if solar and not zf_grid.has_solar: return False

        # how about mass info?
        if masses and not zf_grid.has_masses: return False

        # if we got this far then all the requested info is gridded
        return True

    ####################
    ## extend zf list ##
    ####################
    def extend_zf_list(self, zfs=None):
        """ zfs = ezgal.astro_filter.extend_zf_list( zfs=None )

        Takes a list of zfs and returns a new list which includes all zfs from the previous list,
        plus any zfs that are in this object but weren't in the old list. """

        if zfs is None or len(zfs) == 0:
            if self.nzfs == 0:
                return None
            else:
                return self.zfs

        zfs = np.asarray(zfs)

        # find the distance to the nearest passed formation redshift to each formation redshift in this filter object
        zfs_in = np.asarray(zfs).ravel()
        my_zfs = self.zfs.reshape((self.nzfs, 1))
        dists = np.abs(zfs_in - my_zfs).min(axis=1)

        # which ones don't have a match within the given tolerance?
        m = dists > self.tol

        # if nothing was found, then there is nothing new - return the passed list
        if m.sum() == 0: return zfs.copy()

        # return an extended list
        ret = self.zfs[m].tolist()
        ret.extend(zfs)
        return ret

    ###################
    ## set cosmology ##
    ###################
    def set_cosmology(self, cosmo):
        """ sets the cosmology object for the filter to use for all calculations.  Should be a cosmology.Cosmology() object. """

        # see if the cosmology is changing - if so, we need to dump any stored models
        if self.cosmo is None:
            self.clear_cache()
        elif self.cosmo.Om != cosmo.Om or self.cosmo.Ol != cosmo.Ol or self.cosmo.h != cosmo.h or self.cosmo.w != cosmo.w:
            self.clear_cache()

        # store the new cosmology
        self.cosmo = cosmo

    #################
    ## clear cache ##
    #################
    def clear_cache(self):
        """ ezgal.astro_filter.clear_cache()

        Clears all stored redshift evolution information """

        self.zf_grids = []
        zfs = np.array([])
        self.nzfs = 0
