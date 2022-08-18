#!/usr/bin/python
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

#import collections
import os
import re
import numpy as np
from . import utils, sfhs, weight, astro_filter_light
# more modules are loaded in ezgal.__init__()
# they are split up this way so that ezgal_light doesn't have to import those modules
__ver__ = '2.0'


class ezgal(object):
    """ model = ezgal.ezgal( model_file, is_ised=False, is_fits=False,
    is_ascii=False, has_masses=False, units='a', age_units='gyrs')

    Class for converting galaxy seds as a function of age to magnitudes as a
    function of redshift.
    Reads in bc03 ised files as well as ascii files, effectively replacing
    cm_evolution from the bc03 src files
    Specify a model file.  It should be a bc03 ised file, an ascii file, or a
    model file created with this class (which are stored as fits).
    See ezgal._load_ascii for information on formatting/units for ascii files
    This will automatically try to dtermine which type of file you have passed
    it.  If this fails, you can specify it manually with the is_* flags.

    units, age_units, and has_masses apply only for ascii files.  See
    ezgal._load_ascii()
    """

    # python 3 compatibility
    def __next__(self):
        return self.next()

    # model information
    filename = ''  # the name of the file
    nages = 0  # number of different aged SEDs
    ages = np.array([])  # array of ages (years)
    nvs = 0  # number of frequences in each SED
    vs = np.array([])  # array of frequences for SEDs
    nls = 0  # number of wavelengths in each SED (same as self.nvs)
    ls = np.array([])  # array of wavelengths for SEDs (angstroms)
    seds = np.array(
        [])  # age/SED grid.  Units need to be ergs/cm^2/s.  Size is nvs x nages
    # normalization info
    norm = {'norm': 0, 'z': 0, 'filter': '', 'vega': False, 'apparent': False}
    # mass info
    has_masses = False  # whether or not masses have been set
    masses = np.array([])  # masses as a function of self.ages
    # sfh info
    has_sfh = False  # whether or not a star formation history is set
    sfh = np.array([])  # star formation as a function of self.ages

    # cosmology related stuff
    cosmology_loaded = False  # whether or not a cosmology has been loaded
    cosmo = None  # cosmology object
    zfs = np.array([])  # formation redshifts for models
    nzfs = 0  # number of formation redshifts to model
    zs = np.array([])  # redshifts at which to project models
    nzs = 0  # number of redshifts models should calculated at

    # filter stuff.
    filters = {}  # dictionary of astro_filter objects
    filter_order = []  # list of filter names (in order added)
    nfilters = 0  # number of filters
    current_filter = -1  # counter for iterator

    # info for interpolated model SEDs.  The SEDs are interpolated to a regular grid for each formation redshift.
    # These interpolated models are stored to save time if needed again
    interp_seds = [
    ]  # interpolated seds - each list element is a dictionary with sed info
    interp_zfs = np.array([])  # list of formation redshifts for sed
    tol = 1e-8  # tolerance for determining whether a given zf matches a stored zf

    # additional data
    data_dir = ''  # data directory for filters, models, and reference spectra
    filter_dir = False  # user-set directory for filters
    model_dir = False  # user-set directory for models
    has_vega = False  # whether or not the vega spectrum is found and loaded
    vega = np.array([])  # vega spectrum (nu vs Fnu)
    vega_out = False  # if True then retrieved mags will be in vega mags
    has_solar = False  # whether or not the solar spectrum is found and loaded
    solar = np.array([])  # solar spectrum (nu vs Fnu)

    # meta data
    has_meta_data = False  # whether or not any meta data has been set for this model
    meta_data = {}  # meta data for this model

    # weight for adding together models
    model_weight = 1

    ##########
    ## init ##
    ##########
    def __init__(self,
                 model_file='',
                 is_ised=False,
                 is_fits=False,
                 is_ascii=False,
                 has_masses=False,
                 units='a',
                 age_units='gyrs',
                 skip_load=False):
        """ model = ezgal.ezgal( model_file, is_ised=False, is_fits=False,
        is_ascii=False, has_masses=False, units='a', age_units='gyrs')

        Class for converting galaxy seds as a function of age to magnitudes as
        a function of redshift.
        Reads in bc03 ised files as well as ascii files, effectively replacing
        cm_evolution from the bc03 src files
        Specify a model file.  It should be a bc03 ised file, an ascii file, or
        a model file created with this class (which are stored as fits).
        See ezgal._load_ascii for information on formatting/units for ascii
        files
        This will automatically try to dtermine which type of file you have
        passed it.  If this fails, you can specify it manually with the is_*
        flags.

        units, age_units, and has_masses apply only for ascii files.  See
        ezgal._load_ascii()
        """

        # load additional modules.  Yes, this is strange.  But this way
        # ezgal_light can inherit ezgal.
        # this is necessary because ezgal_light is intended to work without any
        # of these modules
        try:
            from astropy.io import fits as pyfits
        except ImportError:
            import pyfits
        import scipy.integrate as integrate
        from . import cosmology, astro_filter, csp_integrator
        global cosmology
        global pyfits
        global integrate
        global astro_filter
        global csp_integrator

        # load a default cosmology
        self.set_cosmology()

        # clear filter list etc
        self.filters = {}
        self.filter_order = []
        self.interp_seds = []
        self.inter_zs = np.array([])
        self.zfs = np.array([])
        self.zs = np.array([])
        self.norm = {'norm': 0,
                     'z': 0,
                     'filter': '',
                     'vega': False,
                     'apparent': False}

        # save path to data folder: module directory/data
        self.data_dir = os.path.dirname(os.path.realpath(__file__)) + '/data/'

        # how about path to filter and model directories?
        self.filter_dir = False
        self.model_dir = False
        if 'ezgal_filters' in os.environ:
            self.filter_dir = os.environ['ezgal_filters']
        elif 'EZGAL_FILTERS' in os.environ:
            self.filter_dir = os.environ['EZGAL_FILTERS']
        if 'ezgal_models' in os.environ:
            self.model_dir = os.environ['ezgal_models']
        elif 'EZGAL_MODELS' in os.environ:
            self.model_dir = os.environ['EZGAL_MODELS']

        # make sure paths end with a slash
        if self.data_dir[-1] != os.sep: self.data_dir += os.sep
        if self.filter_dir and self.filter_dir[-1] != os.sep:
            self.filter_dir += os.sep
        if self.model_dir and self.model_dir[-1] != os.sep:
            self.model_dir += os.sep

        # attempt to load the vega spectrum
        vega_file = '%srefs/vega.fits' % self.data_dir
        if os.path.isfile(vega_file):
            fits = pyfits.open(vega_file)
            self.vega = np.column_stack(
                (fits[1].data.field('freq'), fits[1].data.field('flux')))
            self.has_vega = True
        else:
            self.vega = np.array([])
            self.has_vega = False

        # attempt to load the solar spectrum
        solar_file = '%srefs/solar.fits' % self.data_dir
        if os.path.isfile(solar_file):
            fits = pyfits.open(solar_file)
            self.solar = np.column_stack(
                (fits[1].data.field('freq'), fits[1].data.field('flux')))
            self.has_solar = True
        else:
            self.solar = np.array([])
            self.has_solar = False

        # load model
        if not skip_load:
            self._load(model_file,
                       is_ised=is_ised,
                       is_fits=is_fits,
                       is_ascii=is_ascii,
                       has_masses=has_masses,
                       units=units,
                       age_units=age_units)

    #####################
    ## return iterator ##
    #####################
    def __iter__(self):
        self.current_filter = -1
        return self

    #########################
    ## next() for iterator ##
    #########################
    def next(self):

        self.current_filter += 1
        if self.current_filter == len(self.filter_order): raise StopIteration

        filt = self.filter_order[self.current_filter]
        return (filt, self.filters[filt])

    #####################
    ## load model file ##
    #####################
    def _load(self,
              model_file,
              is_ised=False,
              is_fits=False,
              is_ascii=False,
              has_masses=False,
              units='a',
              age_units='gyrs'):

        # find the model file
        model_file = self._find_model_file(model_file)
        self.filename = model_file

        # Load input file depending on what type of file it is.

        # test for a bruzual-charlot binary ised file
        if model_file[len(model_file) - 5:] == '.ised' or is_ised:
            self._load_ised(self.filename)
        else:
            # And then the rest.
            print(model_file)
            fp = open(model_file, 'rb')
            start = fp.read(80)
            fp.close()
            if ( re.search( r'SIMPLE\s+=\s+T', start.decode('utf-8'), re.IGNORECASE ) \
                    or is_fits ) and not( is_ascii ):
                self._load_model(model_file)
            elif (start[0:5] == 'EzGal'):
                self._load_ascii_model(model_file)
            else:
                self._load_ascii(model_file,
                                 has_masses=has_masses,
                                 units=units,
                                 age_units=age_units)

        # always include a t=0 SED to avoid out of age interpolation errors later
        # a t=0 SED is also assumed during CSP generation
        if self.nages and self.ages.min() > 0:
            self.ages = np.append(0, self.ages)
            self.seds = np.hstack((np.zeros((self.nvs, 1)), self.seds))
            if self.has_masses: self.masses = np.append(0, self.masses)
            self.nages += 1

    ######################
    ## _find_model_file ##
    ######################
    def _find_model_file(self, file):

        # first check file path
        files = [file]

        # then model_dir if set in environment
        if self.model_dir:
            files.append('%s%s' % (self.model_dir, os.path.basename(file)))

        # finally model directory in data directory
        files.append('%smodels/%s' % (self.data_dir, os.path.basename(file)))

        # now loop through the different files
        for check in files:
            if os.path.isfile(check): return check

        raise ValueError('The specified model file, %s was not found!' % file)
        return False

    #####################
    ## set vega output ##
    #####################
    def set_vega_output(self):
        """ ezgal.set_vega_output()

        :Example:
            >>> import ezgal
            >>> model = ezgal.model( 'bc03_ssp_z_0.02_chab.model' )
            >>> # AB mag for ch1, zf=3, z=1
            >>> model.get_absolute_mags( 3.0, filters='ch1', zs=1.0 )
            5.6135203220610741
            >>> # set Vega output
            >>> model.set_vega_output()
            >>> # Vega mag for ch1, zf=3, z=1
            >>> model.get_absolute_mags( 3.0, filters='ch1', zs=1.0 )
            2.8262012027987748

        Set the default output for the ezgal object to Vega mags for all methods that can return magnitudes in vega or AB units.

        .. note::
            By default all ``EzGal`` methods output AB magnitudes.

        """

        if not self.has_vega:
            raise ValueError(
                'Cannot output vega mags: vega spectrum is not loaded!')
        self.vega_out = True

    ###################
    ## set AB output ##
    ###################
    def set_ab_output(self):
        """ ezgal.set_ab_output()

        :Example:
            >>> import ezgal
            >>> model = ezgal.model( 'bc03_ssp_z_0.02_chab.model' )
            >>> # AB mag for ch1, zf=3, z=1
            >>> model.get_absolute_mags( 3.0, filters='ch1', zs=1.0 )
            5.6135203220610741
            >>> # set Vega output
            >>> model.set_vega_output()
            >>> # Vega mag for ch1, zf=3, z=1
            >>> model.get_absolute_mags( 3.0, filters='ch1', zs=1.0 )
            2.8262012027987748
            >>> # and back to AB mags
            >>> model.set_ab_output()
            >>> model.get_absolute_mags( 3.0, filters='ch1', zs=1.0 )
            5.6135203220610741

        Set the default output for the ezgal object to AB mags for all methods that can return magnitudes in vega or AB units.

        .. note::
            By default all ``EzGal`` methods output AB magnitudes.
        """

        self.vega_out = False

    ################
    ## set masses ##
    ################
    def set_masses(self, ages, masses, age_units='gyrs', grid=True):
        """ ezgal.set_masses( ages, masses, age_units='gyrs', grid=True )

        :param ages: A list of ages
        :param masses: A list of corresponding masses
        :param age_units: The units that ``ages`` are in
        :type ages: list, array
        :type masses: list, array
        :type age_units: str

        Set the age-mass relationship for the model.  Pass a list of ages, the corresponding masses, and the units the ages are in.  Once the age-mass relationship is set, masses and mass-to-light ratios can be fetched using all the standard methods.  Also, age-mass relationships will be stored for any CSPs generated from the ``EzGal`` object.

        .. note::
            Masses are stored in the ezgal object and are gridded for every filter and formation redshift.  This is technically unnecessary since mass is independent of filter.  However, this guarantees that interpolation is done for masses exactly as for luminosity, thus removing potential errors from the mass-to-light ratios. """

        # sort by age
        sinds = ages.argsort()
        ages = ages[sinds]
        masses = masses[sinds]

        mass_ages = utils.convert_time(ages,
                                       incoming=age_units,
                                       outgoing='yrs')

        # is it exactly the same ages as stored in the SEDs?  If not, then interpolate
        if (self.ages.size != mass_ages.size or
                np.abs(self.ages - mass_ages).max() != 0):
            masses = np.interp(self.ages, mass_ages, masses, left=0)

        # store mass information
        self.has_masses = True
        self.masses = masses

        # and grid the filters
        if grid:
            self._grid_filters(masses=True)

    #####################
    ## check cosmology ##
    #####################
    def check_cosmology(self):
        if not self.cosmology_loaded:
            self.set_cosmology()
            print('Default cosmology loaded')

    ###################
    ## set cosmology ##
    ###################
    def set_cosmology(self, Om=0.272, Ol=0.728, h=0.704, w=-1):
        """ ezgal.set_cosmology( Om=0.272, Ol=0.728, h=0.704, w=-1 )

        Set the cosmology.  The default cosmology is from `WMAP 7 <http://adsabs.harvard.edu/abs/2011ApJS..192...18K>`_, Komatsu, et al. 2011, ApJS, 192, 18.  If the cosmology changes, then all filters will be regrided and any stored evolution models will be discarded.
        """

        # is the cosmology changing?
        if self.cosmology_loaded:
            # don't bother doing anything if the cosmology isn't actually changing
            if max(
                    np.abs(self.cosmo.Om - Om), np.abs(self.cosmo.Ol - Ol),
                    np.abs(self.cosmo.h - h),
                    np.abs(self.cosmo.w - w)) < self.tol:
                return True

        self.cosmo = cosmology.Cosmology(Om=Om, Ol=Ol, h=h, w=w)
        self.cosmo.lookup_tol = self.tol
        self.cosmology_loaded = True

        # pass cosmology object to all the filter objects
        for filter in self.filter_order:
            self.filters[filter].set_cosmology(self.cosmo)

        # now clear all the stored information
        self.clear_cache()

        # now force a regrid, since the cosmology has changed
        self._grid_filters(force=True)
        return True

    #############################
    ## set formation redshifts ##
    #############################
    def set_zfs(self, zfs, grid=True):
        """ ezgal.set_zfs( zfs, grid=True )

        :param zfs: A list of formation redshifts
        :type zfs: list, array
        :returns: None

        Set a list of formation redshifts for the model.  When filters are later added to the model, magnitude evolution will automatically be calculated for those filters at the set formation redshifts.  Also, magnitude evolution will be calculated for any already added filters at the given formation redshifts unless ``grid=False``.
        """

        zfs = np.asarray(zfs).ravel()
        if len(zfs.shape) == 0: zfs = np.array([zfs])
        if zfs.min() < 0:
            raise ValueError('Redshifts must be greater than zero!')

        self.zfs = zfs
        self.nzfs = self.zfs.size

        # build the iteration grid for all the filters
        if grid: self._grid_filters()

        self._set_zs()

    ##########################
    ## set redshift outputs ##
    ##########################
    def _set_zs(self, zf=None):
        """ ezgal._set_zs( zf=None )

        Set some default redshifts at which models should be evaluated.
        They will go out to z=zf.  The maximum set zf will be used if not zf is passed. """

        if zf is None and self.nzfs == 0:
            raise ValueError(
                'Please set the formation redshifts with ezgal.set_zfs() before setting redshift outputs')
        if zf is None: zf = self.zfs.max()

        # always keep the most inclusive zs possible
        if self.nzs > 0:
            if zf < max(self.zs): return True

        self.zs = self.get_zs(zf)
        self.nzs = self.zs.size
        return True

    #############################
    ## get apparent magnitudes ##
    #############################
    def get_apparent_mags(self,
                          zf,
                          filters=None,
                          zs=None,
                          normalize=True,
                          vega=None,
                          ab=None,
                          squeeze=True):
        """ mags = ezgal.get_apparent_mags( zf, filters=None, zs=None, normalize=True, vega=None, ab=None, squeeze=True )

        Same as :meth:`ezgal.ezgal.get_absolute_mags` excepts returns apparent magnitudes.

        .. note::
            This is equivalent to the apparent magnitude field ``m_AB_ev`` found in the output of the bruzual and charlot program ``cm_evolution``.
        .. seealso::
            :func:`ezgal.ezgal.get_absolute_mags`
        """

        return self._get_mags(zf,
                              kind='apparent',
                              filters=filters,
                              zs=zs,
                              normalize=normalize,
                              ab=ab,
                              vega=vega,
                              squeeze=squeeze)

    ######################################
    ## get observed absolute magnitudes ##
    ######################################
    def get_observed_absolute_mags(self,
                                   zf,
                                   filters=None,
                                   zs=None,
                                   normalize=True,
                                   vega=None,
                                   ab=None,
                                   squeeze=True):
        """ mags = ezgal.get_observed_absolute_mags( zf, filters=None, zs=None, normalize=True, vega=None, ab=None, squeeze=True )

        Same as :meth:`ezgal.ezgal.get_absolute_mags` excepts returns observed-frame absolute magnitudes.  The observed-frame absolute magnitude is the absolute magnitude of the model after being redshifted to the given redshift.

        .. note::
            This is the equiavlent to ``M_AB_ev`` from the output of the bruzual and charlot program ``cm_evolution``
        .. seealso::
            :meth:`ezgal.ezgal.get_absolute_mags`, :meth:`ezgal.ezgal.set_normalization`, :meth:`ezgal.ezgal.set_vega_output`, :meth:`ezgal.ezgal.set_ab_output`
        """

        return self._get_mags(zf,
                              kind='obs_abs',
                              filters=filters,
                              zs=zs,
                              normalize=normalize,
                              ab=ab,
                              vega=vega,
                              squeeze=squeeze)

    #############################
    ## get absolute magnitudes ##
    #############################
    def get_absolute_mags(self,
                          zf,
                          filters=None,
                          zs=None,
                          normalize=True,
                          ab=None,
                          vega=None,
                          squeeze=True):
        """ mags = ezgal.get_absolute_mags( zf, filters=None, zs=None, normalize=True, ab=None, vega=None, squeeze=True )

        :param zf: The formation redshift of the galaxy
        :param filters: A list of filters to calculate magnitudes for
        :param zs: A list of redshifts to calculate magnitudes at
        :param normalize: Whether or not to normalize the output
        :param ab: Whether or not to output in AB magnitues
        :param vega: Whether or not to output in Vega magnitudes
        :type zf: int, float
        :type filters: string, list of strings
        :type zs: int, float, list, array
        :type normalize: bool
        :type ab: bool
        :type vega: bool
        :returns: Array of magnitudes
        :rtype: array

        :Example:
            >>> import ezgal
            >>> model = ezgal.model( 'bc03_ssp_z_0.02_chab.model' )
            >>> model.add_filter( 'ch1' )
            >>> model.add_filter( 'ch2' )
            >>> model.add_filter( 'ch3' )
            >>> model.add_filter( 'ch4' )
            >>> model.get_absolute_mags( 3.0, zs=[0,1.0,2.0] )
            array([[ 6.39000031  6.83946654  7.25469915  7.80111159]
                   [ 5.61352032  6.07145866  6.49122232  7.04203427]
                   [ 4.48814389  4.87218428  5.30970258  5.85536578]])
            >>> model.get_absolute_mags( 3.0, filters=['ch1','ch2'], zs=[0,1.0,2.0] )
            array([[ 6.39000031,  6.83946654],
                   [ 5.61352032,  6.07145866],
                   [ 4.48814389,  4.87218428]])
            >>> model.get_absolute_mags( 3.0, filters='ch1', zs=[0,1.0,2.0] )
            array([ 6.39000031  5.61352032  4.48814389])
            >>> model.get_absolute_mags( 3.0, filters='ch1', zs=0 )
            6.39000030832


        Fetch the rest-frame absolute magnitudes for the given filter(s) at the given output redshifts (``zs``) for the given formation redshift (``zf``). If no filter is specified, then magnitudes will be fetched for all loaded filters. If the specified filters are not already loaded into ``EzGal``, it will attempt to load them automatically according to the rules in :meth:`ezgal.ezgal.add_filter`.

        If normalize=True then the returned magnitudes will be normalized if a normalization has been set with :meth:`ezgal.ezgal.set_normalization`.

        Returns an array of shape ``(len(zs),len(filters))`` with the absolute magnitudes. If there is only one filter, then it returns an array of shape ``(len(zs))``

        Call with vega=True or ab=True to specify ab or vega output.  If neither is set it will output AB mags unless :meth:`ezgal.ezgal.set_vega_output` has been called.

        .. note::
            If no output redshifts are specified, then the redshifts in ezgal.zs will be used.
        .. note::
            This is the equiavlent to ``M_AB_ev - k_cor_ev`` from the output of the bruzual and charlot program ``cm_evolution``
        .. seealso::
            :func:`ezgal.ezgal.set_normalization`, :func:`ezgal.ezgal.set_vega_output`, :func:`ezgal.ezgal.set_ab_output`
        """

        return self._get_mags(zf,
                              kind='absolute',
                              filters=filters,
                              zs=zs,
                              normalize=normalize,
                              ab=ab,
                              vega=vega,
                              squeeze=squeeze)

    ###################
    ## get kcorrects ##
    ###################
    def get_kcorrects(self, zf, filters=None, zs=None, squeeze=True):
        """ mags = ezgal.get_kcorrects( zf, filters=None, zs=None, squeeze=True )

        Same as :meth:`ezgal.ezgal.get_ecorrects` but returns the kcorrections.

        .. note::
            This is the equiavlent to ``k_cor_ev`` from the output of the bruzual and charlot program ``cm_evolution``.
        .. seealso::
            :func:`ezgal.ezgal.get_ecorrects`
        """

        return self._get_mags(zf,
                              kind='kcorrect',
                              filters=filters,
                              zs=zs,
                              normalize=False,
                              squeeze=squeeze)

    ###################
    ## get ecorrects ##
    ###################
    def get_ecorrects(self, zf, filters=None, zs=None, squeeze=True):
        """ mags = ezgal.get_ecorrects( zf, filters=None, zs=None, squeeze=True )

        :param zf: The formation redshift
        :param filters: The filters to calculate e-corrections for
        :param zs: The redshifts to calculate e-corrections at
        :type zf: int, float
        :type filters: string, list of strings
        :type zs: int, float, list, array
        :returns: The ecorrections
        :rtype: array

        :Example:
            >>> import ezgal
            >>> model = ezgal.model( 'bc03_ssp_z_0.02_chab.model' )
            >>> model.add_filter( 'ch1' )
            >>> model.add_filter( 'ch2' )
            >>> model.add_filter( 'ch3' )
            >>> model.add_filter( 'ch4' )
            >>> model.get_ecorrects( 3.0, zs=[0,0.5,1.0] )
            array([[ 0.        ,  0.        ,  0.        ,  0.        ],
                   [-0.34465634, -0.32867488, -0.32408269, -0.32054988],
                   [-0.77647999, -0.76800788, -0.76347683, -0.75907732]])
            >>> model.get_ecorrects( 3.0, filters=['ch1','ch2'], zs=[0,0.5,1.0] )
            array([[ 0.        ,  0.        ],
                   [-0.34465634, -0.32867488],
                   [-0.77647999, -0.76800788]])
            >>> model.get_ecorrects( 3.0, filters='ch1', zs=[0,0.5,1.0] )
            array([ 0.        , -0.34465634, -0.77647999])
            >>> model.get_ecorrects( 3.0, filters='ch1', zs=0 )
            0.0

        Fetch the e-corrections for the given filter(s) at the given output redshifts (``zs``) for the given formation redshift (``zf``). If no filter is specified, then magnitudes will be fetched for all loaded filters.  If the specified filters are not already loaded into ``EzGal``, it will attempt to load them automatically according to the rules in :meth:`ezgal.ezgal.add_filter`.

        returns an array of shape ``(len(zs),len(filters))`` with the kcorrects.  If there is only one filter, then it returns an array of shape ``(len(zs))``

        .. note::
            If no redshifts are specified, then the redshifts in ``ezgal.zs`` will be used. """

        return self._get_mags(zf,
                              kind='ecorrect',
                              filters=filters,
                              zs=zs,
                              normalize=False,
                              squeeze=squeeze)

    ###################
    ## get ecorrects ##
    ###################
    def get_ekcorrects(self, zf, filters=None, zs=None, squeeze=True):
        """ mags = ezgal.get_ecorrects( zf, filters=None, zs=None )

        Same as :meth:`ezgal.ezgal.get_ecorrects` but returns the e+k corrections.

        .. note::
            This is the equiavlent to ``e+k_cor`` from the output of the bruzual and charlot program ``cm_evolution``.
        .. seealso::
            :func:`ezgal.ezgal.get_ecorrects`
        """

        return self._get_mags(zf,
                              kind='ekcorrect',
                              filters=filters,
                              zs=zs,
                              normalize=False,
                              squeeze=squeeze)

    ##############
    ## get mags ##
    ##############
    def _get_mags(self,
                  zf,
                  kind='absolute',
                  filters=None,
                  zs=None,
                  ab=None,
                  vega=None,
                  normalize=True,
                  squeeze=True):
        """ mags = ezgal._get_mags( zf, kind='absolute', filters=None, zs=None, ab=None, vega=None, normalize=True )

        Fetch the given type of magnitude, using otherwise the same calling sequence as get_apparent_mags()
        """

        # load default values
        zs = self._populate_zs(zs, zf=zf)
        filters = self._populate_filters(filters)
        vega_out = self._get_vega_out(ab=ab, vega=vega)

        # make sure this formation redshift/filter combination is gridded
        if not self._check_grid(filters=filters, zfs=zf):
            raise ValueError(
                'Cannot calculate mags for given filter/zf combination because the seds have not been loaded!')

        # create data table for results
        mags = np.zeros((len(zs), len(filters)))

        # loop through filters one at a time.
        for (i, filter) in enumerate(filters):

            # now fetch the magnitudes
            if kind == 'absolute':
                mags[:, i] = self.filters[filter].get_absolute_mags(
                    zf, zs, vega=vega_out)
            elif kind == 'obs_abs':
                mags[:, i] = self.filters[filter].get_observed_absolute_mags(
                    zf, zs, vega=vega_out)
            elif kind == 'apparent':
                mags[:, i] = self.filters[filter].get_apparent_mags(
                    zf, zs, vega=vega_out)
            elif kind == 'kcorrect':
                mags[:, i] = self.filters[filter].get_kcorrects(zf, zs)
            elif kind == 'ecorrect':
                mags[:, i] = self.filters[filter].get_ecorrects(zf, zs)
            elif kind == 'ekcorrect':
                mags[:, i] = self.filters[filter].get_ekcorrects(zf, zs)

        # normalize models
        if normalize and (kind == 'absolute' or kind == 'apparent'):
            mags += self.get_normalization(zf)

        # squeeze return value if requested
        if squeeze: return self._squeeze(mags, len(filters))
        return mags

    #########################
    ## get rest M/L ratios ##
    #########################
    def get_rest_ml_ratios(self, zf, filters=None, zs=None, squeeze=True):
        """ mls = ezgal.get_rest_ml_ratios( zf, filters=None, zs=None )

        Same as :meth:`ezgal.ezgal.get_observed_ml_ratios` excepts returns the rest-frame M/L ratios.  Rest frame mass-to-light ratios are calculated using the rest-frame luminosity of the sun and the rest-frame luminosity of the model.

        .. seealso::
            :func:`ezgal.ezgal.get_observed_ml_ratios`
        .. warning::
            Only works if masses are set in the model.
        """

        if not self.has_masses:
            raise ValueError(
                'Cannot get mass-to-light ratios: masses were not found in the model file!')

        # load default values
        filters = self._populate_filters(filters)
        zs = self._populate_zs(zs)

        # fetch model magnitudes
        mags = self._get_mags(zf,
                              kind='absolute',
                              filters=filters,
                              zs=zs,
                              ab=True,
                              normalize=False,
                              squeeze=False)

        # calculate masses for these redshifts
        masses = self.get_masses(zf, zs, nfilters=len(filters), squeeze=False)

        # rest-frame absolute magnitude of sun
        solar_mags = self.get_solar_rest_mags(nzs=zs.size,
                                              filters=filters,
                                              ab=True,
                                              squeeze=False)

        # M/L ratio = mass/(L/Lsun) = mass/10**( -0.4*(M - Msun) )
        mls = masses / 10.0**(-0.4 * (mags - solar_mags))

        # return squeezed array?
        return self._squeeze(mls, len(filters)) if squeeze else mls

    #############################
    ## get observed M/L ratios ##
    #############################
    def get_observed_ml_ratios(self, zf, filters=None, zs=None, squeeze=True):
        """ mls = ezgal.get_observed_ml_ratios( zf, filters=None, zs=None )

        :param zf: The formation redshift
        :param filters: The filters to calculate M/L ratios for
        :param zs: The redshifts to calculate M/L ratios at
        :type zf: int, float
        :type filters: string, list of strings
        :type zs: int, float, list, array
        :returns: The M/L ratios
        :rtype: array

        :Example:
            >>> import ezgal
            >>> model = ezgal.model( 'bc03_ssp_z_0.02_chab.model' )
            >>> model.add_filter( 'sloan_u' )
            >>> model.add_filter( 'sloan_i' )
            >>> model.add_filter( 'ch1' )
            >>> model.get_observed_ml_ratios( 3.0, zs=[0,1.0,2.0] )
            array([[ 7.64179498,  2.52235684,  0.6858418 ],
                   [ 0.19372322,  2.24205801,  0.44824818],
                   [        nan,  0.22833461,  0.28408678]])

        Returns the observed-frame mass-to-light ratios as a function of ``zf``, ``z``, and ``filters``.  The observed-frame mass to light ratios is the stellar mass of the model given ``zf`` and ``z`` divided by the luminosity of the model in solar units.  For observed-frame M/L ratios the observed-frame luminosity of the model and the observed-frame luminosity of the sun are used.  The observed-frame luminosity of the sun is calculated by redshifting the solar spectrum to the given redshift and then projecting it through the filters.  This returns an array of shape ``(len(zs),len(filters))`` with the observed-frame mass-to-light ratios.  If there is only one filter, then it returns an array of shape ``(len(zs))``.

        .. note::
            Returns ``nan`` when the redshifted solar spectrum doesn't fully cover the filter.
        .. warning::
            Only works if masses are set in the model. """

        if not self.has_masses:
            raise ValueError(
                'Cannot get mass-to-light ratios: masses were not found in the model file!')

        # load default values
        filters = self._populate_filters(filters)
        zs = self._populate_zs(zs)

        # fetch model magnitudes
        mags = self._get_mags(zf,
                              kind='obs_abs',
                              filters=filters,
                              zs=zs,
                              ab=True,
                              normalize=False,
                              squeeze=False)

        # calculate masses for these redshifts
        masses = self.get_masses(zf, zs, nfilters=len(filters), squeeze=False)

        # observed-frame absolute magnitude of sun
        solar_mags = self.get_solar_observed_mags(zf,
                                                  filters=filters,
                                                  zs=zs,
                                                  ab=True,
                                                  squeeze=False)

        # M/L ratio = mass/(L/Lsun) = mass/10**( -0.4*(M - Msun) )
        mls = masses / 10.0**(-0.4 * (mags - solar_mags))

        # return squeezed array?
        return self._squeeze(mls, len(filters)) if squeeze else mls

    #########################
    ## get solar rest mags ##
    #########################
    def get_solar_rest_mags(self,
                            nzs=1,
                            filters=None,
                            ab=None,
                            vega=None,
                            squeeze=True):
        """ mags = ezgal.get_solar_rest_mags( nzs=None, filters=None, ab=None, vega=None )

        :param nzs: The number of redshifts to return solar rest-frame magnitudes for
        :param filters: The filters to calculate solar rest-frame magnitudes for
        :param zs: The redshifts to calculate M/L ratios at
        :param ab: Whether or not to output in AB magnitues
        :param vega: Whether or not to output in Vega magnitudes
        :type zf: int, float
        :type filters: string, list of strings
        :type zs: int, float, list, array
        :type ab: bool
        :type vega: bool
        :returns: Array of solar rest-frame magnitudes
        :rtype: array

        :Example:
            >>> import ezgal
            >>> model = ezgal.model( 'bc03_ssp_z_0.02_chab.model' )
            >>> model.add_filter( 'ch1' )
            >>> model.add_filter( 'ch2' )
            >>> model.add_filter( 'ch3' )
            >>> model.add_filter( 'ch4' )
            >>> model.get_solar_rest_mags( nzs=3 )
            array([[ 6.06644384,  6.56615057,  7.0455159 ,  7.66857929],
                   [ 6.06644384,  6.56615057,  7.0455159 ,  7.66857929],
                   [ 6.06644384,  6.56615057,  7.0455159 ,  7.66857929]])
            >>> model.get_solar_rest_mags( nzs=3, filters=['ch1','ch2'] )
            array([[ 6.06644384,  6.56615057],
                   [ 6.06644384,  6.56615057],
                   [ 6.06644384,  6.56615057]])
            >>> model.get_solar_rest_mags( nzs=1, filters='ch1' )
            6.0664438415489164

        Return the rest-frame absolute magnitude of the sun through the given filters.  Specify the number of redshifts to return an array of size ``(nzs,len(filters))``.  The solar rest-frame magnitude does not depend on redshift, so values are repeated in the column for each filter in the returned array.  This is done to match the return type of other methods such as :meth:`ezgal.ezgal.get_solar_observed_mags`.

        If nzs is None then returns an array of size ``len(filters)``.  If no filters are specified then it will calculate solar rest-frame magnitudes for all filters which have already been loaded into the model.

        Call with ``vega=True`` or ``ab=True`` to specify ab or vega output.  If neither is set it will output AB mags unless :meth:`ezgal.ezgal.set_vega_output` has been called.

        .. note::
            Returns NaN for undefined solar magnitudes. """

        # load default values
        filters = self._populate_filters(filters)
        vega_out = self._get_vega_out(ab=ab, vega=vega)

        mags = np.empty(len(filters))

        # fetch mags from filters
        for (i, filt) in enumerate(filters):
            # filter must be added
            if filt not in self.filters:
                self.add_filter(filt)
            if filt not in self.filters:
                raise ValueError('The filter %s could not be loaded!' % filt)

            if self.filters[filt].has_solar:
                mags[i] = self.filters[filt].solar
                if vega_out: mags[i] += self.filters[filt].to_vega
            else:
                mags[i] = np.nan

        # repeat nzs times
        mags = mags.reshape((len(filters), 1)).repeat(nzs, axis=1).transpose()

        if not squeeze: return mags
        return self._squeeze(mags, len(filters))

    #############################
    ## get solar observed mags ##
    #############################
    def get_solar_observed_mags(self,
                                zf,
                                filters=None,
                                zs=zs,
                                ab=None,
                                vega=None,
                                squeeze=True):
        """ mags = ezgal.get_solar_observed_mags( zf, filters=None, zs=zs, ab=None, vega=None, squeeze=True )

        :param zf: The formation redshift of the galaxy
        :param filters: A list of filters to calculate magnitudes for
        :param zs: A list of redshifts to calculate magnitudes at
        :param ab: Whether or not to output in AB magnitues
        :param vega: Whether or not to output in Vega magnitudes
        :type zf: int, float
        :type filters: string, list of strings
        :type zs: int, float, list, array
        :type ab: bool
        :type vega: bool
        :returns: Array of solar observed-frame magnitudes
        :rtype: array

        :Example:
            >>> import ezgal
            >>> model = ezgal.model( 'bc03_ssp_z_0.02_chab.model' )
            >>> model.add_filter( 'ch1' )
            >>> model.add_filter( 'ch2' )
            >>> model.add_filter( 'ch3' )
            >>> model.add_filter( 'ch4' )
            >>> model.get_solar_observed_mags( 3.0, zs=[0,1.0,2.0] )
            array([[ 6.06644384,  6.56615057,  7.0455159 ,  7.66857929],
                   [ 4.05625676,  4.44971653,  4.89681114,  5.50676563],
                   [ 3.35937661,  3.43865068,  3.72883098,  4.27894184]])
            >>> model.get_solar_observed_mags( 3.0, filters=['ch1','ch2'], zs=[0,1.0,2.0] )
            array([[ 6.06644384,  6.56615057],
                   [ 4.05625676,  4.44971653],
                   [ 3.35937661,  3.43865068]])
            >>> model.get_solar_observed_mags( 3.0, filters='ch1', zs=[0,1.0,2.0] )
            array([ 6.06644384,  4.05625676,  3.35937661])
            >>> model.get_solar_observed_mags( 3.0, filters='ch1', zs=0 )
            6.0664438415489164

        Return the observed-frame absolute magnitude of the sun given ``zf``, ``filters``, and ``zs``. Returns an array of size (zs.size,filters.size).  The observed-frame magnitude of the sun is the absolute magnitude of the sun after being redshifted to the given redshift.  As such the observed-frame solar magnitude of the sun does not actually depend on ``zf``.  However, ``zf`` is currently included in the calling sequence for consistency with other similar functions, such as :meth:`ezgal.ezgal.get_absolute_mags`.  Returns an array of shape ``(len(zs),len(filters))``. If there is only one filter, then it returns an array of shape ``(len(zs))``

        Call with ``vega=True`` or ``ab=True`` to specify ab or vega output.  If neither is set it will output AB mags unless :meth:`ezgal.ezgal.set_vega_output` has been called.

        .. note::
            Returns NaN for undefined solar magnitudes.
        .. note::
            The solar observed-frame magnitude does not depend on ``zf``, but this is included for consistency with other similar functions.
        .. warning::
            Requires ``zf > zs`` """

        # load default values
        zs = self._populate_zs(zs)
        filters = self._populate_filters(filters)
        vega_out = self._get_vega_out(ab=ab, vega=vega)

        # make sure this formation redshift/filter combination is gridded
        if not self._check_grid(filters=filters, zfs=zf, solar=True):
            raise ValueError(
                'Cannot calculate mags for given filter/zf combination because the seds have not been loaded!')

        # create data table for results
        mags = np.zeros((len(zs), len(filters)))

        # loop through filters one at a time.
        for (i, filter) in enumerate([filters]):
            mags[:, i] = self.filters[filter].get_solar_mags(zf,
                                                             zs,
                                                             vega=vega_out)

        if not squeeze: return mags
        return self._squeeze(mags, len(filters))

    ###################
    ## get kcorrects ##
    ###################
    def get_distance_moduli(self, zs=None, nfilters=None, squeeze=True):
        """ mags = ezgal.get_distance_moduli( zs=None, nfilters=None, squeeze=True )

        :param zs: The redshifts to return distance moduli for
        :param nfilters: The number of filters to return distance moduli for
        :type zs: int, float, list, array
        :type nfilters: int
        :returns: Distance Moduli
        :rtype: array

        :Example:
            >>> import ezgal
            >>> model = ezgal.model( 'bc03_ssp_z_0.02_chab.model' )
            >>> print model.get_distance_moduli( [0.1, 0.5, 1.0], nfilters=1 )
            array([ 38.30736787,  42.27018553,  44.1248392 ])
            >>> print model.get_distance_moduli( [0.1, 0.5, 1.0], nfilters=2 )
            array([[ 38.30736787,  38.30736787],
                   [ 42.27018553,  42.27018553],
                   [ 44.1248392 ,  44.1248392 ]])

        Fetch the distance moduli for the given redshifts.  Specify the number of filters to return an array of size (len(zs),nfilters).  If nfilters is None, then the number of filters will be taken to be the number of filters loaded in the object.  If there is only one filter, then it returns an array of shape (len(zs)).  The distance moduli does not depend on filter so values are repeated when ``nfilters>1``.  This is done for consistency with the return values of methods like :meth:`ezgal.ezgal.get_apparent_mags`.

        .. note::
            If no output redshifts are specified, then the redshifts in ezgal.zs will be used.
        """

        # load defaults
        zs = self._populate_zs(zs)
        if nfilters is None: nfilters = self.nfilters

        # calculate the distance moduli
        dms = np.empty(len(zs))
        for i in range(len(zs)):
            dms[i] = self.cosmo.DistMod(zs[i])

        dms = dms.reshape((len(zs), 1)).repeat(nfilters, axis=1)

        if not squeeze: return dms
        return self._squeeze(dms, nfilters)

    ################
    ## get masses ##
    ################
    def get_masses(self, zf, zs=None, nfilters=None, squeeze=True):
        """ masses = ezgal.get_masses( zf, zs, nfilters=None, normalize=True, squeeze=True )

        :param zf: The formation redshift
        :param zs: Redshifts to calculate masses at
        :param nfilters: The number of filters to return masses for
        :param normalize: Whether or not to normalize the returned masses
        :type zf: int, float
        :type zs: int, float, list, array
        :type nfilters: int
        :type normalize: bool
        :returns: The mass in solar masses
        :rtype: array

        :Example:
            >>> import ezgal
            >>> model.add_filter( 'ch1' )
            >>> model.add_filter( 'ch2' )
            >>> model.add_filter( 'ch3' )
            >>> model.add_filter( 'ch4' )
            >>> model = ezgal.model( 'bc03_ssp_z_0.02_chab.model' )
            >>> model.get_masses( 3.0, zs=[0, 1, 2] )
            array([[ 0.50909797,  0.50909797,  0.50909797,  0.50909797],
                   [ 0.54490264,  0.54490264,  0.54490264,  0.54490264],
                   [ 0.60169807,  0.60169807,  0.60169807,  0.60169807]])
            >>> model.get_masses( 3.0, zs=[0, 1, 2], nfilters=2 )
            array([[ 0.50909797,  0.50909797],
                   [ 0.54490264,  0.54490264],
                   [ 0.60169807,  0.60169807]])
            >>> model.get_masses( 3.0, zs=[0, 1, 2], nfilters=1 )
            array([ 0.50909797,  0.54490264,  0.60169807])
            >>> model.get_masses( 3.0, zs=0, nfilters=1 )
            0.50909797135560608

        Get the stellar mass (in solar masses) as a function of ``zf`` and ``z``.  Specify the number of filters to return an array of size ``(len(zs),nfilters)``.  If ``nfilters`` is None, then the number of filters will be taken to be the number of filters loaded in the object.  If there is only one filter, then it returns an array of shape ``(len(zs))``.  The mass is independent of filter, so masses are repeated across filters.  This is done for consistency with the return type of methods like :meth:`ezgal.ezgal.get_absolute_mags`.

        .. note::
            If no output redshifts are specified, then the redshifts in ezgal.zs will be used. """

        # load defaults
        zs = self._populate_zs(zs)
        if nfilters is None: nfilters = self.nfilters
        if nfilters == 0: nfilters = 1

        # use the gridded astro filter info to get masses for consistency with how luminosities are calculated
        # since masses are filter independent, we can use any filter to do this.  So just grab the first filter
        filt = self.filter_order[0]
        # make sure the masses are gridded
        if not self._check_grid(filters=self.filter_order[0],
                                zfs=zf,
                                masses=True):
            raise ValueError(
                'Cannot calculate mags for given filter/zf combination because the seds have not been loaded!')

        # get the masses and reshape given nfilters
        masses = self.filters[filt].get_masses(zf, zs).reshape(
            (len(zs), 1)).repeat(nfilters, axis=1)

        if not squeeze: return masses
        return self._squeeze(masses, nfilters)

    ##################
    ## get vega out ##
    ##################
    def _get_vega_out(self, ab=None, vega=None):
        """ vega_out = ezgal._get_vega_out( ab=None, vega=None )

        Returns true or false to specify whether or not mags should be in vega """

        if vega is None and ab is None:
            vega_out = True if self.vega_out else False
        elif vega is None:
            vega_out = False if ab else True
        else:
            vega_out = True if vega else False

        return vega_out

    #############
    ## squeeze ##
    #############
    def _squeeze(self, mags, nfilters):
        """ squeezed = ezgal._squeeze( mags, nfilters )

        Squeeze magnitude array according to number of filters """

        # nothing to do for nfilters > 1
        if nfilters > 1: return mags

        # if nfilters == 1 then return a list and not an array
        # however, if there is only one element, then just return that as a float
        if mags.size == 1: return float(mags)

        # squeeze!
        return np.squeeze(mags)

    #######################
    ## set normalization ##
    #######################
    def set_normalization(self, filter, z, mag, vega=False, apparent=False):
        """ ezgal.set_normalization( filter, z, mag, vega=False, apparent=False )

        :param filter: The normalization filter
        :param z: The normalization redshift
        :param mag: The normalization magnitude
        :param vega: Whether or not the normalization is in Vega mags
        :param apparent: Whether or not the normalization is in apparent magnitudes
        :type filter: str
        :type z: int, float
        :type mag: float
        :type vega: bool
        :type apparent: bool
        :returns: None

        :Example:
            >>> import ezgal
            >>> model = ezgal.model( 'bc03_ssp_z_0.02_chab.model' )
            >>> model.get_absolute_mags( 3.0, filters=['ch1','ch2'], zs=0.24 )
            array([[ 6.18394468,  6.63764649]])
            >>> # normalize to match a Dai et al. 2009 M* galaxy.
            >>> model.set_normalization( 'ch1', 0.24, -25.06, vega=True )
            >>> # also set Vega output, to match normalization
            >>> model.set_vega_output()
            >>> model.get_absolute_mags( 3.0, filters=['ch1','ch2'], zs=0.24 )
            array([[-25.06      , -25.07924584]])


        Set the model normalization, given the rest-frame magnitude of a galaxy in a given filter at a given redshift. Assumes AB magnitudes unless vega=True.
        Assumes rest-frame absolute mag unless apparent=True.

        .. note::
            Once set, the normalization can be unset by calling again with mag=0. """

        if not mag:
            self.norm = {'norm': 0,
                         'z': 0,
                         'filter': '',
                         'vega': False,
                         'apparent': False}
            return True

        if filter not in self.filters: self.add_filter(filter)

        self.norm = {'norm': float(mag),
                     'z': float(z),
                     'filter': filter,
                     'vega': bool(vega),
                     'apparent': bool(apparent)}

    #######################
    ## get normalization ##
    #######################
    def get_normalization(self, zf, flux=False):
        """ normalization = ezgal.get_normalization( zf, flux=False )

        :param zf: The formation redshift to assume
        :param flux: Wheter or not to return a multiplicative factor
        :type zf: int, float
        :type flux: bool
        :returns: The normalization
        :rtype: float

        :Example:
            >>> import ezgal
            >>> model = ezgal.model( 'bc03_ssp_z_0.02_chab.model' )
            >>> model.set_normalization( 'ch1', 0.24, -25.06, vega=True )
            >>> model.get_normalization( 3.0 )
            -28.45662555679969
            >>> # when set, normalizations are always applied by default
            >>> model.get_absolute_mags( 3.0, filters='ch1', zs=1.0 )
            -22.843105234738616
            >>> # manually apply normalization
            >>> model.get_absolute_mags( 3.0, filters='ch1', zs=1.0, normalize=False ) + model.get_normalization( 3.0 )
            -22.843105234738616
            >>> model.get_normalization( 3.0, flux=True )
            241351622492.22711
            >>> # calculate the mass (in solar masses) of the normalized galaxy at z=0.24 assuming zf=3.0
            >>> model.get_masses( 3.0, 0.24, nfilters=1 ) * model.get_normalization( 3.0, flux=True )
            124886846296.74387

        Returns the model normalization given formation redshift.  Returns normalization in units of magnitudes.  Set flux=True to return a multiplicative factor for multiplying SEDs or masses.  Returns 0 if no normalization has been set.

        .. seealso::
            :func:`ezgal.ezgal.set_normalization`
        """

        # nothing to do if not normalized.
        if not self.norm['norm'] or not self.norm['z'] or not self.norm[
                'filter']:
            return 1.0 if flux else 0.0

        # fetch the appropriate magnitude of a galaxy at the normalization redshift through the normalization filter.
        if self.norm['apparent']:
            mag = self._get_mags(zf,
                                 kind='apparent',
                                 filters=self.norm['filter'],
                                 zs=self.norm['z'],
                                 normalize=False,
                                 vega=self.norm['vega'])
        else:
            mag = self._get_mags(zf,
                                 kind='absolute',
                                 filters=self.norm['filter'],
                                 zs=self.norm['z'],
                                 normalize=False,
                                 vega=self.norm['vega'])

        # the difference between the normalization magnitude and the magnitude returned above is the normalization
        if not flux: return self.norm['norm'] - float(mag)

        # otherwise the normalization is that same difference converted to flux
        return 10.0**(-0.4 * (self.norm['norm'] - float(mag)))

    ###########################
    ## get mass weighted age ##
    ###########################
    def get_mass_weighted_ages(self, zf, zs=None, units='gyrs'):
        """ ezgal.get_mass_weighted_ages( zf, zs=None, units='gyrs' )

        :param zf: The formation redshift
        :param zs: The redshifts to calculate ages at
        :param units: The units to return ages in
        :type zf: int, float
        :type zs: int, float, list, array
        :type units: string
        :returns: The mass weighted ages
        :rtype: int, float, array

        :Example:
            >>> import ezgal
            >>> csp = ezgal.model( 'bc03_exp_10.0_z_0.02_chab.model' )
            >>> csp.get_mass_weighted_ages( 3.0, [0, 1.0, 2.0, 2.5] )
            array([ 10.6436012 ,   3.47012948,   1.079878  ,   0.44172018])
            >>> # compare to the time between zf and zs:
            >>> csp.get_age( 3.0, [0, 1.0, 2.0, 2.5] )
            array([ 11.55546768,   3.76705514,   1.1586272 ,   0.47989398])

        Returns the mass weighted age as a function of redshift and formation redshift.  Set units of returned ages with ``units``

        .. warning::
            Only works for CSP models generated with ``EzGal``. """

        if not self.has_sfh:
            raise ValueError(
                'Mass weighted ages can only be calculated for models with a stored star formation history!')

        # populate
        zs = self._populate_zs(zs)

        # calculate age at each redshift
        incoming_ages = self.get_age(zf, zs, units='yrs')

        # array to store mean ages
        outgoing_ages = np.zeros(zs.size)

        # loop through redshifts
        for i in range(zs.size):

            # find everything in SFH with age<age(zf,z)
            m = self.ages < incoming_ages[i]

            # make sure we matched something...
            if not m.sum(): continue

            # now calculate the mean age
            outgoing_ages[i] = np.sum((incoming_ages[i] - self.ages[m]) *
                                      self.sfh[m]) / np.sum(self.sfh[m])

        return utils.convert_time(outgoing_ages,
                                  incoming='yrs',
                                  outgoing=units)

    ###################################
    ## check if something is gridded ##
    ###################################
    def _check_grid(self, filters=None, zfs=None, solar=False, masses=False):
        """ ezgal._check_grid( filters=None, zfs=None, solar=False, masses=False )

        Will check to see if the given filters and formation redshifts are gridded.
        Will attempt to grid them if possible, and not already done.
        If solar=True it will also check to see if solar mags have been calculated.
        if masses=True it will also check to see if masses have been calculated. """

        # load defaults
        zfs = self._populate_zfs(zfs)
        filters = self._populate_filters(filters)

        # if we have the seds, just go ahead and grid this formation redshift - it won't recalculate anything already calculated
        if self.nvs and self.nages:
            self._grid_filters(filters=filters,
                               zfs=zfs,
                               solar=solar,
                               masses=masses)
        else:
            # if we don't have the seds, make sure these filters exist and already have the formation redshifts
            for filter in filters:
                if not (filter in self.filters): return False

                for zf in zfs:
                    # check that the filter has the given formation redshift
                    if not (self.filters[filter].has_zf(zf)): return False

        # if we got this far, then we're fine
        return True

    ###################
    ## _grid_filters ##
    ###################
    def _grid_filters(self,
                      filters=None,
                      zfs=None,
                      force=False,
                      solar=False,
                      masses=False):
        """ ezgal._grid_filters( filters=None, zfs=None, solar=False, masses=False )

        This will generate models for the given filters with the given formation redshifts.
        If nothing is specified for filters or zfs, then all filters and formation redshifts will be grided.
        You can pass a filter key or a filename.  In the latter case, the filter will be automatically added.
        If models have already been generated for a given filter and formation redshift, then they will not be regenerated, unless force=True
        If solar=True then solar observed magnitudes will also be gridded.
        If masses=True then masses will also be gridded. """

        filters = self._populate_filters(filters)
        zfs = self._populate_zfs(zfs)

        # is there anything to do?
        if len(filters) == 0: return True
        if len(zfs) == 0: return True

        if self.nages == 0 or self.nvs == 0:
            raise ValueError(
                'Cannot grid filters because no seds were found in the model file!')

        # now start gridding
        for filter in filters:
            # add the filter if it doesn't exist - don't bother gridding, since
            # that's what we're doing anyway
            if filter not in self.filters:
                # try to load it.
                self.add_filter(filter, grid=False)
                # okay, it really doesn't exist...
                if filter not in self.filters:
                    raise ValueError(
                        'The specified filter, %s, has not been loaded and cannot be found!'
                        % filter)

            for zf in zfs:

                # If this is already gridded (and we're not forcing a regrid) then skip gridding
                # and check and see if we need to grid solar magnitudes or masses
                if self.filters[filter].has_zf(zf) and not (force):

                    # if we don't need to check solar mags or masses, then we're all done
                    if not solar and not masses: continue

                    # grid solar magnitudes if requested and needed
                    if solar and not self.filters[filter].has_zf(zf,
                                                                 solar=True):
                        # if ezgal doesn't have solar mags then there is a problem...
                        if not self.has_solar:
                            raise ValueError(
                                'Cannot grid solar magnitudes - solar spectrum is not loaded!')
                        # okay, grid the solar mags
                        self.filters[filter].grid_solar(zf, self.solar[:, 0],
                                                        self.solar[:, 1])

                    # grid masses if requested and needed
                    if masses and not self.filters[filter].has_zf(zf,
                                                                  masses=True):
                        # if ezgal doesn't have masses then there is a problem...
                        if not self.has_masses:
                            raise ValueError(
                                'Cannot grid masses: mass-age relationship is not loaded!')
                        # okay, grid the masses
                        self.filters[filter].grid_masses(zf, self.ages,
                                                         self.masses)

                    # all done
                    continue

                # do the gridding!
                # fetch interpolated SEDs for this formation redshift
                (zs, ages, seds) = self._get_seds(zf)

                # grid this filter and formation redshift
                self.filters[filter].grid(zf,
                                          self.vs,
                                          zs,
                                          ages,
                                          seds,
                                          force=force)

                # grid solar mags if ezgal has them
                if self.has_solar:
                    self.filters[filter].grid_solar(zf, self.solar[:, 0],
                                                    self.solar[:, 1])

                # and grid masses if ezgal has them
                if self.has_masses:
                    self.filters[filter].grid_masses(zf, self.ages,
                                                     self.masses)

    ##################
    ## populate zfs ##
    ##################
    def _populate_zfs(self, zfs=None):
        """ ezgal._populate_zfs( zfs=None )

        Returns a list of formation redshifts - either the passed ones (as a numpy array), or the defaults for the model object """

        if zfs is not None:
            zfs = np.asarray(zfs)
            if len(zfs.shape) == 0: zfs = np.array([zfs])
        else:
            zfs = self.zfs

        return zfs

    #################
    ## populate zs ##
    #################
    def _populate_zs(self, zs=None, zf=5.0):
        """ ezgal._populate_zs( zs=None, zf=5.0 )

        Returns a list of zs - either the passed ones, or the defaults for the model object.
        Pass an optional formation redshift in case the default zs haven't been generated yet """

        if zs is None:
            if self.nzs == 0: self.set_zfs(zf)
            zs = self.zs
            nzs = self.nzs
        else:
            zs = np.asarray(zs)
            if len(zs.shape) == 0: zs = np.array([zs])
            nzs = zs.size

        return zs

    ######################
    ## populate filters ##
    ######################
    def _populate_filters(self, filters=None):
        """ ezgal._populate_filters( filters=None )

        Returns a list of filters - either the passed ones (as a string list), or the filters loaded for the model object """

        if filters is None: filters = self.filter_order
        if type(filters) == type(str('')): filters = [filters]

        return filters

    ################################
    ## retrieve the sed given age ##
    ################################
    def get_sed(self, age, age_units='gyrs', units='Fv'):
        """ sed = ezgal.get_sed( age, age_units='gyrs', units='Fv' )

        :param age: The desired age of the SED
        :param age_units: The units the age is in
        :param units: The output units for the SED
        :type age: int, float
        :type age_units: string
        :type units: string
        :returns: The SED
        :rtype: array

        :Example:
            >>> import ezgal
            >>> model = ezgal.model( 'bc03_ssp_z_0.02_chab.model' )
            >>> sed = model.get_sed( 10, age_units='gyrs', units='Fl' )
            >>> sed.size
            6900
            >>> model.ls.size, model.vs.size
            (6900, 6900)

        Returns the rest-frame sed of the galaxy at a given age.  See :func:`ezgal.utils.to_years` for the available age units.  Returns an array of size (model.nvs).  The frequencies (wavelengths) of each point in the returned array correspond to the frequencies (wavelengths) in model.vs (model.ls).

        Available output units are (case insensitive):

        ========== ====================
        Name       Units
        ========== ====================
        Jy         Jansky
        Fv         ergs/s/cm^2/Hz
        Fl         ergs/s/cm^2/Angstrom
        Flux       ergs/s/cm^2
        Luminosity ergs/s
        ========== ====================

        .. note::
            Uses linear interpolation between two nearest age models.
        """

        # don't interpolate outside of the age range
        minage = utils.to_years(self.ages[0], units=age_units, reverse=True)
        maxage = utils.to_years(self.ages[-1], units=age_units, reverse=True)
        if age < minage or age > maxage:
            raise ValueError('Age must be between %.2f and %.2f %s' %
                             (minage, maxage, age_units))
        age_yr = utils.convert_time(age, incoming=age_units, outgoing='yrs')

        # simple two point interpolation
        # find the closest SED younger than the given age
        yind = np.abs(self.ages - age_yr).argmin()
        if self.ages[yind] > age_yr: yind -= 1
        # ind of closest SED older than the given age
        oind = yind + 1

        # are we at the borders of the age array?  If so return
        if oind == self.nages: return self.seds[:, yind].copy()
        if yind < 0: return self.seds[:, oind].copy()

        # age of older and younger seds
        yage = self.ages[yind]
        oage = self.ages[oind]

        # now interpolate
        sed = self.seds[:, yind] + (self.seds[:, oind] - self.seds[:, yind]
                                    ) * (age_yr - yage) / (oage - yage)

        units = units.lower()
        if units == 'jy': return sed / 1e-23
        if units == 'fv': return sed
        if units == 'fl':
            return sed * self.vs**2.0 / utils.convert_length(utils.c,
                                                             outgoing='a')
        sed *= self.vs
        if units == 'flux': return sed
        if units == 'luminosity':
            return sed * 4.0 * np.pi * utils.convert_length(10,
                                                            incoming='pc',
                                                            outgoing='cm')**2.0
        raise NameError('Units of %s are unrecognized!' % units)

    ###############################
    ## retrieve sed given zf & z ##
    ###############################
    def get_sed_z(self,
                  zf,
                  z,
                  units='Fv',
                  normalize=True,
                  observed=False,
                  return_frequencies=False):
        """ sed = ezgal.get_sed_z( zf, z, units='Fv', normalize=True, observed=False, return_frequencies=False )

        :param zf: The formation redshift for the output SED
        :param z: The redshift for the output SED
        :param units: The output units for the SED
        :param normalize: Whether or not to normalize the output SED
        :param observed: Whether or not to output the observed-frame SED
        :param return_frequencies: Whether or not to return the corresponding frequencies
        :type zf: int, float
        :type z: int, float
        :type units: string
        :type normalize: bool
        :type observed: bool
        :type return_frequencies: bool
        :returns: The SED
        :rtype: array

        :Example:
            >>> import ezgal
            >>> model = ezgal.model( 'bc03_ssp_z_0.02_chab.model' )
            >>> sed = model.get_sed_z( 3.0, 0.5, units='Fl' )
            >>> sed.size
            6900
            >>> model.ls.size, model.vs.size
            (6900, 6900)
            >>> ( ls, sed ) = model.get_sed_z( 3.0, 0.5, units='Fl', return_frequencies=True )


        Returns the rest-frame sed for a galaxy given its formation redshift (``zf``) and current redshift (``z``). If ``normalize=True`` and a normalization has been set with :meth:`ezgal.ezgal.set_normalization`` then the sed will be normalized accordingly.  If ``observed=True`` then the observed frame SED is returned.  If ``return_frequencies=True`` then it also returns the frequencies of the points in the returned SED (or wavelength array in angstroms if ``units='Fl'``).  In this case a tuple is returned, with the first element being the list of frequencies, and the second element the SED.  Otherwise, the frequencies (wavelengths) of each point in the returned array correspond to the frequencies (wavelengths) in model.vs (model.ls).

        .. seealso::
            See ezgal.get_sed() for available output units. """

        units = units.lower()

        # fetch sed
        sed = self.get_sed(self.get_age(zf, z), units=units)

        # and normalize
        if normalize: sed *= self.get_normalization(zf, flux=True)

        # copy out frequencies in case they are going to be returned
        vs = self.vs.copy()
        # reverse if units are Fl so that frequencies are increasing in wavelength
        if units == 'fl':
            vs = utils.to_lambda(vs[::-1], units='a')
            sed = sed[::-1]

        # all done if observed=False
        if not observed:
            if not return_frequencies: return sed
            if units == 'fl': return (vs, sed)
            return (self.vs, sed)

        # we need to do different things depending on the units
        if units == 'fv' or units == 'jy':
            vs /= (1.0 + z)
            sed *= (1.0 + z)
        if units == 'fl':
            vs = vs * (1.0 + z)
            sed /= (1.0 + z)

        # in addition, if units are not luminosity, then we must account for distance modulus
        if units != 'luminosity':
            sed *= 10.0**(-0.4 * self.get_distance_moduli(z, nfilters=1))

        # and that should do it
        if return_frequencies: return (vs, sed)
        return sed

    ###############
    ## _get_seds ##
    ###############
    def _get_seds(self, zf):
        """ ezgal.get_seds( zf )

        Returns a list of SEDs and ages interpolated nicely in redshift space.
        Stores interpolated SEDs in object for quick retrieval later. """

        # Has this zf already been interpolated?
        w = np.where(np.abs(self.interp_zfs - zf) < self.tol)[0]

        # It has!  Return stored SEDs
        if w.size > 0:
            interped = self.interp_seds[w[0]]
            return (interped['zs'], interped['ages'], interped['seds'])

        # get regular redshift grid for this zf
        zs = np.append([0, 0.001], self.get_zs(zf))
        for (i, z) in enumerate(zs):
            # fetch age of sed at z given zf
            age = self.get_age(zf, z, units='yrs')
            # and fetch the SED
            sed = self.get_sed_z(zf, z, normalize=False).reshape((self.nvs, 1))
            # generate SED grid
            if i == 0:
                ages = np.array([age])
                seds = sed
                age_zs = np.array([z])
            else:
                # append so that ages are monotonically increasing
                ages = np.append(age, ages)
                seds = np.hstack((sed, seds))
                age_zs = np.append(z, age_zs)

        # store in the object, reverse redshifts since ages are monotonically increasing
        self.interp_seds.append({'ages': ages, 'seds': seds, 'zs': age_zs})
        self.interp_zfs = np.append(self.interp_zfs, zf)

        # and now return
        return (age_zs, ages, seds)

    #############
    ## get age ##
    #############
    def get_age(self, z1, z2, units='gyrs'):
        """ age = ezgal.get_age( z1, z2, units='gyrs' )

        :param z1: The first redshift
        :param z2: The second redshift
        :param units: The units to return the time in
        :type z1: int, float
        :type z2: int, float, list, array
        :type units: str
        :returns: Time between two redshifts
        :rtype: int, float, list, array

        :Example:
            >>> import ezgal
            >>> model = ezgal.model( 'bc03_ssp_z_0.02_chab.model' )
            >>> print model.get_age( 3.0, [0.0,0.5,1.0] )
            [ 11.55546768   6.49725355   3.76705514]

        returns the time difference between z1 and z2.  z2 can be a list or numpy array.  Primarily used to get the age of a galaxy at redshift z2, given formation redshift z1.  See :func:`ezgal.utils.to_years` for available age units.
        """

        if type(z2) == type([]) or type(z2) == type(np.array([])):
            ages = np.empty(len(z2))
            for i in range(len(z2)):
                if np.abs(z2[i] - z1) < self.tol:
                    ages[i] = 0
                else:
                    ages[i] = utils.to_years(
                        self.cosmo.Tl(z1, yr=True) - self.cosmo.Tl(z2[i],
                                                                   yr=True),
                        units=units,
                        reverse=True)
            return ages

        else:
            if np.abs(z2 - z1) < self.tol:
                return 0
            else:
                return utils.to_years(
                    self.cosmo.Tl(z1, yr=True) - self.cosmo.Tl(z2, yr=True),
                    units=units,
                    reverse=True)

    #################
    ## clear cache ##
    #################
    def clear_cache(self):

        # reset the list of interpolated SEDs
        self.interp_seds = []
        self.interp_zfs = np.array([])

        # reset all stored evolution info in filter objects
        for (filt, filt_obj) in self:
            filt_obj.clear_cache()

    ####################################################
    ## get a redshift grid going out to some redshift ##
    ####################################################
    def get_zs(self, z):
        """ ezgal.get_zs( z )

        :param z: The redshift out which to return redshifts
        :type z: int, float
        :returns: An array of redshifts
        :rtype: array

        :Example:
            >>> import ezgal
            >>> model = ezgal.model( 'bc03_ssp_z_0.02_chab.model' )
            >>> model.get_zs( 0.1 )
            array([ 0.005,  0.01 ,  0.015,  0.02 ,  0.025,  0.03 ,  0.035,  0.04 ,
                    0.045,  0.05 ,  0.055,  0.06 ,  0.065,  0.07 ,  0.075,  0.08 ,
                    0.085,  0.09 ,  0.095])

        Fetch a redshift grid out to redshift ``z``.  Returned redshifts stop just short of ``z``.  Returns more closely spaced redshifts at low redhift."""

        zs = np.arange(0.005, np.min([0.1, z]), 0.005)
        if z > 0.1:
            zs = np.concatenate((zs, np.arange(0.1, np.min([2.0, z]), 0.025)))
        if z > 2.0: zs = np.concatenate((zs, np.arange(2.0, z, 0.1)))
        return zs

    #####################
    ## extract filters ##
    #####################
    def extract_filters(self, filename, filters=None, grid=True):
        """ ezgal.extract_filters( filename, filters=None, grid=True )

        This will extract filter response curves saved in a binary fits file with an ezgal object.
        Pass a list of filters to extract - if no filters are passed, then all the filters found in the file will be added to the object.

        If grid is True, then the models will be generated for all filters at all formation redshifts.
        This can be a bit slow.  If not done now, it will be done on the fly as needed. """

        # load the file
        fits = pyfits.open(filename)

        # make sure it has some filters
        if not (fits[0].header.has_key('nfilters')):
            raise ValueError(
                'Cannot extract filters from specified file because it has none!')
        if fits[0].header['nfilters'] == 0:
            raise ValueError(
                'Cannot extract filters from specified file because it has none!')

        nfilters = fits[0].header['nfilters']

        # where do the response curves start?  If the file contains SEDs, then they start in extension index 3, otherwise 1
        start = 1
        if fits[0].header['has_seds']: start = 3

        # if we haven't asked for any filters then load up the list of filters from the file
        if filters is None:
            filters = []
            for i in range(nfilters):
                filters.append(fits[0].header['filter%d' % (i + 1)])
        # make sure it is an array
        if type(filters) == type(str('')): filters = [filters]

        # now loop through all the filters in the file, and add the ones in our filter list
        for i in range(nfilters):
            ind = i + start
            name = fits[ind].header['name']

            # see if we are keeping this filter
            if not (name in filters): continue

            # add!
            self.add_filter(fits[ind].data, name=name, units='hz')

    ###############################
    ## add filter to filter list ##
    ###############################
    def add_filter(self, file, name=None, units='a', grid=True):
        """ ezgal.add_filter( file, name=None, units='a', grid=True )

        :param file: The filename containing the filter response curve
        :param name: The name to store the filter as
        :param units: The length units for the wavelengths in the file
        :param grid: Whether or not to calculate evolution information when
        first added
        :type file: string
        :type name: string
        :type units: string
        :type grid: bool

        Add a filter for calculating models.  Specify the name of the file
        containing the filter transmission curve.  If the file is not found
        then ``EzGal`` will search for it in the directory specified by the
        ``EZGAL_FILTERS`` environment variable, and then in the
        ``data/filters`` directory in the ``EzGal`` module directory.

        The filter file should have two columns (wavelength,transmission).
        Wavelengths are expected to be in angstroms unless specified otherwise
        with ``units``.  See :func:`ezgal.utils.to_meters` for list of
        available units.

        Specify a name to refer to the filter as later.  If no name is
        specified, the filename is used (excluding path information and
        extension)
        If a filter already exists with that name, the previous filter will be
        replaced.

        If grid is True, then models will be generated for this filter at all
        set formation redshifts.

        You can pass a numpy array directly, instead of a file, but if you do
        this you need to specify the name.
        """

        if name is None:
            if type(file) != type(str('')):
                raise ValueError(
                    'You need to pass a file name or a numpy array and filter name!')
            name = os.path.basename(file)

        # if a file name was passed then search for the file in the various
        # directories
        if type(file) == type(str('')):
            file = self._find_filter_file(file)

        self.filters[name] = astro_filter(str(file),
                                          units=units,
                                          cosmology=self.cosmo,
                                          vega=self.vega,
                                          solar=self.solar)
        self.filters[name].tol = self.tol
        # store its name in self.filter_order
        if not self.filter_order.count(name): self.filter_order.append(name)
        self.nfilters += 1

        if grid: self._grid_filters(name)

    #######################
    ## _find_filter_file ##
    #######################
    def _find_filter_file(self, file):

        # first check file path
        files = [file]

        # then filter_dir if set in environment
        if self.filter_dir:
            files.append('%s%s' % (self.filter_dir, os.path.basename(file)))

        # finally filter directory in data directory
        files.append('%sfilters/%s' % (self.data_dir, os.path.basename(file)))

        # now loop through the different files
        for file in files:
            if os.path.isfile(file): return file

        raise ValueError(
            'The specified filter transmission file was not found!')
        return False

    ##########################
    ## save a model to fits ##
    ##########################
    def save_model(self, model_file, filter_info=True, filter_only=False):
        """ ezgal.save_model( output_file, filter_info=True, filter_only=False )

        :param model_file: Output filename
        :param filter_info: Whether or not to output calculated model evolution
        :param filter_only: If true, only output calculated model evolution.
        :type model_file: str
        :type filter_info: bool
        :type filter_only: bool
        :returns: None

        :Example:
            >>> import ezgal
            >>> model = ezgal.model( 'bc03_ssp_z_0.02_chab.model' )
            >>> model.add_filter( 'ch1' )
            >>> model.add_filter( 'ch2' )
            >>> model.set_zfs( [3,4,5] )
            >>> model.save_model( 'bc03_ssp_z_0.02_chab_with_models.model' )
            >>> new_model = ezgal.model( 'bc03_ssp_z_0.02_chab_with_models.model' )
            >>> new_model.get_absolute_mags( 3.0, zs=[0,1,2] )
            array([[ 6.39000031,  6.83946654],
                   [ 5.61352032,  6.07145866],
                   [ 4.48814389,  4.87218428]])

        Saves the ``EzGal`` model to a multi-extensions fits file which can be read back in later.

        If ``filter_info=True`` then this fits file will also contain the calculated model evolution as a function of filter and formation redshift.  This will be loaded back into the object later, so that you don't have to recalculate the models everytime.

        if ``filter_only=True`` then the SEDs will not be stored - just the calculated filter data.  This can later be loaded and return observables as a function of z for any formation redshifts and filters already computed.  These files are smaller because the model grid is not included, but ``EzGal`` will not be able to calculate observables for any new formation redshifts or filters.
        """

        if not filter_only:
            # primary hdu - sed table
            primary_hdu = pyfits.PrimaryHDU(self.seds)
            primary_hdu.header['units'] = 'ergs/s/cm^2/Hz'
            primary_hdu.header['has_seds'] = True

            # store the list of frequencies in a table
            vs_hdu = pyfits.BinTableHDU.from_columns(pyfits.ColDefs(
                [pyfits.Column(name='vs',
                               array=self.vs,
                               format='D',
                               unit='hertz')]))

            # the list of ages
            cols = [pyfits.Column(name='ages',
                                  array=self.ages,
                                  format='D',
                                  unit='years')]

            # the list of masses (if present)
            if self.has_masses:
                cols.append(pyfits.Column(name='masses',
                                          array=self.masses,
                                          format='D',
                                          unit='m_sun'))
            # and the sfh (if present)
            if self.has_sfh:
                cols.append(pyfits.Column(name='sfh',
                                          array=self.sfh,
                                          format='D'))

            # generate fits HDU
            ages_hdu = pyfits.BinTableHDU.from_columns(pyfits.ColDefs(cols))
            if self.has_masses: ages_hdu.header['has_mass'] = True
            if self.has_sfh: ages_hdu.header['has_sfh'] = True
        else:
            # we are only storing filter info, so use a blank primary hdu with some data in the header
            if self.nfilters == 0:
                raise ValueError(
                    'Cannot output only the filter information if there is no filter information!')
            primary_hdu = pyfits.PrimaryHDU()
            primary_hdu.header['has_seds'] = False
            primary_hdu.header['units'] = ''

        # and some info that we always want...
        primary_hdu.header['nfilters'] = 0
        primary_hdu.header['nzfs'] = 0

        # and meta data if set
        self._save_meta_data(primary_hdu.header)

        # store filter information as well
        if filter_info or filter_only:
            # Store the number of filters in the primary hdu
            primary_hdu.header['nfilters'] = self.nfilters
            # Also store the cosmology
            primary_hdu.header['Om'] = self.cosmo.Om
            primary_hdu.header['Ol'] = self.cosmo.Ol
            primary_hdu.header['w'] = self.cosmo.w
            primary_hdu.header['h'] = self.cosmo.h

            # Store the filter response curves, each in its own fits extension
            # as long as we're looping through, figure out all formation redshifts which have been used
            filter_hdus = []
            zfs = None
            for (i, filter) in enumerate(self.filter_order):
                image = pyfits.ImageHDU(np.column_stack((self.filters[
                    filter].vs, self.filters[filter].tran)))
                image.header['name'] = filter
                filter_hdus.append(image)
                zfs = self.filters[filter].extend_zf_list(zfs)

                # add the list of filter names to the header
                primary_hdu.header['filter%d' % (i + 1)] = filter

            # now loop through each formation redshift and store all necessary info for each filter
            if zfs is not None:
                zfs.sort()
                primary_hdu.header['nzfs'] = len(zfs)
                for zf in zfs:
                    # loop through the filters and retrieve any stored info for this filter at this formation redshift
                    count = 0
                    for filter in self.filter_order:
                        if not (self.filters[filter].has_zf(zf)): continue

                        # fetch the grid object for this filter and formation redshift
                        zf_grid = self.filters[filter].get_zf_grid(zf)

                        # for the first filter, get the redshifts, ages, masses, and start the column definitions
                        if count == 0:
                            zs = zf_grid.zs
                            cols = [pyfits.Column(name='zs',
                                                  array=zs,
                                                  format='D'),
                                    pyfits.Column(name='ages',
                                                  array=zf_grid.ages,
                                                  format='D')]
                            if zf_grid.has_masses:
                                cols.append(pyfits.Column(name='masses',
                                                          array=zf_grid.masses,
                                                          format='E'))

                        # add the rest-frame and observed-frame mags to the list of table columns
                        cols.extend([pyfits.Column(name=filter + '_rest',
                                                   array=zf_grid.rest,
                                                   format='E'),
                                     pyfits.Column(name=filter + '_obs',
                                                   array=zf_grid.obs,
                                                   format='E')])
                        # add observed-frame solar mags if they exists
                        if zf_grid.has_solar:
                            cols.append(pyfits.Column(name=filter + '_solar',
                                                      array=zf_grid.solar,
                                                      format='E'))

                        count += 1

                    # now generate the table extension and add it to the list
                    tbl = pyfits.BinTableHDU.from_columns(pyfits.ColDefs(cols))
                    tbl.header['zf'] = zf
                    filter_hdus.append(tbl)

        # join all the hdus together and write them out!
        hdus = [primary_hdu]
        if not filter_only: hdus.extend([vs_hdu, ages_hdu])
        if filter_info or filter_only: hdus.extend(filter_hdus)

        hdulist = pyfits.HDUList(hdus)
        hdulist.writeto(model_file, clobber=True)

    ############################
    ## load a model from fits ##
    ############################
    def _load_model(self, model_file):
        """ ezgal._load_model( model_file )

        loads a model from a fits file created with ezgal.save_model()
        Saves the model information in the model object """

        if not (os.path.isfile(model_file)):
            raise ValueError('The specified model file was not found!')

        fits = pyfits.open(model_file)

        # was sed information included in this model file?
        if fits[0].header['has_seds']:
            self.seds = fits[0].data
            self.vs = fits[1].data.field('vs')
            self.ls = utils.to_lambda(self.vs)
            self.ages = fits[2].data.field('ages')
            self.nvs = self.vs.size
            self.nls = self.ls.size
            self.nages = self.ages.size
            start_filters = 3
            # how about masses?
            if 'has_mass' in fits[2].header and fits[2].header['has_mass']:
                self.set_masses(self.ages,
                                fits[2].data.field('masses'),
                                age_units='yrs',
                                grid=False)
            # and sfh?
            if 'has_sfh' in fits[2].header and fits[2].header['has_sfh']:
                self.sfh = fits[2].data.field('sfh')
                self.has_sfh = True
        else:
            self.nvs = 0
            self.nls = 0
            self.nages = 0
            start_filters = 1

        # and meta info if set
        self.load_meta_data(fits[0].header)

        # was filter information included in this model file?
        # if so, load it and store it in the object
        if not ('nfilters' in fits[0].header): return True
        if fits[0].header['nfilters'] == 0: return True

        # set cosmology specified in the model file
        self.set_cosmology(Om=fits[0].header['Om'],
                           Ol=fits[0].header['Ol'],
                           h=fits[0].header['h'],
                           w=fits[0].header['w'])

        for i in range(start_filters,
                       start_filters + fits[0].header['nfilters']):
            name = fits[i].header['name']
            self.filters[name] = astro_filter.astro_filter(
                fits[i].data,
                units='hz',
                cosmology=self.cosmo,
                vega=self.vega,
                solar=self.solar)
            self.filter_order.append(name)
            self.nfilters += 1

        # load any stored models
        if not ('nzfs' in fits[0].header): return True

        st = start_filters + fits[0].header['nfilters']
        zfs = []
        for i in range(st, st + fits[0].header['nzfs']):
            data = fits[i].data
            zf = fits[i].header['zf']
            zs = data.field('zs')
            ages = data.field('ages')
            masses = None if not data.names.count('masses') else data.field(
                'masses')

            zfs.append(zf)

            # now loop through filters
            for filter in self.filter_order:
                # check to see if this filter has info stored for this zf
                if data.names.count(filter + '_rest') == 0: continue

                # store the info in the filter object
                solar = None if not data.names.count(
                    filter + '_solar') else data.field(filter + '_solar')
                self.filters[filter].store_grid(zf,
                                                zs,
                                                ages,
                                                data.field(filter + '_rest'),
                                                data.field(filter + '_obs'),
                                                solar=solar,
                                                masses=masses)

        self.set_zfs(zfs, grid=False)

    ######################
    ## save ascii model ##
    ######################
    def save_ascii_model(self, model_file):
        """ ezgal.save_ascii_model( model_file )

        Save the interpolated model information in an ascii file for later retrieval """

        # open the output file
        fp = open(model_file, 'wb')

        # write out basic header info
        fp.write(
            "EzGal ascii model file\nOriginal file:\n%s\nNumber Filters, Number of zfs:\n%d %d\nCosmology (Om, Ol, w, h):\n%.3f %.3f %.3f %.3f\n"
            % (self.filename, self.nfilters, self.nzfs, self.cosmo.Om,
               self.cosmo.Ol, self.cosmo.w, self.cosmo.h))

        # write out the filter names, vega-to-ab conversions, and solar magnitudes
        filters = []
        vegas = []
        solars = []
        for (i, filter) in enumerate(self.filter_order):
            filters.append(filter)
            vegas.append('%.4f' % self.filters[filter].to_vega)
            solars.append('%.4f' % self.filters[filter].solar)
        fp.write('%s\n%s\n%s\n' %
                 (' '.join(filters), ' '.join(vegas), ' '.join(solars)))

        # formation redshifts will be output in reverse order
        zfs = self.zfs.copy()
        zfs.sort()
        zfs = zfs[::-1]

        # write out the formation redshifts
        zf_strings = []
        for zf in zfs:
            zf_strings.append('%.3f' % zf)
        fp.write('%s\n' % ' '.join(zf_strings))

        # fetch dm, zs from the highest zf point, using any filter
        grid = self.filters[self.filter_order[0]].get_zf_grid(self.zfs.max())
        zs = grid.zs
        dms = self.get_distance_moduli(zs, nfilters=1)

        # generate a data array to store rest-frame mag evolution, observed-frame mag evolution, and observed-frame solar mag as a function of filter, zf, and z
        # need age and mass as a function of zf and z
        # also use a mask to denote where there are actually values
        res = np.zeros(
            (zs.size, 2 + self.nfilters * self.nzfs * 3 + 2 * self.nzfs))
        res[:, 0] = zs
        res[:, 1] = dms
        mask = np.zeros(
            (zs.size, 2 + self.nfilters * self.nzfs * 3 + 2 * self.nzfs))
        mask[:, 0] = 1
        mask[:, 1] = 1

        # output formats
        formats = ['%7.4f', '%7.4f']

        # loop through the zfs, then the filters
        for (zfind, zf) in enumerate(zfs):

            # fetch zs for this zf from any filter
            grid = self.filters[self.filter_order[0]].get_zf_grid(zf)
            my_zs = grid.zs

            # store age and mass
            age_ind = 2 + (zfind * self.nfilters) * 3 + (zfind) * 2
            res[0:my_zs.size, age_ind] = grid.ages
            res[0:my_zs.size, age_ind + 1] = grid.masses
            mask[0:my_zs.size, age_ind] = 1
            mask[0:my_zs.size, age_ind + 1] = 1
            formats.extend(['%14.8e', '%12.6e'])

            # now store gridded filter information
            for (find, filter) in enumerate(self.filter_order):

                start = 2 + (zfind * self.nfilters + find) * 3 + (zfind + 1
                                                                  ) * 2
                zf_grid = self.filters[filter].get_zf_grid(zf)

                # store observed-frame, rest-frame, and solar evolution in grid
                res[0:my_zs.size, start] = zf_grid.obs
                res[0:my_zs.size, start + 1] = zf_grid.rest
                res[0:my_zs.size, start + 2] = zf_grid.solar
                mask[0:my_zs.size, start] = 1
                mask[0:my_zs.size, start + 1] = 1
                mask[0:my_zs.size, start + 2] = 1
                formats.extend(['%7.4f', '%7.4f', '%7.4f'])

        # okay, now we just need to output this huge data array...
        for i in range(zs.size):
            # array for storing string data
            data = []
            # find actual data
            w = np.where(mask[i, :] == 1)[0]
            # convert to string
            for j in range(len(w)):
                data.append(formats[w[j]] % res[i, w[j]])
            # and write out line
            fp.write(' '.join(data) + '\n')

        # all done!
        fp.close()

    #######################
    ## _load_ascii_model ##
    #######################
    def _load_ascii_model(self, model_file):
        """ ezgal.load_ascii_model( model_file )

        Loads calculated model info outputted by ezgal to an ascii file.  This does not allow calculation of new models. """

        if not (os.path.isfile(model_file)):
            raise ValueError('The specified model file was not found!')

        fp = open(model_file, 'r')

        # no sed info
        self.nvs = 0
        self.nls = 0
        self.nages = 0

        # first two lines are junk
        j = fp.readline()
        j = fp.readline()

        # next line is filename
        self.filename = fp.readline().strip()

        # more junk, then number of filters and zfs
        j = fp.readline()
        (self.nfilters, self.nzfs) = fp.readline().strip().split()
        self.nfilters = int(self.nfilters)
        self.nzfs = int(self.nzfs)

        # cosmological parameters
        j = fp.readline()
        (om, ol, w, h) = fp.readline().strip().split()
        self.set_cosmology(Om=float(om), Ol=float(ol), h=float(h), w=float(w))

        # now read through and generate filter objects
        filters = fp.readline().strip().split()
        to_vegas = fp.readline().strip().split()
        solars = fp.readline().strip().split()
        for (filter, to_vega, solar) in zip(filters, to_vegas, solars):
            self.filters[filter] = astro_filter_light.astro_filter_light(
                None,
                cosmology=self.cosmo,
                vega=float(to_vega),
                solar=float(solar))
            self.filter_order.append(filter)

        # and formation redshifts
        self.zfs = np.array(fp.readline().strip().split()).astype('float')

        # okay, now start reading in the big data array
        zs = np.array([])
        dms = np.array([])
        c = 0
        while True:
            line = fp.readline()
            if line == '': break

            data = np.array(line.strip().split()).astype('float')
            # copy out redshift, distance moduli, and mag evolution
            zs = np.append(zs, data[0])
            dms = np.append(dms, data[1])
            evol = data[2:]

            # and store in data array
            # also keep mask to track what has data and what doesn't
            if c == 0:
                res = evol
                mask = np.ones(res.size)
                ncols = res.size
            else:
                this = np.zeros(ncols)
                this_mask = np.zeros(ncols)
                this[0:evol.size] = evol
                this_mask[0:evol.size] = 1
                res = np.vstack((res, this))
                mask = np.vstack((mask, this_mask))
            c += 1

        # now we can loop through data array by column and store evolution data in filters
        for (zfind, zf) in enumerate(self.zfs):

            # copy out age-mass relationship for this formation redshift
            age_ind = (zfind * self.nfilters) * 3 + (zfind) * 2

            # find good data
            w = np.where(mask[:, age_ind] > -1)[0]

            ages = res[w, age_ind]
            masses = res[w, age_ind + 1]
            if zfind == 0:
                self.set_masses(ages, masses, age_units='yrs', grid=False)

            for (find, filter) in enumerate(self.filter_order):

                # index for data in array
                start = (zfind * self.nfilters + find) * 3 + (zfind + 1) * 2
                # store in filter
                self.filters[filter].store_grid(
                    zf, zs[w], ages, res[w, start + 1], res[w, start], dms[w])
                zf_grid = self.filters[filter].get_zf_grid(zf)
                zf_grid.store_solar_mags(res[w, start + 2])
                zf_grid.store_masses(masses)

        # all done!
        fp.close()

    ################################
    ## load model from ascii file ##
    ################################
    def _load_ascii(self, file, has_masses=False, units='a', age_units='gyrs'):
        """ ezgal._load_ascii( file, has_masses=False, units='a', age_units='gyrs' )

        Load a model file in ascii format.  The file should be a data array of size (nwavelengths+1,nages+1).
        The first row specifies the age of each column, and the first column specifies the wavelength of each row.
        This means the data value in the first row of the first column is ignored.  However, it still must have SOME value ('0' is fine) as a placeholder
        You can include masses in the file by specifying the mass (in solar masses) for each age in the second row.  If you do this then set has_masses=True
        It loads a bit slow, so you should save it as a fits - see ezgal.save_model() - if you are going to be using it more than once.

        Specify units for the age with 'age_units'.  Default is gyrs.  See ezgal.utils.to_years() for avaialable unit specifications
        Specify units for wavelength & flux with 'units'.  Default is 'a' for Angstroms, with flux units of ergs/s/angstrom.
        Set units='hz' for frequency with flux units of 'ergs/s/hertz'
        You can also set the units as anything else found in ezgal.utils.to_meters() as long as the flux has units of ergs/s/(wavelength units)
        """

        if not (os.path.isfile(file)):
            raise ValueError('The specified model file was not found!')

        model = utils.rascii(file)

        self.vs = model[1:, 0]
        self.nvs = self.vs.size
        self.nls = self.nvs
        self.ages = model[0, 1:]
        self.nages = self.ages.size

        if has_masses:
            self.set_masses(ages, model[1, 1:], age_units=age_units)
            self.seds = model[1:, 2:]
        else:
            self.seds = model[1:, 1:]

        # convert to intermediate units (hz, ergs/s/hz)
        units = units.lower()
        age_units = age_units.lower()
        if units != 'hz':
            self.seds *= self.vs.reshape(
                (self.nvs, 1))**2.0 / utils.convert_length(utils.c,
                                                           outgoing=units)
            self.vs = utils.to_hertz(self.vs, units=units)

        self.ls = utils.to_lambda(self.vs)

        # convert from ergs/s/Hz to ergs/s/Hz/cm^2.0 @ 10pc
        self.seds /= 4.0 * np.pi * utils.convert_length(10,
                                                        incoming='pc',
                                                        outgoing='cm')**2.0

        # convert ages to the proper units
        self.ages = utils.to_years(self.ages, units=age_units)

        # now sort it to make sure that age is increasing
        sind = self.ages.argsort()
        self.ages = self.ages[sind]
        self.seds = self.seds[:, sind]

        # the same for frequency
        sind = self.vs.argsort()
        self.vs = self.vs[sind]
        self.seds = self.seds[sind, :]

    ###############################
    ## load model from ised file ##
    ###############################
    def _load_ised(self, file):
        """ ezgal._load_ised( file )

        Load a bruzual and charlot binary ised file.
        Saves the model information in the model object """

        # read ised file
        (seds, ages, vs) = utils.read_ised(file)

        # store ages
        self.ages = ages
        self.nages = ages.size

        # store frequencies/wavelengths
        self.nvs = vs.size
        self.nls = vs.size
        self.vs = vs
        self.ls = utils.to_lambda(self.vs)

        # store seds
        self.seds = seds

    ###################
    ## set meta data ##
    ###################
    def set_meta_data(self, data):
        """ ezgal.set_meta_data( data )

        :param data: A dictionary containing model information
        :type data: dict
        :returns: None

        :Example:
            >>> import ezgal
            >>> model = ezgal.model( 'bc03_ssp_z_0.02_chab.model' )
            >>> model.set_meta_data( {'model': 'BC03', 'sfh': 'SSP'} )
            >>> model.meta_data
            {'model': 'BC03', 'sfh': 'SSP'}

        Pass a dictionary with key/value pairs containing model information.  This information is stored along with the model data, and is restored when the model is loaded later.  Mainly, this serves as a method for propogating model information throughout python sessions.  It can also be used extensively with the ``Wrapper`` class for searching, sorting, and interpolating between models.

        .. warning::
            Meta data will be stored in the fits header so keep key length <= 8 characters. Keep value length <20 characters. """

        if type(data) != type({}) or len(data.keys()) == 0:
            raise ValueError('Please pass a dictionary of meta data')

        self.has_meta_data = True
        self.meta_data = {}
        # copy to meta data dictionary
        for (key, val) in data.items():
            if len(str(key)) > 8:
                raise ValueError(
                    'Meta data keys must be less than 9 characters long!')
            if len(str(val)) > 20:
                raise ValueError(
                    'Meta data values must be less than 21 characters long!')
            self.meta_data[str(key)] = str(val)

    ####################
    ## save meta data ##
    ####################
    def _save_meta_data(self, hdr):
        """ ezgal._save_meta_data( hdr )

        Stores the SPS model data in a fits header (if meta data is present) """

        if not self.has_meta_data: return hdr

        hdr['has_meta'] = True

        # store meta data in fits header
        for (key, val) in self.meta_data.iteritems():
            hdr[key] = (val, 'meta data')

        return hdr

    ####################
    ## load meta data ##
    ####################
    def load_meta_data(self, hdr):
        """ ezgal.load_meta_data( hdr )

        Saves the meta data in the fits header into the ezgal object (if present) """

        if not 'has_meta' in hdr or not hdr['has_meta']: return False

        # loop through all header cards in the header and look for ones with a comment that says 'meta data'
        self.meta_data = {}
        for key in hdr:
            if hdr.comments[key].strip() == 'meta data':
                self.meta_data[key.lower()] = hdr[key]

        self.has_meta_data = True
        return True

    ##############
    ## make csp ##
    ##############
    def make_csp(self,
                 sfh_function,
                 args=(),
                 dust_function=None,
                 dust_args=(),
                 break_points=None,
                 meta_data={},
                 max_err=0.001,
                 max_iter=200):
        """ new_model = model.make_csp( sfh_function, args=(), dust_function=None, dust_args=(), break_points=None, meta_data={}, max_err=0.001, max_iter=200 )

        :param sfh_function: The star formation rate as a function of age
        :param args: A tuple with additional arguments to pass to sfh_function
        :param dust_function: The dust dimming factor as a function of age and wavelength
        :param dust_args: A tuple with additional arguments to pass to dust_function
        :param break_points: A list of discontinutites in the star formation history
        :param meta_data: A dictionary with meta information to be stored in the new model
        :param max_err: Maximum allowed integration error (in magnitudes)
        :param max_iter: Maximum number of interations allowed when trying to get errors below ``max_err``
        :type sfh_function: function or callable object
        :type args: tuple
        :type dust_function: function or callable object
        :type dust_args: tuple
        :type break_points: list, array
        :type meta_data: dict
        :type max_err: float
        :type max_iter: int
        :returns: ``EzGal`` model object for new CSP
        :rtype: ezgal object

        :Example:
            >>> import ezgal
            >>> import numpy as np
            >>> model = ezgal.model( 'bc03_ssp_z_0.02_chab.model' )
            >>> def exponential( t, tau ):
            >>>    return np.exp( -1.0*t/tau )
            >>> # make a csp with a dust free exponentially decaying star formation history with tau=1.0 gyrs
            >>> exp_1gyr = model.make_csp( exponential, (1.0,) )

        Returns a new EzGal object which is a composite stellar population generated from an arbitrary SFH.  Pass a python function or callable object representing the star formation history (``sfh_function``). The star formation history function will be passed an age (in gyrs) and should return a corresponding weight (relative star formation rate).  ``args`` is an optional tuple and is passed to sfh_function.  The calling sequence for the star formation history function will be::

            weight = sfh_function( time, *args )

        `dust` is an optional parameter and can be any python function or callable object.  It should represent the dust extinction and should accept a time (in gyrs) and an array of wavelengths (in Angstroms).  It should return the dimming factor at all wavelengths in an array of size ``lambdas.size``.  If a tuple is passed to dust_args then these will be passed to dust_function as extra arguments.   The calling sequence for dust_function is::

            factor = dust_function( time, lambdas, *args )

        If the SFH history or dust function has discontinuities in time then the integral can be split up.  Use ``break_points`` to pass a list of ages (in Gyrs) to split the integral at.

        For integration, the SEDs in the model will be interpolated onto a more finely spaced grid in age.  Before integrating an iterative process is used at 3000, 8000, and 12000 angstroms to determine the proper level of age resampling.  The full integral is completed for these wavelength with increasing age resampling and the process stops when the difference in magnitude at all wavelengths from one iteration to the next drops below ``max_err`` (in magnitudes) or until the iteration has run ``max_iter`` times.

        In general finer age sampling (and therefore longer calculation times) are required for star formation histories with sharp bursts (i.e. short bursts of star formation).

        .. seealso::
            See :meth:`ezgal.ezgal.set_meta_data` for information about meta data.
        .. note::
            ``break_points`` is intended to decrease the amount of age resampling required for models with discontinuities, making for shorter execution times.  However, it is still under development and its utility is not yet established.
        .. warning::
            Discontinuties are always a source of trouble, and at the moment it is not clear whether or not ``make_csp`` is properly handling them.  In fact, it seems not to be.  ``break_points`` is my current attempt at dealing with this problem, but is still under development.  If you do model a star formation history with discontinuities then expect ``make_csp`` to hit the maximum iteration limit, and expect your model to take a very long amount of time to calculate.  Moreover, it still might not be correct. Test your results carefully. """

        # basic check - only generate CSPs from SSPs
        if self.has_meta_data and self.meta_data.has_key(
                'sfh') and self.meta_data['sfh'].lower() != 'ssp':
            raise ValueError(
                'Defiantly refusing to generate CSPs from anything other than SSPs. Meta data says this model has an alternate SFH: %s'
                % self.meta_data['sfh'])

        # load up a CSP integrator object (all ages should be in units of Gyrs)
        integrator = csp_integrator.csp_integrator(self.seds,
                                                   self.ls,
                                                   self.ages / 1e9,
                                                   self.cosmo.Tu(gyr=True),
                                                   break_points)

        # pass the star formation history stuff
        integrator.set_sfh(sfh_function, args)

        # the dust stuff
        if dust_function is not None:
            integrator.set_dust(dust_function, dust_args)

        # and the masses
        if self.has_masses: integrator.set_masses(self.masses)

        # calculate the amount of resampling needed
        resampling = integrator.find_resampling(max_err, max_iter)

        # integrate
        (seds, masses, sfh) = integrator.integrate(resampling)

        # and finally return the new model
        return self._return_new(seds, masses, sfh=sfh, meta_data=meta_data)

    ######################
    ## make exponential ##
    ######################
    def make_exponential(self,
                         tau,
                         dust_function=None,
                         dust_args=(),
                         max_err=0.001,
                         max_iter=200):
        """ new_model = model.make_exponential( tau, dust_function=None, dust_args=(), max_err=0.001, max_iter=200 )

        :param tau: Time scale for exponentially decaying burst of star formation (in gyrs).
        :param dust_function: The dust dimming factor as a function of age and wavelength
        :param dust_args: A tuple with additional arguments to pass to dust_function
        :param max_err: Maximum allowed integration error (in magnitudes)
        :param max_iter: Maximum number of interations allowed when trying to get errors below ``max_err``
        :type tau: float
        :type sfh_function: function or callable object
        :type args: tuple
        :type dust_function: function or callable object
        :type dust_args: tuple
        :type max_err: float
        :type max_iter: int
        :returns: ``EzGal`` model object for new CSP
        :rtype: ezgal object

        :Example:
            >>> import ezgal
            >>> import numpy as np
            >>> model = ezgal.model( 'bc03_ssp_z_0.02_chab.model' )
            >>> # dust free exponentially decaying burst with tau=1 gyr
            >>> exp_1 = model.make_exponential( 1.0 )

        Returns a new EzGal object which exponentially decaying SFH with a timescale of ``tau``.

        .. seealso::
            See :meth:`ezgal.ezgal.make_csp` for description of other parameters.
        """

        return self.make_csp(sfhs.exponential,
                             args=(tau, ),
                             meta_data={'sfh': 'Exponential',
                                        'tau': str(tau)},
                             dust_function=dust_function,
                             dust_args=dust_args,
                             max_err=max_err,
                             max_iter=max_iter)

    ################
    ## make burst ##
    ################
    def make_burst(self,
                   length,
                   dust_function=None,
                   dust_args=(),
                   max_err=0.001,
                   max_iter=200):
        """ new_model = model.make_burst( length, dust_function=None, dust_args=(), max_err=0.001, max_iter=200 )

        :param length: Length of burst (in gyrs)
        :param dust_function: The dust dimming factor as a function of age and wavelength
        :param dust_args: A tuple with additional arguments to pass to dust_function
        :param max_err: Maximum allowed integration error (in magnitudes)
        :param max_iter: Maximum number of interations allowed when trying to get errors below ``max_err``
        :type length: float
        :type sfh_function: function or callable object
        :type args: tuple
        :type dust_function: function or callable object
        :type dust_args: tuple
        :type max_err: float
        :type max_iter: int
        :returns: ``EzGal`` model object for new CSP
        :rtype: ezgal object

        :Example:
            >>> import ezgal
            >>> import numpy as np
            >>> model = ezgal.model( 'bc03_ssp_z_0.02_chab.model' )
            >>> # dust free 0.1 gyr constant burst
            >>> short_burst = model.make_burst( 0.1 )

        Returns a new EzGal object with a constant SFR from ``t=0`` to ``t=length``.  Length should be in gyrs.

        .. seealso::
            See :meth:`ezgal.ezgal.make_csp` for description of other parameters.
        .. warning::
            A discontinuity is present for star formation histories which are a constant burst, as the star formation drops from some finite value to zero at the end of the burst.  It is not yet clear if :meth:`ezgal.ezgal.make_csp` is properly handling discontinuities (see the warning there for more info), so test results from this method carefully before using them. """

        return self.make_csp(sfhs.constant,
                             args=(length, ),
                             dust_function=dust_function,
                             dust_args=dust_args,
                             break_points=length,
                             meta_data={'sfh': 'Burst',
                                        'length': str(length)},
                             max_err=max_err,
                             max_iter=max_iter)

    ##################
    ## make numeric ##
    ##################
    def make_numeric(self,
                     ages,
                     sfr,
                     age_units='gyrs',
                     dust_function=None,
                     dust_args=(),
                     break_points=None,
                     max_err=0.001,
                     max_iter=200):
        """ new_model = model.make_numeric( ages, sfr, age_units='gyrs', dust_function=None, dust_args=(), break_points=None, max_err=0.001, max_iter=200 )

        :param ages: A list of ages for the numeric star formation history.
        :param sfr: The star formation rate at each age in ``ages``.
        :param age_units: The age units for the ``ages`` array.
        :param dust_function: The dust dimming factor as a function of age and wavelength
        :param dust_args: A tuple with additional arguments to pass to dust_function
        :param max_err: Maximum allowed integration error (in magnitudes)
        :param max_iter: Maximum number of interations allowed when trying to get errors below ``max_err``
        :type ages: list, array
        :type sfr: list, array
        :type age_units: string
        :type sfh_function: function or callable object
        :type args: tuple
        :type dust_function: function or callable object
        :type dust_args: tuple
        :type max_err: float
        :type max_iter: int
        :returns: ``EzGal`` model object for new CSP
        :rtype: ezgal object

        :Example:
            >>> import ezgal
            >>> import numpy as np
            >>> model = ezgal.model( 'bc03_ssp_z_0.02_chab.model' )
            >>> # some arbitrary SFH
            >>> ages = [0,1,2,5,10,12]
            >>> sfr  = [0,2,5,8,20,0]
            >>> csp = model.make_numeric( ages, sfr, age_units='gyrs' )

        Returns a new EzGal object with a SFH determined from an arbitrary numeric star formation history.  ``ages`` should be an array of ages and ``sfr`` should be an array with the corresponding star formation rates as a function of time.  Expects the ages to be in units of gyrs, if not specify units with ``age_units`` (see :func:`ezgal.utils.to_years` for full list of available age units).  The star formation history does not have to be normalized.

        .. seealso::
            See :meth:`ezgal.ezgal.make_csp` for description of other parameters. """

        ages = np.asarray(ages)
        sfr = np.asarray(sfr)
        sinds = ages.argsort()
        return self.make_csp(
            sfhs.numeric(
                utils.convert_time(ages[sinds],
                                   incoming=age_units,
                                   outgoing='gyrs'),
                sfr[sinds]),
            dust_function=dust_function,
            dust_args=dust_args,
            break_points=break_points,
            meta_data={'sfh': 'Numeric'},
            max_err=max_err,
            max_iter=max_iter)

    ##################
    ## make delayed ##
    ##################
    def make_delayed(self, delay=0.0, age_units='gyrs'):
        """ new_model = model.make_delayed( delay, age_units='gyrs' )

        :param delay: length of delay.
        :param age_units: units of delay.
        :type delay: int, float
        :type age_units: string
        :returns: ``EzGal`` model with delayed star formation history
        :rtype: ``EzGal`` model object.

        :Example:
            >>> import ezgal
            >>> import numpy as np
            >>> model = ezgal.model( 'bc03_ssp_z_0.02_chab.model' )
            >>> delayed_ssp = model.make_delayed( 1.0 )

        Return a new model with a delay before the onset of star formation. Expects ``delay`` to be in units of ``age_units``.  See :func:`ezgal.utils.to_years` for available age units.

        This is intended for generating CSPs with complicated star formation histories, as it allows you to add together bursts of star formation which begin at different times.

        .. note::
            Uses interpolation to add in delay.
        """

        # generate arrays for holding new seds/masses/sfhs
        seds = np.zeros(self.seds.shape)
        masses = np.zeros(self.nages) if self.has_masses else None
        sfh = np.zeros(self.nages) if self.has_sfh else None

        # convert delay to years
        delay = utils.convert_time(delay, incoming=age_units, outgoing='yrs')
        # and check
        if delay < self.ages.min() or delay > self.ages.max():
            raise ValueError(
                "Delay before onset of star formation must be between %f and %f gyrs"
                % (self.ages.min() / 1e9, self.ages.max() / 1e9))

        # now generate new SEDs using interpolation
        w = np.where(self.ages > delay)[0]
        for ind in w:
            seds[:, ind] = self.get_sed(self.ages[ind] - delay,
                                        age_units='yrs')
            if self.has_masses:
                masses[ind] = np.interp(self.ages[ind] - delay, self.ages,
                                        self.masses)
            if self.has_sfh:
                sfh[ind] = np.interp(self.ages[ind] - delay, self.ages,
                                     self.sfh)

        return self._return_new(seds,
                                masses,
                                sfh,
                                meta_data={'delay': utils.convert_time(
                                    delay,
                                    incoming='yrs',
                                    outgoing='gyrs')})

    #################
    ## _return_new ##
    #################
    def _return_new(self,
                    seds,
                    masses=None,
                    sfh=None,
                    meta_data=None,
                    weight=1):
        """ new_model = model._return_new( seds, masses=None, sfh=None, meta_data=meta_data, weight=1 )

        Return a new EzGal object with the given SEDs, masses, and sfh but with everything else the same.
        seds.shape must equal model.seds.shape.  Does not maintain filter information or stored models. """

        # generate a blank EzGal object to work with
        model = ezgal(skip_load=True)

        # store all SED info, and make sure to actually copy
        model.nages = self.nages
        model.nvs = self.nvs
        model.nls = self.nls
        model.ages = self.ages.copy()
        model.vs = self.vs.copy()
        model.ls = self.ls.copy()
        model.seds = seds

        # cosmology
        model.set_cosmology(Om=self.cosmo.Om,
                            Ol=self.cosmo.Ol,
                            h=self.cosmo.h,
                            w=self.cosmo.w)

        # star formation history
        if sfh is not None:
            model.has_sfh = True
            model.sfh = sfh

        # masses
        if masses is not None:
            model.has_masses = True
            model.masses = masses

        # meta data
        items = []
        if self.has_meta_data: items += self.meta_data.items()
        # do new meta data last so that it overwrites old meta data
        if meta_data is not None and type(meta_data) == type({}):
            items += meta_data.items()
        if len(items): model.set_meta_data(dict(items))

        # weight
        model.model_weight = weight

        return model

    ##########
    ## copy ##
    ##########
    def copy(self):
        """ copy = model.copy()

        :returns: A copy of the model
        :rtype: ``EzGal`` model object

        :Example:
            >>> import ezgal
            >>> model = ezgal.model( 'bc03_ssp_z_0.02_chab.model' )
            >>> model_copy = model.copy()

        Returns a copy of the ``EzGal`` object without any stored filter information."""

        masses = self.masses.copy() if self.has_masses else None
        sfh = self.sfh.copy() if self.has_sfh else None

        return self._return_new(self.seds.copy(),
                                masses,
                                sfh,
                                weight=self.model_weight)

    ############
    ## weight ##
    ############
    def weight(self, weight):
        """ new_model = model.weight( weight )

        :param weight: Weight to apply to model
        :type weight: int, float
        :returns: Weighted ``EzGal`` object
        :rtype: ``EzGal`` model object

        :Example:
            >>> import ezgal
            >>> model = ezgal.model( 'bc03_ssp_z_0.02_chab.model' )
            >>> heavy = model.weight( 2 )
            >>> heavier = heavy.weight( 4 )
            >>> model.model_weight, heavy.model_weight, heavier.model_weight
            (1, 2, 8)

        Return a copy of the EzGal model weighted by an additional factor of ``weight``. """

        new = self.copy()
        new.model_weight *= weight
        return new

    #############
    ## __mul__ ##
    #############
    def __mul__(self, obj):
        if type(obj) == type(self):
            return self.weight(obj.model_weight)
        elif type(obj) == type(weight.weight(1)):
            return self.weight(obj.weight)
        else:
            return self.weight(obj)

    ##############
    ## __imul__ ##
    ##############
    def __imul__(self, obj):
        if type(obj) == type(self):
            self.model_weight *= obj.model_weight
        elif type(obj) == type(weight.weight(1)):
            self.model_weight *= obj.weight
        else:
            self.model_weight *= obj
        return self

    #############
    ## __add__ ##
    #############
    def __add__(self, obj):
        new = self.copy()
        new += obj
        return new

    ##############
    ## __iadd__ ##
    ##############
    def __iadd__(self, obj):
        if type(obj) != type(self):
            raise TypeError(
                'EzGal model objects can only be added with other EzGal model objects!')
        if self.seds.shape != obj.seds.shape or np.max(np.abs(
                self.ages - obj.ages)) > 0 or np.max(np.abs(self.vs -
                                                            obj.vs)) > 0:
            raise TypeError(
                'EzGal model objects can only be added if they have the same age/wavelength grid!')

        # add together SEDs
        tot_weight = self.model_weight + obj.model_weight
        self.seds *= self.model_weight
        self.seds += obj.model_weight * obj.seds
        self.seds /= tot_weight

        # add together masses
        if self.has_masses and obj.has_masses:
            self.masses *= self.model_weight
            self.masses += obj.model_weight * obj.masses
            self.masses /= tot_weight
        else:
            self.has_masses = False
            self.masses = []

        # and SFHs
        if self.has_sfh and obj.has_sfh:
            self.sfh *= self.model_weight
            self.sfh += obj.model_weight * obj.sfh
            self.sfh /= tot_weight
        else:
            self.has_sfh = False
            self.sfh = []

        # add together weights and return
        self.model_weight = tot_weight
        return self


def usage():
    print(
        "ezgal.py [-k -d -b -p --by_filter --prefix=prefix] model_file zf1 zf2 ... zfn filter1 filter2 ... filtern")
    print("")
    print("all command line flags must come first")
    print(
        "pass the input model file, a list of formation redshifts, and a list of filter transmission curves (filenames)")
    print("specify an optional prefix for the output filenames")
    print(
        "by default it outputs distance moduli, kcorrections, absolute mags, and apparent mags for each filter")
    print(
        "use the -k, -b, -p, or -d  options to have it output kcorrections, absolute mags, apparent mags, or distance moduli")
    print(
        "If you specify the --by_filter flag, and are only outputting one column mag data (-k,-b,-p) then an output file will be created for each filter instead of each formation redshift")
    print("")
    print("Use these flags:")
    print(
        "--norm=mag --norm_filter=filter --norm_z=z --norm_vega --norm_apparent")
    print(
        "to set the normalization to be a magnitude of norm through filter norm_filter at redshift norm_z.")
    print(
        "Normalization mags are assumed to be in absolute AB mags unless --norm_vega or --norm_apparent is set.")

################################
## let's make this executable ##
################################
if __name__ == '__main__':
    import sys, getopt, re

    # check arguments
    try:
        (opts, args) = getopt.getopt(sys.argv[1:], 'fr:m:z:t:aVvwlhkbpd', [
            'by_filter', 'prefix=', 'norm=', 'norm_z=', 'norm_filter=',
            'norm_apparent', 'norm_vega', 'vega', 'Om=', 'Ol=', 'h='
        ])
    except getopt.GetoptError as err:
        print(str(err))
        usage()
        sys.exit(2)

    if len(sys.argv) < 2:
        usage()
        sys.exit()

    # make sure the model file exists
    model_file = args[0]
    if not (os.path.isfile(model_file)):
        print('The specified model file does not exist!')
        usage()
        sys.exit()

    # check flags
    norm = 0.0
    norm_z = 0.0
    norm_filter = ''
    norm_apparent = False
    group = 'zf'
    out = 'all'
    prefix = ''
    norm_vega = False
    vega = False
    om = 0.279
    ol = 0.721
    h = 0.701
    for (o, a) in opts:
        if o == '-f' or o == '--by_filter': group = 'filter'
        if o == '-k': out = 'kcor'
        if o == '-b': out = 'abs'
        if o == '-p': out = 'app'
        if o == '-d': out = 'dm'
        if o == '-r' or o == '--prefix': prefix = a
        if o == '-m' or o == '--norm': norm = float(a)
        if o == '-z' or o == '--norm_z': norm_z = float(a)
        if o == '-t' or o == '--norm_filter': norm_filter = a
        if o == '-a' or o == '--norm_apparent': norm_apparent = True
        if o == '-V' or o == '--norm_vega': norm_vega = True
        if o == '-v' or o == '--vega': vega = True
        if o == '-w' or o == '--Om': om = float(a)
        if o == '-l' or o == '--Ol': ol = float(a)
        if o == '-h' or o == '--h': h = float(a)

    # load the model
    model = ezgal(model_file)

    # set the cosmology
    model.set_cosmology(Om=om, Ol=ol, h=h)

    # loop through arguments and find formation redshifts/filters
    zfs_in = []
    filters = []
    nfilters = 0
    for arg in args[1:]:
        # are we still looking for formation redshifts?
        if nfilters == 0:
            # is this a number?
            if re.match('[\-+]?\d{1,}\.?\d{,}([eEdD]\d+)?$', arg):
                zfs_in.append(arg)
                continue

        # if the filter isn't already in the model then make sure there is a file for it
        if not (model.filters.has_key(arg)) and not (os.path.isfile(arg)):
            raise ValueError('Could not load the filter %s!' % arg)
        filters.append(arg)
        nfilters += 1

    # convert to float and sort formation redshifts
    zfs = np.asarray(zfs_in).astype('float')
    sinds = zfs.argsort()
    zfs = zfs[sinds]
    zfs_in = np.asarray(zfs_in)[sinds]

    # time to actually do the calculations
    if len(zfs_in): model.set_zfs(zfs, grid=False)
    for filter in filters:
        if not (model.filters.has_key(filter)): model.add_filter(filter)

    # was normalization info passed?
    if norm and norm_z and norm_filter:
        model.set_normalization(norm_filter,
                                norm_z,
                                norm,
                                vega=norm_vega,
                                apparent=norm_apparent)

    # output vega mags?
    if vega: model.set_vega_output()

    # if no filters were passed, see if we can load them from the model
    if nfilters == 0:
        if model.nfilters == 0:
            usage()
            sys.exit()

        filters = model.filter_order
        nfilters = model.nfilters

    # if no zfs were passed, see if we can load them from the model
    if len(zfs_in) == 0:
        if model.nzfs == 0:
            usage()
            sys.exit()

        zfs = model.zfs
        zfs_in = [''] * zfs.size
        for i in range(zfs.size):
            zfs_in[i] = '%.3f' % zfs[i]

    system = 'vega' if vega else 'ab'

    # create an output file for each filter
    if group == 'filter' and out != 'all' and out != 'dm':
        # add a prefix if there isn't one, otherwise the filter files will be overwritten
        if prefix == '': prefix = 'ev_'

        for (find, filter) in enumerate([filters]):
            # file header
            display = 'Apparent Mags'
            if out == 'kcor': display = 'Kcorrects'
            if out == 'abs': display = 'Absolute Mags'
            header = '# %s, filter = %s, Om = %.4f, Ol = %.4f, h = %.4f, w = %.4f\n# input file = %s\n# All mangitudes are on the %s system\n#  zf= ' % (
                display, filter, model.cosmo.Om, model.cosmo.Ol, model.cosmo.h,
                model.cosmo.w, model.filename, system)
            header += ' '.join('%7s' % zf for zf in zfs_in)

            # now load data
            output = np.empty((model.zs.size, zfs.size + 1))
            output[:, 0] = model.zs
            for (zfind, zf) in enumerate(zfs):
                if out == 'app':
                    output[:, zfind + 1] = model.get_apparent_mags(
                        zf, filters=filter)
                if out == 'abs':
                    output[:, zfind + 1] = model.get_absolute_mags(
                        zf, filters=filter)
                if out == 'kcor':
                    output[:, zfind + 1] = model.get_kcorrects(zf,
                                                               filters=filter)

            # don't output stuff if everything is zero
            m = np.abs(output[:, 1:]).max(axis=1) != 0

            # and output
            utils.wascii(output[m, :],
                         '%s%s' % (prefix, filter),
                         formats='%7.4f',
                         header=header)

    else:
        if group == 'filter':
            print(
                'Ignoring group by filter specification - that only works for single column ouput!\n')

        # create an output file for each formation redshift
        for (zfind, zf) in enumerate(zfs):
            header = '# zf = %s, Om = %.4f, Ol = %.4f, h = %.4f, w = %.4f\n# input file = %s\n# All magnitudes are on the %s system\n#  1: z' % (
                zfs_in[zfind], model.cosmo.Om, model.cosmo.Ol, model.cosmo.h,
                model.cosmo.w, model.filename, system)
            zs = model.zs.reshape((model.nzs, 1))

            # output apparent magnitudes
            if out == 'app':
                output = np.hstack(
                    (zs, model.get_apparent_mags(zf, filters=filters).reshape(
                        (model.nzs, len(filters)))))
                for (i, filter) in enumerate(model.filter_order):
                    header += '\n# %2d: Apparent Mag %s' % (i + 2, filter)

    # output kcorrections
            if out == 'kcor':
                output = np.hstack(
                    (zs, model.get_kcorrects(zf, filters=filters).reshape(
                        (model.nzs, len(filters)))))
                for (i, filter) in enumerate(model.filter_order):
                    header += '\n# %2d: Kcorrect %s' % (i + 2, filter)

    # output absolute mags
            if out == 'abs':
                output = np.hstack(
                    (zs, model.get_absolute_mags(zf, filters=filters).reshape(
                        (model.nzs, len(filters)))))
                for (i, filter) in enumerate(model.filter_order):
                    header += '\n# %2d: Absolute Mag %s' % (i + 2, filter)

    # output distance moduli
            if out == 'dm' or out == 'all':
                output = np.hstack(
                    (zs, model.get_distance_moduli(nfilters=1).reshape(
                        (model.nzs, 1))))
                header += '\n#  2: Distance Modulus'

            if out == 'all':
                for (i, filter) in enumerate([filters]):
                    filter_dat = np.column_stack(
                        (model.get_kcorrects(zf, filters=filter),
                         model.get_absolute_mags(zf, filters=filter),
                         model.get_apparent_mags(zf, filters=filter)))
                    output = np.hstack((output, filter_dat))
                    header += '\n# %2d: Kcorrect %s\n# %2d: Absolute Mag %s\n# %2d: Apparent Mag %s (2 + %d + %d)' % (
                        i * 3 + 3, filter, i * 3 + 4, filter, i * 3 + 5,
                        filter, i * 3 + 3, i * 3 + 4)
                    if i == 0: wgood = np.abs(filter_dat).max(axis=1) != 0
            else:
                # figure out what is outside the valid interpolation range
                wgood = np.abs(output[:, 1:]).max(axis=1) != 0

    # now write out a file
            utils.wascii(output[wgood, :],
                         prefix + 'zf_%s' % zfs_in[zfind],
                         formats='%7.4f',
                         header=header)
