from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import ezgal
import numpy as np


class wrapper(object):

    models = []  # list of ezgal model objects
    nmodels = 0  # number of models in wrapper
    meta_data = {}  # meta data from model objects
    meta_keys = []  # list of meta data keys
    current_model = -1  # counter for iterator

    # these are used to determine if models match precisely and can be interpolated
    sed_shape = ()  # shape of sed array in models
    sed_ages = np.array([])  # list of sed ages
    sed_ls = np.array([])  # list of sed wavelengths
    is_matched = True  # whether or not all models match in SED array shape and ages
    has_masses = True  # whether or not all models have masses
    has_sfh = True  # whether or not all models have star formation histories

    ##########
    ## init ##
    ##########
    def __init__(self, models, extra_data={}, extra_name=''):
        """ wrapper_obj = ezgal.wrapper( [model_list], extra_data={}, extra_name='' )
		
		Initialize a wrapper object for handling multiple ezgal objects at once.  Model list
		should be a list of ezgal objects or filenames of ezgal compatible model files.
		See wrapper.add_model() for meaning of extra_data and extra_name parameters """

        # models should be a list
        if type(models) != type([]): models = [models]

        self.models = []
        self.nmodels = 0
        self.meta_data = {}
        self.meta_keys = []
        self.has_masses = True
        self.has_sfh = True

        # normalize any extra data passed to the wrapper object
        extra_data = self._normalize_data(extra_data,
                                          extra_name,
                                          return_list=True,
                                          require_length=len(models))

        # loop through models and make sure it is an ezgal object or filename
        for (i, model) in enumerate(models):
            # break up extra data to pass one set at a time to add_model()
            my_extra_data = {}
            if extra_data:
                for (key, val) in extra_data.iteritems():
                    my_extra_data[key] = val[i]

            self.add_model(model, my_extra_data)

    ############
    ## length ##
    ############
    def __len__(self):
        return self.nmodels

    #####################
    ## return iterator ##
    #####################
    def __iter__(self):
        self.current_model = -1
        return self

    #########################
    ## next() for iterator ##
    #########################
    def next(self):

        self.current_model += 1
        if self.current_model == self.nmodels: raise StopIteration

        return self.models[self.current_model]

    #############
    ## getitem ##
    #############
    def __getitem__(self, key):

        # index with tuple (assume a np.where() result)
        if type(key) == type(()):
            key = key[0]

        # index with numpy array
        if type(key) == type(np.array([])):
            if key.dtype == 'bool':
                return wrapper([self.models[ind] for ind in np.where(key)[0]])
            else:
                return wrapper([self.models[ind] for ind in key])

        # index with list
        if type(key) == type([]):
            return wrapper([self.models[ind] for ind in key])

        # index with string
        if type(key) == type(''):
            if not self.meta_data.has_key(key):
                raise IndexError('Meta key %s does not exist!' % key)
            return np.asarray(self.meta_data[key])

        # index with scalar
        if key >= self.nmodels:
            raise IndexError('Indexes must be less than %d' % self.nmodels)
        if key < 0: raise IndexError('Indexes must be >0')

        return self.models[key]

    #########
    ## add ##
    #########
    def __add__(self, more):

        new = self.copy()
        new += more
        return new

    ##########
    ## iadd ##
    ##########
    def __iadd__(self, more):

        if type(more) != type(self):
            raise TypeError('EzGal wrappers can only be added with eachother!')

        # loop through models in new object
        for (i, model) in enumerate(more):

            # break up extra data to pass one set at a time to add_model()
            my_extra_data = {}
            for key in more.meta_keys:
                my_extra_data[key] = more.meta_data[key][i]

            self.add_model(model, my_extra_data)

        return self

    ##########
    ## copy ##
    ##########
    def copy(self):
        return wrapper(self.models, self.meta_data)

    ############
    ## aslist ##
    ############
    def aslist(self):
        return [model for model in self.models]

    ###############
    ## add model ##
    ###############
    def add_model(self, model, extra_data={}, extra_name=''):
        """ wrapper_obj.add_model( model )
		
		:param model: A filename/object to add to the wrapper object
		:type model: filename, EzGal object
		
		Add an ezgal object to the wrapper object.  ``model`` can be an ezgal model object or the filename of an ezgal compatible model file."""

        # load the model if it is a filename
        if type(model) == type(''): model = ezgal.ezgal(model)

        # and make sure we have an ezgal object
        if type(model) != type(ezgal.ezgal(skip_load=True)):
            raise ValueError(
                'Must pass a filename or EzGal model to the wrapper object!')

        # store the model
        self.models.append(model)

        # store sed ages/shapes or check and make sure everything matches
        if self.nmodels == 0:
            self.sed_shape = model.seds.shape
            self.sed_ages = model.ages.copy()
            self.sed_ls = model.ls.copy()
        else:
            if self.sed_shape != model.seds.shape:
                self.is_matched = False
            if self.sed_ages.size != model.ages.size or np.max(np.abs(
                    self.sed_ages - model.ages)) > 1e-8:
                self.is_matched = False
            if self.sed_ls.size != model.ls.size or np.max(np.abs(
                    self.sed_ls - model.ls)) > 1e-8:
                self.is_matched = False

        # see if this has masses and a sfh
        if not model.has_masses: self.has_masses = False
        if not model.has_sfh: self.has_sfh = False

        # finally deal with meta data
        # normalize any extra data passed to the wrapper object
        extra_data = self._normalize_data(extra_data, extra_name)

        # the wrapper objects keeps a list of all meta data for all loaded ezgal objects for easy lookup.
        # Get the list of meta data from this model (including any additional info passed with the model)
        # and store the data in the wrapper object
        this_meta_keys = extra_data.keys()
        this_meta_vals = extra_data.values()
        if model.has_meta_data:
            this_meta_keys += model.meta_data.keys()
            this_meta_vals += model.meta_data.values()
            extra_data = dict(extra_data.items() + model.meta_data.items())

        # no meta data at all?
        if not len(this_meta_keys) and not len(self.meta_keys):
            self.nmodels += 1
            return True

        # okay, now merge meta data info.

        # First add any meta data that the model and the wrapper have in common
        for key in set(self.meta_keys).intersection(this_meta_keys):
            self.meta_data[key].append(extra_data[key])

        # now add blank meta data for any meta keys this model doesn't have data for
        for key in set(self.meta_keys).difference(this_meta_keys):
            self.meta_data[key].append('')

        # slightly tricker: add blank meta data for everything else if this model has meta data but the wrapper doesn't
        for key in set(this_meta_keys).difference(self.meta_keys):
            self.meta_data[key] = [''] * self.nmodels if self.nmodels else []
            self.meta_data[key].append(extra_data[key])
            self.meta_keys.append(key)

        self.nmodels += 1

    ####################
    ## normalize data ##
    ####################
    def _normalize_data(self,
                        extra_data,
                        extra_name,
                        return_list=False,
                        require_length=False):
        """ wrapper_obj._normalize_data( extra_data, extra_name )
		
		The wrapper object wants all extra data to be a dictionary of scalar or lists. """

        # if the extra data is already a dictionary then there is nothing to do
        if type(extra_data) == type({}):

            # if no extra data or we don't need to check the length then we are all done
            if not extra_data or not require_length: return extra_data

            # check length of extra data for consistency
            for val in extra_data.values():
                if len(val) != require_length:
                    raise ValueError('Mismatched extra data list length!')

            return extra_data

        # otherwise there should be a name...
        if type(extra_name) != type('') or not extra_name:
            raise ValueError(
                'When passing extra data a name must be provided if a list or scalar is passed')

        # require a list?
        if return_list and type(extra_data) != type([]):
            extra_data = [extra_data]

        # make a dictionary
        extra_data = {extra_name: extra_data}

        # length check?
        if require_length:
            for val in extra_data.values():
                if len(val) != require_length:
                    raise ValueError('Mismatched extra data list length!')

        return extra_data

    #######################
    ## get meta data set ##
    #######################
    def get_meta_data_set(self):
        """ wrapper_obj.get_meta_data_set()
		
		Returns a meta data dictionary containing 'Mixed' where models have different values,
		and the common value otherwise """

        new_data = {}
        # loop through each key
        for (key, vals) in self.meta_data.iteritems():

            # if there is only one value for all models then return that value
            # otherwise return 'Mixed'
            if np.unique(vals).size == 1:
                new_data[key] = vals[0]
            else:
                new_data[key] = 'Mixed'

        return new_data

        return True

    ##########
    ## find ##
    ##########
    def find(self, keys, vals):
        """ model = wrapper_obj.find( keys, vals, silent_fail=False )
		
		Returns a boolean mask designating which models have
		the given meta values for the given meta keys.
		Keys and vals can be lists of the same size or scalar values. """

        if type(keys) != type([]) and type(keys) != type(np.array([])):
            keys = [keys]
        if type(vals) != type([]) and type(vals) != type(np.array([])):
            vals = [vals]

        # generate a mask of matching models
        mask = np.ones(self.nmodels, dtype='bool')

        # make sure the keys all exist
        for key in keys:
            if not self.meta_data.has_key(key):
                raise ValueError("Meta data key %s does not exist!" % key)

        # now loop through keys/values and find things that don't match
        for (key, val) in zip(keys, vals):
            m = np.asarray(self.meta_data[key]) != val
            if m.sum(): mask[m] = False

        return mask

    ###############
    ## get model ##
    ###############
    def get_models(self, keys, vals, silent_fail=False):
        """ new_wrapper = wrapper_obj.get_models( keys, vals, return_wrapper=True, silent_fail=False )
		
		Returns a new wrapper object containing the models that have
		the given meta values for the given meta keys.
		Keys and vals can be lists of the same size or scalar values.
		
		If nothing matches an error will be raised unless silent_fail == True, in which
		case an empty list will be returned. """

        m = self.find(keys, vals)

        # raise a warning if nothing matched...
        if not m.sum():
            if silent_fail: return []
            raise ValueError("No models matched the listed criteria!")

        return self[m]

    #############
    ## argsort ##
    #############
    def argsort(self, key):
        """ sort_indexes = wrapper_obj.argsort( key )
		
		:param key: The name of a meta keyword by which to sort
		:type key: string
		:returns: List of indexes for soring wrapper object
		:rtype: list
		
		:Example:
			>>> import ezgal
			>>> wrapper = ezgal.wrapper( ['bc03_ssp_z_0.02_chab.model','bc03_ssp_z_0.008_chab.model'] )
			>>> print wrapper.argsort( 'met' )
			[1 0]
		
		Return a numpy array of indexes to sort the models in the wrapper object. Sorting is done numerically by values in meta data keyword ``key``. Data in given meta key must contain only numeric data. """

        if not self.meta_data.has_key(key):
            raise ValueError('Meta data key %s does not exist!' % key)

        # is this numeric data?
        try:
            data = [float(val) for val in self[key]]
        except ValueError:
            # nope.
            data = self[key]

        return np.asarray(data).argsort()

    ##########
    ## sort ##
    ##########
    def sort(self, key):
        """ sorted_wrapper = wrapper_obj.sort( key )
		
		Returns a new wrapper object with models sorted numerically according to values in
		meta data keyword `key` """

        return self[self.argsort(key)]

    #################
    ## interpolate ##
    #################
    def interpolate(self, key, values, return_wrapper=True):
        """ wrapper_obj.interpolate( key, values, return_wrapper=True ):
		
		Interpolate among stored models and return new EzGal objects at interpolated values.
		Pass a meta key which tells which values to interpolate between, and a list of
		values that you want the new models to be interpolated at.
		
		 If return_wrapper == True then it will return the models as an ezgal.wrapper object.
		 Otherwise, it will return them as a list of models, or as a single model (if a scalar value is passed in `values`). """

        # make sure we can interpolate
        if not self.is_matched:
            raise ValueError(
                "Can't interpolate among the models because the models have different age/wavelength points in their SEDs!")

        # make sure the key actually exists in our meta data
        if not self.meta_data.has_key(key):
            raise ValueError(
                "Can't interpolate by %s because that key wasn't found in the wrapper object's meta data!"
                % key)

        # get meta data for the given key and convert to float
        meta_values = np.array([float(val) for val in self.meta_data[key]])
        # finally get sort indexes
        sinds = meta_values.argsort()

        # convert values to numpy array
        is_scalar = False
        if type(values) != type([]) and type(values) != type(np.array([])):
            values = np.array([values])
            is_scalar = True
        else:
            values = np.asarray(values)

        # make sure the chosen value is bounded by the meta data values
        if values.min() < meta_values.min() or values.max() > meta_values.max(
        ):
            raise ValueError(
                'At least one passed interpolation value is not within the range of model values!')

        # okay, now we can interpolate

        # store a big array with all sed values together from models
        all_seds = np.empty(
            (self.sed_shape[0], self.sed_shape[1], self.nmodels))
        if self.has_masses:
            all_masses = np.empty((self.sed_ages.size, self.nmodels))
        if self.has_sfh:
            all_shfs = np.empty((self.sed_ages.size, self.nmodels))
        for (i, model) in enumerate(self.models):
            all_seds[:, :, i] = self.models[i].seds
            if self.has_masses: all_masses[:, i] = self.models[i].masses
            if self.has_sfh: all_sfhs[:, i] = self.models[i].sfh

        # make a big array to store interpolated seds/sfhs
        new_seds = np.empty(
            (self.sed_shape[0], self.sed_shape[1], len(values)))
        if self.has_masses:
            new_masses = np.empty((self.sed_ages.size, len(values)))
        if self.has_sfh: new_sfhs = np.empty((self.sed_ages.size, len(values)))

        # now loop through sed grid and use np.interp() for each age/wavelength point
        for a in range(self.sed_ages.size):

            # interpolate masses/sfh
            if self.has_masses:
                new_masses[a, :] = np.interp(values, meta_values[sinds],
                                             all_masses[a, sinds])
            if self.has_sfh:
                new_sfhs[a, :] = np.interp(values, meta_values[sinds],
                                           all_sfhs[a, sinds])

            # and SED
            for l in range(self.sed_ls.size):
                new_seds[l, a, :] = np.interp(values, meta_values[sinds],
                                              all_seds[l, a, sinds])

        # get a new meta data array combining everything stored in the wrapper object
        new_meta_data = self.get_meta_data_set()

        # great! Now just loop through each new sed and store the new ezgal object
        models = []
        for i in range(len(values)):

            # star formation history
            this_sfh = new_sfhs[:, i] if self.has_sfh else None
            # masses
            this_masses = new_masses[:, i] if self.has_masses else None

            models.append(self.models[0]._return_new(new_seds[:, :, i],
                                                     this_masses,
                                                     sfh=this_sfh,
                                                     meta_data={key: str(
                                                         values[i])}))

        # finally return
        if return_wrapper:
            return wrapper(models)
        elif is_scalar:
            return models[0]
        else:
            return models

    #####################
    ## set vega output ##
    #####################
    def set_vega_output(self):
        """ wrapper.set_vega_output()
		
		Calls model.set_vega_output() on all loaded models. """

        for model in self:
            model.set_vega_output()
        return True

    ###################
    ## set AB output ##
    ###################
    def set_ab_output(self):
        """ wrapper.set_ab_output()
		
		Calls model.set_ab_output() on all loaded models. """

        for model in self:
            model.set_ab_output()

    ##############
    ## get data ##
    ##############
    def _get_data(self,
                  zf,
                  kind='',
                  filters=None,
                  zs=None,
                  normalize=True,
                  ab=None,
                  vega=None):
        """ wrapper.get_data( zf, filters=None, zs=None, normalize=True, ab=None, vega=None ) """

        # pre-populate calling parameters to ensure everything is done the same way
        zs = self.models[0]._populate_zs(zs, zf=zf)
        filters = self.models[0]._populate_filters(filters)
        vega_out = self.models[0]._get_vega_out(ab=ab, vega=vega)
        ab = False
        vega = False
        if vega_out:
            vega = True
        else:
            ab = True

        # build data cube for storing results
        data = np.empty((self.nmodels, len(zs), len(filters)))

        # finally populate array
        for (i, model) in enumerate(self):
            if kind == 'absolute' or kind == 'apparent':
                data[i, :, :] = model._get_mags(zf,
                                                kind=kind,
                                                filters=filters,
                                                zs=zs,
                                                ab=ab,
                                                vega=vega,
                                                squeeze=False,
                                                normalize=normalize)
            elif kind == 'kcorrect' or kind == 'ecorrect' or kind == 'ekcorrect':
                data[i, :, :] = model._get_mags(zf,
                                                kind=kind,
                                                filters=filters,
                                                zs=zs,
                                                ab=ab,
                                                vega=vega,
                                                squeeze=False,
                                                normalize=False)
            elif kind == 'rest_ml':
                data[i, :, :] = model.get_rest_ml_ratios(zf,
                                                         filters=filters,
                                                         zs=zs,
                                                         squeeze=False)
            elif kind == 'obs_ml':
                data[i, :, :] = model.get_observed_ml_ratios(zf,
                                                             filters=filters,
                                                             zs=zs,
                                                             squeeze=False)
            elif kind == 'solar_obs':
                data[i, :, :] = model.get_solar_observed_mags(zf,
                                                              filters=filters,
                                                              zs=zs,
                                                              ab=ab,
                                                              vega=vega,
                                                              squeeze=False)
            elif kind == 'obs_abs':
                data[i, :, :] = model.get_observed_absolute_mags(
                    zf,
                    filters=filters,
                    zs=zs,
                    ab=ab,
                    vega=vega,
                    squeeze=False,
                    normalize=normalize)

        return data

    #######################
    ## get absolute mags ##
    #######################
    def get_absolute_mags(self,
                          zf,
                          filters=None,
                          zs=None,
                          normalize=True,
                          ab=None,
                          vega=None):
        """ wrapper.get_absolute_mags( zf, filters=None, zs=None, normalize=True, ab=None, vega=None )
		
		:param zf: The formation redshift
		:param filters: List of filters
		:param zs: List of zs
		:param normalize: Normalize according to model normalization, if set
		:param ab: Whether or not to return AB mags
		:param vega: Whether or not to return vega mags
		:type zf: int, float
		:type filters: string, list
		:type zs: list, array, int, float
		:type normalize: bool
		:type ab: bool
		:type vega: bool
		:returns: Data cube with absolute magnitudes
		:rtype: array
		
		:Example:
			>>> import ezgal
			>>> wrapper = ezgal.wrapper( ['bc03_ssp_z_0.02_chab.model','bc03_ssp_z_0.008_chab.model'] )
			>>> mags = wrapper.get_absolute_mags( 3.0, filters=['ch1','ch2','ch3'], zs=[0,1,2,2.5] )
			>>> print mags.shape
			( 2, 4, 3 )
			>>> print mags
			[[[ 6.39000031  6.83946654  7.25469915]
			  [ 5.61352032  6.07145866  6.49122232]
			  [ 4.48814389  4.87218428  5.30970258]
			  [ 4.49875516  4.90041299  5.34425979]]
			 [[ 6.43424729  6.92860144  7.35495047]
			  [ 5.75053844  6.21216407  6.64700271]
			  [ 4.73228287  5.13875043  5.59090961]
			  [ 4.55125868  4.95278808  5.40624718]]]
		
		Calls :meth:`ezgal.ezgal.get_absolute_mags` on all stored model objects and returns results in a data cube of shape ``(len( wrapper ),len( zs ),len( nfilters ))``
		
		.. seealso::
			:meth:`ezgal.ezgal.get_absolute_mags`
		.. warning::
			If not already done, this will calculate the redshift evolution for all the specified filters at the given formation redshifts.  If many ``EzGal`` objects have been loaded into the wrapper object, then this can take a lot of time.  If working with many models it is often best to pre-calculate the redshift evolution for the filters and formation redshifts of interest, and save them in the ``EzGal`` model files.
		"""

        return self._get_data(zf,
                              kind='absolute',
                              filters=filters,
                              zs=zs,
                              normalize=normalize,
                              ab=ab,
                              vega=vega)

    #######################
    ## get apparent mags ##
    #######################
    def get_apparent_mags(self,
                          zf,
                          filters=None,
                          zs=None,
                          normalize=True,
                          ab=None,
                          vega=None):
        """ wrapper.get_apparent_mags( zf, filters=None, zs=None, normalize=True, ab=None, vega=None )
		
		Same as :meth:`ezgal.wrapper.get_absolute_mags` but calls :meth:`ezgal.ezgal.get_apparent_mags`.
		
		.. seealso::
			:meth:`ezgal.wrapper.get_absolute_mags`, :meth:`ezgal.ezgal.get_apparent_mags`
		"""

        return self._get_data(zf,
                              kind='apparent',
                              filters=filters,
                              zs=zs,
                              normalize=normalize,
                              ab=ab,
                              vega=vega)

    #########################
    ## get distance moduli ##
    #########################
    def get_distance_moduli(self, zs=None, nfilters=None):
        """ wrapper.get_distance_moduli( zs=None, nfilters=None )
		
		Same as :meth:`ezgal.wrapper.get_absolute_mags` but calls :meth:`ezgal.ezgal.get_distance_moduli`.
		
		.. seealso::
			:meth:`ezgal.wrapper.get_absolute_mags`, :meth:`ezgal.ezgal.get_distance_moduli`
		"""

        zs = self.models[0]._populate_zs(zs)
        if nfilters is None: nfilters = self.models[0].nfilters

        data = np.empty((self.nmodels, len(zs), nfilters))

        for (i, model) in enumerate(self):
            data[i, :, :] = model.get_distance_moduli(zs,
                                                      nfilters=nfilters,
                                                      squeeze=False)

        return data

    ###################
    ## get kcorrects ##
    ###################
    def get_kcorrects(self, zf, filters=None, zs=None):
        """ wrapper.get_kcorrects( zf, filters=None, zs=None )
		
		Same as :meth:`ezgal.wrapper.get_absolute_mags` but calls :meth:`ezgal.ezgal.get_kcorrects`.
		
		.. seealso::
			:meth:`ezgal.wrapper.get_absolute_mags`, :meth:`ezgal.ezgal.get_kcorrects`
		"""

        return self._get_data(zf,
                              kind='kcorrect',
                              filters=filters,
                              zs=zs,
                              normalize=False)

    ###################
    ## get ecorrects ##
    ###################
    def get_ecorrects(self, zf, filters=None, zs=None):
        """ wrapper.get_ecorrects( zf, filters=None, zs=None )
		
		Same as :meth:`ezgal.wrapper.get_absolute_mags` but calls :meth:`ezgal.ezgal.get_ecorrects`.
		
		.. seealso::
			:meth:`ezgal.wrapper.get_absolute_mags`, :meth:`ezgal.ezgal.get_ecorrects`
		"""

        return self._get_data(zf,
                              kind='ecorrect',
                              filters=filters,
                              zs=zs,
                              normalize=False)

    ####################
    ## get ekcorrects ##
    ####################
    def get_ekcorrects(self, zf, filters=None, zs=None):
        """ wrapper.get_ekcorrects( zf, filters=None, zs=None )
		
		Same as :meth:`ezgal.wrapper.get_absolute_mags` but calls :meth:`ezgal.ezgal.get_ekcorrects`.
		
		.. seealso::
			:meth:`ezgal.wrapper.get_absolute_mags`, :meth:`ezgal.ezgal.get_ekcorrects`
		"""

        return self._get_data(zf,
                              kind='ekcorrect',
                              filters=filters,
                              zs=zs,
                              normalize=False)

    #########################
    ## get rest M/L ratios ##
    #########################
    def get_rest_ml_ratios(self, zf, filters=None, zs=None):
        """ wrapper.get_rest_ml_ratios( zf, filters=None, zs=None )
		
		Same as :meth:`ezgal.wrapper.get_absolute_mags` but calls :meth:`ezgal.ezgal.get_rest_ml_ratios`.
		
		.. seealso::
			:meth:`ezgal.wrapper.get_absolute_mags`, :meth:`ezgal.ezgal.get_rest_ml_ratios`
		"""

        return self._get_data(zf, kind='rest_ml', filters=filters, zs=zs)

    #############################
    ## get observed M/L ratios ##
    #############################
    def get_observed_ml_ratios(self, zf, filters=None, zs=None):
        """ wrapper.get_observed_ml_ratios( zf, filters=None, zs=None )
		
		Same as :meth:`ezgal.wrapper.get_absolute_mags` but calls :meth:`ezgal.ezgal.get_observed_ml_ratios`.
		
		.. seealso::
			:meth:`ezgal.wrapper.get_absolute_mags`, :meth:`ezgal.ezgal.get_observed_ml_ratios`
		"""

        return self._get_data(zf, kind='obs_ml', filters=filters, zs=zs)

    #############################
    ## get solar observed mags ##
    #############################
    def get_solar_observed_mags(self,
                                zf,
                                filters=None,
                                zs=None,
                                ab=None,
                                vega=None):
        """ wrapper.get_observed_ml_ratios( zf, filters=None, zs=None, ab=None, vega=None )
		
		Same as :meth:`ezgal.wrapper.get_absolute_mags` but calls :meth:`ezgal.ezgal.get_solar_observed_mags`.
		
		.. seealso::
			:meth:`ezgal.wrapper.get_absolute_mags`, :meth:`ezgal.ezgal.get_solar_observed_mags`
		"""

        return self._get_data(zf,
                              kind='solar_obs',
                              filters=filters,
                              zs=zs,
                              ab=ab,
                              vega=vega)

    ################################
    ## get observed absolute mags ##
    ################################
    def get_observed_absolute_mags(self,
                                   zf,
                                   filters=None,
                                   zs=None,
                                   normalize=True,
                                   ab=None,
                                   vega=None):
        """ wrapper.get_observed_absolute_mags( zf, filters=None, zs=None, normalize=True, ab=None, vega=None )
		
		Same as :meth:`ezgal.wrapper.get_absolute_mags` but calls :meth:`ezgal.ezgal.get_observed_absolute_mags`.
		
		.. seealso::
			:meth:`ezgal.wrapper.get_absolute_mags`, :meth:`ezgal.ezgal.get_observed_absolute_mags`
		"""

        return self._get_data(zf,
                              kind='obs_abs',
                              filters=filters,
                              zs=zs,
                              ab=ab,
                              vega=vega)

    ################
    ## get masses ##
    ################
    def get_masses(self, zf, zs=None, nfilters=None):
        """ wrapper.get_masses( zf, zs=zs, nfilters=None )
		
		Same as :meth:`ezgal.wrapper.get_absolute_mags` but calls :meth:`ezgal.ezgal.get_masses`.
		
		.. seealso::
			:meth:`ezgal.wrapper.get_absolute_mags`, :meth:`ezgal.ezgal.get_masses`
		"""

        zs = self.models[0]._populate_zs(zs, zf=zf)
        if nfilters is None: nfilters = self.models[0].nfilters

        data = np.empty((self.nmodels, len(zs), nfilters))

        for (i, model) in enumerate(self):
            data[i, :, :] = model.get_masses(zf, zs, squeeze=False)

        return data

    ############
    ## get zs ##
    ############
    def get_zs(self, z):
        """ zs = wrapper.get_zs( z )
		
		:param z: Upper limit for list of redshifts
		:type z: int, float
		:returns: Array with list of redshifts
		:rtype: array
		
		Shortcut for wrapper[0].get_zs( z )
		
		.. seealso::
			:meth:`ezgal.ezgal.get_zs`
		"""

        return self[0].get_zs(z)

    #############
    ## get age ##
    #############
    def get_age(self, z1, z2, units='gyrs'):
        """ ages = wrapper.get_age( z1, z2, units='gyrs' )
		
		:param z1: The first redshift
		:param z2: The second redshift
		:param units: The units to return the time in
		:type z1: int, float
		:type z2: int, float, list, array
		:type units: str
		:returns: Time between two redshifts
		:rtype: int, float, list, array
		
		Shortcut for wrapper[0].get_age( z1, z2, units=units )
		
		.. seealso::
			:meth:`ezgal.ezgal.get_age`
		"""

        return self[0].get_age(z1, z2, units=units)

    #######################
    ## get normalization ##
    #######################
    def get_normalization(self, zf, flux=False):
        """ normalizations = wrapper.get_normalization( zf, flux=False )
		
		:param zf: The formation redshift to assume
		:param flux: Wheter or not to return a multiplicative factor
		:type zf: int, float
		:type flux: bool
		:returns: The normalizations
		:rtype: array
		
		:Example:
			>>> import ezgal
			>>> wrapper = ezgal.wrapper( ['bc03_ssp_z_0.02_chab.model','bc03_ssp_z_0.008_chab.model'] )
			>>> wrapper.set_normalization( 'ch1', 0.24, -25.06, vega=True )
			>>> wrapper.get_normalization( 3.0 )
			array([-28.45662556, -28.5216577 ])
			>>> wrapper.get_normalization( 3.0, flux=True )
			array([  2.41351622e+11,   2.56249531e+11])
		
		Returns an array of size ``len( wrapper )`` containing the normalization for each model in the wrapper object.
		
		.. seealso::
			:meth:`ezgal.ezgal.get_normalization`, :meth:`ezgal.wrapper.set_normalization`
		"""

        return np.asarray(
            [model.get_normalization(zf, flux) for model in self])

    #######################
    ## set normalization ##
    #######################
    def set_normalization(self, filter, z, mag, vega=False, apparent=False):
        """ wrapper.set_normalization( filter, z, mag, vega=False, apparent=False )
		
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
		
		Calls :meth:`ezgal.ezgal.set_normalization` on all models loaded in the wrapper.
		
		.. seealso::
			:meth:`ezgal.ezgal.set_normalization`, :meth:`ezgal.wrapper.get_normalization`
		"""

        for model in self:
            model.set_normalization(filter,
                                    z,
                                    mag,
                                    vega=vega,
                                    apparent=apparent)
