import os,utils,astro_filter_light,re,ezgal
import numpy as np

class ezgal_light(ezgal.ezgal):
	""" model = ezgal.ezgal_light( model_file )
	
	Stripped down ezgal class for working on systems with only a basic python installation.
	Will only work with ezgal ascii model files
	"""

	# model information
	filename = ''			# the name of the file
	nages = 0			# number of different aged SEDs
	ages = np.array( [] )		# array of ages
	nvs = 0				# number of frequences in each SED
	vs = np.array( [] )		# array of frequences for SEDs
	nls = 0				# number of wavelengths in each SED (same as self.nvs)
	ls = np.array( [] )		# array of wavelengths for SEDs (angstroms)
	seds = np.array( [] )		# age/SED grid.  Units need to be ergs/cm^2/s.  Size is nvs x nages
	# normalization info
	norm = { 'norm': 0, 'z': 0, 'filter': '', 'vega': False, 'apparent': False }
	# mass info
	mass = { 'has_masses': False, 'masses': np.array( [] ), 'mass_ages': np.array( [] ), 'interp': [] }

	# cosmology related stuff
	cosmology_loaded = False	# whether or not a cosmology has been loaded
	cosmo = None			# cosmology object
	zfs = np.array( [] )		# formation redshifts for models
	nzfs = 0			# number of formation redshifts to model
	zs = np.array( [] )		# redshifts at which to project models
	nzs = 0				# number of redshifts models should calculated at

	# filter stuff.
	filters = {}			# dictionary of astro_filter objects
	filter_order = []		# list of filter names (in order added)
	nfilters = 0			# number of filters
	current_filter = -1		# counter for iterator

	# info for interpolated model SEDs.  The SEDs are interpolated to a regular grid for each formation redshift.
	# These interpolated models are stored to save time if needed again
	interp_seds = []		# interpolated seds - each list element is a dictionary with sed info
	interp_zfs = np.array( [] )	# list of formation redshifts for sed
	tol = 1e-8			# tolerance for determining whether a given zf matches a stored zf

	# additional data
	data_dir = ''			# data directory for filters, models, and reference spectra
	has_vega = False		# whether or not the vega spectrum is found and loaded
	vega = np.array( [] )		# vega spectrum (nu vs Fnu)
	vega_out = False		# if True then retrieved mags will be in vega mags
	has_solar = False		# whether or not the solar spectrum is found and loaded
	solar = np.array( [] )		# solar spectrum (nu vs Fnu)

	##########
	## init ##
	##########
	def __init__( self, model_file ):
		""" model = ezgal.ezgal_light( model_file )
		
		Stripped down ezgal class for working on systems with only a basic python installation.
		Will only work with ezgal ascii model files
		"""

		# load a default cosmology
		self.set_cosmology()

		# clear filter list etc
		self.filters = {}
		self.filter_order = []
		self.interp_seds = []
		self.inter_zs = np.array( [] )
		self.zfs = np.array( [] )
		self.zs = np.array( [] )
		self.norm = { 'norm': 0, 'z': 0, 'filter': '', 'vega': False, 'apparent': False }

		# save path to data folder - either set by environment (ezgal) or default to module directory/data
		if os.environ.has_key('ezgal'):
			self.data_dir = os.environ['ezgal']
		elif os.environ.has_key('EZGAL'):
			self.data_dir = os.environ['EZGAL']
		elif os.environ.has_key('EzGal'):
			self.data_dir = os.environ['EzGal']
		else:
			self.data_dir = os.path.dirname( os.path.realpath( __file__ ) ) + '/data/'

		# make sure data path ends with a slash
		if self.data_dir[-1] != os.sep: self.data_dir += os.sep

		# load model
		self._load( model_file )

	#####################
	## load model file ##
	#####################
	def _load( self, model_file, is_ised=False, is_fits=False, is_ascii=False, has_masses=False, units='a', age_units='gyrs' ):

		# make sure model file exists
		if not( os.path.isfile( model_file ) ):
			# is it in the data directory?
			model_file = '%smodels/%s' % (self.data_dir,os.path.basename(model_file))
			if not( os.path.isfile( model_file ) ):
				raise ValueError( 'The specified model file was not found!' )
			else:
				print 'loading file from: %s' % model_file

		self.filename = model_file

		self._load_ascii_model( model_file )

	#####################
	## check cosmology ##
	#####################
	def check_cosmology( self ):
		if not self.cosmology_loaded:
			self.set_cosmology()
			print 'Default cosmology loaded'

	###################
	## set cosmology ##
	###################
	def set_cosmology( self, Om=0.279, Ol=0.721, h=0.701, w=-1 ):
		""" ezgal.set_cosmology( Om=0.279, Ol=0.721, h=0.701, w=-1 )
		
		Set the cosmology.  The default cosmology is from WMAP 5, Hinshaw, G. et al. 2009, ApJS, 180, 225
		If the cosmology changes, then all filters will be regrided and any stored evolution models will be discarded.
		"""

		self.cosmo = cosmology_light( Om=Om, Ol=Ol, h=h, w=w )
		self.cosmology_loaded = True

		# pass cosmology object to all the filter objects
		for filter in self.filter_order: self.filters[filter].set_cosmology( self.cosmo )

	#########################
	## get distance moduli ##
	#########################
	def get_distance_moduli( self, zf, zs=None, nfilters=None, squeeze=True ):
		""" mags = ezgal_ligt.get_distance_moduli( zf, zs=None, nfilters=None )

		fetch the distance moduli for the given redshifts and formation redshift
		Specify the number of filters to return an array of size (len(zs),nfilters)
		if nfilters is None, then the number of filters will be taken to be the number of filters loaded in the object
		If there is only one filter, then it returns an array of shape (len(zs))
		If no output redshifts are specified, then the redshifts in ezgal.zs will be used.
		"""

		# load defaults
		zs = self._populate_zs( zs )
		if nfilters is None: nfilters = self.nfilters

		# fetch the distance moduli - any filter will do
		dms = self.filters[self.filter_order[0]].get_zf_grid( zf ).get_dms( zs )

		dms = dms.reshape( (len(zs),1) ).repeat( nfilters, axis=1 )

		if not squeeze: return dms
		return self._squeeze( dms, nfilters )

	#############################
	## get observed M/L ratios ##
	#############################
	def get_observed_ml_ratios( self, zf, filters=None, zs=None, squeeze=True ):
		""" mls = ezgal.get_observed_ml_ratios( zf, filters=None, zs=None )
		
		Returns the observed-frame mass-to-light ratios as a function of redshift and formation redshift.
		filters is a list of filters to calculate mass-to-light ratios for.
		If rest=True returns rest-frame mass-to-light ratios, otherwise observed-frame
		
		returns an array of shape (len(zs),len(filters)) with the mass-to-light ratios.
		If there is only one filter, then it returns an array of shape (len(zs))
		
		Returns nan for undefined mass-to-light ratios (which means the solar spectrum doesn't fully cover the filter).
		Only works if masses were found in the model. """

		if not self.has_masses: raise ValueError( 'Cannot get mass-to-light ratios: masses were not found in the model file!' )

		# load default values
		filters = self._populate_filters( filters )
		zs = self._populate_zs( zs )

		# fetch model magnitudes
		mags = self._get_mags( zf, kind='apparent', filters=filters, zs=zs, ab=True, normalize=False, squeeze=False ) - self.get_distance_moduli( zf, zs=zs, nfilters=len( filters ), squeeze=False )

		# calculate masses for these redshifts
		masses = self.get_masses( zf, zs, nfilters=len( filters ), squeeze=False, normalize=False )

		# observed-frame absolute magnitude of sun
		solar_mags = self.get_solar_observed_mags( zf, filters=filters, zs=zs, ab=True, squeeze=False )

		# M/L ratio = mass/(L/Lsun) = mass/10**( -0.4*(M - Msun) )
		mls = masses/10.0**( -0.4*(mags-solar_mags) )

		# return squeezed array?
		return self._squeeze( mls, len( filters ) ) if squeeze else mls

	#############
	## get age ##
	#############
	def get_age( self, z1, z2, units='gyrs' ):
		""" age = ezgal.get_age( z1, z2, units='gyrs' )
		
		returns the time difference between z1 and z2.  z2 can be a list or numpy array.
		See ezgal.utils.to_years for available output units
		"""

		# need to pass an array
		is_scalar = False
		if type( z2 ) != type( [] ) and type( z2 ) != type( np.array( [] ) ):
			is_scalar = True
			z2 = np.array( [z2] )

		# z/age grids are stored in the filter grid objects
		to_return = utils.convert_time( self.filters[self.filter_order[0]].get_zf_grid( z1 ).get_ages( z2 ), incoming='yrs', outgoing=units )

		if is_scalar:
			return to_return[0]
		return to_return

# stripped down cosmology class - just stores cosmology
class cosmology_light:
    def __init__(self, Om=0.279, Ol=0.721, h=0.701, w=-1):
        self.Om = Om
        self.Ol = Ol
        self.w  = w
        self.h  = h
