import numpy as np

class zf_grid(object):
	""" object = ezal.zf_grid( zf, zs, rest, obs )
	
	This object class stores grid information calculated by astro_filter.astro_filter for a particular formation redshift.
	It creates and stores interpolation objects to quickly calculate mags and kcors as a function of redshift.
	"""

	zf = 0			# formation redshift
	zs = np.array( [] )	# zs
	rest = np.array( [] )	# rest-frame absolute mags
	obs = np.array( [] )	# kcorrections
	ages = np.array( [] )	# ages

	has_dms = False		# distance moduli stored?
	dms = np.array( [] )	# distance moduli

	has_solar = False	# solar mags stored?
	solar = np.array( [] )	# solar mags

	has_masses = False	# masses stored?
	masses = np.array( [] )	# masses

	##############
	## __init__ ##
	##############
	def __init__( self, zf, zs, ages, rest, obs, dms=None ):

		# store everything.  Sort zs to make sure it is ascending (it probably isn't)
		self.zf = zf

		sinds = zs.argsort()
		self.zs = zs[sinds]
		self.ages = ages[sinds]
		self.rest = rest[sinds]
		self.obs = obs[sinds]
		if dms is not None:
			self.dms = dms[sinds]
			self.has_dms = True
		else:
			self.dms = np.array( [] )
			self.has_dms = False

	######################
	## store solar mags ##
	######################
	def store_solar_mags( self, mags ):

		if mags.size != self.zs.size: raise ValueError( 'Cannot store solar grid: wrong number of magnitudes!' )

		self.has_solar = True
		self.solar = mags

	##################
	## store masses ##
	##################
	def store_masses( self, masses ):

		if masses.size != self.zs.size: raise ValueError( 'Cannot store masses: wrong number of masses!' )

		self.has_masses = True
		self.masses = masses

	###################
	## get rest mags ##
	###################
	def get_rest_mags( self, zs ):
		""" mag = ezal.zf_grid.get_rest_mags( zs )
		
		Use the interpolation object to calculate rest-frame absolute magnitude as a function of zs.
		Return NaN if out of bounds """

		# return nan for anything outside the interpolation range
		zs = np.asarray( zs )
		m = ( zs >= self.zs.min() ) & ( zs <= self.zs.max() )

		res = np.empty( zs.size )
		res[:] = np.nan

		res[m] = np.interp( zs[m], self.zs, self.rest )
		return res

	##################
	## get obs mags ##
	##################
	def get_obs_mags( self, zs ):
		""" mag = ezal.zf_grid.get_obs_mags( zs )
		
		Use the interpolation object to calculate observed-frame absolute magnitude as a function of zs.
		Return NaN if out of bounds """

		# return nan for anything outside the interpolation range
		zs = np.asarray( zs )
		m = ( zs >= self.zs.min() ) & ( zs <= self.zs.max() )

		res = np.empty( zs.size )
		res[:] = np.nan

		res[m] = np.interp( zs[m], self.zs, self.obs )
		return res

	##############
	## get ages ##
	##############
	def get_ages( self, zs ):
		""" mag = ezal.zf_grid.get_ages( zs )
		
		Use the interpolation object to calculate age as a function of zs.
		Return NaN if out of bounds """

		# return nan for anything outside the interpolation range
		zs = np.asarray( zs )
		m = ( zs >= self.zs.min() ) & ( zs <= self.zs.max() )

		res = np.empty( zs.size )
		res[:] = np.nan

		res[m] = np.interp( zs[m], self.zs, self.ages )
		return res

	####################
	## get solar mags ##
	####################
	def get_solar_mags( self, zs ):
		""" mag = ezal.zf_grid.get_solar_mags( zs )
		
		Use the interpolation object to calculate observed-frame absolute magnitude of the sun as a function of zs.
		Return NaN if out of bounds """

		if not self.has_solar: raise ValueError( 'Solar mags have not been set!' )

		# return nan for anything outside the interpolation range
		zs = np.asarray( zs )
		m = ( zs >= self.zs.min() ) & ( zs <= self.zs.max() )

		res = np.empty( zs.size )
		res[:] = np.nan

		res[m] = np.interp( zs[m], self.zs, self.solar )
		return res

	################
	## get masses ##
	################
	def get_masses( self, zs ):
		""" masses = ezal.zf_grid.get_masses( zs )
		
		Use the interpolation object to calculate stellar mass as a function of zs.
		Return NaN if out of bounds """

		# return nan for anything outside the interpolation range
		if not self.has_masses: raise ValueError( 'Masses have not been set!' )

		zs = np.asarray( zs )
		m = ( zs >= self.zs.min() ) & ( zs <= self.zs.max() )

		res = np.empty( zs.size )
		res[:] = np.nan

		res[m] = np.interp( zs[m], self.zs, self.masses )
		return res

	#############
	## get dms ##
	#############
	def get_dms( self, zs ):
		""" mag = ezgal.zf_grid.dms( zs )
		
		Use interpolation object to calculate dm as a function of zs.
		Return NaN if out of bounds """

		# return nan for anything outside the interpolation range
		zs = np.asarray( zs )
		m = ( zs >= self.zs.min() ) & ( zs <= self.zs.max() )

		res = np.empty( zs.size )
		res[:] = np.nan

		res[m] = np.interp( zs[m], self.zs, self.dms )
		return res