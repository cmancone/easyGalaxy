import utils,os,astro_filter,zf_grid
import numpy as np

class astro_filter_light(astro_filter.astro_filter):
	""" filter = ezgal.astro_filter_light( filename, units='a', cosmology=None ) """

	##########
	## init ##
	##########
	def __init__( self, filename, units='a', cosmology=None, vega=False, solar=False ):

		# very basic load - no filter response curve
		self.npts = 0
		self.to_vega = vega
		self.has_vega = True
		if cosmology is not None: self.cosmo = cosmology
		self.solar = solar
		self.has_solar = not np.isnan( solar )
		return

	#######################
	## get apparent mags ##
	#######################
	def get_apparent_mags( self, zf, zs, vega=False ):
		""" mag = ezgal.astro_filter.get_apparent_mags( zf, zs, vega=False )
		
		Returns the apparent magnitude of the model at the given redshifts, given the formation redshift.
		Uses the zf_grid object to speed up calculations.  Can only be used for formation redshifts that have been gridded.
		Outputs vega mags if vega=True
		"""

		zf_grid = self.get_zf_grid( zf )
		if zf_grid == False: raise ValueError( 'Cannot fetch mag for given formation redshift because it has not been gridded!' )

		to_vega = self.to_vega if vega else 0.0

		return zf_grid.get_obs_mags( zs ) + zf_grid.get_dms( zs ) + to_vega