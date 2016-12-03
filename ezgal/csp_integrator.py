import dusts,sfhs
import numpy as np
import scipy.integrate as integrate
__ver__ = '1.0'

class csp_integrator(object):
	""" csp integrator: a class used to perform CSP integration and return SEDs as a function of age
	for arbitrary star formation histories and dust laws.  This uses csp_region_integrator objects
	to do the actual integration and split up the integral in the case of break points """

	seds = np.array( [] )	# SED wavelength/age grid
	ls = np.array( [] )	# SED wavelengths
	ages = np.array( [] )	# SED ages (gyrs)
	breaks = None		# list of break points
	Tu = 0.0

	regions = []		# list of region objects
	sfh_func = ''		# star formation history function
	has_sfh = False		# whether or not the sfh function has been loaded
	dust_func = ''		# dust function
	has_dust = False	# whether or not a dust function is loaded
	masses = np.array( [] )	# list of masses
	has_masses = False	# whether or not masses are loaded

	def __init__( self, seds, ls, ages, Tu, break_points=None ):

		self.seds = seds
		self.ls = ls
		self.ages = ages

		# sort the break points as a list
		if break_points is None:
			break_points = [0,Tu]

		breaks = np.sort( break_points )

		# if a scalar was passed, convert to a proper numpy array
		if len( breaks.shape ) == 0: breaks = np.array( [break_points] )

		# first break point is t=0
		if breaks[0] != 0: breaks = np.append( 0, breaks )

		# and last is Tu
		if breaks[-1] < Tu:
			breaks = np.append( breaks, Tu )
		else:
			breaks[-1] = Tu

		self.breaks = breaks

	def set_sfh( self, function, args ):
		# wrap up the sfh function so I don't have to worry about the exact calling sequence
		self.sfh_func = sfhs.sfh_wrapper( function, args )
		self.has_sfh = True

	def set_dust( self, function, args ):
		# wrap up the dust function so I don't have to worry about the exact calling sequence
		self.dust_func = dusts.dust_wrapper( function, args )
		self.has_dust = True

	def set_masses( self, masses ):
		# store in each region integrator
		self.masses = masses
		self.has_masses = True

	def generate_regions( self, resampling=1, ls=None ):

		# if ls is not None then we only want to give the integrators the wavelenghts in ls
		limit = False
		if ls is not None:
			limit = True
			inds = np.array( [ np.argmin( np.abs( self.ls-l ) ) for l in ls ] )
			nls = inds.size
		else:
			nls = self.ls.size

		ls = self.ls[inds] if limit else self.ls

		if resampling == 1:
			seds = self.seds[inds,:] if limit else self.seds
			ages = self.ages
			if self.has_masses: masses = self.masses
		else:
			# resample age grid
			ages = self.resample_ages( resampling )
			if self.has_masses: masses = np.interp( ages, self.ages, self.masses )

			# resample SEDs
			seds = np.empty( (nls,ages.size) )
			# use different interpolation method depending on whether limit is True
			if limit:
				for ( i, ind ) in enumerate( inds ): seds[i,:] = np.interp( ages, self.ages, self.seds[ind,:] )
			else:
				for ( i, age ) in enumerate( ages ): seds[:,i] = self.interp_sed( age )

		# generate a list of region integrator objects and store
		self.regions = []
		for i in range( self.breaks.size-1 ):
			region = csp_region_integrator( seds, ls, ages, self.breaks[i], self.breaks[i+1] )
			if self.has_sfh: region.set_sfh( self.sfh_func )
			if self.has_dust: region.set_dust( self.dust_func )
			if self.has_masses: region.set_masses( masses )
			self.regions.append( region )

		return nls

	def resample_ages( self, factor ):
		return np.interp( np.arange( 0, self.ages.size-1 + 1.0/factor, 1.0/factor ), np.arange( self.ages.size ), self.ages )

	def interp_sed( self, age ):

		# simple two point interpolation
		# find the closest SED younger than the given age
		yind = np.abs( self.ages - age ).argmin()
		if self.ages[yind] > age: yind -= 1
		# ind of closest SED older than the given age
		oind = yind + 1

		# are we at the borders of the age array?  If so return
		if oind >= self.ages.size: return self.seds[:,yind].copy()
		if yind < 0: return self.seds[:,oind].copy()

		# age of older and younger seds
		yage = self.ages[yind]
		oage = self.ages[oind]

		# now interpolate
		return self.seds[:,yind] + (self.seds[:,oind]-self.seds[:,yind])*(age-yage)/(oage-yage)

	def integrate( self, resampling=1, ls=None, return_all=True ):

		if not self.has_sfh: raise ValueError( 'Set a star formation history before integrating!' )

		# generate regions for integration
		nls = self.generate_regions( resampling, ls )

		# generate arrays for keeping new SEDs and age/mass relationship
		csp_seds = np.zeros( ( nls, self.ages.size ) )
		csp_masses = np.zeros( self.ages.size )

		# get normalization of star formation history
		norm = 0
		for region in self.regions:
			norm += region.get_sfh_normalization()
		if norm <= 0: raise ValueError( 'There is no star formation in the star formation history!' )

		# and calculate sfh as a function of time
		sfh = self.sfh_func( self.ages )/norm

		# loop through all but the first age and calculate CSP SED
		# the first age is skipped because everything equals zero at t=0
		for age_ind in range( 1, self.ages.size ):

			# let the regions do all the work
			for region in self.regions:
				( seds, mass ) = region.integrate( self.ages[age_ind] )
				csp_seds[:,age_ind] += seds
				csp_masses[age_ind] += mass

		# normalize
		csp_seds /= norm
		csp_masses /= norm

		if return_all:
			return ( csp_seds, csp_masses, sfh )
		else:
			return csp_seds

	def find_resampling( self, max_err, max_iter=200 ):

		# wavelengths to check integral over
		ls = [3000,8000,12000]
		nls = len( ls )

		err = np.inf
		niter = 0

		while True:

			# adjust resampling factor
			resampling = 1 if niter == 0 else resampling + 1

			# and integrate
			this = self.integrate( resampling=resampling, ls=ls, return_all=False )

			niter += 1	
			if niter == 1:
				last = this
				continue

			# filter out first and last rows, as well as zero values
			this_compare = this[:,1:-1]
			last_compare = last[:,1:-1]
			m = ( this_compare != 0 ) & ( last_compare != 0 )
			err = np.max( np.abs( -2.5*np.log10( (this_compare[m]/last_compare[m]) ) ) )
	
			last = this
	
			if (err < max_err) or (niter > max_iter): break
	
		if niter > max_iter:
			print "WARNING: Reached iteration limit of %d iterations.  Expect errors in final magnitudes of %.4f mags.\nIf this is unacceptable try increasing max_iter." % (max_iter,err)

		return resampling

class csp_region_integrator(object):

	seds = np.array( [] )		# SED wavelength/age grid
	ls = np.array( [] )		# SED wavelengths
	ages = np.array( [] )		# SED ages (gyrs)
	minage = 0.0			# lower bound for age integration
	maxage = 0.0			# upper bound for age integration
	sfh_func = ''			# star formation history function
	has_sfh = False			# whether or not the sfh function is set
	dust_func = ''			# dust function
	has_dust = False		# whether or not a dust function is set
	masses = np.array( [] )		# masses
	has_masses = False		# whether or not masses are set

	def __init__( self, seds, ls, ages, minage, maxage ):
		self.seds = seds
		self.ls = ls
		self.ages = ages
		self.minage = minage
		self.maxage = maxage

	def set_sfh( self, wrapper ):
		self.sfh_func = wrapper
		self.has_sfh = True

	def set_dust( self, wrapper ):
		self.dust_func = wrapper
		self.has_dust = True

	def set_masses( self, masses ):
		self.masses = masses
		self.has_masses = True

	def get_sfh_normalization( self ):
		if not self.has_sfh: raise ValueError( 'You must set the SFH function before getting a normalization!' )
		return integrate.quad( self.sfh_func, self.minage, self.maxage )[0]

	def integrate( self, age ):

		# calculate range to integrate over
		lower_bound = age - self.maxage
		upper_bound = age - self.minage

		# if upper_bound is negative then this is a later region and we are working on early ages
		# If so, return zeros
		if upper_bound < 0: return ( np.zeros( self.ls.size ), 0 )

		# find things in the age range (include a little extra to account for rounding errors)
		inds = np.where( (self.ages >= lower_bound-age*1e-5) & (self.ages <= upper_bound+age*1e-5) )[0]

		# simpsons rule is based on intervals, so include the SED one age lower if it exists
		# otherwise one interval will be missed at every boundary
		if inds[0] > 0 and np.abs( self.ages[inds[0]] - lower_bound ) > 1e-5*age:
			inds = np.append( inds[0]-1, inds )

		weights = self.sfh_func( age-self.ages[inds] )

		# if weights are all zero then there is no star formation in this region and therefore no need to integrate
		if max( weights ) <= 0:
			return ( np.zeros( self.ls.size ), 0 )

		if self.has_dust:
			# integrate weights*sed*dust
			seds = integrate.simps( weights*self.seds[:,inds]*self.dust_func( self.ages[inds], self.ls ), x=self.ages[inds], even='avg' )
		else:
			# integrate weights*sed
			seds = integrate.simps( weights*self.seds[:,inds], x=self.ages[inds], even='avg' )

		# integrate weights*mass
		mass = integrate.simps( weights*self.masses[inds], x=self.ages[inds], even='avg' ) if self.has_masses else 0

		return ( seds, mass )