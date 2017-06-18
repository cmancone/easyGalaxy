from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from . import utils
__ver__ = '1.0'

class dust_wrapper(object):
	""" dust_wrapper class.  EzGal wraps this class around the dust function.  It takes care of the 
	details of passing or not passing parameters """

	func = ''		# sfh function
	args = ()		# extra arguments to pass on call
	has_args = False	# whether or not there are actually any extra arguments

	def __init__( self, function, args ):

		self.func = function

		if type( args ) == type( () ) and len( args ) > 0:
			self.has_args = True
			self.args = args

	def __call__( self, time, ls ):

		if self.has_args:
			return self.func( time, ls, *self.args )
		else:
			return self.func( time, ls )

class charlot_fall(object):
	""" callable-object implementation of the Charlot and Fall (2000) dust law """
	tau1 = 0.0
	tau2 = 0.0
	tbreak = 0.0

	def __init__( self, tau1=1.0, tau2=0.5, tbreak=0.01 ):
		""" dust_obj = charlot_fall( tau1=1.0, tau2=0.3, tbreak=0.01 )
		Return a callable object for returning the dimming factor as a function of age
		for a Charlot and Fall (2000) dust law.  The dimming is:
		
		np.exp( -1*Tau(t)(lambda/5500angstroms) )
		
		Where Tau(t) = `tau1` for t < `tbreak` (in gyrs) and `tau2` otherwise. """

		self.tau1 = tau1
		self.tau2 = tau2
		self.tbreak = tbreak

	def __call__( self, ts, ls ):

		ls = np.asarray( ls )
		ts = np.asarray( ts )
		ls.shape = (ls.size,1)
		ts.shape = (1,ts.size)

		taus = np.asarray( [self.tau1]*ts.size )
		m = (ts > self.tbreak).ravel()
		if m.sum(): taus[m] = self.tau2

		return np.exp( -1.0*taus*(ls/5500.0)**-0.7 )

class calzetti(object):
	""" callable-object implementation of the Calzetti et al. (2000) dust law """
	av = 0.0
	rv = 0.0
	ebv = 0.0
	esbv = 0.0
	
	def __init__( self, av=1.0, rv=4.05 ):
		""" dust_obj = calzetti( av=1.0, rv=4.05 )
		Return a callable object for returning the dimming factor as a function of age
		for a Calzetti et al. (2000) dust law.  The dimming is:
		
		 """
		
		self.av = av
		self.rv = rv
		self.ebv = self.av/self.rv
		self.esbv = self.ebv*0.44
		
	def __call__( self, ts, ls ):
		
		# calzetti was fit in microns...
		ls = utils.convert_length( np.asarray( ls ), incoming='a', outgoing='um' )
		
		ks = np.zeros( ls.size )
		s = ls < .63
		if s.any(): ks[s] = 2.659*( -2.156 + 1.509/ls[s] - 0.198/ls[s]**2.0 + 0.011/ls[s]**3.0 ) + self.rv
		l = ~s
		if l.any(): ks[l] = 2.659*( -1.857 + 1.040/ls[l] ) + self.rv
		
		# calculate dimming factor as a function of lambda
		factors = 10.0**( -0.4*self.esbv*ks )
		
		# need to return an array of shape (nls,nts).  Therefore, repeat
		return factors.reshape( (ls.size,1) ).repeat( len( ts ), axis=1 )
