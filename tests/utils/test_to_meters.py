import unittest
import ezgal.utils as utils
import numpy as np

class test_to_meters(unittest.TestCase):
	
	def test_angstroms( self ):
		self.assertEqual( utils.to_meters( 1e10, units='a' ), 1 )
	
	def test_nanometers( self ):
		self.assertEqual( utils.to_meters( 1e9, units='nm' ), 1 )
	
	def test_microns( self ):
		self.assertEqual( utils.to_meters( 1e6, units='um' ), 1 )
	
	def test_milimeters( self ):
		self.assertEqual( utils.to_meters( 1e3, units='mm' ), 1 )
	
	def test_centimeters( self ):
		self.assertEqual( utils.to_meters( 1e2, units='cm' ), 1 )
	
	def test_meters( self ):
		self.assertEqual( utils.to_meters( 1, units='m' ), 1 )
	
	def test_kilometers( self ):
		self.assertEqual( utils.to_meters( 1, units='km' ), 1e3 )
	
	def test_astronomical_units( self ):
		self.assertEqual( utils.to_meters( 1, units='au' ), 1.49598e11 )
	
	def test_parsecs( self ):
		# assertAlmostEqual does not work for large numbers,
		# so we have to divide by 1e16 to make it work out
		self.assertEqual( utils.to_meters( 1, units='pc' )/1e16, 1.49598e11*3600*180/np.pi/1e16, 7 )
	
	def test_kiloparsecs( self ):
		# assertAlmostEqual does not work for large numbers,
		# so we have to divide by 1e19 to make it work out
		self.assertEqual( utils.to_meters( 1, units='kpc' )/1e19, 1.49598e14*3600*180/np.pi/1e19, 7 )
	
	def test_megaparsecs( self ):
		# assertAlmostEqual does not work for large numbers,
		# so we have to divide by 1e22 to make it work out
		self.assertAlmostEqual( utils.to_meters( 1, units='mpc' )/1e22, 1.49598e17*3600*180/np.pi/1e22, 7 )
	
	# to hertz, to_lambda, and convert_length are all small variations of
	# to_meters, which is why I have gone ahead and included them here.
	# We don't need to test all of the units though, since it all just
	# boils down to some simple arithmetic on results from to_meters,
	# which is fully tested above
	def test_to_hertz( self ):
		self.assertAlmostEqual( utils.to_hertz( 1000, units='a' )/1e15, 2.99792458, 8 )
	
	def test_to_lambda( self ):
		self.assertAlmostEqual( utils.to_lambda( 2.99792458e15, units='a' ), 1000, 8 )
	
	def test_convert_length( self ):
		self.assertEqual( utils.convert_length( 10, incoming='km', outgoing='m' ), 10000 )
	
if __name__ == '__main__':
	unittest.main()