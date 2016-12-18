import unittest
import ezgal.cosmology
import tests
import numpy as np

class test_getz(unittest.TestCase):
	
	cosmo = None
	
	def setUp( self ):
		
		# standard cosmology
		self.cosmo = ezgal.cosmology.Cosmology( Om=0.272, Ol=0.728, h=0.704, w=-1 )
	
	def test_getz_array_high( self ):
		
		self.assertTrue( np.allclose(
			self.cosmo.GetZ( [1e10,1e9,1e8], 3 ),
			[ 0.1217726,  2.1012914, 2.8822294],
			1e-7
		) );
	
	def test_getz_scalar_high( self ):
		
		self.assertTrue( np.allclose(
			self.cosmo.GetZ( 5e9, 3 ),
			[ 0.7386323 ],
			1e-7
		) );
	
	def test_getz_cache_check( self ):
		
		# getz caches results for a formation redshift
		# make sure that works properly by calling twice
		# so the cache gets used.
		self.cosmo.GetZ( [5e9,1e9,1e8], 1 )
		
		self.assertTrue( np.allclose(
			self.cosmo.GetZ( [2e9,5e8], 1 ),
			[ 0.60802307,  0.88482757],
			1e-7
		) );
	
	def test_getz_no_negative( self ):
		
		with self.assertRaises( ValueError ):
			self.cosmo.GetZ( -1, 3 )
	
	def test_getz_no_old( self ):
		
		with self.assertRaises( ValueError ):
			self.cosmo.GetZ( self.cosmo.Tl( 3, yr=True )+1, 3 )
	
if __name__ == '__main__':
	unittest.main()