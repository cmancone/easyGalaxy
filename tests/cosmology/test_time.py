import unittest
import ezgal.cosmology
import tests
import numpy as np

class test_lengths(unittest.TestCase):
	
	cosmo = None
	
	def setUp( self ):
		
		# standard cosmology
		self.cosmo = ezgal.cosmology.Cosmology( Om=0.272, Ol=0.728, h=0.704, w=-1 )
	
	def test_Tl( self ):
		
		self.assertAlmostEqual( self.cosmo.Tl( 1 )/1e17, 2.4578361, 7 )
	
	def test_Tl_s( self ):
		
		self.assertAlmostEqual( self.cosmo.Tl( 1, s=True )/1e17, 2.4578361, 7 )
	
	def test_Tl_year( self ):
		
		self.assertAlmostEqual( self.cosmo.Tl( 1, yr=True )/1e9, 7.7884125, 7 )
	
	def test_Tl_myr( self ):
		
		self.assertAlmostEqual( self.cosmo.Tl( 1, myr=True )/1e3, 7.7884125, 7 )
	
	def test_Tl_gyr( self ):
		
		self.assertAlmostEqual( self.cosmo.Tl( 1, gyr=True ), 7.7884125, 7 )
	
if __name__ == '__main__':
	unittest.main()