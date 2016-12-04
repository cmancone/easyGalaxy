import unittest
import ezgal.astro_filter
import tests
import numpy as np

class test_filter_properties(unittest.TestCase):
	
	astro_filter = None
	
	def setUp( self ):
		
		self.astro_filter = ezgal.astro_filter( tests.v_filter )
	
	def test_mean( self ):
		
		self.assertAlmostEqual( self.astro_filter.mean, 5389.0620285, 7 )
	
	def test_pivot( self ):
		
		self.assertAlmostEqual( self.astro_filter.pivot, 5408.2341463, 7 )
	
	def test_average( self ):
		
		self.assertAlmostEqual( self.astro_filter.average, 5417.9103809, 7 )
	
	def test_sig( self ):
		
		self.assertAlmostEqual( self.astro_filter.sig, 0.0591858709, 10 )
	
	def test_width( self ):
		
		self.assertAlmostEqual( self.astro_filter.width, 751.0847588, 7 )
	
	def test_equivalent_width( self ):
		
		self.assertAlmostEqual( self.astro_filter.equivalent_width, 846.9297972, 7 )
	
	def test_rectangular_width( self ):
		
		self.assertAlmostEqual( self.astro_filter.rectangular_width, 846.9297972, 7 )
	
if __name__ == '__main__':
	unittest.main()