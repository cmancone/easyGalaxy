import unittest
import ezgal.sfhs
import numpy as np

class test_numeric(unittest.TestCase):
	
	def setUp( self ):
		
		self.numeric = ezgal.sfhs.numeric( [0,5], [0,5] )
	
	def test_numeric_0( self ):
		
		self.assertAlmostEqual( self.numeric( 0 ), 0, 7 )
	
	def test_numeric_25( self ):
		
		self.assertAlmostEqual( self.numeric( 2.5 ), 2.5, 7 )
	
	def test_numeric_5( self ):
		
		self.assertAlmostEqual( self.numeric( 5 ), 5, 7 )
	
if __name__ == '__main__':
	unittest.main()