import unittest
import ezgal.sfhs
import numpy as np

class test_wrapper(unittest.TestCase):
	
	def setUp( self ):
		
		# test a wrapper with arguments (multiply by two)
		self.wrapper_with_args = ezgal.sfhs.sfh_wrapper( lambda x, y : x * y, ( 2, ) )
		
		# test a wrapper without arguments (square)
		self.wrapper_without_args = ezgal.sfhs.sfh_wrapper( lambda x : x ** 2, () )
	
	def test_args( self ):
		
		self.assertEqual( self.wrapper_with_args( 3 ), 6 )
	
	def test_no_args( self ):
		
		self.assertEqual( self.wrapper_without_args( 3 ), 9 )

	
if __name__ == '__main__':
	unittest.main()