import unittest
import ezgal.dusts
import numpy as np

class test_dust_wrapper(unittest.TestCase):
	
	def setUp( self ):
		
		# test a wrapper with arguments
		self.wrapper_with_args = ezgal.dusts.dust_wrapper( lambda x, y, z : x * y * z, ( 2, ) )
		
		# test a wrapper without arguments
		self.wrapper_without_args = ezgal.dusts.dust_wrapper( lambda x, y : x * y, () )
	
	def test_args( self ):
		
		self.assertEqual( self.wrapper_with_args( 3, 6 ), 36 )
	
	def test_no_args( self ):
		
		self.assertEqual( self.wrapper_without_args( 3, 6 ), 18 )

	
if __name__ == '__main__':
	unittest.main()