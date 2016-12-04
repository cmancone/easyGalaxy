import unittest
import ezgal.utils as utils
import numpy as np
import StringIO

class test_rascii(unittest.TestCase):
	
	def test_one_d( self ):
		
		test_file = StringIO.StringIO( '1 2 3 4' )
		answer = np.asarray( [ [ 1., 2, 3, 4 ] ] )
		
		self.assertTrue( np.array_equal( utils.rascii( test_file, True ), answer ) )
	
	def test_two_d( self ):
		
		test_file = StringIO.StringIO( '1 2 3 4\n5 6 7 8' )
		answer = np.asarray( [ [ 1., 2, 3, 4 ], [ 5, 6, 7, 8 ] ] )
		
		self.assertTrue( np.array_equal( utils.rascii( test_file, True ), answer ) )
	
	def test_comments( self ):
		
		test_file = StringIO.StringIO( '# "comments" are ignored\nI should get one line\n1 2 3 4' )
		answer = np.asarray( [ [ 1., 2, 3, 4 ] ] )
		
		self.assertTrue( np.array_equal( utils.rascii( test_file, True ), answer ) )
	
	def test_scientific( self ):
		
		# scientific notation needs to be properly supported
		test_file = StringIO.StringIO( '-1e0 +2.0E1' )
		answer = np.asarray( [ [ -1., 20 ] ] )
		
		self.assertTrue( np.array_equal( utils.rascii( test_file, True ), answer ) )
	
if __name__ == '__main__':
	unittest.main()