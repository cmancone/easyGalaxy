import unittest
import ezgal.zf_grid
import numpy as np
import math

# I put the test data for the zf_grid tests in
# tests.zf_grid instead of in tests because
# there is a lot of data but it is all
# specific for this test.
import tests.zf_grid

class test_get_masses(tests.zf_grid.test_zf_grid):
	
	def test_get_masses( self ):
		
		self.assertTrue( np.allclose(
			self.zf_grid.get_masses( tests.zf_grid.test_zs ),
			[ 0.55,  0.65,  0.75,  0.85,  0.925, 0.9667],
			1e-4
		) )
	
	def test_get_masses_lower_bound( self ):
		
		# if we go lower than our lowest grided z then
		# we should get a nan
		vals = self.zf_grid.get_masses( [-1] )
		
		self.assertTrue( math.isnan( vals[0] ) )
	
	def test_get_masses_upper_bound( self ):
		
		# if we go lower than our lowest grided z then
		# we should get a nan
		vals = self.zf_grid.get_masses( [4] )
		
		self.assertTrue( math.isnan( vals[0] ) )
	
	def test_get_masses_empty_failure( self ):
		
		with self.assertRaises( ValueError ):
			self.zf_grid_empty.get_masses( tests.zf_grid.test_zs )
	
	
if __name__ == '__main__':
	unittest.main()