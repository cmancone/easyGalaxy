import unittest
import ezgal.zf_grid
import numpy as np
import math

# I put the test data for the zf_grid tests in
# tests.zf_grid instead of in tests because
# there is a lot of data but it is all
# specific for this test.
import tests.zf_grid

class test_get_solar_mags(tests.zf_grid.test_zf_grid):
	
	def test_get_solar_mags( self ):
		
		self.assertTrue( np.allclose(
			self.zf_grid.get_solar_mags( tests.zf_grid.test_zs ),
			[ 0.275,  0.75,  1.25,  1.75,  2.25, 2.778],
			1e-4
		) )
	
	def test_get_solar_mags_lower_bound( self ):
		
		# if we go lower than our lowest grided z then
		# we should get a nan
		vals = self.zf_grid.get_solar_mags( [-1] )
		
		self.assertTrue( math.isnan( vals[0] ) )
	
	def test_get_solar_mags_upper_bound( self ):
		
		# if we go lower than our lowest grided z then
		# we should get a nan
		vals = self.zf_grid.get_solar_mags( [4] )
		
		self.assertTrue( math.isnan( vals[0] ) )
	
	def test_get_solar_mags_empty_failure( self ):
		
		with self.assertRaises( ValueError ):
			self.zf_grid_empty.get_solar_mags( tests.zf_grid.test_zs )

	
	
if __name__ == '__main__':
	unittest.main()