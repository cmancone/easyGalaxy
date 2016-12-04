import unittest
import ezgal.utils as utils
import numpy as np

class test_to_years(unittest.TestCase):
	
	def test_from_gyrs( self ):
		self.assertEqual( utils.to_years( 20, units='gyrs' ), 2e10 )
	
	def test_to_gyrs( self ):
		self.assertEqual( utils.to_years( 2e10, units='gyrs', reverse=True ), 20 )
	
	def test_from_myrs( self ):
		self.assertEqual( utils.to_years( 15, units='myrs' ), 15e6 )
	
	def test_to_myrs( self ):
		self.assertEqual( utils.to_years( 15e6, units='myrs', reverse=True ), 15 )
	
	def test_from_yrs( self ):
		self.assertEqual( utils.to_years( 15, units='yrs' ), 15 )
	
	def test_to_yrs( self ):
		self.assertEqual( utils.to_years( 15, units='yrs', reverse=True ), 15 )
	
	def test_from_days( self ):
		self.assertEqual( utils.to_years( 365, units='days' ), 1 )
	
	def test_to_days( self ):
		self.assertEqual( utils.to_years( 1, units='days', reverse=True ), 365 )
	
	def test_from_seconds( self ):
		self.assertEqual( utils.to_years( 86400*365, units='secs' ), 1 )
	
	def test_to_seconds( self ):
		self.assertEqual( utils.to_years( 1, units='secs', reverse=True ), 86400*365 )
	
	def test_from_log( self ):
		self.assertEqual( utils.to_years( 9, units='log' ), 1e9 )
	
	def test_to_log( self ):
		self.assertEqual( utils.to_years( 1e9, units='log', reverse=True ), 9 )
	
	def test_invalid_name( self ):
		with self.assertRaises( NameError ):
			utils.to_years( 1e9, 'non-existent units' ) 

if __name__ == '__main__':
	unittest.main()