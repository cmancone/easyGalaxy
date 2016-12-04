#!/usr/bin/python
import unittest

tests = unittest.TestLoader().discover( 'tests' )
testRunner = unittest.runner.TextTestRunner()
testRunner.run(tests)