#!/usr/bin/python
import unittest

import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

tests = unittest.TestLoader().discover('tests')
testRunner = unittest.runner.TextTestRunner()
testRunner.run(tests)
