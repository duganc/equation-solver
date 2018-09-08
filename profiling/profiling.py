import cProfile
import unittest
from tests.transformation_tests import TransformationTests

suite = unittest.TestSuite()
suite.addTest(TransformationTests("test_single_variable_solver_handles_distribution"))
runner = unittest.TextTestRunner()
cProfile.run("runner.run(suite)")
