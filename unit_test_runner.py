import unittest

TEST_FOLDER_NAME = "Tests"

###############
#Run unit tests
###############
suite = unittest.TestLoader().discover(start_dir = TEST_FOLDER_NAME, pattern="*_tests.py")
test_runner = unittest.TextTestRunner(verbosity = 2, failfast = True)
test_result = test_runner.run(suite)
assert test_result.wasSuccessful(), "At least one test failed."
