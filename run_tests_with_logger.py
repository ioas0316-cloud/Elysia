# run_tests_with_logger.py
import unittest
from tests.test_logger import JsonTestResult

def run_tests():
    """
    Discovers and runs all tests, using the JsonTestResult class to log results.
    """
    suite = unittest.defaultTestLoader.discover('tests')
    runner = unittest.TextTestRunner(resultclass=JsonTestResult)
    runner.run(suite)

if __name__ == '__main__':
    run_tests()
