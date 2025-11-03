# tests/test_logger.py
import unittest
import json
import time
import os

class JsonTestResult(unittest.TestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log = []
        self.log_file = 'test_log.json'
        # Clear the log file at the beginning of the test run
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def startTest(self, test):
        super().startTest(test)
        self.log.append({
            'event': 'start_test',
            'test': test.id(),
            'timestamp': time.time()
        })
        self._write_log()

    def stopTest(self, test):
        super().stopTest(test)
        self.log.append({
            'event': 'stop_test',
            'test': test.id(),
            'timestamp': time.time()
        })
        self._write_log()

    def addSuccess(self, test):
        super().addSuccess(test)
        self.log.append({
            'event': 'add_success',
            'test': test.id(),
            'timestamp': time.time()
        })
        self._write_log()

    def addError(self, test, err):
        super().addError(test, err)
        self.log.append({
            'event': 'add_error',
            'test': test.id(),
            'error': str(err),
            'timestamp': time.time()
        })
        self._write_log()

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.log.append({
            'event': 'add_failure',
            'test': test.id(),
            'error': str(err),
            'timestamp': time.time()
        })
        self._write_log()

    def _write_log(self):
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.log, f, indent=2)
