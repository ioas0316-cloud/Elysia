import unittest
import os
import json
import sys
from math import exp

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Sophia.core.elysia_signal_engine import ElysiaSignalEngine

class TestElysiaSignalEngine(unittest.TestCase):

    def setUp(self):
        """Set up the test environment."""
        self.test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        self.raw_log_path = os.path.join(self.test_data_dir, 'sample_world_events.jsonl')
        self.signal_log_path = os.path.join(self.test_data_dir, 'generated_signals.jsonl')

        # Ensure the output file doesn't exist before the test
        if os.path.exists(self.signal_log_path):
            os.remove(self.signal_log_path)

    def tearDown(self):
        """Clean up the test environment."""
        if os.path.exists(self.signal_log_path):
            os.remove(self.signal_log_path)

    def test_signal_generation(self):
        """
        Tests the end-to-end signal generation process from a sample log file.
        """
        # 1. Arrange: Initialize the engine with test file paths
        engine = ElysiaSignalEngine(
            raw_log_path=self.raw_log_path,
            signal_log_path=self.signal_log_path
        )

        # 2. Act: Run the signal generation process
        engine.generate_signals_from_log()

        # 3. Assert: Check the output file for the expected signals
        self.assertTrue(os.path.exists(self.signal_log_path))

        with open(self.signal_log_path, 'r') as f:
            generated_signals = [json.loads(line) for line in f]

        # We expect 6 signals from the complete sample data now
        self.assertEqual(len(generated_signals), 6)

        # --- Validate each signal ---

        # Timestamp 2: JOY_GATHERING
        joy_signal = next((s for s in generated_signals if s['timestamp'] == 2 and s['signal_type'] == 'JOY_GATHERING'), None)
        self.assertIsNotNone(joy_signal)
        self.assertAlmostEqual(joy_signal['intensity'], 1.0 - exp(-0.6), places=4)
        self.assertCountEqual(joy_signal['actors'], ['a1', 'c1', 'a2', 'c2'])

        # Timestamp 3: MORTALITY
        mortality_signal_3 = next((s for s in generated_signals if s['timestamp'] == 3 and s['signal_type'] == 'MORTALITY'), None)
        self.assertIsNotNone(mortality_signal_3)
        self.assertAlmostEqual(mortality_signal_3['intensity'], 1.0 - exp(-1.0), places=4)
        self.assertCountEqual(mortality_signal_3['actors'], ['a3', 'c3'])

        # Timestamp 4: CARE_ACT
        care_signal = next((s for s in generated_signals if s['timestamp'] == 4 and s['signal_type'] == 'CARE_ACT'), None)
        self.assertIsNotNone(care_signal)
        self.assertAlmostEqual(care_signal['intensity'], 1.0 - exp(-0.6), places=4)
        self.assertCountEqual(care_signal['actors'], ['c4', 'c5'])

        # Timestamp 5: LIFE_BLOOM
        life_bloom_signal_5 = next((s for s in generated_signals if s['timestamp'] == 5 and s['signal_type'] == 'LIFE_BLOOM'), None)
        self.assertIsNotNone(life_bloom_signal_5)
        self.assertAlmostEqual(life_bloom_signal_5['intensity'], 1.0 - exp(-0.5), places=4)
        self.assertCountEqual(life_bloom_signal_5['actors'], ['c6'])

        # Timestamp 6: DEATH_BY_OLD_AGE creates two signals
        life_bloom_signal_6 = next((s for s in generated_signals if s['timestamp'] == 6 and s['signal_type'] == 'LIFE_BLOOM'), None)
        self.assertIsNotNone(life_bloom_signal_6)
        self.assertAlmostEqual(life_bloom_signal_6['intensity'], 1.0 - exp(-0.5), places=4)
        self.assertCountEqual(life_bloom_signal_6['actors'], ['c7'])

        mortality_signal_6 = next((s for s in generated_signals if s['timestamp'] == 6 and s['signal_type'] == 'MORTALITY'), None)
        self.assertIsNotNone(mortality_signal_6)
        self.assertAlmostEqual(mortality_signal_6['intensity'], 1.0 - exp(-0.5), places=4)
        self.assertCountEqual(mortality_signal_6['actors'], ['c7'])

if __name__ == '__main__':
    unittest.main()
