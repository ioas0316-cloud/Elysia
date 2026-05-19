import unittest
import sys
import os

# Ensure Core is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from Core.Monad.ouroboros_loop import OuroborosLoop, SovereignVector

class MockEngine:
    def read_field_state(self):
        return {'torque': 0.8, 'enthalpy': 0.6}

class TestOuroborosLoop(unittest.TestCase):
    def setUp(self):
        self.engine = MockEngine()
        self.log_messages = []

        # Capture logs
        def mock_logger(msg):
            self.log_messages.append(msg)

        self.loop = OuroborosLoop(self.engine, log_callback=mock_logger)

    def test_feed_output_as_input(self):
        # Create a mock 21D vector
        vec1 = SovereignVector([1.0] * 21)
        self.loop.feed_output_as_input(vec1)

        # Check if resonance was set
        self.assertIsNotNone(self.loop.residual_resonance)
        self.assertEqual(len(self.loop.residual_resonance.data), 21)
        self.assertTrue(any("Output fed back into Input" in msg for msg in self.log_messages))

        # Feed another vector to test attenuation
        vec2 = SovereignVector([0.0] * 21)
        self.loop.feed_output_as_input(vec2)
        # Assuming attenuation factor 0.8: new = 0.2*vec1 + 0.8*vec2 = 0.2
        self.assertAlmostEqual(self.loop.residual_resonance.data[0], 0.2)

    def test_dream_cycle(self):
        # Initial dream cycle with no prior resonance
        self.loop.dream_cycle()
        self.assertTrue(self.loop.is_dreaming)
        self.assertGreater(self.loop.dream_depth, 0.0)
        self.assertEqual(len(self.loop.dream_history), 1)
        self.assertIsNotNone(self.loop.residual_resonance)

        initial_history_len = len(self.loop.dream_history)

        # Second cycle should build upon the first
        self.loop.dream_cycle()
        self.assertEqual(len(self.loop.dream_history), initial_history_len + 1)
        self.assertTrue(any("💭 [DREAM DEPTH" in msg for msg in self.log_messages))

if __name__ == '__main__':
    unittest.main()
