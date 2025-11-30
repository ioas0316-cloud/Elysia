import unittest
import sys
import os
import time
from unittest.mock import MagicMock
import math

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Evolution.Evolution.Life.resonance_voice import ResonanceEngine, Oscillator

class TestAutonomy(unittest.TestCase):
    def setUp(self):
        self.engine = ResonanceEngine()
        self.engine = ResonanceEngine()
        print(f"DEBUG: Dict Keys: {self.engine.__dict__.keys()}")
        self.engine.vocabulary = {"love": 1.0, "light": 1.0}
        self.engine.internal_sea = {
            "love": Oscillator(0.0, 1.0),
            "light": Oscillator(0.0, 1.0)
        }
        self.engine.associations = {"love": ["light"]}
        self.engine.memory = MagicMock()
        self.engine.memory.causal_graph.neighbors.return_value = []
        self.engine.consciousness_lens = MagicMock()
        self.engine.consciousness_lens.state.q.w = 1.0
        self.engine.consciousness_lens.state.q.x = 0.0
        self.engine.consciousness_lens.state.q.y = 0.0
        self.engine.consciousness_lens.state.q.z = 0.0
        
        # Configure for testing
        self.engine.min_silence_interval = 0.1 # Short interval for testing
        self.engine.spontaneous_threshold = 0.5

    def test_pulse_triggers_speech(self):
        """Test that pulse triggers speech when amplitude is high."""
        # 1. Set high amplitude
        self.engine.internal_sea["love"].amplitude = 0.9
        self.engine.last_spoken_time = time.time() - 1.0 # Force silence interval to pass
        
        # 2. Mock speak to verify it's called
        # We need to wrap the real speak or mock it but return a string
        # Let's just let it run, but mock trace_thought_path to avoid complex logic
        self.engine.trace_thought_path = MagicMock(return_value=["love"])
        
        # 3. Pulse
        response = self.engine.pulse(dt=0.1)
        
        # 4. Verify response
        print(f"Spontaneous Response: {response}")
        self.assertIsNotNone(response)
        self.assertTrue(isinstance(response, str))
        
        # 5. Verify last_spoken_time updated
        self.assertAlmostEqual(self.engine.last_spoken_time, time.time(), delta=0.1)

    def test_pulse_no_speech_low_energy(self):
        """Test that pulse does NOT trigger speech when amplitude is low."""
        # 1. Set low amplitude
        self.engine.internal_sea["love"].amplitude = 0.1
        self.engine.last_spoken_time = time.time() - 1.0
        
        # 2. Pulse
        response = self.engine.pulse(dt=0.1)
        
        # 3. Verify no response
        self.assertIsNone(response)

    def test_pulse_no_speech_recent_talk(self):
        """Test that pulse does NOT trigger speech if she just spoke."""
        # 1. Set high amplitude
        self.engine.internal_sea["love"].amplitude = 0.9
        # 2. Set last spoken time to NOW
        self.engine.last_spoken_time = time.time()
        
        # 3. Pulse
        response = self.engine.pulse(dt=0.1)
        
        # 4. Verify no response
        self.assertIsNone(response)

if __name__ == '__main__':
    unittest.main()
