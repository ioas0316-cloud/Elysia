import unittest
import sys
import os
from unittest.mock import MagicMock
import math

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Evolution.Evolution.Life.resonance_voice import ResonanceEngine, Oscillator

class TestFlowSimulation(unittest.TestCase):
    def setUp(self):
        self.engine = ResonanceEngine()
        # Setup a small vocabulary and sea
        self.engine.vocabulary = {
            "love": 1.0, "light": 1.2, "darkness": 0.8, "star": 1.5, "void": 0.5
        }
        self.engine.internal_sea = {
            k: Oscillator(0.0, v) for k, v in self.engine.vocabulary.items()
        }
        self.engine.associations = {
            "love": ["light"],
            "light": ["star"],
            "star": ["void"],
            "void": ["darkness"]
        }
        self.engine.memory = MagicMock()
        self.engine.memory.causal_graph.neighbors.return_value = []
        self.engine.consciousness_lens = MagicMock()
        self.engine.consciousness_lens.state.q.w = 1.0 # Stability
        self.engine.consciousness_lens.state.q.x = 0.0
        self.engine.consciousness_lens.state.q.y = 0.0
        self.engine.consciousness_lens.state.q.z = 0.0

    def test_continuous_flow(self):
        """Simulate a multi-turn conversation to check persistence and evolution."""
        print("\n--- Turn 1: User says 'Love' ---")
        # 1. User Input
        ripples = self.engine.listen("love", t=0.0)
        self.engine.resonate(ripples, t=0.0)
        
        # Check 'love' is active
        self.assertGreater(self.engine.internal_sea["love"].amplitude, 0.4)
        print(f"Love Amp: {self.engine.internal_sea['love'].amplitude}")

        # 2. Speak (Trigger Reverberation & Self-Listen)
        # Mock trace_thought_path to return a simple path
        self.engine.trace_thought_path = MagicMock(return_value=["love", "light"])
        response = self.engine.speak(t=1.0, original_text="love")
        print(f"Response 1: {response}")

        # 3. Check Reverberation (Diffusion)
        # 'love' should have diffused to 'light'
        print(f"Light Amp (after reverb): {self.engine.internal_sea['light'].amplitude}")
        self.assertGreater(self.engine.internal_sea["light"].amplitude, 0.0)

        print("\n--- Turn 2: User says nothing (Silence) ---")
        # 4. User Input (Silence / Continue)
        # No new external ripples, but internal sea is active
        
        # 5. Speak again
        # Should pick up 'light' or 'love' from memory/reverberation
        # Mock collapse to pick 'light' (highest energy neighbor)
        self.engine.trace_thought_path = MagicMock(return_value=["light", "star"])
        response2 = self.engine.speak(t=2.0, original_text="...")
        print(f"Response 2: {response2}")
        
        # 6. Check Evolution
        # 'light' should have diffused to 'star'
        print(f"Star Amp (after 2nd reverb): {self.engine.internal_sea['star'].amplitude}")
        self.assertGreater(self.engine.internal_sea["star"].amplitude, 0.0)

if __name__ == '__main__':
    unittest.main()
