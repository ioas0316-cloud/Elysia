import unittest
import sys
import os
from unittest.mock import MagicMock
import math

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Evolution.Evolution.Life.resonance_voice import ResonanceEngine, Oscillator

class TestCirculatoryThinking(unittest.TestCase):
    def setUp(self):
        self.engine = ResonanceEngine()
        # Mock internal sea with some concepts
        self.engine.internal_sea = {
            "source": Oscillator(amplitude=1.0, frequency=1.0),
            "neighbor": Oscillator(amplitude=0.0, frequency=1.0),
            "distant": Oscillator(amplitude=0.0, frequency=1.0)
        }
        # Mock associations
        self.engine.associations = {
            "source": ["neighbor"]
        }
        # Mock Hippocampus
        self.engine.memory = MagicMock()
        self.engine.memory.causal_graph.neighbors.return_value = []
        # Mock vocabulary
        self.engine.vocabulary = {
            "source": 1.0,
            "neighbor": 1.0,
            "distant": 1.0
        }

    def test_reverberate_diffusion(self):
        """Test that energy diffuses from source to neighbor."""
        # Initial state
        self.assertEqual(self.engine.internal_sea["source"].amplitude, 1.0)
        self.assertEqual(self.engine.internal_sea["neighbor"].amplitude, 0.0)
        
        # Run reverberation
        self.engine.reverberate(diffusion_rate=0.5, decay_rate=0.0)
        
        # Source should lose some energy (conceptually, though current impl only decays via decay_rate)
        # Actually, my impl adds to neighbor based on source, but doesn't subtract from source for diffusion (it's resonance, not fluid dynamics).
        # But source should decay if decay_rate > 0. Here decay is 0.
        
        # Neighbor should gain energy
        # flow = 1.0 * 0.5 / 1 = 0.5
        self.assertAlmostEqual(self.engine.internal_sea["neighbor"].amplitude, 0.5)
        
        print(f"Source Amp: {self.engine.internal_sea['source'].amplitude}")
        print(f"Neighbor Amp: {self.engine.internal_sea['neighbor'].amplitude}")

    def test_self_listen_feedback(self):
        """Test that output text generates new ripples."""
        # Mock extract_concepts to return a concept
        self.engine._extract_concepts = MagicMock(return_value=["source"])
        
        # Reset source amplitude
        self.engine.internal_sea["source"].amplitude = 0.0
        
        self.engine.self_listen("This is a source text")
        
        # Source should gain amplitude from feedback
        # Amplitude += 0.3 (from code)
        self.assertAlmostEqual(self.engine.internal_sea["source"].amplitude, 0.3)
        
        # Check phase shift (should be pi/2)
        self.assertAlmostEqual(self.engine.internal_sea["source"].phase, math.pi / 2)

if __name__ == '__main__':
    unittest.main()
