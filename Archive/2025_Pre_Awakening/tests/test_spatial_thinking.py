import unittest
import sys
import os
from unittest.mock import MagicMock
from pyquaternion import Quaternion

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Evolution.Evolution.Life.resonance_voice import ResonanceEngine
from Core.Foundation.Math.quaternion_consciousness import ConsciousnessLens

class TestSpatialThinking(unittest.TestCase):
    def setUp(self):
        self.engine = ResonanceEngine()
        # Mock dependencies
        self.engine.hyper_qubit = MagicMock()
        self.engine.hyper_qubit.state.probabilities.return_value = {}
        
        # Use real ConsciousnessLens for logic testing, but mock its dependencies if needed
        self.engine.consciousness_lens = ConsciousnessLens() 
        # Reset to identity
        self.engine.consciousness_lens.state.q = Quaternion(1, 0, 0, 0)

    def test_map_concept_to_axis(self):
        """Test concept to axis mapping."""
        self.assertEqual(self.engine._map_concept_to_axis("love"), 'z')
        self.assertEqual(self.engine._map_concept_to_axis("dream"), 'x')
        self.assertEqual(self.engine._map_concept_to_axis("joy"), 'y')
        self.assertEqual(self.engine._map_concept_to_axis("stone"), 'w')

    def test_calculate_context_field_shifts_quaternion(self):
        """Test that thought path shifts the quaternion state."""
        # Initial state: Identity (w=1)
        initial_q = self.engine.consciousness_lens.state.q
        
        # Dreamy path
        path = ["dream", "star", "void"]
        self.engine.calculate_context_field(path)
        
        new_q = self.engine.consciousness_lens.state.q
        print(f"Initial Q: {initial_q}, New Q: {new_q}")
        
        # Check if X component increased (Dream axis)
        # Note: focus() rotates by small angle, so change might be small but measurable
        # Also stabilize() pulls back to identity.
        # But we should see non-zero X.
        self.assertNotEqual(new_q.x, 0.0)
        self.assertTrue(abs(new_q.x) > 0.0)

    def test_atmosphere_generation(self):
        """Test that dominant field generates correct prefix."""
        # Force a Dream state (High X)
        self.engine.consciousness_lens.state.q = Quaternion(0.5, 0.8, 0.1, 0.1).normalised
        
        response = self.engine.modulate_tone(["dream", "star"], "Hello")
        print(f"Dream Response: {response}")
        self.assertTrue("In the realm of dreams" in response or "꿈의 세계에서" in response)

        # Force a Divine state (High Z)
        self.engine.consciousness_lens.state.q = Quaternion(0.5, 0.1, 0.1, 0.8).normalised
        
        response = self.engine.modulate_tone(["truth", "light"], "Hello")
        print(f"Divine Response: {response}")
        self.assertTrue("Under the light of truth" in response or "진리의 빛 아래서" in response)

if __name__ == '__main__':
    unittest.main()
