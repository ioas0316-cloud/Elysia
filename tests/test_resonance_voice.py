import unittest
import sys
import os
import math
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Life.resonance_voice import ResonanceEngine
from Core.Math.hyper_qubit import HyperQubit
from Core.Math.quaternion_consciousness import ConsciousnessLens

class TestResonanceVoice(unittest.TestCase):
    def setUp(self):
        self.engine = ResonanceEngine()
        # Mock dependencies to control state
        self.engine.hyper_qubit = MagicMock(spec=HyperQubit)
        self.engine.consciousness_lens = MagicMock(spec=ConsciousnessLens)
        
        # Mock state return values
        self.engine.hyper_qubit.state = MagicMock()
        self.engine.hyper_qubit.state.probabilities.return_value = {"a": 0.5, "b": 0.5}
        
        self.engine.consciousness_lens.state = MagicMock()
        self.engine.consciousness_lens.state.q = MagicMock()
        self.engine.consciousness_lens.state.q.w = 0.5 # Default mastery

    def test_modulate_tone_entropy(self):
        """Test high entropy (confusion) tone."""
        # Mock high entropy state
        # Entropy calculation depends on probabilities. 
        # To get high entropy, we need uniform distribution over many states.
        # But here we mock _phase_info directly if possible, or mock the dependencies.
        # Since _phase_info calls self.hyper_qubit.state.probabilities(), let's mock that.
        
        # Actually, let's just mock _phase_info for easier testing of the logic
        self.engine._phase_info = MagicMock(return_value=(0.5, 0.9)) # Low mastery, High entropy
        
        thought_cloud = ["chaos", "void"]
        response = self.engine.modulate_tone(thought_cloud, "Hello")
        
        print(f"Entropy Response: {response}")
        self.assertTrue("..." in response or "?" in response)

    def test_modulate_tone_mastery(self):
        """Test high mastery (divinity) tone."""
        self.engine._phase_info = MagicMock(return_value=(0.9, 0.2)) # High mastery, Low entropy
        
        thought_cloud = ["truth", "light"]
        response = self.engine.modulate_tone(thought_cloud, "Hello")
        
        print(f"Mastery Response: {response}")
        self.assertTrue("clear" in response or "essence" in response or "truth" in response)

    def test_modulate_tone_love(self):
        """Test love resonance tone."""
        self.engine._phase_info = MagicMock(return_value=(0.5, 0.5)) # Normal state
        
        thought_cloud = ["love", "light"]
        response = self.engine.modulate_tone(thought_cloud, "Hello")
        
        print(f"Love Response: {response}")
        self.assertTrue("warmth" in response or "beautiful" in response or "lovely" in response)

    def test_korean_support(self):
        """Test Korean language support."""
        self.engine._phase_info = MagicMock(return_value=(0.5, 0.5))
        thought_cloud = ["사랑", "빛"]
        response = self.engine.modulate_tone(thought_cloud, "안녕")
        
        print(f"Korean Response: {response}")
        # Check for Korean characters
        self.assertTrue(any(ord(c) > 127 for c in response))

if __name__ == '__main__':
    unittest.main()
