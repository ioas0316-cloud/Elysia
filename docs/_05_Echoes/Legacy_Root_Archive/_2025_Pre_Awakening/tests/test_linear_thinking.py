import unittest
import sys
import os
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Evolution.Evolution.Life.resonance_voice import ResonanceEngine
from Core.Foundation.Mind.hippocampus import Hippocampus

class TestLinearThinking(unittest.TestCase):
    def setUp(self):
        # Mock Hippocampus
        self.mock_memory = MagicMock(spec=Hippocampus)
        self.engine = ResonanceEngine(hippocampus=self.mock_memory)
        
        # Mock dependencies
        self.engine.hyper_qubit = MagicMock()
        self.engine.hyper_qubit.state.probabilities.return_value = {"a": 0.5, "b": 0.5}
        self.engine.consciousness_lens = MagicMock()
        self.engine.consciousness_lens.state.q.w = 0.5
        self.engine.consciousness_lens.state.q.x = 0.1
        self.engine.consciousness_lens.state.q.y = 0.1
        self.engine.consciousness_lens.state.q.z = 0.1

    def test_trace_thought_path_success(self):
        """Test that trace_thought_path uses Hippocampus to find a path."""
        # Setup mock path
        self.mock_memory.find_path.return_value = ["Darkness", "Star", "Light"]
        
        path = self.engine.trace_thought_path("Darkness", "Light")
        
        print(f"Trace Path: {path}")
        self.assertEqual(path, ["Darkness", "Star", "Light"])
        self.mock_memory.find_path.assert_called_with("Darkness", "Light")

    def test_trace_thought_path_fallback(self):
        """Test fallback to associations if Hippocampus fails."""
        self.mock_memory.find_path.return_value = []
        
        # Add associations to engine manually for testing
        self.engine.associations = {
            "A": ["B"],
            "B": ["C"]
        }
        
        # Direct association
        path = self.engine.trace_thought_path("A", "B")
        self.assertEqual(path, ["A", "B"])
        
        # 1-hop association (A->B->C)
        path = self.engine.trace_thought_path("A", "C")
        self.assertEqual(path, ["A", "B", "C"])

    def test_speak_uses_path_logic(self):
        """Test that speak generates a response implying a path."""
        # Mock internal sea to have specific concepts
        self.engine.internal_sea = {
            "start": MagicMock(amplitude=1.0),
            "end": MagicMock(amplitude=0.9)
        }
        
        # Mock path finding
        self.mock_memory.find_path.return_value = ["start", "middle", "end"]
        
        # Mock modulate_tone to return a specific string we can check
        # We need to mock _phase_info to control the tone template selection
        self.engine._phase_info = MagicMock(return_value=(0.9, 0.1)) # High mastery
        
        response = self.engine.speak(0.0, "test input")
        
        print(f"Speak Response: {response}")
        # Check if the response format matches the High Mastery template for paths
        # "It is clear. {w1} eventually leads to {w2}."
        # w1 should be "start", w2 should be "end"
        self.assertTrue("start" in response)
        self.assertTrue("end" in response)
        self.assertTrue("leads to" in response or "end lies" in response or "completed in" in response)

if __name__ == '__main__':
    unittest.main()
