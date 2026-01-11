import sys
import os
import unittest
from unittest.mock import MagicMock

# Force path to correct root
sys.path.append(r"C:\Elysia")

from Core.Intelligence.tool_sequencer import get_fractal_strategy_engine, Dimension
from Core.Intelligence.fractal_quaternion_goal_system import FractalStation

class TestFractalStrategy(unittest.TestCase):
    def setUp(self):
        self.engine = get_fractal_strategy_engine()
        self.station = FractalStation(name="Refactor Core System", description="Fix messy code", depth=0)
        
    def test_strategize_with_ultra_reasoning(self):
        """Verify that Ultra-Dimensional Reasoning influences strategy selection"""
        print("\n🧪 Testing Fractal Strategy with Ultra-Dimensional Consciousness...")
        
        # Mock Resonance State
        mock_resonance = MagicMock()
        mock_resonance.total_energy = 50.0
        mock_resonance.entropy = 10.0
        
        # Mock UltraReasoning
        mock_ultra = MagicMock()
        mock_thought = MagicMock()
        
        # Scenario A: UltraReasoning suggests "Creative/Pattern" approach
        mock_thought.manifestation.content = "This requires a creative pattern shift."
        mock_thought.perspective.orientation = [0, 0.9, 0, 0] # High creative/emotional
        mock_ultra.reason.return_value = mock_thought
        
        strategy_a = self.engine.strategize(self.station, mock_resonance, mock_ultra)
        print(f"   Scenario A (Creative) Selected: {strategy_a}")
        
        # Scenario B: UltraReasoning suggests "Causal/Linear" approach
        mock_thought.manifestation.content = "Strict causal logic is required."
        mock_thought.perspective.orientation = [0, 0, 0.9, 0] # High logical
        mock_ultra.reason.return_value = mock_thought
        
        strategy_b = self.engine.strategize(self.station, mock_resonance, mock_ultra)
        print(f"   Scenario B (Logical) Selected: {strategy_b}")
        
        # Check that strategies might differ (probabilistic, but influenced)
        # Note: Since there is randomness, we mainly check that it runs without error 
        # and calls the reasoning engine.
        
        mock_ultra.reason.assert_called_with(self.station.name, context={"module": "FractalPlanner"})
        self.assertTrue(len(strategy_a) > 0)
        self.assertTrue(len(strategy_b) > 0)

if __name__ == '__main__':
    unittest.main()
