
import unittest
import sys
import os
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Planning.planning_cortex import PlanningCortex, Plan

class TestPlanningIntegration(unittest.TestCase):
    def setUp(self):
        self.mock_hippocampus = MagicMock()
        self.cortex = PlanningCortex(hippocampus=self.mock_hippocampus)

    def test_synthesize_intent_known_drive(self):
        # Mock resonance: "Hunger" is dominant
        resonance = {"Hunger": 0.9, "Sleep": 0.1}
        intent = self.cortex.synthesize_intent(resonance)
        self.assertEqual(intent, "Find Energy Source")

    def test_synthesize_intent_unknown_concept(self):
        # Mock resonance: "QuantumPhysics" is dominant
        resonance = {"QuantumPhysics": 0.8, "Math": 0.2}
        intent = self.cortex.synthesize_intent(resonance)
        self.assertEqual(intent, "Focus on QuantumPhysics")

    def test_time_awareness(self):
        current_time = self.cortex.get_current_time()
        self.assertIsNotNone(current_time)
        print(f"Time Perception Verified: {current_time}")

    def test_plan_generation_from_resonance(self):
        # 1. Simulate Resonance Input
        resonance = {"Curiosity": 0.95}
        
        # 2. Synthesize Intent
        intent = self.cortex.synthesize_intent(resonance)
        self.assertEqual(intent, "Explore Unknown Area")
        
        # 3. Generate Plan
        plan = self.cortex.generate_plan(intent)
        
        # 4. Verify Plan Structure
        self.assertIsInstance(plan, Plan)
        self.assertEqual(plan.intent, "Explore Unknown Area")
        self.assertTrue(len(plan.steps) > 0)
        print(f"Generated Plan: {plan.intent} with {len(plan.steps)} steps.")

if __name__ == '__main__':
    unittest.main()
