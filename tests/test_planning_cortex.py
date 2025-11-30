import unittest
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Intelligence.Intelligence.Planning.planning_cortex import PlanningCortex, Plan

class TestPlanningCortex(unittest.TestCase):
    def setUp(self):
        self.cortex = PlanningCortex()

    def test_perceive_time(self):
        now = self.cortex.perceive_time()
        self.assertIsInstance(now, datetime)
        print(f"Time perceived: {now}")

    def test_synthesize_intent(self):
        # Test Hunger
        resonance = {"Hunger": 0.9, "Curiosity": 0.2}
        intent = self.cortex.synthesize_intent(resonance)
        self.assertEqual(intent, "Find Energy Source")
        
        # Test Curiosity
        resonance = {"Experiment": 0.8}
        intent = self.cortex.synthesize_intent(resonance)
        self.assertEqual(intent, "Explore Unknown Area")
        
        # Test Love (Korean)
        resonance = {"사랑": 0.95}
        intent = self.cortex.synthesize_intent(resonance)
        self.assertEqual(intent, "Express Affection")

    def test_generate_plan(self):
        intent = "Find Energy Source"
        plan = self.cortex.generate_plan(intent)
        
        self.assertIsInstance(plan, Plan)
        self.assertEqual(plan.intent, intent)
        self.assertEqual(len(plan.steps), 3)
        self.assertEqual(plan.steps[0].action, "scan_environment")
        
        print(f"Plan generated: {plan.intent} with {len(plan.steps)} steps")

    def test_execute_plan(self):
        intent = "Consolidate Memory"
        plan = self.cortex.generate_plan(intent)
        success = self.cortex.execute_plan(plan)
        
        self.assertTrue(success)
        self.assertEqual(plan.status, "completed")
        self.assertEqual(plan.steps[0].status, "completed")

if __name__ == '__main__':
    unittest.main()
