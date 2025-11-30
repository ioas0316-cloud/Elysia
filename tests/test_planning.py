import sys
import os
import unittest
from unittest.mock import MagicMock

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Intelligence.Intelligence.Planning.planning_cortex import PlanningCortex
from Tools.time_tools import get_current_time

class TestPlanningPhase2(unittest.TestCase):
    
    def setUp(self):
        # Mock dependencies for PlanningCortex
        self.mock_hippocampus = MagicMock()
        self.mock_conscience = MagicMock()
        # Allow all actions by default
        self.mock_conscience.evaluate_action.return_value = True
        
        self.planner = PlanningCortex(self.mock_hippocampus, self.mock_conscience)

    def test_time_tools(self):
        """Test if get_current_time returns a valid ISO string."""
        time_str = get_current_time()
        print(f"Current time: {time_str}")
        self.assertIsInstance(time_str, str)
        self.assertIn("T", time_str) # ISO format check

    def test_planning_cortex_write_file(self):
        """Test heuristic planner for 'write file' goal."""
        goal = "Write a poem to poem.txt"
        plan = self.planner.develop_plan(goal)
        
        self.assertTrue(len(plan) > 0)
        self.assertEqual(plan[0]['tool'], 'write_to_file')
        self.assertEqual(plan[0]['parameters']['filename'], 'poem.txt')
        print(f"Plan for '{goal}': {plan}")

    def test_planning_cortex_search(self):
        """Test heuristic planner for 'search' goal."""
        goal = "Research quantum physics"
        plan = self.planner.develop_plan(goal)
        
        self.assertTrue(len(plan) > 0)
        self.assertEqual(plan[0]['tool'], 'web_search')
        print(f"Plan for '{goal}': {plan}")

if __name__ == '__main__':
    unittest.main()
