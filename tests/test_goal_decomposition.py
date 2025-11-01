import unittest
import os
from unittest.mock import MagicMock

# Add project root to path to allow direct imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from Project_Sophia.goal_decomposition_cortex import GoalDecompositionCortex
from tools.kg_manager import KGManager

class TestGoalDecomposition(unittest.TestCase):

    def setUp(self):
        """Set up a mock KGManager and the cortex for each test."""
        # Create a mock KGManager
        self.mock_kg_manager = MagicMock(spec=KGManager)

        # Instantiate the cortex with the mock manager
        self.gdc = GoalDecompositionCortex(self.mock_kg_manager)

    def test_decompose_rule_based_goal_website(self):
        """
        Tests if a predefined, rule-based goal (create website) is decomposed correctly.
        """
        goal = "Can you create a website for me?"
        plan = self.gdc.decompose_goal(goal)

        self.assertEqual(len(plan), 3)
        self.assertEqual(plan[0]['action'], 'create_file')
        self.assertEqual(plan[1]['action'], 'create_file')
        self.assertEqual(plan[2]['action'], 'run_server')
        self.assertEqual(plan[0]['step'], 1)

    def test_decompose_kg_based_goal_socrates(self):
        """
        Tests if a goal requiring KG lookup is decomposed correctly by querying the mock KG.
        """
        # Configure the mock KGManager to return specific data for this test
        socrates_node = {"id": "socrates", "position": {"x": 0, "y": 0, "z": 0}}
        socrates_edges = [
            {"source": "socrates", "target": "human", "relation": "is_a"},
            {"source": "plato", "target": "socrates", "relation": "was_student_of"}
        ]

        self.mock_kg_manager.get_node.return_value = socrates_node
        self.mock_kg_manager.kg = {'edges': socrates_edges} # Simulate the kg structure

        goal = "Tell me about socrates"
        plan = self.gdc.decompose_goal(goal)

        # Verify the mock was called correctly
        self.mock_kg_manager.get_node.assert_called_with("socrates")

        # Verify the generated plan
        self.assertEqual(len(plan), 3)
        self.assertEqual(plan[0]['action'], 'summarize')
        self.assertEqual(plan[1]['action'], 'state_fact')
        self.assertEqual(plan[2]['action'], 'state_fact')
        self.assertIn("'socrates' is_a 'human'", plan[1]['details'])
        self.assertIn("'plato' was_student_of 'socrates'", plan[2]['details'])

    def test_decompose_unknown_goal(self):
        """
        Tests if the cortex returns a clarification plan for an unknown goal.
        """
        # Ensure the mock returns None for any KG lookup
        self.mock_kg_manager.get_node.return_value = None

        goal = "Could you bake a pie?"
        plan = self.gdc.decompose_goal(goal)

        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]['action'], 'clarify_goal')
        self.assertIn("I am not yet sure how to achieve the goal", plan[0]['details'])


if __name__ == '__main__':
    unittest.main()
