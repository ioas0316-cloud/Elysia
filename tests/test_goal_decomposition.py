import unittest
from unittest.mock import patch
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Sophia.goal_decomposition_cortex import GoalDecompositionCortex

class TestGoalDecompositionCortex(unittest.TestCase):

    def setUp(self):
        """Set up a fresh cortex for each test."""
        self.cortex = GoalDecompositionCortex()

    @patch('Project_Sophia.goal_decomposition_cortex.generate_text')
    def test_decompose_simple_goal(self, mock_generate_text):
        """
        Tests if the cortex can correctly decompose a simple, single-step goal.
        """
        # 1. Mock the LLM's response to simulate a successful plan generation
        mock_response = """
        [
            {
                "tool_name": "search_web",
                "parameters": {
                    "query": "Elysia Project"
                }
            }
        ]
        """
        mock_generate_text.return_value = mock_response

        # 2. Define a simple goal
        goal = "Find information about the Elysia Project."

        # 3. Decompose the goal
        plan = self.cortex.decompose_goal(goal)

        # 4. Assertions
        # It should call the LLM once
        mock_generate_text.assert_called_once()

        # The plan should be a list with one step
        self.assertIsInstance(plan, list)
        self.assertEqual(len(plan), 1)

        # Check the content of the step
        self.assertEqual(plan[0]['tool_name'], 'search_web')
        self.assertEqual(plan[0]['parameters']['query'], 'Elysia Project')

    @patch('Project_Sophia.goal_decomposition_cortex.generate_text')
    def test_decompose_multi_step_goal(self, mock_generate_text):
        """
        Tests if the cortex can correctly decompose a more complex, multi-step goal.
        """
        # 1. Mock the LLM's response for a two-step plan
        mock_response = """
        [
            {
                "tool_name": "read_file",
                "parameters": { "filepath": "meeting_notes.txt" }
            },
            {
                "tool_name": "summarize_text",
                "parameters": { "text": "<content of meeting_notes.txt>" }
            }
        ]
        """
        mock_generate_text.return_value = mock_response

        # 2. Define a multi-step goal
        goal = "Read the meeting notes file and then summarize it."

        # 3. Decompose the goal
        plan = self.cortex.decompose_goal(goal)

        # 4. Assertions
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan[0]['tool_name'], 'read_file')
        self.assertEqual(plan[1]['tool_name'], 'summarize_text')
        self.assertEqual(plan[1]['parameters']['text'], '<content of meeting_notes.txt>')

    @patch('Project_Sophia.goal_decomposition_cortex.generate_text')
    def test_handle_unachievable_goal(self, mock_generate_text):
        """
        Tests if the cortex returns an empty list for a goal that cannot be
        achieved with the available tools.
        """
        # 1. Mock the LLM's response to indicate failure
        mock_response = "[]"
        mock_generate_text.return_value = mock_response

        # 2. Define an unachievable goal
        goal = "Make a cup of coffee."

        # 3. Decompose the goal
        plan = self.cortex.decompose_goal(goal)

        # 4. Assert that the plan is empty
        self.assertEqual(plan, [])

    @patch('Project_Sophia.goal_decomposition_cortex.generate_text')
    def test_handle_invalid_json_response(self, mock_generate_text):
        """
        Tests if the cortex gracefully handles a malformed JSON response from the LLM.
        """
        # 1. Mock a broken JSON response
        mock_response = "[{'tool_name': 'search', 'parameters': {'query': 'test'}" # Missing closing brace and quotes
        mock_generate_text.return_value = mock_response

        # 2. Decompose a goal
        goal = "Search for something."

        # 3. Decompose the goal
        plan = self.cortex.decompose_goal(goal)

        # 4. Assert that the plan is empty
        self.assertEqual(plan, [])

if __name__ == '__main__':
    unittest.main()
