import unittest
from unittest.mock import patch
import json
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Sophia.meta_cognition_cortex import MetaCognitionCortex

class TestMetaCognitionCortex(unittest.TestCase):

    def setUp(self):
        """Set up a fresh cortex for each test."""
        self.cortex = MetaCognitionCortex()

    @patch('Project_Sophia.meta_cognition_cortex.generate_text')
    def test_reflection_on_success(self, mock_generate_text):
        """
        Tests if the cortex correctly summarizes a successful execution
        and generates no new learning goal.
        """
        # 1. Mock the LLM's response for a successful reflection
        mock_response = json.dumps({
            "summary": "To summarize a file, I must first read its content.",
            "new_learning_goal": None
        })
        mock_generate_text.return_value = mock_response

        # 2. Define the successful execution data
        goal = "Summarize the project plan."
        plan = [
            {"tool_name": "read_file", "parameters": {"filepath": "plan.txt"}},
            {"tool_name": "summarize_text", "parameters": {"text": "<step_1_output>"}}
        ]
        result = "목표를 성공적으로 달성했습니다. 최종 결과: Summary: The project plan is..."

        # 3. Perform reflection
        reflection = self.cortex.reflect(goal, plan, result)

        # 4. Assertions
        mock_generate_text.assert_called_once()
        self.assertIn("Briefly summarize what was learned", mock_generate_text.call_args[0][0])
        self.assertEqual(reflection["summary"], "To summarize a file, I must first read its content.")
        self.assertIsNone(reflection["new_learning_goal"])

    @patch('Project_Sophia.meta_cognition_cortex.generate_text')
    def test_reflection_on_failure_generates_goal(self, mock_generate_text):
        """
        Tests if the cortex analyzes a failure and generates a new,
        relevant learning goal.
        """
        # 1. Mock the LLM's response for a failed reflection
        mock_response = json.dumps({
            "summary": "The plan failed because I lack the ability to read websites. Therefore, I need to learn this skill.",
            "new_learning_goal": "Learn how to read the content of a website using Python."
        })
        mock_generate_text.return_value = mock_response

        # 2. Define the failed execution data
        goal = "Summarize the content of the website 'example.com'."
        plan = [{"tool_name": "read_website", "parameters": {"url": "http://example.com"}}]
        result = "Error: Execution failed at step 1 (read_website). Tool 'read_website' not found."

        # 3. Perform reflection
        reflection = self.cortex.reflect(goal, plan, result)

        # 4. Assertions
        mock_generate_text.assert_called_once()
        self.assertIn("You are reflecting on a FAILED goal.", mock_generate_text.call_args[0][0])
        self.assertIn("I lack the ability to read websites", reflection["summary"])
        self.assertEqual(reflection["new_learning_goal"], "Learn how to read the content of a website using Python.")

if __name__ == '__main__':
    unittest.main()
