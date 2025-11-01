import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Sophia.execution_cortex import ExecutionCortex

class TestExecutionCortex(unittest.TestCase):

    def setUp(self):
        """Set up a fresh cortex for each test."""
        # We patch the ToolExecutor within each test to ensure isolation
        pass

    @patch('Project_Sophia.execution_cortex.ToolExecutor')
    def test_successful_multi_step_execution(self, MockToolExecutor):
        """
        Tests if the cortex can execute a multi-step plan, passing context between steps.
        """
        mock_executor_instance = MockToolExecutor.return_value
        # Configure side_effect specifically for this test
        mock_executor_instance.execute.side_effect = [
            "Elysia is a project about AI.",
            "AI project."
        ]
        cortex = ExecutionCortex()

        plan = [
            {"tool_name": "read_file", "parameters": {"filepath": "project_intro.txt"}},
            {"tool_name": "summarize_text", "parameters": {"text": "<step_1_output>"}}
        ]

        result = cortex.execute_plan(plan)

        self.assertEqual(mock_executor_instance.execute.call_count, 2)
        mock_executor_instance.execute.assert_called_with("summarize_text", {"text": "Elysia is a project about AI."})
        self.assertIn("목표를 성공적으로 달성했습니다.", result)
        self.assertIn("AI project.", result)

    @patch('Project_Sophia.execution_cortex.ToolExecutor')
    def test_file_not_found_help_request(self, MockToolExecutor):
        """
        Tests if the cortex asks for help correctly when a file is not found.
        """
        mock_executor_instance = MockToolExecutor.return_value
        # Configure side_effect specifically for this test
        mock_executor_instance.execute.side_effect = FileNotFoundError("No such file or directory: 'non_existent_file.txt'")
        cortex = ExecutionCortex()

        plan = [{"tool_name": "read_file", "parameters": {"filepath": "non_existent_file.txt"}}]

        result = cortex.execute_plan(plan)

        self.assertIn("경로 'non_existent_file.txt'에서 파일을 찾을 수 없습니다.", result)
        self.assertIn("다른 경로를 시도해볼까요?", result)

    @patch('Project_Sophia.execution_cortex.ToolExecutor')
    def test_missing_parameter_help_request(self, MockToolExecutor):
        """
        Tests if the cortex asks for help when a required parameter is missing from the context.
        """
        # No mock needed for execute, as the error happens before the call
        cortex = ExecutionCortex()

        plan = [
            {"tool_name": "read_file", "parameters": {"filepath": "some_file.txt"}},
            # This step depends on a non-existent output from a previous step
            {"tool_name": "summarize_text", "parameters": {"text": "<step_99_output>"}}
        ]

        result = cortex.execute_plan(plan)

        # More flexible assertion: check for the key phrases
        self.assertIn("정보가 부족합니다", result)
        self.assertIn("<step_99_output>", result)
        self.assertIn("알려주시겠어요?", result)


if __name__ == '__main__':
    unittest.main()
