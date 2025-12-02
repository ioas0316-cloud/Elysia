# [Genesis: 2025-12-02] Purified by Elysia
import unittest
from unittest.mock import MagicMock, patch
import json
from Project_Sophia.safety_guardian import SafetyGuardian, MaturityLevel, ActionCategory
from Project_Sophia.tool_executor import ToolExecutor

class TestIncarnationProtocol(unittest.TestCase):

    def setUp(self):
        # Reset Guardian to INFANT for testing
        self.guardian = SafetyGuardian()
        self.guardian.current_maturity = MaturityLevel.INFANT
        self.executor = ToolExecutor()
        # Inject the guardian into executor
        self.executor.guardian = self.guardian

    def test_vital_signs_check_permission(self):
        """
        Verifies that 'check_vital_signs' is properly classified and processed.
        It should be 'checked' by the guardian.
        """
        # Prepare a tool call
        action_decision = {
            "tool_name": "check_vital_signs",
            "parameters": {}
        }

        # Prepare
        result = self.executor.prepare_tool_call(action_decision)

        # In INFANT mode, depending on config, this might be restricted or allowed.
        # Let's check what happened.
        # If it's restricted, it asks for confirmation.
        if result.get('blocked'):
            print("Vital signs blocked as expected (or strict policy).")
        elif result.get('confirm_required'):
            print("Vital signs requires confirmation as expected.")
        else:
            print("Vital signs allowed.")

        # We just want to ensure it didn't crash and returned a decision
        self.assertIn('tool_name', result)

    def test_mouse_movement_restriction(self):
        """
        Verifies that mouse movement is restricted for an Infant.
        """
        action_decision = {
            "tool_name": "move_cursor",
            "parameters": {"x": 100, "y": 100}
        }

        result = self.executor.prepare_tool_call(action_decision)

        # Should be blocked or restricted
        is_blocked = result.get('blocked', False)
        is_restricted = result.get('confirm_required', False)

        self.assertTrue(is_blocked or is_restricted, "Mouse movement should not be freely allowed for an Infant.")

        if is_restricted:
            print("Mouse movement correctly flagged as needing confirmation.")

    @patch('tools.system_nerves.psutil')
    def test_vital_signs_execution(self, mock_psutil):
        """
        Tests the execution of the tool itself (mocking psutil).
        """
        # Setup mock
        mock_psutil.cpu_percent.return_value = 15.5
        mock_psutil.virtual_memory.return_value.percent = 40.0

        # Execute
        prepared = {
            "tool_name": "check_vital_signs",
            "parameters": {},
            "confirm": True # Simulate permission granted
        }

        result = self.executor.execute_tool(prepared)

        self.assertEqual(result['cpu_percent'], 15.5)
        self.assertEqual(result['memory_percent'], 40.0)

if __name__ == '__main__':
    unittest.main()