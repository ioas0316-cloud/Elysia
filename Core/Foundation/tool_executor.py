"""
Tool Executor for Elysia.

This module is responsible for taking a tool-use decision from the ActionCortex
and preparing it for execution by the environment. It also handles the result
of the execution.
"""
from typing import Dict, Any, Optional, Tuple
try:
    from infra.telemetry import Telemetry
except Exception:
    Telemetry = None
try:
    from Core.Foundation.safety_guardian import SafetyGuardian, ActionCategory
except Exception:
    SafetyGuardian = None
    ActionCategory = None

class ToolExecutor:
    """
    The ToolExecutor validates and prepares a tool call. In a real environment,
    it would interface with the system to actually execute the tool. Here, it
    acts as a marshaller for the tool call data.
    """
    def __init__(self):
        # In the future, this could load tool schemas for validation.
        self.telemetry = Telemetry() if Telemetry else None
        self.guardian = SafetyGuardian() if SafetyGuardian else None

    def prepare_tool_call(self, action_decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates the decision and prepares it for the execution environment.
        For now, it's a simple pass-through.

        Args:
            action_decision: The decision from ActionCortex.

        Returns:
            A dictionary representing the final tool call to be executed.
        """
        # (Logging placeholder)

        # Future validation logic would go here, e.g.:
        # - Check if tool_name is valid.
        # - Check if all required parameters are present.
        # - Check if parameter types are correct.

        return action_decision

    def execute_tool(self, prepared: Dict[str, Any]) -> Any:
        """
        Executes a small set of built-in tools safely.
        """
        try:
            if not prepared: return {}
            tool = prepared.get('tool_name')
            params = prepared.get('parameters', {})
            if not tool:
                return {'error': 'No tool specified'}

            # Minimal built-in tools stub
            if tool == 'read_file':
                return "File reading not implemented in this minimal executor."

            return {'error': f'Unknown tool: {tool}'}
        except Exception as e:
            return {'error': f'execute_tool failed: {e}'}

    # --- Helpers ---
    def _classify_tool(self, tool: Optional[str], params: Dict[str, Any]) -> Optional[Tuple[Any, str]]:
        """
        Maps a tool decision to (ActionCategory, action) for guardian checks.
        """
        if not tool or not ActionCategory:
            return None
        # Simplified for now
        return None

    def process_tool_result(self, tool_output: Any) -> str:
        """
        Processes the result from a tool execution and converts it into a
        natural language string to be fed back into the cognition pipeline.
        """
        if isinstance(tool_output, dict) and 'error' in tool_output:
            return f"An error occurred during tool execution: {tool_output['error']}"

        result_str = str(tool_output)
        if len(result_str) > 1000:
            result_str = result_str[:1000] + "... (result truncated)"

        return f"The result of the tool execution is: {result_str}"
