"""
Tool Executor for Elysia.

This module is responsible for taking a tool-use decision from the ActionCortex
and preparing it for execution by the environment. It also handles the result
of the execution.
"""
from typing import Dict, Any, Optional

class ToolExecutor:
    """
    The ToolExecutor validates and prepares a tool call. In a real environment,
    it would interface with the system to actually execute the tool. Here, it
    acts as a marshaller for the tool call data.
    """
    def __init__(self):
        # In the future, this could load tool schemas for validation.
        pass

    def prepare_tool_call(self, action_decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates the decision and prepares it for the execution environment.
        For now, it's a simple pass-through.

        Args:
            action_decision: The decision from ActionCortex.

        Returns:
            A dictionary representing the final tool call to be executed.
        """
        print(f"[ToolExecutor] Preparing tool call for: {action_decision}")
        
        # Future validation logic would go here, e.g.:
        # - Check if tool_name is valid.
        # - Check if all required parameters are present.
        # - Check if parameter types are correct.

        return action_decision

    def process_tool_result(self, tool_output: Any) -> str:
        """
        Processes the result from a tool execution and converts it into a
        natural language string to be fed back into the cognition pipeline.

        Args:
            tool_output: The raw output from the tool execution.

        Returns:
            A natural language summary of the tool's result.
        """
        # This is a simple formatter. More complex tools might need more
        # sophisticated result processing.
        if isinstance(tool_output, dict) and 'error' in tool_output:
            return f"An error occurred during tool execution: {tool_output['error']}"
        
        # Truncate long results to keep the context manageable.
        result_str = str(tool_output)
        if len(result_str) > 1000:
            result_str = result_str[:1000] + "... (result truncated)"

        return f"The result of the tool execution is: {result_str}"
