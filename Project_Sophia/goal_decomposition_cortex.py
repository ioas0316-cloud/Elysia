"""
Goal Decomposition Cortex for Elysia.

This module is responsible for breaking down high-level, complex goals into a
sequence of concrete, executable steps (tool calls). It leverages an external
LLM to understand the user's intent and formulate a logical plan.
"""
import json
import logging
import os
from .gemini_api import generate_text
from tools.tool_manager import ToolManager

# --- Logging Configuration ---
log_file_path = os.path.join(os.path.dirname(__file__), 'goal_decomposition.log') # Changed to a more general name
logging.basicConfig(
    level=logging.DEBUG, # Changed to DEBUG to capture all info
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# --- End Logging Configuration ---

class GoalDecompositionCortex:
    """
    Decomposes a high-level goal into a structured plan of tool calls.
    """
    def __init__(self):
        self.tool_manager = ToolManager()

    def decompose_goal(self, goal: str) -> list[dict]:
        """
        Takes a high-level goal and breaks it down into a sequence of tool calls
        using an external LLM.

        Args:
            goal: The user's high-level goal.

        Returns:
            A list of dictionaries, where each dictionary represents a tool call.
            Returns an empty list if a plan cannot be formed.
        """
        logger.info(f"Decomposing goal: {goal}")

        available_tools = self.tool_manager.get_tool_signatures()
        tools_description = "\n".join([f"- {sig}" for sig in available_tools])

        prompt = f"""
        You are an expert planning module for an AI named Elysia.
        Your task is to decompose a high-level user goal into a precise, step-by-step plan.
        The plan must consist of a sequence of calls to the available tools.

        **Available Tools:**
        {tools_description}

        **User Goal:**
        "{goal}"

        **Instructions:**
        1. Think step-by-step to understand the user's intent.
        2. Create a logical sequence of tool calls to achieve the goal.
        3. The output MUST be a JSON array of objects, where each object has "tool_name" and "parameters" keys.
        4. The "parameters" value must be an object of key-value pairs.
        5. If a tool is not available to perform a step, you cannot include it.
        6. If the goal is simple and can be achieved with a single tool call, the plan should contain just that one call.
        7. If the goal cannot be achieved with the available tools, return an empty JSON array.

        **Example Output:**
        [
            {{
                "tool_name": "read_file",
                "parameters": {{
                    "filepath": "research_paper.txt"
                }}
            }},
            {{
                "tool_name": "summarize_text",
                "parameters": {{
                    "text": "<content of research_paper.txt>"
                }}
            }}
        ]

        **Plan:**
        """
        logger.debug(f"Generated LLM Prompt:\n{prompt}")

        try:
            response_text = generate_text(prompt)
            logger.debug(f"Received LLM Response:\n{response_text}")

            # Clean the response to extract only the JSON part
            json_match = response_text[response_text.find('['):response_text.rfind(']')+1]
            if not json_match:
                logger.warning(f"Could not find a valid JSON array in the LLM response for goal: {goal}")
                return []

            plan = json.loads(json_match)
            logger.info(f"Successfully decomposed goal into a plan with {len(plan)} steps.")
            return plan

        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from LLM response: {response_text}")
            return []
        except Exception as e:
            logger.error(f"An unexpected error occurred during goal decomposition: {e}")
            return []
