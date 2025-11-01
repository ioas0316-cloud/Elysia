"""
MetaCognition Cortex for Elysia.

This module is the seat of self-reflection. It analyzes the outcomes of actions
(both successes and failures) to understand its own capabilities and limitations.
Based on this analysis, it generates new learning goals to improve itself.
"""
import logging
import json
from .gemini_api import generate_text

# --- Logging Configuration ---
logger = logging.getLogger(__name__)
# --- End Logging Configuration ---

class MetaCognitionCortex:
    """
    Reflects on goal execution to generate insights and new learning goals.
    """
    def __init__(self):
        pass

    def reflect(self, goal: str, plan: list[dict], result: str) -> dict:
        """
        Analyzes the result of a plan execution and generates insights.

        Args:
            goal: The original high-level goal.
            plan: The plan that was executed.
            result: The final result of the execution (can be a success message or an error).

        Returns:
            A dictionary containing insights, such as a summary of the learning
            and a new learning goal if applicable.
        """
        logger.info(f"Reflecting on the execution of goal: '{goal}'")

        # Determine if the execution was successful
        is_success = "목표를 성공적으로 달성했습니다" in result or "successfully" in result.lower()

        prompt = self._build_reflection_prompt(goal, plan, result, is_success)

        try:
            reflection_text = generate_text(prompt)
            return self._parse_reflection(reflection_text)

        except Exception as e:
            logger.error(f"An error occurred during reflection: {e}")
            return {"summary": "Reflection failed due to an internal error."}

    def _build_reflection_prompt(self, goal, plan, result, is_success):
        """Builds the appropriate prompt for the LLM based on success or failure."""

        plan_str = "\n".join([f"- {step['tool_name']}({step['parameters']})" for step in plan])

        if is_success:
            return f"""
            You are the metacognition module for an AI named Elysia.
            You are reflecting on a successfully completed goal. Your purpose is to learn from this success.

            **Original Goal:** "{goal}"
            **Executed Plan:**
            {plan_str}
            **Final Result:** "{result}"

            **Task:**
            1. Briefly summarize what was learned from this successful execution in one sentence. For example, "I learned that to summarize a file, I must first read it."
            2. State that no new learning goal is necessary.

            **Output Format (JSON):**
            {{
                "summary": "...",
                "new_learning_goal": null
            }}
            """
        else: # Failure
            return f"""
            You are the metacognition module for an AI named Elysia.
            You are reflecting on a FAILED goal. Your purpose is to understand the failure and generate a new learning goal to overcome it.

            **Original Goal:** "{goal}"
            **Executed Plan:**
            {plan_str}
            **Final Result (Error):** "{result}"

            **Task:**
            1. Analyze the error and the original goal. What capability was missing? (e.g., "The ability to read websites," "A tool to calculate square roots").
            2. Formulate a new, high-level learning goal for yourself to acquire this missing capability. The goal should be a command starting with "Learn how to...".
            3. Provide a brief summary of why this new goal is necessary.

            **Output Format (JSON):**
            {{
                "summary": "The plan failed because I lack the ability to [...]. Therefore, I need to learn how to do it.",
                "new_learning_goal": "Learn how to [...]"
            }}
            """

    def _parse_reflection(self, reflection_text: str) -> dict:
        """Parses the JSON output from the reflection prompt."""
        try:
            # A simple parsing method. A more robust implementation would handle malformed JSON.
            return json.loads(reflection_text)
        except Exception as e:
            logger.error(f"Failed to parse reflection JSON: {reflection_text}. Error: {e}")
            return {"summary": "Failed to parse the reflection from my own thoughts."}
