import json
import logging
from typing import List, Dict, Any
from Core.Foundation.gemini_api import generate_text
from tools.time_tool import get_current_time

class PlanningCortex:
    """
    Breaks down complex, high-level goals into a sequence of executable tool calls.
    This cortex is the heart of Elysia's ability to form and execute multi-step plans.
    It uses Gemini to reason about the goal and available tools, with awareness of the current time.
    """

    def __init__(self, core_memory, action_cortex):
        """
        Initializes the Planning Cortex.

        Args:
            core_memory: An interface to Elysia's core memory system.
            action_cortex: The action cortex to decide on individual tool calls.
        """
        self.core_memory = core_memory
        self.action_cortex = action_cortex
        self.logger = logging.getLogger("PlanningCortex")

    def develop_plan(self, goal: str) -> List[Dict[str, Any]]:
        """
        Develops a step-by-step plan to achieve a given goal.

        Args:
            goal (str): The high-level goal to achieve.

        Returns:
            list: A list of tool calls (dictionaries) representing the plan.
        """
        self.logger.info(f"Developing plan for goal: {goal}")
        
        current_time = get_current_time()
        
        # Construct the prompt for Gemini
        prompt = f"""
        You are the Planning Cortex of an advanced AI named Elysia.
        Your task is to break down a high-level goal into a sequence of specific tool calls.
        
        Current Time: {current_time}
        Goal: {goal}
        
        Available Tools (Abstract):
        - read_file(filepath)
        - write_to_file(filepath, content)
        - list_dir(directory_path)
        - google_search(query)
        - view_text_website(url)
        - get_current_time()
        
        Instructions:
        1. Analyze the goal and determine the necessary steps.
        2. If the goal involves time, use the Current Time to make decisions.
        3. Output the plan as a JSON list of objects, where each object represents a step.
        4. Each step should have a 'tool_name' and 'parameters'.
        5. If a step requires thinking or internal processing without a tool, use 'tool_name': 'thought' and 'parameters': {{'content': '...'}}.
        
        Example Output Format:
        [
            {{
                "tool_name": "get_current_time",
                "parameters": {{}}
            }},
            {{
                "tool_name": "write_to_file",
                "parameters": {{
                    "filepath": "c:/Elysia/log.txt",
                    "content": "Log entry..."
                }}
            }}
        ]
        
        Generate the JSON plan now. Do not include markdown formatting like ```json ... ```. Just the raw JSON string.
        """
        
        try:
            response_text = generate_text(prompt)
            # Clean up potential markdown formatting if Gemini adds it despite instructions
            cleaned_text = response_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
                
            plan = json.loads(cleaned_text)
            
            self.logger.info(f"Plan generated with {len(plan)} steps.")
            return plan
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse plan JSON: {e}. Response was: {response_text}")
            return []
        except Exception as e:
            self.logger.error(f"Error developing plan: {e}")
            return []

    def _decompose_goal(self, goal):
        """
        Legacy method, kept for compatibility if needed, but develop_plan is preferred.
        """
        return self.develop_plan(goal)