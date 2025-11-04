# c:/Elysia/Project_Sophia/intent_analysis_cortex.py
from typing import Dict, Any
import json
import re
from Project_Sophia.gemini_api import generate_text, APIKeyError, APIRequestError

class IntentAnalysisCortex:
    """
    사용자의 자연어 입력을 분석하여 구조화된 '목표 객체(Goal Object)'로 변환합니다.
    """
    def __init__(self, llm_cortex=None):
        self.llm_cortex = llm_cortex
        self._goal_id_counter = 0

    def _generate_goal_id(self) -> str:
        self._goal_id_counter += 1
        return f"goal_{self._goal_id_counter:04d}"

    def _create_default_goal(self, user_input: str) -> Dict[str, Any]:
        """Creates a default goal when LLM analysis fails."""
        return {
            "goal_id": self._generate_goal_id(),
            "description": "General conversation or undefined task.",
            "type": "CONVERSATION",
            "parameters": {"topic": user_input},
            "source_input": user_input
        }

    def analyze(self, user_input: str) -> Dict[str, Any]:
        """
        LLM을 사용하여 사용자 입력을 분석하고 구조화된 목표 객체를 생성합니다.
        """
        prompt = f"""
        Analyze the user's high-level purpose and convert it into a structured JSON goal object.

        User's purpose: "{user_input}"

        The JSON object should have the following fields:
        - "description": A concise, one-sentence summary of the goal in English.
        - "type": The most appropriate goal type from the list: [ENHANCE_CAPABILITY, ACQUIRE_KNOWLEDGE, PERFORM_ACTION, USER_INTERACTION, SELF_REFLECTION].
        - "parameters": A dictionary of key parameters extracted from the purpose. Examples:
            - For ENHANCE_CAPABILITY: {{"target_area": "reasoning", "metric": "depth"}}
            - For ACQUIRE_KNOWLEDGE: {{"topic": "causal reasoning", "scope": "latest research"}}
            - For PERFORM_ACTION: {{"action": "write_code", "file": "main.py"}}

        Respond ONLY with the JSON object.

        JSON Response:
        """
        try:
            response_text = generate_text(prompt)
            # Extract JSON from the response, handling potential markdown code blocks
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)

            goal_data = json.loads(response_text)

            # Validate the structure of the parsed JSON
            if not all(k in goal_data for k in ["description", "type", "parameters"]):
                raise ValueError("Missing required keys in LLM response.")

            goal_data["goal_id"] = self._generate_goal_id()
            goal_data["source_input"] = user_input
            return goal_data

        except (APIKeyError, APIRequestError) as e:
            print(f"[IntentAnalysisCortex] API Error: {e}. Falling back to default goal.")
            return self._create_default_goal(user_input)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[IntentAnalysisCortex] Failed to parse or validate LLM response: {e}. Falling back to default goal.")
            return self._create_default_goal(user_input)
