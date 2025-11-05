"""
Action Cortex for Elysia.

This module is responsible for deciding which tool to use to respond to a user's
request or to achieve a specific goal. It bridges the gap between 'thinking'
and 'acting'.
"""
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional

from .wave_mechanics import WaveMechanics
from tools.kg_manager import KGManager
from .gemini_api import generate_text  # Import the LLM function


class ActionCortex:
    """
    The ActionCortex decides which action (tool) to take based on a given prompt
    or internal goal. It uses the Wave Principle on a dedicated tool KG and then
    uses an LLM to extract parameters.
    """

    def __init__(self):
        tools_kg_path = Path("data/tools_kg.json")
        self.tools_kg_manager = KGManager(filepath=tools_kg_path)
        self.wave_mechanics = WaveMechanics(self.tools_kg_manager)
        self.tool_schemas = self._load_tool_schemas()

    def _load_tool_schemas(self) -> Dict:
        """Loads the schemas for tools that define their parameters."""
        # In a real system, this would be more robust.
        # For now, we hardcode the schema for the 'read_file' tool.
        return {
            "read_file": {
                "description": "Reads the content of a specified file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "The path to the file that needs to be read.",
                        }
                    },
                    "required": ["filepath"],
                },
            },
            "http_request": {
                "description": "Fetches a URL and returns a simple summary (status, title/text).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to fetch"}
                    },
                    "required": ["url"]
                }
            },
            "fetch_url": {
                "description": "Alias of http_request",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"}
                    },
                    "required": ["url"]
                }
            }
        }

    def _find_best_tool(self, prompt: str) -> Optional[str]:
        """
        Uses the Wave Principle to find the most relevant tool by aggregating
        activation from all relevant prompt tokens.
        """
        prompt_tokens = set(re.findall(r'\w+', prompt.lower()))
        all_node_ids = {node['id'] for node in self.tools_kg_manager.kg['nodes']}

        stimulus_nodes = prompt_tokens.intersection(all_node_ids)
        if not stimulus_nodes:
            return None

        tool_ids = {edge['target'] for edge in self.tools_kg_manager.kg['edges'] if edge.get('relation') == 'activates'}
        source_ids = {edge['source'] for edge in self.tools_kg_manager.kg['edges'] if edge.get('relation') == 'activates'}
        tool_ids.update(all_node_ids - source_ids)

        final_echo = {}
        for start_node in stimulus_nodes:
            echo = self.wave_mechanics.spread_activation(start_node)
            for node, energy in echo.items():
                final_echo[node] = final_echo.get(node, 0) + energy

        if not final_echo:
            return None

        tool_echo = {node: energy for node, energy in final_echo.items() if node in tool_ids}
        if not tool_echo:
            return None

        best_tool = max(tool_echo, key=tool_echo.get)
        return best_tool

    def _extract_parameters(self, prompt: str, tool_name: str) -> Dict[str, Any]:
        """
        Uses an LLM to extract parameters for a given tool from the prompt.
        """
        schema = self.tool_schemas.get(tool_name)
        if not schema:
            # If the tool has no parameters, return an empty dict.
            return {}

        extraction_prompt = f"""
        Given the user's prompt and the tool schema, extract the required parameters.
        Respond with a JSON object containing the parameters.

        User Prompt: "{prompt}"

        Tool: "{tool_name}"
        Schema: {json.dumps(schema, indent=2)}

        Extracted parameters (JSON):
        """

        try:
            response_text = generate_text(extraction_prompt)
            # Clean the response to get only the JSON part
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                extracted_json = json_match.group(0)
                return json.loads(extracted_json)
            else:
                print(f"[ActionCortex] Could not parse JSON from LLM response: {response_text}")
                return {}
        except Exception as e:
            print(f"[ActionCortex] Error during parameter extraction: {e}")
            return {}

    def decide_action(self, prompt: str, app=None) -> Optional[Dict[str, Any]]:
        """
        Based on the user's prompt, decide which tool to use and what parameters to use.
        """
        print(f"[ActionCortex] Deciding action for prompt: {prompt}")

        best_tool = self._find_best_tool(prompt)

        if not best_tool:
            print("[ActionCortex] No relevant tool found using Wave Mechanics.")
            return None

        print(f"[ActionCortex] Best tool found via Wave Mechanics: {best_tool}")

        # Step 2: Extract parameters for the chosen tool
        parameters = self._extract_parameters(prompt, best_tool)

        return {
            "tool_name": best_tool,
            "parameters": parameters
        }
