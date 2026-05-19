from typing import List, Dict, Any
from .essence_mapper import EssenceMapper
import logging

logger = logging.getLogger(__name__)

class AlchemyCortex:
    """
    The Synthesizer.
    Combines multiple concepts into a single Genesis Action definition.
    It uses the EssenceMapper to extract logic from concepts and merges them.
    """

    def __init__(self):
        self.mapper = EssenceMapper()

    def synthesize_action(self, concepts: List[str]) -> Dict[str, Any]:
        """
        Synthesizes a new action from a list of concept IDs.
        Example: ['wind', 'punch'] -> Wind Punch Action
        """
        combined_id = "_".join(concepts)

        # Default Template
        action_def = {
            "id": f"action:{combined_id}",
            "type": "action",
            "label": " ".join(concepts).title(),
            "logic": {
                "cost": {},
                "conditions": [],
                "effects": [],
                "selection_rule": {
                    "trigger": "general",
                    "priority": 1.0
                }
            }
        }

        logic = action_def["logic"]

        for concept in concepts:
            essence = self.mapper.get_essence(concept)
            if not essence:
                logger.warning(f"AlchemyCortex: No essence found for '{concept}'. Skipping.")
                continue

            # Merge Cost
            if "cost" in essence:
                for res, val in essence["cost"].items():
                    logic["cost"][res] = logic["cost"].get(res, 0) + val

            # Merge Conditions
            if "base_type" in essence and "logic_template" in essence:
                # It's a base verb (e.g., Attack), assume its template structure
                template = essence["logic_template"]
                if "target_type" in template:
                    logic["target_type"] = template["target_type"]
                if "conditions" in template:
                    logic["conditions"].extend(template["conditions"])
                if "effects" in template:
                    logic["effects"].extend(template["effects"])

            elif "effects" in essence:
                # It's an element/modifier (e.g., Fire)
                logic["effects"].extend(essence["effects"])

        # Clean up: Ensure we have at least one effect
        if not logic["effects"]:
            logic["effects"].append({"op": "log", "template": f"{combined_id} performed!"})

        logger.info(f"AlchemyCortex: Synthesized action '{action_def['id']}'")
        return action_def
