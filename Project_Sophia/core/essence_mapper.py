from typing import Dict, Any, Optional

# Type alias for a Logic Fragment
Essence = Dict[str, Any]

class EssenceMapper:
    """
    The Periodic Table of Logic.
    Maps semantic concept IDs to GenesisEngine primitives.
    This acts as the bridge between abstract meaning and concrete mechanics.
    """

    def __init__(self):
        self._registry: Dict[str, Essence] = {
            # --- Elements ---
            "불": {
                "type_modifier": "fire",
                "effects": [{"op": "damage", "multiplier": 0.5, "type": "fire"}],
                "cost": {"ki": 5}
            },
            "fire": { # English alias
                "type_modifier": "fire",
                "effects": [{"op": "damage", "multiplier": 0.5, "type": "fire"}],
                "cost": {"ki": 5}
            },
            "물": {
                "type_modifier": "water",
                "effects": [{"op": "heal", "amount": 5}],
                "cost": {"mana": 5}
            },
            "water": {
                "type_modifier": "water",
                "effects": [{"op": "heal", "amount": 5}],
                "cost": {"mana": 5}
            },
            "바람": {
                "type_modifier": "wind",
                "effects": [{"op": "modify_stat", "stat": "agility", "value": 5}],
                "cost": {"ki": 3}
            },
            "wind": {
                "type_modifier": "wind",
                "effects": [{"op": "modify_stat", "stat": "agility", "value": 5}],
                "cost": {"ki": 3}
            },
            "빛": {
                "type_modifier": "light",
                "effects": [{"op": "log", "template": "A blinding light flashes!"}],
                "cost": {"mana": 10}
            },

            # --- Actions/Verbs ---
            "공격": {
                "base_type": "action",
                "logic_template": {
                    "target_type": "entity",
                    "conditions": [{"check": "stat_ge", "stat": "strength", "value": 1}],
                    "effects": [{"op": "damage", "multiplier": 1.0}]
                }
            },
            "punch": {
                "base_type": "action",
                "logic_template": {
                    "target_type": "entity",
                    "conditions": [{"check": "stat_ge", "stat": "strength", "value": 1}],
                    "effects": [{"op": "damage", "multiplier": 1.0}, {"op": "log", "template": "{actor} punches {target}!"}]
                }
            },
            "치유": {
                "base_type": "action",
                "logic_template": {
                    "target_type": "entity", # Can target self or others
                    "conditions": [{"check": "stat_ge", "stat": "wisdom", "value": 1}],
                    "effects": [{"op": "heal", "amount": 10}]
                }
            }
        }

    def get_essence(self, concept_id: str) -> Optional[Essence]:
        """
        Retrieves the logic essence for a given concept.
        Supports substring matching if exact match fails (e.g., 'big fire' -> 'fire').
        """
        # 1. Exact match
        if concept_id in self._registry:
            return self._registry[concept_id]

        # 2. Substring match (primitive fuzzy logic)
        for key, essence in self._registry.items():
            if key in concept_id:
                return essence

        return None
