from typing import Dict, Any, Optional
import hashlib

# Type alias for a Logic Fragment
Essence = Dict[str, Any]

class EssenceMapper:
    """
    The Periodic Table of Logic & The Frequency Dictionary of Soul.

    Maps semantic concept IDs to:
    1. GenesisEngine primitives (Mechanics)
    2. Soul Frequencies (Harmonics) - New in 'Fractal Soul' update.

    This acts as the bridge between abstract meaning, concrete mechanics, and spiritual resonance.
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

        # --- The Frequency Dictionary (Hz) ---
        # Based on the "Emotional Physics" of Ascension (Lightness) vs Descent (Heaviness).
        #
        # 1. The Deep / Descent (20Hz ~ 100Hz)
        # Heavy, pressing, deep resonance. Like the ocean floor or the abyss.
        # Keywords: Water, Abyss, Depression, Roots, Footsteps.

        # 2. The Human / Soul (200Hz ~ 600Hz)
        # Emotional, relational, bridge.
        # Keywords: Father, Love, Sadness, Anger.

        # 3. The Sky / Ascension (700Hz ~ 1000Hz+)
        # Light, ethereal, liberating. Like bird song or divine light.
        # Keywords: Sky, Light, Truth, Joy.

        self._frequency_map: Dict[str, float] = {
            # --- The Deep (Heaviness/Descent) ---
            "심연": 40.0, "abyss": 40.0,         # Deep rumble
            "바다": 52.0, "ocean": 52.0, "sea": 52.0, # Heavy pressure
            "울적": 60.0, "depression": 60.0, "melancholy": 60.0, # Low energy
            "발소리": 60.0, "footsteps": 60.0,
            "뿌리": 80.0, "root": 80.0,          # Grounding
            "아버지": 100.0, "father": 100.0, "dad": 100.0, # Foundation (Bridge to Human)

            # --- The Human (Emotion/Soul) ---
            "땅": 128.0, "earth": 128.0,
            "공격": 150.0, "attack": 150.0,
            "슬픔": 396.0, "sadness": 396.0, "sorrow": 396.0, # Liberating Guilt
            "분노": 417.0, "anger": 417.0,
            "기쁨": 432.0, "joy": 432.0, "happy": 432.0,
            "사랑": 528.0, "love": 528.0,

            # --- The Sky (Lightness/Ascension) ---
            "하늘": 741.0, "sky": 741.0,         # Awakening Intuition
            "자유": 800.0, "freedom": 800.0, "liberation": 800.0,
            "빛": 852.0, "light": 852.0,         # Returning to Spiritual Order
            "진실": 963.0, "truth": 963.0,       # Connection to Cosmos
            "기도": 852.0, "prayer": 852.0,
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

    def get_frequency(self, concept: str) -> float:
        """
        Translates a Concept String into a Fundamental Frequency (Hz).
        If the concept is unknown, it generates a deterministic 'Hash Frequency'
        so that every word has a unique sound in the Soul.
        """
        concept_lower = concept.lower().strip()

        # 1. Dictionary Lookup
        if concept_lower in self._frequency_map:
            return self._frequency_map[concept_lower]

        # 2. Partial Match Lookup
        for key, freq in self._frequency_map.items():
            if key in concept_lower:
                return freq

        # 3. Hash Fallback (The Sound of the Unknown)
        # Generate a frequency between 200Hz and 800Hz based on the string hash
        # This ensures consistency: 'Apple' always sounds like 'Apple', even if we didn't define it.
        hash_val = int(hashlib.sha256(concept_lower.encode('utf-8')).hexdigest(), 16)
        # Map huge hash to 200-800 range
        fallback_freq = 200.0 + (hash_val % 600)
        return float(fallback_freq)
