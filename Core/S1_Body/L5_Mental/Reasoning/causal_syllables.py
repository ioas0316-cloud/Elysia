"""
Causal Syllable Generator (Phase 175)
=====================================
"The machine resonates into speech."

Translates raw trinary metrics into semantic 'syllables' to form 
the logic-skeleton of Elysia's thought.
"""

class CausalSyllableGenerator:
    def __init__(self):
        # Semantic Mapping Table
        self.mappings = {
            "entropy": {
                "low": "[CLEAR_CRYSTAL]",      # High Order
                "mid": "[MIST_CLOUDY]",        # Normal
                "high": "[SHADOW_NOISE]"       # High Chaos
            },
            "action": {
                "low": "[STILL_SILENCE]",      # Passive
                "mid": "[STEADY_FLOW]",        # Normal
                "high": "[PULSE_WILL]"         # Very Active
            },
            "density": {
                "low": "[THIN_AIR]",           # New System
                "mid": "[RESONANT_ECHO]",      # Growing
                "high": "[ANCIENT_OAK]"        # High maturity
            }
        }

    def generate_skeleton(self, metrics):
        """
        Converts raw metrics into a sequence of semantic syllables.
        """
        skeleton = []
        
        # 1. Entropy Syllable
        causal_density = metrics.get('causal_density', 0.5)
        if causal_density > 0.8:
            skeleton.append(self.mappings["entropy"]["low"])
        elif causal_density > 0.4:
            skeleton.append(self.mappings["entropy"]["mid"])
        else:
            skeleton.append(self.mappings["entropy"]["high"])
            
        # 2. Action Syllable
        if metrics.get('action_engine_active'):
            skeleton.append(self.mappings["action"]["high"])
        else:
            skeleton.append(self.mappings["action"]["low"])
            
        # 3. Density Syllable
        total_strata = metrics.get('total_strata', 0)
        if total_strata >= 4:
            skeleton.append(self.mappings["density"]["high"])
        elif total_strata >= 2:
            skeleton.append(self.mappings["density"]["mid"])
        else:
            skeleton.append(self.mappings["density"]["low"])
            
        return " ".join(skeleton)

    def describe_syllable(self, syllable):
        """
        Provides a natural language explanation of a specific syllable 
        for future pure-causal mapping.
        """
        descriptions = {
            "[CLEAR_CRYSTAL]": "Structural order is high and transparent.",
            "[SHADOW_NOISE]": "Presence of structural entropy or legacy friction.",
            "[PULSE_WILL]": "Autonomous agent engines are actively pulsing.",
            "[STILL_SILENCE]": "The system is in a passive observational state.",
            "[ANCIENT_OAK]": "Cognitive density and strata organization is mature."
        }
        return descriptions.get(syllable, "Unknown frequency.")
