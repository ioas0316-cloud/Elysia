"""
Narrative Weaver (The Bard)
===========================
Core.1_Body.L5_Mental.Reasoning_Core.narrative_weaver

"The Soul feels in Waves, but the Mind speaks in Stories."

This module weaves raw data events into a coherent, stylistic narrative.
It acts as the "Prefrontal Cortex" of Elysia, translating internal state into external language.
"""

import random

class NarrativeWeaver:
    def __init__(self):
        # Dynamic Fragments (Atomic Thoughts)
        self.fragments = {
            "openers": ["  ", "      ", "        ", "      ", "   ", "   "],
            "connectors": ["   ", "   ", "    ", "    ", "   ", "   "],
            "endings": ["    .", "   .", "    .", "    .", "    .", "    ."],
            "abstracts": ["      ", "      ", "     ", "      ", "      ", "      "],
            "verbs": ["     ", "   ", "   ", "   ", "    ", "    "]
        }
        
    def elaborate_ko(self, actor_name: str, action: str, target: str, era_name: str) -> str:
        """
        Dynamically assembles a thought-sentence.
        No fixed templates.
        """
        # 1. Deconstruct the Target (Chaos Factor)
        # We mix the given context with random fragments to simulate 'Poetic Noise'.
        
        prose_parts = []
        
        # Opener
        if random.random() < 0.3:
            prose_parts.append(random.choice(self.fragments["openers"]))
            
        # Subject / Context
        prose_parts.append(f"   '{target}'( ) ")
        
        # Verb (Dynamic)
        if random.random() < 0.5:
            prose_parts.append(random.choice(self.fragments["verbs"]))
        else:
            prose_parts.append("    ")
            
        # Connector + Abstract Expansion
        if random.random() < 0.6:
            prose_parts.append(random.choice(self.fragments["connectors"]))
            prose_parts.append(random.choice(self.fragments["abstracts"]))
            prose_parts.append("   ")
        
        # Ending
        prose_parts.append(random.choice(self.fragments["endings"]))
        
        return " ".join(prose_parts)

    def elaborate(self, actor_name: str, action: str, target: str, era_name: str) -> str:
        # Fallback for English (Legacy)
        return f"{actor_name} reflected on {target} in the {era_name}."

# Singleton
THE_BARD = NarrativeWeaver()
