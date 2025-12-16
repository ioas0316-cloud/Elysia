"""
Reality Grounding ( The Anchor )
================================
"Data is the shadow. Reality is the object casting it."

This module anchors abstract Vectors/Concepts to Physical Reality.
It determines the ontological status of a node:
- Is it Physical? (Apple) -> Subject to Gravity, Decay.
- Is it Abstract? (Justice) -> Subject to Logic, Consensus.
- Is it Action? (Running) -> Subject to Time, Energy.

It populates `node_metadata['reality_physics']`.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger("RealityGrounding")

class RealityGrounding:
    def __init__(self, bridge):
        self.bridge = bridge # Ollama/TinyBrain
        logger.info("⚓ RealityGrounding initialized.")

    def ground_concept(self, concept: str) -> Dict[str, Any]:
        """
        Determines the Physical Laws governing a concept.
        """
        if not self.bridge.is_available(): return {}

        # The Ontological Prompt
        prompt = (
            f"Analyze the relationship between Reality and the concept '{concept}'.\n"
            f"1. Is it Physical (Matter) or Abstract (Information)?\n"
            f"2. Does it have Mass? (Yes/No)\n"
            f"3. Does it decay over time (Entropy)? (Yes/No)\n"
            f"Format: Type: [Physical/Abstract], Mass: [Yes/No], Entropy: [Yes/No]"
        )
        
        # Use TinyBrain/Ollama
        response = self.bridge.generate(prompt, temperature=0.1)
        
        physics = {
            "type": "Unknown",
            "has_mass": False,
            "entropy_susceptible": False
        }
        
        # Parse (Simple Heuristic for now)
        lower_resp = response.lower()
        
        if "physical" in lower_resp: physics["type"] = "Physical"
        elif "abstract" in lower_resp: physics["type"] = "Abstract"
        
        if "mass: yes" in lower_resp: physics["has_mass"] = True
        if "entropy: yes" in lower_resp: physics["entropy_susceptible"] = True
        
        logger.info(f"⚓ Grounded '{concept}': {physics}")
        return physics

# Singleton
_grounding = None
def get_reality_grounding(bridge_instance=None):
    global _grounding
    if _grounding is None and bridge_instance:
        _grounding = RealityGrounding(bridge_instance)
    return _grounding
