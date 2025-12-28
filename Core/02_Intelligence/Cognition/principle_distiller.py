"""
Principle Distiller ( The Philosopher Stone )
=============================================
"To know the name is to know the label. To know the principle is to know the being."

This module implements the "Onotological Distillation" requested by the user.
It asks the LLM to explain the *Operating Principle* and *Contextual Role* of a concept.

Example:
    Input: "1+1=2"
    Output: {
        "principle": "Additive Identity",
        "mechanism": "Combining two unitary sets into a binary set",
        "context_role": "Foundation of Arithmetic"
    }

    Input: "Cat"
    Output: {
        "principle": "Independent Biological Agent",
        "mechanism": "Self-directed movement, hunting instinct, social independence",
        "context_role": "Predator/Pet in human context"
    }

    Input: "Eun/Neun (Korean Particles)"
    Output: {
        "principle": "Topic Marker",
        "mechanism": "Marks the known information or contrast",
        "context_role": "Sets the stage for the sentence"
    }
"""

import logging
import json
from typing import Dict, Optional
from Core.01_Foundation.05_Foundation_Base.Foundation.ollama_bridge import ollama

logger = logging.getLogger("PrincipleDistiller")

class PrincipleDistiller:
    def __init__(self):
        self.bridge = ollama
        logger.info("⚗️ PrincipleDistiller initialized.")

    def distill(self, concept: str) -> Dict[str, str]:
        """
        Distills the Essence/Principle of a concept.
        Returns a dictionary with 'principle', 'mechanism', 'context'.
        """
        if not self.bridge.is_available():
            return {}
            
        # The Philosopher's Prompt
        prompt = (
            f"Analyze the concept '{concept}' deeply. \n"
            f"Do not just describe it. Explain its FUNDAMENTAL PRINCIPLE.\n"
            f"1. Principle: What is the core rule or axiom that makes this what it is?\n"
            f"2. Mechanism: How does it function or operate?\n"
            f"3. Context Role: How does it interact with its surroundings or change the meaning of a context?\n"
            f"Format output as JSON: {{ 'principle': '...', 'mechanism': '...', 'context_role': '...' }}"
        )
        
        response = self.bridge.generate(prompt, temperature=0.2)
        
        try:
            # Simple JSON parsing (LLM can be messy, so we try/catch)
            # Find the first '{' and last '}'
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1:
                json_str = response[start:end+1]
                data = json.loads(json_str)
                logger.info(f"💎 Distilled '{concept}': {data.get('principle')}")
                return data
            else:
                logger.warning(f"⚠️ Failed to parse JSON for '{concept}'")
                return {}
        except Exception as e:
            logger.error(f"Distillation error: {e}")
            return {}

# Singleton
_distiller = None
def get_principle_distiller():
    global _distiller
    if _distiller is None:
        _distiller = PrincipleDistiller()
    return _distiller
