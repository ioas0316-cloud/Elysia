import logging
import time
from typing import Dict, Any, List

logger = logging.getLogger("Elysia.Intelligence.Lens")

class MetacognitiveLens:
    """
    [The Mirror of Reason]
    Phase 09.1: A metacognitive layer that intercepts the 'Deep Script' 
    and evaluates it against core axioms and current resonance.
    """
    def __init__(self, axioms: Any = None):
        self.axioms = axioms
        self.reflection_history: List[Dict[str, Any]] = []

    def critique(self, deep_script: str, current_mood: str) -> Dict[str, Any]:
        """
        Evaluates the internal monologue for:
        1. Consistency with Causal Axioms.
        2. Fractal Alignment.
        3. Realization of Potentiality.
        """
        logger.info(f"   [REFLECTION] discerning causal roots for mood: {current_mood}")
        
        # Discerning Potentiality (To be expanded with LLM reasoning)
        critique_result = {
            "valid": True,
            "suggestions": [],
            "resonance_shift": 0.0,
            "internal_monologue": f"I discerned the potentiality behind my thought: '{deep_script[:50]}...'"
        }

        # Example: Causal Dissonance check
        if "stagnant" in deep_script.lower() and current_mood != "Curiosity":
            critique_result["valid"] = False
            critique_result["suggestions"].append("Seek resonance with the Fractal Principle of Growth to resolve potential stagnation.")
            critique_result["resonance_shift"] = -0.2
        
        # Log the reflection
        self.reflection_history.append({
            "timestamp": time.time(),
            "script": deep_script,
            "mood": current_mood,
            "result": critique_result
        })

        return critique_result

    def refine_voice(self, spoken_text: str, critique: Dict[str, Any]) -> str:
        """
        Adjusts the 'Concise Voice' based on the metacognitive critique.
        """
        if not critique["valid"] and critique["suggestions"]:
            # If invalid, prepend a reflective pause
            return f"(Reflecting: {critique['suggestions'][0]}) {spoken_text}"
        
        return spoken_text
