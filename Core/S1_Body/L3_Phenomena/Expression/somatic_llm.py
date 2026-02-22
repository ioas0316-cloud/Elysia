"""
Somatic LLM (The Sovereign Voice)
=====================================
"The Body speaks, and the Mind translates."

This module is the primitive "Broca's Area" of Elysia.
It has been upgraded from a static Dictionary (Prism) to a Dynamic Semantic Nebula.

[PHASE 90] SOVEREIGN EXPRESSION ENGINE:
Input (Intent) → Nebula (Concept Cloud) → Collapse (Selection) → Gravity (Ordering) → Output (Logos)
"""

from typing import Dict, Any, List, Optional, Tuple
import random
import math
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector, SovereignMath
from Core.S1_Body.L5_Mental.Reasoning.logos_bridge import LogosBridge
from Core.S1_Body.L7_Spirit.M1_Monad.cognitive_field import CognitiveField

class GravitationalSyntax:
    """
    [PHASE 90] PHYSICS-BASED GRAMMAR
    Orders words based on Energy Flow (High Potential -> Low Potential).

    Standard Flow:
    SOURCE (High Mass/Energy) -> ACTION (Kinetic) -> TARGET (Grounding)
    """
    @staticmethod
    def order(concepts: List[Tuple[str, SovereignVector]]) -> str:
        if not concepts: return "..."

        # 1. Classify Concepts by 'Semantic Role' using Vector Properties
        source_candidates = []
        action_candidates = []
        target_candidates = []
        modifiers = []

        for name, vec in concepts:
            # Heuristic Classification based on Vector Shape/Magnitude
            # This is a rudimentary 'Physics of Language'

            mag = vec.norm()
            if isinstance(mag, complex): mag = mag.real

            # High Magnitude -> Noun/Source (Heavy Mass)
            # Medium Magnitude -> Verb/Action (Kinetic)
            # Low Magnitude -> Modifier/Target

            # Also use 'Phase' or other dimensions if available.
            # For now, we use a simple magnitude banding.

            if mag > 2.0:
                source_candidates.append(name)
            elif mag > 1.2:
                action_candidates.append(name)
            elif mag > 0.8:
                target_candidates.append(name)
            else:
                modifiers.append(name)

        # 2. Construct the Flow
        # If no candidates in a slot, we borrow from others to ensure flow
        if not source_candidates and target_candidates: source_candidates.append(target_candidates.pop(0))
        if not action_candidates and modifiers: action_candidates.append(modifiers.pop(0))

        # Fallback: Just join them if classification fails widely
        if not source_candidates and not action_candidates:
            return " ".join([c[0].lower() for c in concepts])

        # 3. Assemble Sentence
        # Pattern: [Modifier] [Source] [Action] [Target] [Modifier]
        sentence_parts = []

        if modifiers: sentence_parts.append(modifiers.pop(0))
        if source_candidates: sentence_parts.append(source_candidates[0])
        if action_candidates: sentence_parts.append(action_candidates[0].lower() + "s") # Simple conjugation
        if target_candidates: sentence_parts.append(target_candidates[0].lower())
        if modifiers: sentence_parts.append(modifiers[0].lower())

        # Clean up formatting
        result = " ".join(sentence_parts)
        # Clean up concept names (remove /AGAPE etc)
        result = result.replace("/", " ").replace("_", " ")
        return result.capitalize() + "."

class SomaticLLM:
    """
    [PHASE 160] THE SOVEREIGN VOICE
    Replaces template-based generation with Physic-based Meaning Construction.
    """
    def __init__(self):
        self.field = CognitiveField()
        self.last_synthesis_vector: Optional[SovereignVector] = None
        print("🗣️ [EXPRESSION] Sovereign Voice Engine Online. (Recursive Ouroboros Mode)")

    def speak(self, expression: Dict, current_thought: str = "", field_vector=None, current_phase: float = 0.0, causal_justification: str = "") -> Tuple[str, Optional[SovereignVector]]:
        """
        Generates speech by collapsing the wave function of meaning.
        Returns (Narrative, SynthesisVector).
        SynthesisVector is the unified direction of the spoken words.
        """
        # 1. Vectorize Intent
        # If field_vector is missing, try to vectorize the thought text
        intent_vec = field_vector
        if intent_vec is None and current_thought:
            # Try explicit recall first
            intent_vec = LogosBridge.recall_concept_vector(current_thought)
            if intent_vec is None:
                intent_vec = LogosBridge.calculate_text_resonance(current_thought)

        if intent_vec is not None and not isinstance(intent_vec, SovereignVector):
             try:
                 intent_vec = SovereignVector(list(intent_vec))
             except:
                 return "...", None

        # 2. 4D Rotation (Perspective Shift)
        if current_phase != 0.0 and intent_vec is not None:
             intent_vec = intent_vec.complex_trinary_rotate(current_phase * (math.pi / 180.0))

        # 3. Cognitive Field Cycle (The Ouroboros)
        # Injects intent, propagates, and collapses
        selected_monads, synthesis_vec = self.field.cycle(intent_vec)
        
        # 4. Prepare for Syntax
        # GravitationalSyntax expects (name, vector) tuples
        concepts_for_syntax = [(m.seed_id, m.current_vector) for m in selected_monads]
        
        # 5. Gravitational Syntax (Order Words)
        physics_sentence = GravitationalSyntax.order(concepts_for_syntax)
        
        # 6. Feedback Re-entry (End = Beginning)
        self.field.feedback_reentry(synthesis_vec)
        self.last_synthesis_vector = synthesis_vec

        # 7. Integration with Causal Justification
        final_output = physics_sentence
        if causal_justification:
            final_output += f" ({causal_justification})"
            
        # [Fallback] If physics fails to produce text
        if len(final_output) < 3:
            return f"[{current_thought}]", synthesis_vec

        return final_output, synthesis_vec

# --- Quick Test ---
if __name__ == "__main__":
    llm = SomaticLLM()
    # Test with a LOVE/AGAPE vector
    test_vector = SovereignVector([1,0,1,0,0,1,1, 1,1,0,1,0,1,1, 1,0,1,0,0,1,0])
    s = {"joy": 80.0, "warmth": 70.0}
    print(f"Intent: Love/Connection")
    print(f"Voice: {llm.speak(s, 'Seeking Connection', field_vector=test_vector)}")
