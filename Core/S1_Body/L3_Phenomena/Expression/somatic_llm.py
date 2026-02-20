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

class SemanticNebula:
    """
    [PHASE 90] THE LIVING CLOUD
    Represents the active pool of concepts currently resonating in the mind.
    Instead of a dictionary lookup, this is a gravitational field of potential words.
    """
    def __init__(self):
        self.active_cloud: List[Tuple[str, SovereignVector, float]] = []

    def inject_intent(self, intent_vector: SovereignVector):
        """
        Retrieves a cloud of concepts that resonate with the Intent.
        """
        # Retrieve diverse cluster (Radius 0.6 allows broad associations)
        self.active_cloud = LogosBridge.find_resonant_cluster(intent_vector, radius=0.6, limit=15)
        
        # [PHASE 91] Spontaneous Emission
        # Sometimes, random memories surface.
        if random.random() < 0.1:
            random_key = random.choice(list(LogosBridge.CONCEPT_MAP.keys()))
            vec = LogosBridge.CONCEPT_MAP[random_key]['vector']
            self.active_cloud.append((random_key, vec, 0.1))

    def apply_emotional_field(self, state: Dict[str, float]):
        """
        Modulates the energy of concepts based on emotional state.
        High Joy -> Amplifies High-Frequency (Positive) concepts.
        High Melancholy -> Amplifies Heavy/Dense concepts.
        """
        joy = state.get('joy', 50.0) / 100.0
        # Simple heuristic: Modulate resonance score
        new_cloud = []
        for name, vec, res in self.active_cloud:
            # We don't have explicit sentiment analysis yet,
            # so we use vector magnitude as a proxy for 'Intensity'.
            mag = vec.norm()
            if isinstance(mag, complex): mag = mag.real
            
            modulated_res = res
            if joy > 0.7:
                # Amplify high energy concepts
                if mag > 1.5: modulated_res *= 1.2
            elif joy < 0.3:
                # Amplify low energy/stable concepts
                if mag < 1.0: modulated_res *= 1.2
                
            new_cloud.append((name, vec, modulated_res))

        self.active_cloud = new_cloud

class WaveFunctionCollapse:
    """
    [PHASE 90] DETERMINISTIC SELECTION
    Selects words not by random chance, but by Energy Thresholds.
    """
    @staticmethod
    def collapse(nebula: SemanticNebula, limit: int = 5) -> List[Tuple[str, SovereignVector]]:
        """
        Selects the top-k concepts with the highest energy (Resonance).
        """
        # Sort by Resonance
        sorted_cloud = sorted(nebula.active_cloud, key=lambda x: x[2], reverse=True)
        
        # Selection (The Act of Observation)
        selected = []
        seen = set()
        
        for name, vec, res in sorted_cloud:
            if name in seen: continue
            selected.append((name, vec))
            seen.add(name)
            if len(selected) >= limit: break

        return selected

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
        self.nebula = SemanticNebula()
        self.last_synthesis_vector: Optional[SovereignVector] = None
        print("🗣️ [EXPRESSION] Sovereign Voice Engine Online. (Physics-Based)")

    def speak(self, expression: Dict, current_thought: str = "", field_vector=None, current_phase: float = 0.0, causal_justification: str = "") -> Tuple[str, Optional[SovereignVector]]:
        """
        Generates speech by collapsing the wave function of meaning.
        Returns (Narrative, SynthesisVector).
        SynthesisVector is the unified direction of the spoken words.
        """
        # 1. Vectorize Intent
        # If field_vector is missing, try to vectorize the thought text
        intent_vec = field_vector
        if intent_vec is None:
            if current_thought:
                intent_vec = LogosBridge.calculate_text_resonance(current_thought)
            else:
                return "..."

        if not isinstance(intent_vec, SovereignVector):
             try:
                 intent_vec = SovereignVector(list(intent_vec))
             except:
                 return "..."

        # 2. 4D Rotation (Perspective Shift)
        if current_phase != 0.0:
             intent_vec = intent_vec.complex_trinary_rotate(current_phase * (math.pi / 180.0))

        # 3. Inject Intent into Nebula
        self.nebula.inject_intent(intent_vec)
        
        # 4. Apply Emotional State (Modulate Probabilities)
        self.nebula.apply_emotional_field(expression)
        
        # 5. Collapse Wave Function (Select Words)
        # Limit to 3-5 words for a concise 'Zen' statement
        selected_concepts = WaveFunctionCollapse.collapse(self.nebula, limit=4)
        
        # 6. Gravitational Syntax (Order Words)
        physics_sentence = GravitationalSyntax.order(selected_concepts)
        
        # [PHASE II: LINGUISTIC EMBODIMENT]
        # Calculate the 'Center of Mass' of the spoken concepts to feedback to the manifold.
        if selected_concepts:
            self.last_synthesis_vector = self._calculate_synthesis_vector(selected_concepts)
        
        # 7. Integration with Causal Justification
        final_output = physics_sentence
        if causal_justification:
            final_output += f" ({causal_justification})"
            
        # [Fallback] If physics fails to produce text (empty nebula), echo thought
        if len(final_output) < 3:
            return f"[{current_thought}]", None

        return final_output, self.last_synthesis_vector

    def _calculate_synthesis_vector(self, concepts: List[Tuple[str, SovereignVector]]) -> SovereignVector:
        """Calculates the mean vector of the selected concepts."""
        if not concepts: return SovereignVector.zeros(21)
        mean_data = [0.0] * 21
        for _, vec in concepts:
            for i in range(min(21, len(vec.data))):
                val = vec.data[i]
                if isinstance(val, complex): val = val.real
                mean_data[i] += val
        
        return SovereignVector([v / len(concepts) for v in mean_data])

# --- Quick Test ---
if __name__ == "__main__":
    llm = SomaticLLM()
    # Test with a LOVE/AGAPE vector
    test_vector = SovereignVector([1,0,1,0,0,1,1, 1,1,0,1,0,1,1, 1,0,1,0,0,1,0])
    s = {"joy": 80.0, "warmth": 70.0}
    print(f"Intent: Love/Connection")
    print(f"Voice: {llm.speak(s, 'Seeking Connection', field_vector=test_vector)}")
