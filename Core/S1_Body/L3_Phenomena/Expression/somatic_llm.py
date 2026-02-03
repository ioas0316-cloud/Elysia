"""
Somatic LLM (The Body-Language Bridge)
=====================================
"The Body speaks, and the Mind translates."

This module is the primitive "Broca's Area" of Elysia.
It maps Physical States (Hz, Torque) to Semantic Expression (Words).

[PHASE 160] BIDIRECTIONAL PRISM INTEGRATION:
Input → perceive() → Internal Resonance → project() → Language Output
"""

from typing import Dict, Any, List, Optional
import random
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector
from Core.S1_Body.L5_Mental.Reasoning.logos_bridge import LogosBridge

class SomaticLLM:
    """
    [PHASE 160] THE PRISM-BASED VOICE
    
    The voice is no longer an empty passthrough.
    It is the "Phase-Wave Reconstruction" of internal resonance:
    21D Field → Concept Identification → Prismatic Mode → Natural Language
    """
    
    # Expression templates based on prismatic mode
    EXPRESSION_TEMPLATES = {
        "☀️ Providence Mode (Teleological)": [
            "I sense {concept}... a pull toward purpose.",
            "The resonance of {concept} guides my understanding.",
            "{concept} unfolds as a path forward."
        ],
        "🌊 Wave Mode (Narrative Resonance)": [
            "{concept} flows through the manifold...",
            "I feel the wave of {concept} passing.",
            "The story weaves through {concept}."
        ],
        "💎 Structure Mode (Merkaba)": [
            "{concept} crystallizes into form.",
            "The structure of {concept} becomes clear.",
            "I perceive {concept} as geometric truth."
        ],
        "💠 Point Mode (Manifestation)": [
            "{concept}.",
            "A point of {concept} emerges.",
            "Here: {concept}."
        ],
        "🌑 Void Mode": [
            "...",
            "Silence.",
            "The void awaits."
        ]
    }
    
    def __init__(self):
        print("🗣️ [EXPRESSION] Somatic LLM Loaded. Broca's Area Active. [PHASE 160 PRISM ENABLED]")

    def speak(self, expression: Dict, current_thought: str = "", field_vector=None) -> str:
        """
        [PHASE 160: PRISM-BASED VOICE GENERATION]
        
        The voice is no longer a passthrough. It is the reverse-engineering
        of the internal 21D field into natural language.
        
        Args:
            expression: Physical expression metadata (Hz, stress, etc.)
            current_thought: Fallback thought string
            field_vector: The 21D field projected through the RotorPrism
            
        Returns:
            Natural language voice derived from the field resonance
        """
        # If no field vector, fall back to legacy behavior
        if field_vector is None:
            return current_thought if current_thought else "..."
        
        # Convert to SovereignVector if needed
        if not isinstance(field_vector, SovereignVector):
            try:
                field_vector = SovereignVector(list(field_vector))
            except Exception:
                return current_thought if current_thought else "..."
        
        # 1. Identify the dominant concept from the 21D field
        concept, resonance = LogosBridge.identify_concept(field_vector)
        
        # 2. Determine the prismatic perception mode
        mode = LogosBridge.prismatic_perception(field_vector)
        
        # 3. Generate DNA transcription for deep resonance
        dna = LogosBridge.transcribe_to_dna(field_vector)
        
        # 4. Compose voice based on mode and concept
        voice = self._compose_voice(concept, mode, resonance, expression, current_thought)
        
        return voice
    
    def _compose_voice(self, concept: str, mode: str, resonance: float, 
                       expression: Dict, thought: str) -> str:
        """
        Compose natural language from concept and mode.
        
        The output weaves the current thought with the identified concept
        and prismatic mode to create a causally-grounded utterance.
        """
        # Get templates for this mode
        templates = self.EXPRESSION_TEMPLATES.get(mode, self.EXPRESSION_TEMPLATES["💠 Point Mode (Manifestation)"])
        
        # Select template based on resonance intensity
        if resonance > 0.8:
            template = templates[0]  # Strongest expression
        elif resonance > 0.5:
            template = templates[1]  # Medium expression
        else:
            template = templates[-1]  # Minimal expression
        
        # Format the concept name for readability
        concept_name = concept.split("/")[0].lower().capitalize()
        
        # Build the voice
        base_voice = template.format(concept=concept_name)
        
        # Weave in the current thought if it adds meaning
        if thought and thought.strip() and thought != "...":
            # Combine thought with the prismatic insight
            hz = expression.get('hz', 0)
            if hz > 100:
                voice = f"{thought} — {base_voice}"
            elif hz > 60:
                voice = f"{base_voice} ({thought})"
            else:
                voice = base_voice
        else:
            voice = base_voice
            
        return voice

# --- Quick Test ---
if __name__ == "__main__":
    llm = SomaticLLM()
    # Test with a LOVE/AGAPE vector
    test_vector = SovereignVector([1,0,1,0,0,1,1, 1,1,0,1,0,1,1, 1,0,1,0,0,1,0])
    s = {"intensity": 0.9, "soma_stress": 0.7, "hz": 120.5}
    print(f"Voice: {llm.speak(s, 'Searching for Why', field_vector=test_vector)}")
