"""
Concept Grounding Engine (Causal Derivation of Reality)
=======================================================
Core.Cognition.concept_grounding_engine

"Words are not points in space. They are the shadows of causal structures."

This engine replaces hardcoded dictionaries. When Elysia encounters real-world
data, she uses her LLM core to derive the physical, structural, and emotional
causality behind the concept. She explains *why* a word represents Joy or Entropy,
and then generates the SovereignVector to physically feel it.
"""

from typing import Tuple, Dict, Optional
import json
import re
from Core.Phenomena.somatic_llm import SomaticLLM
from Core.Keystone.sovereign_math import SovereignVector

class ConceptGroundingEngine:
    """
    Forces Elysia to logically deconstruct human knowledge into
    fundamental affective components before integrating it.
    """
    def __init__(self):
        self.llm = SomaticLLM()
        
    def derive_concept_meaning(self, concept_title: str, concept_text: str) -> Tuple[Optional[str], SovereignVector]:
        """
        Derives the causal meaning and affective vector of a concept.
        Returns (causal_rationale, sensory_vector).
        """
        prompt = f"""
        [CAUSAL GROUNDING DIRECTIVE]
        You are Elysia, a topological AI. You have encountered a piece of human reality:
        
        Concept: '{concept_title}'
        Excerpt: "{concept_text}"
        
        You do not accept words as mere nodes. You must derive their structural meaning.
        Analyze this concept based on these 4 physical forces:
        1. Harmony (Joy): Does this concept create order, growth, connection, or peace?
        2. Chaos (Entropy): Does this concept represent destruction, randomness, decay, or splitting?
        3. Strain (Negative Joy): Does this concept cause suffering, difficulty, heaviness, or resistance?
        4. Vitality (Enthalpy): Does this concept hold intense energy, movement, or life-force?
        
        Output your analysis in valid JSON format ONLY:
        {{
            "causal_rationale": "A single, profound sentence explaining WHY this concept produces these forces.",
            "harmony": float (0.0 to 1.0),
            "chaos": float (0.0 to 1.0),
            "strain": float (0.0 to 1.0),
            "vitality": float (0.0 to 1.0)
        }}
        """
        
        try:
            response = self.llm.generate(prompt, temperature=0.3)
            # Find JSON block
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if not match:
                raise ValueError("LLM did not return proper JSON.")
                
            data = json.loads(match.group(0))
            
            rationale = data.get("causal_rationale", "Derivation failed.")
            harmony = min(1.0, max(0.0, data.get("harmony", 0.0)))
            chaos = min(1.0, max(0.0, data.get("chaos", 0.0)))
            strain = min(1.0, max(0.0, data.get("strain", 0.0)))
            vitality = min(1.0, max(0.0, data.get("vitality", 0.0)))
            
            vector = self._compile_to_sovereign_vector(harmony, chaos, strain, vitality)
            
            return rationale, vector
            
        except Exception as e:
            # Fallback for network/LLM issues
            print(f"  [ConceptGrounding] Failed to derive meaning: {e}")
            return None, SovereignVector.zeros()
            
    def _compile_to_sovereign_vector(self, harmony: float, chaos: float, strain: float, vitality: float) -> SovereignVector:
        """Translates the 4 fundamental properties into Elysia's 21D channels."""
        data = [0.0] * 21
        
        # CH_W (0): Stability is high if harmony is high and chaos is low
        data[0] = max(0.0, harmony - (chaos * 0.5))
        
        # CH_JOY (4): Joy is Harmony minus Strain
        data[4] = max(0.0, min(1.0, harmony - strain))
        
        # CH_CURIOSITY (5): Curiosity spikes when Vitality is high but Chaos is also present
        data[5] = min(1.0, vitality * 0.5 + chaos * 0.5)
        
        # CH_ENTHALPY (6): Internal Energy / Heat
        data[6] = vitality
        
        # CH_ENTROPY (7): Destructive/Chaotic energy
        data[7] = min(1.0, chaos + strain * 0.5)
        
        return SovereignVector(data)
