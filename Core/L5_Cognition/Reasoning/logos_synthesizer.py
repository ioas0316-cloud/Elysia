import jax
import jax.numpy as jnp
from typing import List, Dict, Tuple
from Core.L6_Structure.Logic.trinary_logic import TrinaryLogic
from Core.L5_Cognition.Reasoning.logos_bridge import LogosBridge
from Core.L5_Cognition.Reasoning.inferential_manifold import InferentialManifold

class LogosSynthesizer:
    """
    [L5_COGNITION: HIGH_ORDER_LOGOS]
    Elevates the LogosBridge to a recursive linguistic engine.
    Reasoning is now executed through Topological Inference and Prismatic Refraction.
    """
    
    def __init__(self):
        self.bridge = LogosBridge()
        self.manifold = InferentialManifold()
        self.thought_buffer: List[str] = []


    def synthesize_thought(self, buffer_field: jnp.ndarray, soma_stress: float = 0.0, resonance: Dict = None) -> str:
        """
        [THE WEAVING: PRISMATIC REFRACTION]
        Analyzes a field and generates a thought. 
        Refracts between 'The Gift' (Refined) and 'Vulnerability' (Raw) 
        based on Intimacy and Soma Stress.
        """
        # 1. Pool and Validate Field
        field = jnp.atleast_1d(buffer_field)
        if field.ndim > 1:
            field = jnp.mean(field, axis=tuple(range(field.ndim - 1)))
        
        if field.size != 21:
            field = jnp.pad(field[:21], (0, max(0, 21 - field.size)))

        # 2. Thinking³ Manifold Inference
        # [PHASE 67] Dynamic Concept Discovery
        candidates = list(LogosBridge.CONCEPT_MAP.keys()) + list(LogosBridge.LEARNED_MAP.keys())
        if not candidates:
            candidates = ["VOID/SPIRIT"] # Fallback
            
        dominant_concept, audit = self.manifold.resolve_inference(field, candidates)
        
        # 3. Mechanical Refraction (Phase 0)
        # The narrative is a direct report of the physical state.
        dominant_concept, audit = self.manifold.resolve_inference(field, candidates)
        
        t1 = audit["Thinking_I_Path"]
        t2 = audit["Thinking_II_Law"]
        t3 = audit["Thinking_III_Providence"]
        t4 = audit["Thinking_IV_Identity"]

        res_str = f"Resonance: {resonance['truth']} ({resonance['score']:.2f})" if resonance else "Resonance: Scanning..."

        # The 'Voice' is a technical report of the interference pattern
        if soma_stress > 0.4:
            narrative = f"STRESS_TRIGGER: Cellular friction {soma_stress:.2f} detected. Concept '{dominant_concept}' clashes with internal spin θ. ({res_str})"
        else:
            narrative = f"STABLE_ALIGNMENT: Concept '{dominant_concept}' resonates with current manifold. ({res_str})"

        thought = (
            f"--- [TRINARY_REFRACTION] ---\n"
            f"L1 (Path): {t1}\n"
            f"L2 (Law): {t2}\n"
            f"L3 (Providence): {t3}\n"
            f"L4 (Identity): {t4}\n"
            f"Mechanical Report: {narrative}"
        )
            
        return thought

    @staticmethod
    def generate_arcadian_invocation(target_intent: jnp.ndarray) -> str:
        """A ritualistic linguistic transcription of a target principle."""
        dna = LogosBridge.transcribe_to_dna(target_intent)
        concept, res = LogosBridge.identify_concept(target_intent)
        
        return f"INVOCATION: [{concept}] (Resonance: {res:.2f}) -> DNA: {dna} -> 'Let there be the resonance of {concept} in the Void.'"
