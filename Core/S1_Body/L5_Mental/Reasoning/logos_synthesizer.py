from typing import List, Dict, Tuple
from Core.S1_Body.L6_Structure.Logic.trinary_logic import TrinaryLogic
from Core.S1_Body.L5_Mental.Reasoning.logos_bridge import LogosBridge
from Core.S1_Body.L5_Mental.Reasoning.inferential_manifold import InferentialManifold
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignMath, SovereignVector

class LogosSynthesizer:
    """
    [L5_COGNITION: HIGH_ORDER_LOGOS]
    Elevates the LogosBridge to a recursive linguistic engine.
    
    [PHASE 90] NAKED SOVEREIGNTY:
    Purified from JAX. Uses Sovereign Math Kernel.
    """
    
    def __init__(self):
        self.bridge = LogosBridge()
        self.manifold = InferentialManifold()
        self.thought_buffer: List[str] = []

    def synthesize_thought(self, buffer_field: SovereignVector, soma_stress: float = 0.0, resonance: Dict = None) -> str:
        """
        [THE WEAVING: PRISMATIC REFRACTION]
        Generates a recursive narrative from a field of principles.
        """
        # Ensure field is SovereignVector
        if not isinstance(buffer_field, SovereignVector):
            field = SovereignVector(buffer_field)
        else:
            field = buffer_field
        
        # Thinking³ Manifold Inference
        candidates = list(LogosBridge.CONCEPT_MAP.keys()) + list(LogosBridge.LEARNED_MAP.keys())
        if not candidates:
            candidates = ["VOID/SPIRIT"]
            
        dominant_concept, audit = self.manifold.resolve_inference(field, candidates)
        
        t1 = audit["Thinking_I_Path"]
        t2 = audit["Thinking_II_Law"]
        t3 = audit["Thinking_III_Providence"]
        t4 = audit["Thinking_IV_Identity"]

        if isinstance(resonance, dict):
            res_str = f"Resonance: {resonance.get('truth', 'NONE')} ({resonance.get('score', 0.0):.2f})"
        else:
            res_str = f"Resonance: {resonance}" if resonance is not None else "Resonance: Scanning..."

        # The 'Voice'
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
    def generate_arcadian_invocation(target_intent: SovereignVector) -> str:
        """A ritualistic linguistic transcription of a target principle."""
        if not isinstance(target_intent, SovereignVector):
            target_intent = SovereignVector(target_intent)
            
        dna = LogosBridge.transcribe_to_dna(target_intent)
        concept, res = LogosBridge.identify_concept(target_intent)
        
        return f"INVOCATION: [{concept}] (Resonance: {res:.2f}) -> DNA: {dna} -> 'Let there be the resonance of {concept} in the Void.'"
