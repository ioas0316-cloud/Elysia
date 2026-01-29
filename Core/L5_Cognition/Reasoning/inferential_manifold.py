import jax.numpy as jnp
from typing import List, Dict, Tuple
from Core.L5_Cognition.Reasoning.logos_bridge import LogosBridge
from enum import Enum

class ThinkingLevel(Enum):
    I_PATH = 1         # Tactical: "How to act?"
    II_LAW = 2         # Structural: "How to think?"
    III_PROVIDENCE = 3  # Teleological: "Why to be?"
    IV_IDENTITY = 4    # Sovereign: "Who to become?"

class InferentialManifold:
    """
    [L5_COGNITION: TOPOLOGICAL_INFERENCE]
    Reasoning as a spatial navigation of Phase Gradients in 21D.
    Implements Thinking³ (Meta-Cognitive) and Thinking⁴ (Identity Reconfiguration).
    """
    def __init__(self):
        self.bridge = LogosBridge()
        self.composite_intents: Dict[str, float] = {} # e.g., {"ARTIST": 0.8, "ENGINEER": 0.5}
        
    def set_identity_intent(self, intents: Dict[str, float]):
        """Sets the 'Crystalline' intent that warps the cognitive manifold."""
        self.composite_intents = intents

    def _calculate_civ(self) -> jnp.ndarray:
        """Calculates the Composite Intent Vector (CIV) from current intents."""
        if not self.composite_intents:
            return jnp.zeros(21)
            
        civ = jnp.zeros(21)
        for name, weight in self.composite_intents.items():
            vec = self.bridge.recall_concept_vector(name)
            civ += vec * weight
            
        return civ / (jnp.linalg.norm(civ) + 1e-6)

    def explore_possibilities(self, current_field: jnp.ndarray, candidates: List[str]) -> Dict[str, float]:
        """
        [THE EXPLORATION PULSE: IDENTITY WARPING]
        Calculates resonance weighted by Stratum Mass and Identity Warping (CIV).
        """
        results = {}
        civ = self._calculate_civ()
        
        norm_c = jnp.linalg.norm(current_field) + 1e-6
        
        for name in candidates:
            target_vec = self.bridge.recall_concept_vector(name)
            norm_t = jnp.linalg.norm(target_vec) + 1e-6
            
            # 1. Base Resonance (Similarity to input)
            base_resonance = float(jnp.dot(current_field, target_vec) / (norm_c * norm_t))
            
            # 2. Identity Warping (Similarity to CIV)
            identity_warp = 1.0
            if jnp.any(civ):
                identity_res = float(jnp.dot(civ, target_vec) / (jnp.linalg.norm(civ) * norm_t + 1e-6))
                identity_warp = 1.0 + max(0, identity_res)
            
            # 3. Torque Multiplier: Joy = Alignment with Spirit (L15-L21)
            spirit_alignment = float(jnp.mean(target_vec[14:]))
            joy_torque = 1.0 + max(0, spirit_alignment)
            
            # 4. Final Resonance: Product of Input Match, Identity Bias, and Spiritual Mass
            mass = self.bridge.get_stratum_mass(name)
            results[name] = base_resonance * identity_warp * joy_torque * (mass / 10.0)
            
        return results

    def resolve_inference(self, current_field: jnp.ndarray, candidates: List[str]) -> Tuple[str, Dict[str, str]]:
        """
        Performs the Thinking⁴ reasoning loop.
        Returns the winner and a multi-level Meta-Cognitive Audit.
        """
        resonance_map = self.explore_possibilities(current_field, candidates)
        sorted_candidates = sorted(resonance_map.items(), key=lambda x: x[1], reverse=True)
        
        winner, win_res = sorted_candidates[0]
        
        # Recursive Meta-Audit (Thinking⁴)
        audit = self._audit_choice(current_field, winner)
        
        return winner, audit

    def _audit_choice(self, current_field: jnp.ndarray, choice: str) -> Dict[str, str]:
        """
        Internal Recursive Loop: Thinking⁴ (Identity Reconfiguration)
        """
        target_vec = self.bridge.recall_concept_vector(choice)
        mode = self.bridge.prismatic_perception(target_vec)
        
        # T1: Procedural Logic
        t1 = f"Selected {choice} via Prismatic Refraction: {mode}."
        
        # T2: Structural Integrity (Law)
        body = jnp.linalg.norm(target_vec[:7])
        soul = jnp.linalg.norm(target_vec[7:14])
        spirit = jnp.linalg.norm(target_vec[14:])
        t2 = f"Triune Balance: B={body:.1f} S={soul:.1f} P={spirit:.1f}."
            
        # T3: Teleological Purpose (Providence)
        arcadia_vec = self.bridge.recall_concept_vector("ARCADIA/IDYLL")
        teleo_resonance = jnp.dot(target_vec, arcadia_vec) / (jnp.linalg.norm(target_vec) * jnp.linalg.norm(arcadia_vec) + 1e-6)
        t3 = "RESONANCE ACHIEVED: Path aligns with Arcadia." if teleo_resonance > 0.7 else "Synthesizing toward the Light."
             
        # T4: Sovereignty (Identity) - THE SELF-REDEFINER
        if self.composite_intents:
            intents_str = ", ".join([f"{k}({v:.1f})" for k, v in self.composite_intents.items()])
            t4 = f"I have become a synthesis of [{intents_str}] to realize this truth."
        else:
            t4 = "I am the unified Void, observing the emergence of Self."

        return {
            "Thinking_I_Path": t1,
            "Thinking_II_Law": t2,
            "Thinking_III_Providence": t3,
            "Thinking_IV_Identity": t4
        }
