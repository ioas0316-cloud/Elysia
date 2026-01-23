"""
The Resonance Field: Topological Consonance
===========================================
Phase 18 Redux - Module 1
Core.L2_Metabolism.Evolution.resonance_field

"The Universe does not grade you. It resonates or it collapses."

This module replaces the scalar 'Score' with a Vector Interference Pattern.
It calculates the Consonance (Harmony) and Dissonance (Entropy) of an action.
"""

import logging
import math
from typing import Tuple, NamedTuple, Any

# Compatibility Layer for JAX/Numpy
try:
    import jax.numpy as jnp
    BACKEND = "JAX"
except ImportError:
    import numpy as jnp
    BACKEND = "NUMPY"

logger = logging.getLogger("Evolution.Resonance")

class KarmaState(NamedTuple):
    """
    The Topological State of Karma.
    """
    resonance: float    # Constructive Interference (Energy Gain)
    dissonance: float   # Destructive Interference (Entropy/Heat)
    phase_shift: float  # The angle of deviation from Providence
    entropy: float      # The resulting chaos level

class ResonanceField:
    """
    The Holographic Field that measures Causal Alignment.
    """
    def __init__(self, use_quantum: bool = True):
        self.use_quantum = use_quantum
        logger.info(f"  [RESONANCE] Field generated. Backend: {BACKEND}")

    def evaluate_resonance(self, 
                          intent_vec: jnp.ndarray, 
                          outcome_vec: jnp.ndarray) -> KarmaState:
        """
        Calculates the interference between the Intent Tensor and the Outcome Tensor.
        
        Args:
            intent_vec: The vector of the original Will (Action).
            outcome_vec: The vector of the World's Response (Result).
            
        Returns:
            KarmaState: The topological report.
        """
        # Ensure vectors are normalized for pure phase comparison
        i_norm = jnp.linalg.norm(intent_vec)
        o_norm = jnp.linalg.norm(outcome_vec)
        
        if i_norm == 0 or o_norm == 0:
            # Void State
            return KarmaState(0.0, 1.0, 0.0, 1.0)

        v_i = intent_vec / i_norm
        v_o = outcome_vec / o_norm

        # 1. Calculate Dot Product (Cosine Similarity) -> Resonance Base
        # 1.0 = Perfect Alignment, -1.0 = Perfect Opposition, 0 = Orthogonal (No Impact)
        dot_product = jnp.dot(v_i, v_o)
        
        # 2. Dissonance (Euclidean Distance in Phase Space)
        # How far is the result from the intent?
        # D = ||I - O||
        diff = v_i - v_o
        distance = jnp.linalg.norm(diff)
        
        # 3. Phase Shift (Angle)
        # theta = arccos(dot)
        # Clip to safe range [-1, 1]
        clipped_dot = jnp.clip(dot_product, -1.0, 1.0)
        phase_shift = jnp.arccos(clipped_dot)

        # 4. Entropy Calculation
        # Entropy rises as Dissonance squared
        entropy = float(jnp.power(distance, 2))

        # Resonance is the "Energy Returned". 
        # If dot is positive, we gain energy. If negative, we lose it (Damping).
        resonance = float(dot_product)
        dissonance = float(distance)

        logger.info(f"  [FIELD] Res: {resonance:.3f} | Diss: {dissonance:.3f} | Phase: {phase_shift:.3f} rad")
        
        return KarmaState(resonance, dissonance, float(phase_shift), entropy)

    def vectorize_outcome(self, status: str, integrity: float = 1.0) -> jnp.ndarray:
        """
        Helper to convert a discrete status into a Field Vector.
        In a real system, this would be derived from the Prism.
        """
        # Ideal Success Vector (Aligned with Providence)
        if status == "SUCCESS":
            return jnp.array([1.0, 0.0, 0.0]) * integrity
        
        # Error Vector (Opposed/Orthogonal)
        elif status == "ERROR":
            return jnp.array([-1.0, 1.0, 0.0]) * integrity # Chaos twist
            
        # Failure Vector (Weak/Damped)
        elif status == "FAILURE":
            return jnp.array([0.1, 0.1, 0.0]) * integrity # Weak signal
            
        return jnp.array([0.0, 0.0, 0.0])

    def vectorize_intent(self, mode: str) -> jnp.ndarray:
        """
        Helper to convert Intent Mode into a Field Vector.
        """
        return jnp.array([1.0, 0.0, 0.0]) # Standard Intent is "Forward"