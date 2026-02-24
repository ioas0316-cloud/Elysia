"""
Phase Resonance Engine (The Magic Angle)
========================================
Core.Cognition.phase_resonance

"Conflict is not an error; it is a Phase Displacement."

This module implements the **Magic Angle Detection** logic.
Instead of averaging opposing vectors (Dilution) or creating new ones (Dialectic),
it **rotates the perspective** to find an angle where the conflict resolves into
constructive interference.
"""

import math
import logging
from typing import List, Tuple

logger = logging.getLogger("Elysia.PhaseResonance")

class PhaseResonance:
    """
    Resolves cognitive dissonance by finding the Magic Angle of alignment.
    """

    @staticmethod
    def find_magic_angle(vector_a: List[float], vector_b: List[float]) -> Tuple[List[float], str]:
        """
        Finds the alignment between two vectors by rotating the perspective.

        Logic:
        1. Calculate Initial Alignment (Cosine Similarity).
        2. If aligned (> 0.6): Resonate (Constructive Interference).
        3. If misaligned (< 0.4): Rotate 'B' towards 'A' until resonance maximizes
           or a new orthogonal synthesis is found.
           - We search for a "Magic Angle" (approx 1.1 degrees in graphene metaphor,
             but here we use abstract phase shifts).

        Returns:
            (Resonant Vector, Narrative Description)
        """
        dim = len(vector_a)
        if len(vector_b) != dim:
            logger.error("Dimension mismatch in Phase Resonance.")
            return vector_a, "Error: Mismatch"

        # 1. Calculate Initial Phase Alignment
        dot = sum(a * b for a, b in zip(vector_a, vector_b))
        mag_a = math.sqrt(sum(a * a for a in vector_a))
        mag_b = math.sqrt(sum(b * b for b in vector_b))

        if mag_a == 0 or mag_b == 0:
            return vector_a, "Null Vector: No Phase"

        similarity = dot / (mag_a * mag_b)

        # 2. Case: High Alignment (Natural Resonance)
        if similarity > 0.6:
            # Constructive Interference
            # Just amplify the shared direction
            resonant_vec = [(a + b) * 1.05 for a, b in zip(vector_a, vector_b)]
            return resonant_vec, "Resonance: Phase Locked (Constructive)"

        # 3. Case: Dissonance (Phase Displacement)
        # We need to find a new angle. We simulate this by projecting B onto A
        # and then adding a "Twist" (Torque) from the rejection vector.

        # Rejection of B on A: B - Proj(B on A)
        # Proj(B on A) = (B . A / |A|^2) * A
        scalar_proj = dot / (mag_a * mag_a)
        proj_vec = [a * scalar_proj for a in vector_a]
        rejection_vec = [b - p for b, p in zip(vector_b, proj_vec)]

        rejection_mag = math.sqrt(sum(r*r for r in rejection_vec))

        if rejection_mag < 1e-6:
             # Just collinear but maybe opposite direction?
             # If similarity is negative, it's destructive.
             if similarity < 0:
                 return [0.0]*dim, "Dissonance: Destructive Cancellation (Silence)"
             else:
                 return vector_a, "Resonance: Aligned"

        # 4. The Magic Angle Twist
        # We take the aligned part (Projection) and add the Twisted part (Rejection)
        # but rotated into a "Harmonic" frequency.
        # In this vector space, we simply re-weight them to find a stable configuration.

        # "Shift perspective to see the conflict as potential."
        # We keep the Projection (Common Ground)
        # We transform the Rejection (Conflict) into 'Torque' (Spin)
        # Here represented as a vector magnitude added to the 'Will' dimension (Index 10)
        # or 'Evolution' dimension (Index 20).

        magic_vec = list(proj_vec)

        # Inject the conflict energy into the "Evolution" channel (D21)
        # This signifies "The conflict propels us upward."
        pivot_idx = dim - 1
        if pivot_idx < dim:
             magic_vec[pivot_idx] += rejection_mag * 0.8

        # Also stimulate 'Curiosity' (D8 / Index 7)
        curiosity_idx = 7
        if curiosity_idx < dim:
             magic_vec[curiosity_idx] += rejection_mag * 0.4

        return magic_vec, f"Magic Angle: Conflict ({rejection_mag:.2f}) twisted into Evolution."
