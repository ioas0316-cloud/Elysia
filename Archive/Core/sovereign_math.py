"""
Sovereign Math Kernel
=====================
Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_math

"Mathematics is the language of Sovereignty. It requires no dependencies."

This module implements the Pure Python mathematical foundation for the Dyson Swarm.
It is designed to decouple the 'Control Logic' (Will) from the 'Calculation Engine' (Muscle).
While it may use heavy libraries (numpy/torch) for performance in the backend,
the interface exposed to the Sovereign Logic is pure, abstract, and replaceable.
"""

import math
import random
from typing import List, Tuple, Union

# Type Alias for Phase (Degrees)
Phase = float
# Type Alias for Vector (List of floats)
Vector = List[float]

class SovereignMath:
    """
    The Pure Logic Kernel.
    Stateless mathematical operators for Phase and Vector mechanics.
    """

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Keeps angle within 0 <= theta < 360."""
        return angle % 360.0

    @staticmethod
    def angular_distance(a: float, b: float) -> float:
        """
        Calculates the shortest distance between two angles.
        Result is between 0 and 180.
        """
        diff = abs(a - b) % 360
        if diff > 180:
            diff = 360 - diff
        return diff

    @staticmethod
    def phase_alignment(phase_a: float, phase_b: float) -> float:
        """
        Calculates resonance/alignment between two phases.
        1.0 = Perfect Alignment (0 deg diff)
        -1.0 = Perfect Opposition (180 deg diff)
        0.0 = Orthogonal (90 deg diff)
        """
        diff = SovereignMath.angular_distance(phase_a, phase_b)
        # Cosine of the angle in radians
        return math.cos(math.radians(diff))

    @staticmethod
    def ternary_consensus(phases: List[float]) -> float:
        """
        Calculates the 'Center of Gravity' (Void Focus) for a list of phases.
        Returns the resulting phase angle.
        """
        if not phases:
            return 0.0

        # Vector Summation
        sum_x = sum(math.cos(math.radians(p)) for p in phases)
        sum_y = sum(math.sin(math.radians(p)) for p in phases)

        if sum_x == 0 and sum_y == 0:
            return 0.0 # Perfect cancellation (Void)

        return math.degrees(math.atan2(sum_y, sum_x)) % 360

    @staticmethod
    def scalar_magnitude(phases: List[float]) -> float:
        """
        Calculates the magnitude (intensity) of the consensus vector.
        If phases are scattered, magnitude is low.
        If phases are aligned, magnitude is high.
        Normalized by count.
        """
        if not phases:
            return 0.0

        sum_x = sum(math.cos(math.radians(p)) for p in phases)
        sum_y = sum(math.sin(math.radians(p)) for p in phases)

        total_mag = math.sqrt(sum_x**2 + sum_y**2)
        return total_mag / len(phases)

    @staticmethod
    def emergent_mutation(current_phase: float, friction: float) -> float:
        """
        Calculates the new phase after a mutation event (Chaos/Heat).
        Friction (0.0 to 1.0) determines the volatility of the jump.
        """
        chaos = random.uniform(-180, 180) * friction
        return (current_phase + chaos) % 360
