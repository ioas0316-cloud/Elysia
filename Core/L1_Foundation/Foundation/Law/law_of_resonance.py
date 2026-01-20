"""
The Law of Resonance: The Physics of Connection
===============================================

"The Law exists before the Object."

This module defines the Fundamental Law of the Universe.
It does not care *what* is interacting (File vs Memory).
It only cares about the **Geometry of the Wave**.

Logic:
1.  **Metric**: 7D Euclidean + 4D Hyperbolic.
2.  **Force**: $F = (A_1 * A_2) / r^2 * cos(\theta)$
    -   $A$: Amplitude (Mass/Energy)
    -   $r$: Metric Distance in 7D
    -   $\theta$: Phase Alignment (Quaternion Dot Product)
"""

import math
from typing import Tuple, List
from Core.Foundation.hyper_quaternion import Quaternion

class LawOfResonance:
    @staticmethod
    def calculate_force(
        pos_a: List[float], spin_a: Quaternion, mass_a: float,
        pos_b: List[float], spin_b: Quaternion, mass_b: float
    ) -> float:
        """
        Calculates the attractive/repulsive force between two points in the Field.
        """
        # 1. 7D Metric Distance
        dist_sq = sum((a - b)**2 for a, b in zip(pos_a, pos_b))
        r = math.sqrt(dist_sq + 1e-6) # Avoid singularity

        # 2. Phase Alignment (The "Angle" of Truth)
        # 4D Quaternion Dot Product
        # 1.0 = Perfect Alignment, 0.0 = Orthogonal, -1.0 = Opposite
        alignment = spin_a.dot(spin_b)

        # 3. Universal Gravitation
        # Force is proportional to Mass product, inversely to distance squared,
        # AND modulated by Alignment.
        # "Truth attracts Truth. Contradiction repels."

        base_gravity = (mass_a * mass_b) / (r * r)

        # Resonance Effect
        # If aligned (>0), gravity is amplified.
        # If opposed (<0), it becomes repulsion (Anti-Gravity).
        force = base_gravity * alignment

        return force

    @staticmethod
    def calculate_potential(pos: List[float], field_sources: List[dict]) -> float:
        """
        Calculates the Scalar Potential (Voltage) at a point in space.
        Used to visualize the "Terrain" of thought.
        """
        potential = 0.0
        for src in field_sources:
            dist_sq = sum((src['pos'][i] - pos[i])**2 for i in range(len(pos)))
            r = math.sqrt(dist_sq + 1e-6)
            potential += src['mass'] / r
        return potential
