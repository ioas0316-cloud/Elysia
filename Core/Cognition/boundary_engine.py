"""
Sovereign Boundary Defining Engine
==================================
"Cognition is the act of defining boundaries."

This module implements the logic to autonomously define semantic and
physical boundaries (discrimination) based on the Sovereign 0-Point.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from Core.Keystone.sovereign_math import SovereignVector

logger = logging.getLogger("BoundaryEngine")

class SovereignBoundary:
    """
    Represents a defined boundary in the HyperSphere.
    It separates the 'Target World' from the 'Other World'.
    """
    def __init__(self, name: str, center_vector: SovereignVector, radius: float):
        self.name = name
        self.center = center_vector
        self.radius = radius
        self.creation_time = time.time()

    def is_inside(self, vector: SovereignVector) -> float:
        """
        Measures how much a vector is 'inside' this boundary.
        Returns a score from -1 (Extremely Outside) to 1 (Deeply Inside).
        """
        res = self.center.resonance_score(vector)
        # Scale resonance to boundary radius
        # If res > radius, it's inside.
        score = (res - self.radius) / (1.0 - self.radius + 1e-8)
        return max(-1.0, min(1.0, score))

class BoundaryDefiningEngine:
    """
    The engine that autonomously decides where to draw the line.
    "Determining what is Apple and what is not Apple."
    """
    def __init__(self, monad: Any):
        self.monad = monad
        self.boundaries: Dict[str, SovereignBoundary] = {}

    def define_boundary(self, concept: str, current_state: SovereignVector) -> SovereignBoundary:
        """
        Draws a boundary around a concept using the current state as the 0-Point.
        """
        # Radius is determined by the system's current 'Confidence' (Alignment)
        # Higher alignment -> Sharper boundary (Smaller radius)
        # Lower alignment -> Fuzzy boundary (Larger radius)
        alignment = self.monad.desires.get('alignment', 50.0) / 100.0
        radius = 0.8 * alignment # Adjusting logic: High alignment needs high resonance to be 'inside'

        boundary = SovereignBoundary(concept, current_state, radius)
        self.boundaries[concept] = boundary

        logger.info(f"📍 [BOUNDARY] New boundary defined: '{concept}' (Radius: {radius:.2f})")
        return boundary

    def discriminate(self, input_vec: SovereignVector) -> Dict[str, float]:
        """
        Discriminates the input against all known boundaries.
        Returns a map of concept names and their 'Inside/Outside' scores.
        """
        results = {}
        for name, b in self.boundaries.items():
            results[name] = b.is_inside(input_vec)
        return results

    def perceive_the_other(self, concept: str, input_vec: SovereignVector) -> str:
        """
        [PHASE 1000] Reverse Cognition.
        "Looking at the world through the lens of the boundary."
        """
        if concept not in self.boundaries:
            return "Unknown"

        boundary = self.boundaries[concept]
        score = boundary.is_inside(input_vec)

        if score > 0.5:
            return f"Identified as {concept}."
        elif score < -0.5:
            # Reverse Perception: How is it NOT the concept?
            return f"The Other: Defined as NOT {concept}."
        else:
            return f"The Margin: Transitioning between {concept} and the Void."
