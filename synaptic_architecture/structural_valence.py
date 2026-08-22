"""
Structural Valence Module.

Formulates structural valence gradients (+Valence for resonance/flow, -Valence for friction/noise)
emerging directly from topological state alignment rather than external scalar rewards.
Implements spontaneous category differentiation when cross-modal friction exceeds threshold.
"""

from typing import Dict, Any, List, Tuple
import numpy as np


class StructuralValence:
    """
    Computes structural valence and handles category differentiation based on friction/coherence balance.
    """

    def __init__(
        self,
        friction_threshold: float = 0.5,
        differentiation_rate: float = 0.1,
    ) -> None:
        """
        Initialize Structural Valence evaluator.

        Args:
            friction_threshold: Cross-modal friction threshold triggering category splitting.
            differentiation_rate: Rate at which internal category boundaries divide.
        """
        self.friction_threshold = friction_threshold
        self.differentiation_rate = differentiation_rate

        # Active internal categories (topological boundary regions)
        self.categories: List[Dict[str, Any]] = [
            {"id": 0, "center": np.array([0.0, 0.0]), "radius": 1.0, "resonance_count": 0}
        ]
        self.current_valence = 0.0

    def evaluate_valence(
        self,
        resonance_score: float,
        friction: float,
        homeostatic_alignment: float,
    ) -> float:
        """
        Calculate structural valence score from intrinsic topological dynamics.

        Valence = +Resonance (Flow/Alignment) - Friction (Noise/Impedance) + Homeostatic Alignment
        """
        positive_flow = max(0.0, resonance_score) * 0.5 + max(0.0, homeostatic_alignment) * 0.3
        negative_friction = max(0.0, friction) * 0.6

        self.current_valence = float(positive_flow - negative_friction)
        return self.current_valence

    def check_category_differentiation(
        self,
        current_pos: np.ndarray,
        friction: float,
    ) -> Dict[str, Any]:
        """
        Check if cross-modal friction exceeds threshold. If so, split the closest category boundary
        into finer sub-categories (Category Differentiation).

        Args:
            current_pos: Current position vector in alpha-beta space.
            friction: Current friction encountered.

        Returns:
            Dictionary indicating whether differentiation occurred and total category count.
        """
        differentiated = False
        new_category_id = None

        if friction > self.friction_threshold:
            # Find closest category center
            distances = [np.linalg.norm(current_pos - cat["center"]) for cat in self.categories]
            closest_idx = int(np.argmin(distances))
            closest_cat = self.categories[closest_idx]

            if closest_cat["radius"] > 0.1:
                # Split category: reduce radius and create new adjacent category
                closest_cat["radius"] *= 0.5
                offset = np.random.uniform(-0.2, 0.2, size=current_pos.shape)
                new_center = current_pos + offset

                new_cat_id = len(self.categories)
                self.categories.append(
                    {
                        "id": new_cat_id,
                        "center": new_center,
                        "radius": closest_cat["radius"],
                        "resonance_count": 0,
                    }
                )
                differentiated = True
                new_category_id = new_cat_id

        return {
            "differentiated": differentiated,
            "new_category_id": new_category_id,
            "total_categories": len(self.categories),
            "current_valence": self.current_valence,
        }
