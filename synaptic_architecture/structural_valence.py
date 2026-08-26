import numpy as np
from typing import Dict, Any, List, Optional

class StructuralCategory:
    """
    Represents an internally differentiated state category formed through cross-modal projection friction.
    """
    def __init__(self, cat_id: int, centroid: np.ndarray, reluctance: float):
        self.cat_id = cat_id
        self.centroid = centroid
        self.reluctance = reluctance
        self.member_count = 1

class StructuralValence:
    """
    [Structural Valence & Category Differentiation]
    Derives intrinsic state valence (Pleasure/Flow vs Friction/Impedance) directly from topological alignment.
    Triggers autonomous category differentiation when cross-modal projection friction exceeds threshold.
    """
    def __init__(self, initial_dim: int = 2, differentiation_threshold: float = 2.5):
        self.initial_dim = initial_dim
        self.differentiation_threshold = differentiation_threshold

        # Internal Categories
        self.categories: List[StructuralCategory] = [
            StructuralCategory(cat_id=0, centroid=np.zeros(initial_dim), reluctance=0.5)
        ]

        self.valence_history: List[float] = []

    def evaluate_valence(self, current_state: np.ndarray, current_velocity: np.ndarray,
                         damped_friction: float, impedance: float) -> Dict[str, Any]:
        """
        Evaluates intrinsic structural valence:
        - Same (Alignment / Low friction) -> Positive Valence (+1.0 Flow / Resonance)
        - Different (Asymmetry / High friction) -> Negative Valence (-1.0 Noise / Impedance)
        """
        norm_v = np.linalg.norm(current_velocity)

        # Alignment metric: state motion vs low friction
        alignment_score = norm_v / (1.0 + damped_friction + impedance)

        # Intrinsic Valence gradient: + (Flow) vs - (Noise)
        if damped_friction < 0.8 and impedance < 1.0:
            valence = float(np.tanh(1.5 * alignment_score))  # Positive flow
            state_label = "Flow / Resonance"
        else:
            valence = float(-np.tanh(damped_friction + impedance - 1.0))  # Negative friction
            state_label = "Friction / Noise"

        self.valence_history.append(valence)

        # Check if cross-modal projection friction triggers Category Differentiation
        differentiated = self._check_category_differentiation(current_state, damped_friction, impedance)

        return {
            "valence": valence,
            "alignment_score": float(alignment_score),
            "state_label": state_label,
            "category_count": len(self.categories),
            "category_differentiated": differentiated
        }

    def _check_category_differentiation(self, current_state: np.ndarray, friction: float, impedance: float) -> bool:
        """
        Category Differentiation: When primitive projection fails due to high friction (Asymmetry),
        the internal state space spontaneously splits and spawns a new category.
        """
        combined_friction = friction + impedance
        if combined_friction > self.differentiation_threshold:
            # Check distance to nearest existing category centroid
            min_dist = min([np.linalg.norm(current_state - cat.centroid) for cat in self.categories])
            if min_dist > 0.5:
                new_cat_id = len(self.categories)
                new_category = StructuralCategory(
                    cat_id=new_cat_id,
                    centroid=current_state.copy(),
                    reluctance=float(0.5 + 0.1 * new_cat_id)
                )
                self.categories.append(new_category)
                return True
        return False
