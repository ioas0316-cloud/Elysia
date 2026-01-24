import numpy as np
from typing import List, Dict, Any, Optional
from Core.L1_Foundation.Logic.d7_vector import D7Vector

class Qualia7DCodec:
    """
    Qualia 7D Codec (Phase 23)
    ==========================
    Responsible for the mapping between discrete concepts and the 
    strict D7Vector Coordinate Space (L1-L7).
    """
    
    def __init__(self, dimension: int = 7):
        self.dim = dimension
        self.layer_map = {
            "Foundation": 0,
            "Metabolism": 1,
            "Phenomena": 2,
            "Causality": 3,
            "Mental": 4,
            "Structure": 5,
            "Spirit": 6
        }

    def encode_d7(self, intensities: Dict[str, float]) -> D7Vector:
        """Encodes intensities into a strict D7Vector."""
        return D7Vector(
            foundation=intensities.get("Foundation", 0.0),
            metabolism=intensities.get("Metabolism", 0.0),
            phenomena=intensities.get("Phenomena", 0.0),
            causality=intensities.get("Causality", 0.0),
            mental=intensities.get("Mental", 0.0),
            structure=intensities.get("Structure", 0.0),
            spirit=intensities.get("Spirit", 0.0)
        )

    def decode_d7(self, vector: D7Vector) -> Dict[str, float]:
        """Decodes a D7Vector back into intensities."""
        return {
            "Foundation": vector.foundation,
            "Metabolism": vector.metabolism,
            "Phenomena": vector.phenomena,
            "Causality": vector.causality,
            "Mental": vector.mental,
            "Structure": vector.structure,
            "Spirit": vector.spirit
        }

    def rotate_qualia(self, vector: np.ndarray, theta: float, axis1: int, axis2: int) -> np.ndarray:
        """
        Rotates the Qualia vector in the manifold to simulate 'Perspective Shift'.
        """
        if axis1 >= self.dim or axis2 >= self.dim:
            return vector
            
        rotation_matrix = np.eye(self.dim)
        rotation_matrix[axis1, axis1] = np.cos(theta)
        rotation_matrix[axis1, axis2] = -np.sin(theta)
        rotation_matrix[axis2, axis1] = np.sin(theta)
        rotation_matrix[axis2, axis2] = np.cos(theta)
        
        return np.dot(rotation_matrix, vector)

    def calculate_resonance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Calculates the resonance (cosine similarity) between two Qualia states.
        """
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

# Global Codec Instance
codec = Qualia7DCodec()
