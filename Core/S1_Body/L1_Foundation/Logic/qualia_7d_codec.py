import numpy as np
from typing import List, Dict, Any, Optional
from Core.S1_Body.L1_Foundation.Logic.d7_vector import D7Vector

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

    def encode_sequence(self, vector: np.ndarray, threshold: float = 0.3) -> str:
        """
        [DNA ENCODER] 
        Converts a continuous weight vector into a trinary DNA sequence (R, V, A).
        A: Attract (Harmony)
        R: Repel (Dissonance)
        V: Void (Neutral)
        """
        from Core.S1_Body.L1_Foundation.Logic.resonance_gate import ResonanceGate, ResonanceState
        sequence = []
        for val in vector:
            state = ResonanceGate.collapse_to_state(val, threshold)
            if state == ResonanceState.ATTRACT: sequence.append("A")
            elif state == ResonanceState.REPEL: sequence.append("R")
            else: sequence.append("V")
        return "".join(sequence)

    def decode_sequence(self, sequence: str, vitality: float = 1.0) -> np.ndarray:
        """
        [DNA DECODER]
        Converts a Tri-Base DNA sequence (A, R, V) back into a 7D vector with dynamic Vitality.
        Supports legacy H (Harmony -> A) and D (Dissonance -> R).
        
        Args:
            sequence: Trinary sequence (A, R, V)
            vitality: The 'Energy Density' or 'Life Force' of this sequence.
        """
        vector = np.zeros(self.dim, dtype=np.float32)
        for i, base in enumerate(sequence[:self.dim]):
            if base in ["A", "H"]: vector[i] = 1.0
            elif base in ["R", "D"]: vector[i] = -1.0
            else: vector[i] = 0.0
            
        norm = np.linalg.norm(vector)
        if norm > 0:
            # Normalize to unit vector then scale by vitality (Energy Density)
            return (vector / norm) * vitality
        return vector

# Global Codec Instance
codec = Qualia7DCodec()
