"""
Universal Wave Encoder
======================
Translates the 'Old World' (Text, Sense) into the 'New World' (Hypersphere).

Function:
1. Maps Concepts -> Hyperspherical Coordinates (Semantics -> Location)
2. Maps Qualities -> Resonance Patterns (Feeling -> Vibration)
"""

import hashlib
import math
from typing import Tuple, Dict, Any
from Core.L5_Mental.Intelligence.Memory.hypersphere_memory import HypersphericalCoord, ResonancePattern

class UniversalWaveEncoder:
    """
    The Bridge between Digital Data and Hypersphere Memory.
    """

    def __init__(self):
        # Basis seeds for deterministic mapping
        self.logic_seed = b'logic_basis'
        self.emotion_seed = b'emotion_basis'
        self.intent_seed = b'intent_basis'

    def _hash_to_float(self, text: str, seed: bytes) -> float:
        """Determinisitcally maps text+seed to 0.0~1.0"""
        h = hashlib.sha256(seed + text.encode('utf-8')).hexdigest()
        # Take first 8 chars -> int -> normalize
        val = int(h[:8], 16)
        return val / 0xFFFFFFFF

    def encode_concept(self, concept: str) -> Tuple[HypersphericalCoord, Dict[str, Any]]:
        """
        Converts a concept string (e.g., "Apple") into:
        1. A coordinate (Where it lives in meaning-space)
        2. A default resonance pattern (Its intrinsic vibration)
        """

        # 1. Calculate Dials (Semantic Hashing)
        # In a real system, this would use an Embedding Model (BERT/SBERT)
        # Here we use deterministic hashing for stability without heavy ML models.

        # Theta (Logic): Based on the word structure
        theta_val = self._hash_to_float(concept, self.logic_seed)
        theta = theta_val * 2 * math.pi

        # Phi (Emotion): Simulated via different hash
        phi_val = self._hash_to_float(concept, self.emotion_seed)
        phi = phi_val * 2 * math.pi

        # Psi (Intent): Simulated
        psi_val = self._hash_to_float(concept, self.intent_seed)
        psi = psi_val * 2 * math.pi

        # R (Depth): Frequency of usage or abstractness (Default 1.0)
        r = 1.0

        coord = HypersphericalCoord(theta, phi, psi, r)

        # 2. Calculate Pattern
        # Omega (Rotation Speed) derived from hash
        omega_x = (theta_val - 0.5) * 2.0
        omega_y = (phi_val - 0.5) * 2.0
        omega_z = (psi_val - 0.5) * 2.0

        pattern_meta = {
            'omega': (omega_x, omega_y, omega_z),
            'phase': 0.0,
            'topology': 'point', # Single concepts are points
            'trajectory': 'static'
        }

        return coord, pattern_meta

    def encode_sensory(self, input_type: str, data: Any) -> Tuple[HypersphericalCoord, Dict[str, Any]]:
        """
        Encodes sensory data (Audio/Visual placeholder).
        """
        # Placeholder logic
        coord = HypersphericalCoord(0, 0, 0, 0.5) # Sensory is usually 'shallower' (r < 1)
        meta = {
            'topology': 'field' if input_type == 'audio' else 'plane',
            'trajectory': 'flow'
        }
        return coord, meta