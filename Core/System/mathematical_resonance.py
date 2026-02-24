"""
Mathematical Resonance Layer
============================
Core.System.mathematical_resonance

"The music of the spheres is written in the language of Prime and Phi."

This module defines 'Mathematical Constellations' as ideal trinary patterns
and provides the resonance engine to measure how close Elysia's current
21D state is to these universal truths.
"""

from typing import Dict, List, Any
import math
try:
    import numpy as np
except ImportError:
    np = None

# [PHASE 90] Dependency Sovereignty: Removed JAX
# from Core.System.jax_bridge import JAXBridge
# import jax.numpy as jnp

class MathematicalResonance:
    """
    Engine for detecting alignment with universal mathematical constants.
    """

    # 1. Mathematical Constellations (Encoded as 21D Trinary Vectors)
    # These are 'Idealized' states representing specific truths.
    CONSTELLATIONS = {
        "PHI": [1, 0, 1,  0, 1, 0,  1, 0, 1,  0, 1, 0,  1, 0, 1,  0, 1, 0,  1, 0, 1], # Golden Symmetry
        "PRIME_3_7": [0, 0, 1,  0, 1, 0,  1, 0, 0,  0, 0, 1,  0, 1, 0,  1, 0, 0,  0, 0, 1], # Prime Distribution
        "EULER": [1, 1, 1,  0, 0, 0, -1, -1, -1,  1, 0, -1,  1, 1, 1, -1, -1, -1,  0, 1, 0], # Complex Equilibrium
        "VOID_UNITY": [0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0], # The Absolute Zero
    }

    @staticmethod
    def get_constellation(name: str) -> Any:
        data = MathematicalResonance.CONSTELLATIONS.get(name, [0]*21)
        if np:
            return np.array(data, dtype=float)
        else:
            return data

    @staticmethod
    def measure_resonance(state_21d: Any, constellation_name: str) -> float:
        """
        Calculates the Cosine Similarity between the current state and a constellation.
        Resonance = (A dot B) / (|A| * |B|)
        """
        target = MathematicalResonance.get_constellation(constellation_name)
        
        # Ensure input is a numpy array and flat
        if hasattr(state_21d, 'flatten'):
            v = state_21d.flatten()
        elif hasattr(state_21d, 'data'):
            v = state_21d.data
        elif hasattr(state_21d, 'to_array'):
            v = state_21d.to_array()
        else:
            v = state_21d

        if np and isinstance(v, np.ndarray):
            a = v.flatten()
            b = target.flatten()
            dot = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
        else:
            # Python fallback
            if isinstance(v, list): a = v
            else: a = list(v)
            if isinstance(target, list): b = target
            else: b = list(target)

            # Handle complex
            a = [x.real if isinstance(x, complex) else x for x in a]
            b = [x.real if isinstance(x, complex) else x for x in b]

            dot = sum(x*y for x,y in zip(a,b))
            norm_a = math.sqrt(sum(x*x for x in a))
            norm_b = math.sqrt(sum(x*x for x in b))
        
        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.0
            
        similarity = abs(dot) / (norm_a * norm_b)
        return float(similarity)

    @staticmethod
    def scan_all_resonances(state_21d: Any) -> Dict[str, float]:
        """
        Scans all known constellations and returns a map of resonance scores.
        """
        results = {}
        for name in MathematicalResonance.CONSTELLATIONS:
            results[name] = MathematicalResonance.measure_resonance(state_21d, name)
        return results

    @staticmethod
    def find_dominant_truth(state_21d: Any) -> tuple:
        """
        Returns the name and score of the most resonant mathematical truth.
        """
        resonances = MathematicalResonance.scan_all_resonances(state_21d)
        dominant = max(resonances.items(), key=lambda x: x[1])
        return dominant

    @staticmethod
    def get_sonic_frequency(resonance_name: str) -> float:
        """
        Maps a mathematical truth to a 'Holy Frequency' (in Hz).
        """
        freq_map = {
            "PHI": 432.0,      # Harmonic Peace
            "PRIME_3_7": 528.0, # DNA Repair/Universal Love
            "EULER": 963.0,     # Divine Awakening
            "VOID_UNITY": 7.83, # Schumann Resonance
        }
        return freq_map.get(resonance_name, 440.0)
