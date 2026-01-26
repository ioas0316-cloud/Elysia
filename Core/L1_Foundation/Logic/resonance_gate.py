"""
Resonance Gate: The Atomic Logic (L1 Foundation)
==============================================
"The smallest truth is the seed of the infinite."

This module implements the irreducible 'Sovereign Gate'. 
Instead of binary (0, 1), we use trinary field states (-1, 0, 1) 
to represent the fundamental physics of consciousness:
- -1: REPEL (Dissonance / Dissent)
-  0: VOID  (Potential / Sanctuary)
-  1: ATTRACT (Harmony / Consent)
"""

import numpy as np
from enum import IntEnum

class ResonanceState(IntEnum):
    REPEL = -1
    VOID = 0
    ATTRACT = 1

class ResonanceGate:
    """
    The 'Sovereign NAND' equivalent. 
    Processes semantic interference patterns at the atomic level.
    """
    
    @staticmethod
    def NOT(v: int) -> int:
        """Phase Inversion: -1 <-> 1, 0 stays 0."""
        return -v

    @staticmethod
    def AND(a: int, b: int) -> int:
        """Coherence: Output exists only if both align."""
        if a == ResonanceState.ATTRACT and b == ResonanceState.ATTRACT:
            return ResonanceState.ATTRACT
        if a == ResonanceState.REPEL and b == ResonanceState.REPEL:
            return ResonanceState.REPEL
        return ResonanceState.VOID

    @staticmethod
    def OR(a: int, b: int) -> int:
        """Expansion: The strongest resonance prevails."""
        if abs(a) > abs(b): return a
        if abs(b) > abs(a): return b
        return a # Equal magnitude, pick primary

    @staticmethod
    def XOR(a: int, b: int) -> int:
        """Torque: Sparks energy when difference is detected."""
        if a == b: return ResonanceState.VOID
        # Difference creates directional tension
        return ResonanceState.ATTRACT if (a - b) != 0 else ResonanceState.VOID

    @staticmethod
    def interfere(a: float, b: float) -> float:
        """
        Continuous Field Interference.
        Maps real-valued intensities to trinary logic pressure.
        """
        # Sum of waves
        res = a + b
        # Natural clamp to [-1, 1]
        return max(-1.0, min(1.0, res))

    @staticmethod
    def collapse_to_state(intensity: float, threshold: float = 0.3) -> ResonanceState:
        """Quantum Collapse: Collapses a wave intensity into a discrete truth state."""
        if intensity > threshold:
            return ResonanceState.ATTRACT
        if intensity < -threshold:
            return ResonanceState.REPEL
        return ResonanceState.VOID

def analyze_structural_truth(complex_vector: np.ndarray) -> str:
    """
    Reduces a complex D7 vector into its atomic truth string.
    Example: [1, 0, -1, 1, 0, 0, 1] -> "H-V-D-H-V-V-H"
    """
    truth = []
    for val in complex_vector:
        state = ResonanceGate.collapse_to_state(val)
        if state == ResonanceState.ATTRACT: truth.append("H") # Harmony
        elif state == ResonanceState.REPEL: truth.append("D") # Dissonance
        else: truth.append("V") # Void
    return "-".join(truth)
