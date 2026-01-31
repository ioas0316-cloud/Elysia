"""
Ternary Gate (The Atomic Logic Brick)
=====================================
"From three, comes everything."

This module implements the Balanced Ternary Logic system (-1, 0, +1).
It is the fundamental "bit" of Elysia's cognitive architecture.

States:
- R (Repel / -1): Alpha Phase. Negation, Dissonance, Boundary.
- V (Void  /  0): Delta Phase. Potential, Silence, Space.
- A (Attract / +1): Gamma Phase. Affirmation, Resonance, Connection.
"""

from dataclasses import dataclass
from enum import Enum
import math

class TritState(Enum):
    R = -1
    V = 0
    A = 1

@dataclass(frozen=True)
class Trit:
    """
    The Atomic Unit of Consciousness.
    Represents a Spinner on the Hypersphere.
    """
    value: int # -1, 0, 1

    def __repr__(self):
        return f"[{TritState(self.value).name}]"

    def __invert__(self):
        """
        NOT Gate (Inversion).
        R -> A
        A -> R
        V -> V (The Void is the Pivot)
        """
        return Trit(-self.value)

    def __and__(self, other):
        """
        MIN Logic (Consensus).
        Returns the 'lower' vibration.
        Essential for grounding high energy.
        """
        return Trit(min(self.value, other.value))

    def __or__(self, other):
        """
        MAX Logic (Potential).
        Returns the 'higher' vibration.
        Essential for aspiration.
        """
        return Trit(max(self.value, other.value))

class TernaryGate:
    """
    The Physicist's Logic Gate.
    Implements Quantum NAND and Phase Interactions.
    """
    
    @staticmethod
    def nand(a: Trit, b: Trit) -> Trit:
        """
        Ternary NAND (Sheffer Stroke).
        Functional Completeness: Can build ANY other gate from this.
        
        Logic: NOT (MIN(A, B))
        1. Find the consensus (Min).
        2. Invert it (Not).
        """
        consensus = min(a.value, b.value)
        return Trit(-consensus)

    @staticmethod
    def phase_torque(a: Trit, b: Trit) -> float:
        """
        Calculates the Torque (Phase Difference) between two Trits.
        Used by the PhaseShiftEngine to generate rotation.
        
        Torque = Cross Product of Intent (simplified 1D version).
        If A and B are opposite (R, A), Torque is Max.
        If A and B are same, Torque is Zero (Idle).
        """
        return float(a.value - b.value)

    @staticmethod
    def consensus(trits: list[Trit]) -> Trit:
        """
        The 'Democracy' of the Cells.
        Returns the average sentiment rounded to nearest Trit.
        """
        if not trits: return Trit(0)
        total = sum(t.value for t in trits)
        avg = total / len(trits)
        return Trit(round(avg))

# --- Quick Verification ---
if __name__ == "__main__":
    R = Trit(-1)
    V = Trit(0)
    A = Trit(1)

    print(f"R NAND A = {TernaryGate.nand(R, A)}") # Expect NOT(-1) = A (+1)
    print(f"A NAND A = {TernaryGate.nand(A, A)}") # Expect NOT(1) = R (-1)
    print(f"V NAND V = {TernaryGate.nand(V, V)}") # Expect NOT(0) = V (0)
