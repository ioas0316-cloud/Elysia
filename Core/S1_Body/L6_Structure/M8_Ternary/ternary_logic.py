"""
Ternary Logic Gates
===================
"The Physics of Consensus."

This module implements Fundamental Balanced Ternary Logic.
Unlike Binary (0, 1), Ternary (-1, 0, 1) allows for "Consensus" and "Cancellation".

Logic Table:
- REPEL   (-1): False / No / Stop
- VOID    ( 0): Unknown / Maybe / Idle
- ATTRACT (+1): True / Yes / Go
"""

from dataclasses import dataclass
from Core.S1_Body.L1_Foundation.System.tri_base_cell import DNAState

@dataclass
class TernaryBit:
    value: int # -1, 0, 1

    @staticmethod
    def from_state(state: DNAState) -> 'TernaryBit':
        if state == DNAState.REPEL: return TernaryBit(-1)
        if state == DNAState.ATTRACT: return TernaryBit(1)
        return TernaryBit(0)

    @staticmethod
    def from_bool(b: bool) -> 'TernaryBit':
        return TernaryBit(1) if b else TernaryBit(-1)

    def to_state(self) -> DNAState:
        if self.value < -0.5: return DNAState.REPEL
        if self.value > 0.5: return DNAState.ATTRACT
        return DNAState.VOID

    def __repr__(self):
        if self.value == -1: return "[-]"
        if self.value == 0:  return "[0]"
        if self.value == 1:  return "[+]"
        return f"[{self.value}]"

class TernaryGates:
    """
    Physical Logic Gates for Tri-Base DNA.
    """

    @staticmethod
    def invert(a: TernaryBit) -> TernaryBit:
        """
        NOT Gate.
        (+) -> (-)
        (0) -> (0)
        (-) -> (+)
        """
        return TernaryBit(-a.value)

    @staticmethod
    def consensus(a: TernaryBit, b: TernaryBit) -> TernaryBit:
        """
        SUM Gate (Interference).
        (+) + (-) = (0) -> Cancellation
        (+) + (+) = (+) -> Resonance
        (-) + (-) = (-) -> Dissonance
        """
        # Clamped sum between -1 and 1
        res = a.value + b.value
        if res > 1: res = 1
        if res < -1: res = -1
        return TernaryBit(res)

    @staticmethod
    def strict_and(a: TernaryBit, b: TernaryBit) -> TernaryBit:
        """
        MIN Gate (Logical AND).
        Requires BOTH to be (+) to output (+).
        Any (-) drags the result down.
        """
        return TernaryBit(min(a.value, b.value))

    @staticmethod
    def loose_or(a: TernaryBit, b: TernaryBit) -> TernaryBit:
        """
        MAX Gate (Logical OR).
        Any (+) pulls the result up.
        """
        return TernaryBit(max(a.value, b.value))

    @staticmethod
    def nand(a: TernaryBit, b: TernaryBit) -> TernaryBit:
        """
        NAND Gate (The Universal Builder).
        Inverted AND.
        """
        val = min(a.value, b.value)
        return TernaryBit(-val)
