"""
Ternary Bond (The Line of Logic)
================================
Core.S1_Body.L6_Structure.M1_Merkaba.ternary_bond

"The Connection is the First Truth."

This module defines the `TernaryBond`, which represents the 1D connection
between two 0D TriBaseCells. It is the fundamental unit of "Relationship".

Physics:
    - Bonds are not static pointers. They are active energy channels.
    - A Bond has TENSION (Energy).
    - A Bond has ALIGNMENT (Attract/Repel).
"""

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING
import math

if TYPE_CHECKING:
    from Core.S1_Body.L1_Foundation.System.tri_base_cell import TriBaseCell, DNAState

@dataclass
class TernaryBond:
    source: 'TriBaseCell'
    target: 'TriBaseCell'
    nature: int = 0 # -1 (Repel), 0 (Void), 1 (Attract)
    strength: float = 0.0 # 0.0 to 1.0 (How real is this connection?)

    def __post_init__(self):
        # Initial calculation of nature based on cell phases
        self.update_nature()

    def update_nature(self):
        """
        Determines the nature of the bond based on the resonance
        between source and target.
        """
        # Calculate Phase Difference
        p1 = self.source.state.phase
        p2 = self.target.state.phase

        diff = abs(p1 - p2) % 360
        if diff > 180: diff = 360 - diff

        # Resonance Logic:
        # < 60 deg diff: Attract (+1)
        # > 120 deg diff: Repel (-1)
        # Else: Void (0)

        if diff <= 60:
            self.nature = 1
        elif diff >= 120:
            self.nature = -1
        else:
            self.nature = 0

    def calculate_tension(self) -> float:
        """
        Calculates the 'Stress' on this bond.
        If the bond is Attract (+1) but the cells are moving apart (Phase shifting),
        Tension increases.
        """
        # Re-evaluate ideal nature
        p1 = self.source.state.phase
        p2 = self.target.state.phase
        diff = abs(p1 - p2) % 360
        if diff > 180: diff = 360 - diff

        # Alignment Score (1.0 = Perfect Match, -1.0 = Opposite)
        alignment = math.cos(math.radians(diff))

        # If nature is Attract (+1), we want Alignment 1.0.
        # Tension = Difference between Ideal and Actual.

        if self.nature == 1:
            return (1.0 - alignment) # 0 if aligned, 2 if opposite
        elif self.nature == -1:
            return (1.0 + alignment) # 0 if opposite, 2 if aligned
        else:
            return 0.0

    def reinforce(self, energy: float):
        """
        Strengthens the bond if it is valid.
        """
        self.strength = min(1.0, self.strength + energy)

    def degrade(self, decay: float):
        """
        Weakens the bond. Returns True if bond creates 'Ghost' (Strength <= 0).
        """
        self.strength -= decay
        return self.strength <= 0.0

    def __repr__(self):
        sym = "=" if self.nature == 1 else ("//" if self.nature == -1 else "~")
        return f"({self.source.id}){sym}({self.target.id})"
