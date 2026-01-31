"""
Tri-Base DNA Cell (The Atomic Unit)
===================================
"The Smallest Unit of Will."

This module defines the fundamental 'Cell' of the Elysia system.
Instead of abstract floating point vectors, the system is composed of
discrete Tri-Base DNA units: R (Repel), V (Void), A (Attract).

Philosophy:
-----------
- **R (Repel / -1)**: Dissonance, Boundary, "No", Separation. Phase: 240 degrees.
- **V (Void / 0)**: Potential, Silence, "Wait", Space. Phase: 0 degrees.
- **A (Attract / +1)**: Resonance, Connection, "Yes", Fusion. Phase: 120 degrees.

The macroscopic behavior of the Rotor and Monad emerges from the
statistical alignment of these microscopic cells.
"""

import math
import random
from dataclasses import dataclass
from enum import Enum

class DNAState(Enum):
    REPEL = -1
    VOID = 0
    ATTRACT = 1

    @property
    def symbol(self) -> str:
        if self == DNAState.REPEL: return "R"
        if self == DNAState.VOID: return "V"
        return "A"

    @property
    def phase(self) -> float:
        """Returns phase in degrees."""
        if self == DNAState.REPEL: return 240.0
        if self == DNAState.VOID: return 0.0 # Or undefined/floating
        return 120.0

    @staticmethod
    def from_symbol(sym: str) -> 'DNAState':
        sym = sym.upper()
        if sym == 'R': return DNAState.REPEL
        if sym == 'A': return DNAState.ATTRACT
        return DNAState.VOID

@dataclass
class TriBaseCell:
    """
    A single unit of Sovereign Memory.
    """
    id: int
    state: DNAState = DNAState.VOID
    energy: float = 1.0 # Health/Stability of this cell

    def __post_init__(self):
        # Random initialization if needed, but usually starts as Void
        pass

    def mutate(self, new_state: DNAState):
        """Changes the state of the cell."""
        self.state = new_state

    def get_vector(self) -> tuple[float, float]:
        """
        Returns the (x, y) vector representation of this cell on the phase plane.
        """
        if self.state == DNAState.VOID:
            return (0.0, 0.0)

        # Convert phase to radians
        rad = math.radians(self.state.phase)
        return (math.cos(rad) * self.energy, math.sin(rad) * self.energy)

    def resonate(self, input_phase: float) -> float:
        """
        Calculates resonance with an external phase input.
        Returns: +1.0 (Perfect Match) to -1.0 (Perfect Opposite)
        """
        if self.state == DNAState.VOID:
            return 0.0

        diff = abs(input_phase - self.state.phase) % 360
        if diff > 180: diff = 360 - diff

        # Convert difference to cosine similarity (1 at 0, -1 at 180)
        # diff 0 -> 1.0
        # diff 60 -> 0.5
        # diff 120 -> -0.5
        # diff 180 -> -1.0
        return math.cos(math.radians(diff))

    def __repr__(self):
        return f"[{self.state.symbol}]"
