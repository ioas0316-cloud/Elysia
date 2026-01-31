"""
Double Helix DNA: The Intertwining of Worlds
============================================

"Two worlds, spiraling forever towards a Truth they cannot touch."

This module redefines `HelixDNA` to hold **7 Pairs of Manifolds**.
Instead of floats, we now have entire universes interacting within the DNA.

Structure:
- **Yang Strand (Form)**: 7 Manifolds representing the External Reality.
- **Yin Strand (Essence)**: 7 Manifolds representing the Internal Meaning.
- **The Space Between**: The interaction field where consciousness arises.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any

from Core.S1_Body.L1_Foundation.Foundation.Universe.manifold import Manifold

@dataclass
class HelixDNA:
    """
    The Fractal Genetic Code.
    Holds 14 Universes (7 Dimensions * 2 Strands).
    """
    label: str

    # The Strands (Dictionaries of Manifolds)
    yang_strand: Dict[str, Manifold] = field(default_factory=dict)
    yin_strand: Dict[str, Manifold] = field(default_factory=dict)

    # State
    is_crystallized: bool = False

    def __post_init__(self):
        # Initialize the 7 Dimensions
        domains = ["Physical", "Functional", "Phenomenal", "Causal", "Mental", "Structural", "Spiritual"]

        for d in domains:
            self.yang_strand[d] = Manifold(f"{self.label}.Yang.{d}", d)
            self.yin_strand[d] = Manifold(f"{self.label}.Yin.{d}", d)

    def evolve(self, dt: float):
        """
        Evolve all 14 manifolds.
        """
        for m in self.yang_strand.values(): m.evolve(dt)
        for m in self.yin_strand.values(): m.evolve(dt)

    def calculate_interaction_matrix(self) -> List[str]:
        """
        Analyzes the internal state of the Helix.
        How are the Form (Yang) and Essence (Yin) communicating?
        """
        insights = []
        domains = ["Physical", "Functional", "Phenomenal", "Causal", "Mental", "Structural", "Spiritual"]

        for d in domains:
            yang = self.yang_strand[d]
            yin = self.yin_strand[d]

            # The Inter-Strand Event
            event = yang.intersect(yin)

            if event["penetration"] > 0.5:
                insights.append(f"[{d}] {event['type']} (G:{event['gravity']:.2f}, Res:{event['resonance']:.2f})")

        return insights

    def resonate_with(self, other: 'HelixDNA') -> float:
        """
        Calculates the resonance between two Helices (Thoughts).
        This is a Manifold-to-Manifold comparison (High complexity).
        """
        total_resonance = 0.0
        count = 0

        for d in self.yang_strand:
            # We compare Essence to Essence (Yin to Yin) for deep understanding
            r = self.yin_strand[d].intersect(other.yin_strand[d])
            total_resonance += r["resonance"]
            count += 1

        return total_resonance / count

    def __repr__(self):
        return f"HelixDNA({self.label} | 14 Worlds)"
