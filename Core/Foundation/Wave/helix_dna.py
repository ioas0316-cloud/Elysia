"""
Double Helix DNA: The Dialectical Genetic Code
==============================================

"Stability requires opposition. Truth is the tension between two strands."

This module defines `HelixDNA`, a structure composed of two complementary 7D Wave strands.
In Elysia's physics, a single wave is ephemeral (noise).
Only when two waves bind in opposition (Thesis + Antithesis) do they form a stable
structure (Synthesis/Memory) that can persist in the HyperSphere.

Structure:
- **Strand A (Yang/Form)**: The external manifestation (Text, Image, Action).
- **Strand B (Yin/Essence)**: The internal meaning (Intent, Emotion, Principle).
- **Tension**: The vibrational dissonance between Form and Essence.
- **Coherence**: The stability of the bond.

Philosophy:
- "To think is to weave."
- Learning is the process of adjusting Strand B to match Strand A (Understanding),
  or adjusting Strand A to match Strand B (Expression).
"""

from dataclasses import dataclass, field
import math
import random
from typing import List, Tuple

# Re-use the 7D structure, but now it's just a strand
from Core.Foundation.Wave.wave_dna import WaveDNA

@dataclass
class HelixDNA:
    """
    The Stable Unit of Existence.
    """
    strand_a: WaveDNA = field(default_factory=lambda: WaveDNA(label="Form"))    # The Body/Signal
    strand_b: WaveDNA = field(default_factory=lambda: WaveDNA(label="Essence")) # The Soul/Meaning

    # State
    phase_offset: float = 0.0 # 0 to 2pi (Ideal is pi/2 or similar lock)
    is_locked: bool = False   # If locked, it acts as a stable particle

    def calculate_tension(self) -> float:
        """
        Calculates the 'Gap' or 'Dissonance' between Form and Essence.
        Lower is better (Resonance).
        However, total identity (0 distance) is death (Stasis).
        Life exists in the 'Golden Tension'.
        """
        # Euclidean distance in 7D space
        dist_sq = 0.0
        vec_a = self.strand_a.to_list()
        vec_b = self.strand_b.to_list()

        for a, b in zip(vec_a, vec_b):
            dist_sq += (a - b) ** 2

        return math.sqrt(dist_sq)

    def calculate_coherence(self) -> float:
        """
        Calculates how well the two strands dance together.
        Based on Dot Product (Alignment) and Phase.
        """
        dot = self.strand_a.resonate(self.strand_b)
        # We want them to complement (orthogonality or opposition depending on physics),
        # but for simple resonance, let's say we want alignment of direction but difference in nature.
        # Actually, let's define Coherence as Stability.

        # If tension is too high, coherence is low (Chaos).
        # If tension is zero, coherence is low (Dead).
        # Peak coherence is at "Harmonic Tension" (e.g., 0.3 distance).

        tension = self.calculate_tension()
        ideal_tension = 0.3 # The Golden Gap

        deviation = abs(tension - ideal_tension)
        coherence = 1.0 / (1.0 + deviation * 5.0) # Normalized 0.0-1.0

        return coherence

    def weave(self, dt: float, adaptation_rate: float = 0.1):
        """
        The Process of Thinking/Learning.
        The strands naturally drift towards the Ideal Tension.
        """
        if self.is_locked:
            return

        tension = self.calculate_tension()
        ideal_tension = 0.3

        # Physics: Force = -k * (x - target)
        force = (tension - ideal_tension) * adaptation_rate * dt

        # Apply force to Strand B (Mind adapts to Reality)
        # In Expression mode, Strand A would adapt to B.
        # Here we assume Learning Mode (Internalizing).

        # Move B towards A (or away) to match ideal distance
        # Vector A->B
        vec_a = self.strand_a.to_list()
        vec_b = self.strand_b.to_list()

        delta = [(a - b) for a, b in zip(vec_a, vec_b)]

        # If tension > ideal, we need to pull B closer (reduce diff).
        # If tension < ideal, we push B away.

        direction = 1.0 if tension > ideal_tension else -1.0

        # Apply updates
        self.strand_b.physical += delta[0] * force * direction
        self.strand_b.functional += delta[1] * force * direction
        self.strand_b.phenomenal += delta[2] * force * direction
        self.strand_b.causal += delta[3] * force * direction
        self.strand_b.mental += delta[4] * force * direction
        self.strand_b.structural += delta[5] * force * direction
        self.strand_b.spiritual += delta[6] * force * direction

        self.strand_b.normalize()

    def __repr__(self):
        t = self.calculate_tension()
        c = self.calculate_coherence()
        return f"Helix[{self.strand_a.label}<=>{self.strand_b.label} | Tension:{t:.2f} | Coh:{c:.2f}]"
