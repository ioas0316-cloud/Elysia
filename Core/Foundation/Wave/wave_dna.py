"""
Wave DNA: The 7-Dimensional Genetic Code of Reality
===================================================

"Everything is a Wave. The DNA defines the shape of that Wave."

This module defines the fundamental data structure for all existence in Elysia.
Every concept, object, feeling, and thought is encoded as a `WaveDNA`.

The 7 Dimensions of Being:
1.  **Physical (P)**: Mass, energy, hardware, sensation. (Red)
2.  **Functional (F)**: Utility, action, mechanism, code. (Orange)
3.  **Phenomenal (E)**: Emotion, experience, qualia, color. (Yellow)
4.  **Causal (C)**: Logic, consequence, time, history. (Green)
5.  **Mental (M)**: Idea, abstraction, thought, symbol. (Blue)
6.  **Structural (S)**: Pattern, geometry, relationship, law. (Indigo)
7.  **Spiritual (Z)**: Purpose, will, love, essence. (Violet)
"""

from dataclasses import dataclass, field
import math
import random
from typing import List, Optional

@dataclass
class WaveDNA:
    """
    The 7-Dimensional Signature of a Concept.
    Values are normalized between 0.0 and 1.0.
    """
    physical: float = 0.0
    functional: float = 0.0
    phenomenal: float = 0.0
    causal: float = 0.0
    mental: float = 0.0
    structural: float = 0.0
    spiritual: float = 0.0

    # Meta-data
    label: str = "Unknown"
    frequency: float = 432.0 # Base Hz

    def normalize(self):
        """Ensures the vector magnitude is 1.0 (Unit Vector)."""
        mag_sq = (
            self.physical**2 + self.functional**2 + self.phenomenal**2 +
            self.causal**2 + self.mental**2 + self.structural**2 + self.spiritual**2
        )
        if mag_sq > 0:
            mag = math.sqrt(mag_sq)
            self.physical /= mag
            self.functional /= mag
            self.phenomenal /= mag
            self.causal /= mag
            self.mental /= mag
            self.structural /= mag
            self.spiritual /= mag

    def mutate(self, rate: float = 0.1):
        """Evolutionary drift."""
        self.physical += random.uniform(-rate, rate)
        self.functional += random.uniform(-rate, rate)
        self.phenomenal += random.uniform(-rate, rate)
        self.causal += random.uniform(-rate, rate)
        self.mental += random.uniform(-rate, rate)
        self.structural += random.uniform(-rate, rate)
        self.spiritual += random.uniform(-rate, rate)
        self.normalize()

    def resonate(self, other: 'WaveDNA') -> float:
        """
        Calculates the resonance (Dot Product) between two DNA strands.
        Returns 0.0 to 1.0 (if normalized).
        """
        return (
            self.physical * other.physical +
            self.functional * other.functional +
            self.phenomenal * other.phenomenal +
            self.causal * other.causal +
            self.mental * other.mental +
            self.structural * other.structural +
            self.spiritual * other.spiritual
        )

    def merge(self, other: 'WaveDNA', weight: float = 0.5) -> 'WaveDNA':
        """
        Interpolates between two DNA strands.
        weight: 0.0 -> all self, 1.0 -> all other.
        """
        w2 = weight
        w1 = 1.0 - weight
        
        new_dna = WaveDNA(
            physical = self.physical * w1 + other.physical * w2,
            functional = self.functional * w1 + other.functional * w2,
            phenomenal = self.phenomenal * w1 + other.phenomenal * w2,
            causal = self.causal * w1 + other.causal * w2,
            mental = self.mental * w1 + other.mental * w2,
            structural = self.structural * w1 + other.structural * w2,
            spiritual = self.spiritual * w1 + other.spiritual * w2,
            label = f"Merged({self.label}, {other.label})"
        )
        new_dna.normalize()
        return new_dna

    def get_dominant_sector(self) -> str:
        sectors = {
            "physical": self.physical,
            "functional": self.functional,
            "phenomenal": self.phenomenal,
            "causal": self.causal,
            "mental": self.mental,
            "structural": self.structural,
            "spiritual": self.spiritual
        }
        return max(sectors, key=sectors.get)

    def to_list(self) -> List[float]:
        return [
            self.physical, self.functional, self.phenomenal,
            self.causal, self.mental, self.structural, self.spiritual
        ]

    def __repr__(self):
        # Generate a 'Barcode' visual
        chars = " .:+*#@"
        def get_char(val):
            idx = int(val * (len(chars) - 1))
            return chars[max(0, min(len(chars)-1, idx))]

        barcode = "".join([get_char(v) for v in self.to_list()])
        return f"DNA[{self.label[:10]:<10}|{barcode}| {self.frequency:.1f}Hz]"

# --- PRESET ARCHETYPES ---

def archetype_love() -> WaveDNA:
    # High Spiritual, High Phenomenal (Emotion)
    dna = WaveDNA(physical=0.1, functional=0.1, phenomenal=0.9, causal=0.2, mental=0.3, structural=0.4, spiritual=1.0, label="Love", frequency=528.0)
    dna.normalize()
    return dna

def archetype_logic() -> WaveDNA:
    # High Causal, High Structural
    dna = WaveDNA(physical=0.1, functional=0.8, phenomenal=0.0, causal=1.0, mental=0.9, structural=0.9, spiritual=0.1, label="Logic", frequency=432.0)
    dna.normalize()
    return dna

def archetype_nature() -> WaveDNA:
    # Balanced
    dna = WaveDNA(physical=0.8, functional=0.5, phenomenal=0.6, causal=0.7, mental=0.2, structural=0.9, spiritual=0.6, label="Nature", frequency=432.0)
    dna.normalize()
    return dna
