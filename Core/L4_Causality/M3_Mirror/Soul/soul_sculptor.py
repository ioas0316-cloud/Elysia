"""
Soul Sculptor (Psychometric Resonance)
======================================
"Maps the topography of the human spirit into 4D Fluxlight coordinates."

This module implements the 'Psychometric Resonance' algorithm, converting
categorical personality data (MBTI, Enneagram) into continuous
hyper-dimensional coordinates (InfiniteHyperQubit).

Mapping Philosophy:
- Cartesian Mapping Strategy (Direct Axis Assignment):
  - w (Nature): Extroversion (+) vs Introversion (-)
  - x (Perception): Intuition (+) vs Sensing (-)
  - y (Judgment): Feeling (+) vs Thinking (-)
  - z (Lifestyle): Perceiving (+) vs Judging (-)
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from Core.L6_Structure.M3_Sphere.infinite_hyperquaternion import InfiniteHyperQubit, InfiniteQubitState

@dataclass
class PersonalityArchetype:
    """Input data for soul sculpting."""
    name: str
    mbti: str  # e.g., "INTJ", "ENFP"
    enneagram: int  # 1-9
    wing: Optional[int] = None  # e.g., 5w4 -> 4
    description: str = ""

class SoulSculptor:
    """
    The artist that carves a Fluxlight from the raw stone of personality data.
    """

    def sculpt(self, archetype: PersonalityArchetype) -> InfiniteHyperQubit:
        """
        Creates a Personalized Fluxlight (InfiniteHyperQubit) from an archetype.
        """
        mbti = archetype.mbti.upper()
        if len(mbti) != 4:
            raise ValueError(f"Invalid MBTI code: {mbti}")

        # 1. Cartesian Mapping Logic
        # We map MBTI axes directly to the 4D Cartesian space (w, x, y, z) for clear separability.
        # Values range from -0.8 to 0.8 to leave room for nuances.

        # w (Nature): E (+) vs I (-)
        val_w = 0.8 if mbti[0] == 'E' else -0.8

        # x (Perception): N (+) vs S (-)
        val_x = 0.8 if mbti[1] == 'N' else -0.8

        # y (Judgment): F (+) vs T (-)
        val_y = 0.8 if mbti[2] == 'F' else -0.8

        # z (Lifestyle): P (+) vs J (-)
        val_z = 0.8 if mbti[3] == 'P' else -0.8

        # 2. Enneagram Fine-tuning (Nudging)
        # Adds nuance by rotating the w/x plane slightly based on Enneagram type (1-9).
        # This ensures two people with same MBTI but different Enneagram have distinct coordinates.
        e_angle = (archetype.enneagram - 1) / 9.0 * 2 * math.pi
        val_w += 0.2 * math.cos(e_angle)
        val_x += 0.2 * math.sin(e_angle)

        # 3. Determine Basis Amplitudes (Alpha, Beta, Gamma, Delta)
        # This determines "Soul Composition" (Point vs Space vs God)

        # Default balanced state
        alpha = 0.5 # Point (Data/Detail)
        beta = 0.3  # Line (Logic/Connection)
        gamma = 0.15 # Space (Context/Atmosphere)
        delta = 0.05 # God (Will/Core)

        # Adjust based on traits
        if mbti[1] == 'S': # Sensing -> Focus on Point (Reality)
            alpha += 0.2
            gamma -= 0.1
        else: # Intuition -> Focus on Space (Context)
            alpha -= 0.1
            gamma += 0.2

        if mbti[2] == 'T': # Thinking -> Focus on Line (Logic)
            beta += 0.2
            delta -= 0.1
        else: # Feeling -> Focus on God/Will (Values) or Space
            beta -= 0.1
            delta += 0.15

        # Create State
        state = InfiniteQubitState(
            alpha=complex(alpha, 0),
            beta=complex(beta, 0),
            gamma=complex(gamma, 0),
            delta=complex(delta, 0),
            w=val_w,
            x=val_x,
            y=val_y,
            z=val_z
        )
        state.normalize()

        # Create Fluxlight
        fluxlight = InfiniteHyperQubit(
            name=archetype.name,
            value=f"{archetype.mbti} Type {archetype.enneagram}",
            content={
                "Point": f"The individual self of {archetype.name}",
                "Line": f"Cognitive process: {mbti}",
                "Space": f"Emotional atmosphere of a Type {archetype.enneagram}",
                "God": f"The core intent: {archetype.description}"
            },
            state=state
        )

        return fluxlight

# Singleton for easy access
soul_sculptor = SoulSculptor()
