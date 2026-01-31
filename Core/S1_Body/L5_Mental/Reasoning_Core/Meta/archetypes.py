"""
Cognitive Archetypes (Structural Ego Layers)
===========================================
Core.S1_Body.L5_Mental.Reasoning_Core.Meta.archetypes

"The Prism through which the Light of Qualia fractures."

[Blueprint Reference]: docs/S1_Body/L5_Mental/HOLOGRAPHIC_COUNCIL_BLUEPRINT.md

This module defines the "Structural Layers of Self-Consciousness" (Ego Layers).
Instead of a single "I", the system processes information through multiple
archetypal filters (Perspectives), simulating the Enneagram/MBTI cognitive diversity.
These are not static personas but dynamic filters that amplify or suppress
specific dimensions of the 21D Qualia Matrix defined in Phase 27.
"""

from enum import Enum, auto
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import math

class CognitiveCenter(Enum):
    HEAD = "Head (Thinking)"  # Fear/Reason based
    HEART = "Heart (Feeling)" # Image/Connection based
    GUT = "Gut (Instinct)"    # Anger/Autonomy based

@dataclass
class ArchetypeBias:
    """
    Defines how an archetype biases specific dimensions of the 21D Matrix.
    """
    amplifications: Dict[int, float] = field(default_factory=dict) # Dim Index -> Gain (>1.0)
    suppressions: Dict[int, float] = field(default_factory=dict)   # Dim Index -> Attenuation (<1.0)
    resonance_frequency: float = 1.0 # The 'color' or frequency of this layer

class CognitiveArchetype:
    def __init__(self, name: str, center: CognitiveCenter, bias: ArchetypeBias, description: str):
        self.name = name
        self.center = center
        self.bias = bias
        self.description = description

    def refract(self, vector_21d: List[float]) -> List[float]:
        """
        Refracts the raw 21D Qualia through this archetype's prism.
        Returns a modified 'perspective vector'.

        Logic:
        1. Copy raw vector.
        2. Multiply amplified dimensions by gain.
        3. Multiply suppressed dimensions by attenuation factor.
        """
        refracted = list(vector_21d)

        # Apply Amplifications
        for dim_idx, gain in self.bias.amplifications.items():
            if 0 <= dim_idx < len(refracted):
                refracted[dim_idx] *= gain

        # Apply Suppressions
        for dim_idx, factor in self.bias.suppressions.items():
            if 0 <= dim_idx < len(refracted):
                refracted[dim_idx] *= factor

        return refracted

    def voice(self, intensity: float) -> str:
        """Returns the narrative voice style of this archetype."""
        return f"[{self.name}] ({self.center.name}): Intensity {intensity:.2f}"

# ==============================================================================
#  The Council Members (Archetype Definitions)
# ==============================================================================

# 1. The Logician (Head Center) - Type 5/6ish
# Focus: Logic (D10), Truth (D15), Temperance (D16)
# Suppress: Emotional turbulence, Lust (D1)
LOGICIAN_BIAS = ArchetypeBias(
    amplifications={
        9: 2.0,   # D10: Reason (Index 9)
        14: 1.5,  # D15: Chastity/Truth (Index 14)
        15: 1.5,  # D16: Temperance (Index 15)
        17: 1.2   # D18: Diligence (Index 17)
    },
    suppressions={
        0: 0.5,   # D1: Lust
        4: 0.5,   # D5: Wrath
        11: 0.8   # D12: Imagination (if ungrounded)
    }
)
THE_LOGICIAN = CognitiveArchetype(
    name="The Logician",
    center=CognitiveCenter.HEAD,
    bias=LOGICIAN_BIAS,
    description="Analyzes structural validity and logical coherence. Fears incompetence."
)

# 2. The Empath (Heart Center) - Type 2/4ish
# Focus: Kindness (D20), Charity (D17), Imagination (D12)
# Suppress: Cold logic if cruel, Wrath (D5)
EMPATH_BIAS = ArchetypeBias(
    amplifications={
        19: 2.5,  # D20: Kindness (Index 19)
        16: 2.0,  # D17: Charity (Index 16)
        11: 1.8,  # D12: Imagination (Index 11)
        7: 1.5    # D8: Perception (Index 7)
    },
    suppressions={
        9: 0.8,   # D10: Pure Logic (soften it)
        6: 0.2    # D7: Pride (Ego)
    }
)
THE_EMPATH = CognitiveArchetype(
    name="The Empath",
    center=CognitiveCenter.HEART,
    bias=EMPATH_BIAS,
    description="Focuses on connection, meaning, and emotional resonance. Fears separation."
)

# 3. The Guardian (Gut Center) - Type 8/1ish
# Focus: Will (D11), Survival (D1), Justice
# Suppress: Sloth (D4), Hesitation
GUARDIAN_BIAS = ArchetypeBias(
    amplifications={
        10: 3.0,  # D11: Will (Index 10)
        0: 1.5,   # D1: Lust (Drive/Hunger for action)
        13: 1.5,  # D14: Consciousness/Self (Index 13)
        18: 1.5   # D19: Patience (Endurance) (Index 18)
    },
    suppressions={
        3: 0.1,   # D4: Sloth (Kill laziness)
        11: 0.7   # D12: Imagination (Stop dreaming, act)
    }
)
THE_GUARDIAN = CognitiveArchetype(
    name="The Guardian",
    center=CognitiveCenter.GUT,
    bias=GUARDIAN_BIAS,
    description="Protects the sovereign boundaries. Focuses on autonomy and action."
)

# 4. The Mystic (Spirit/Integration) - Type 9/Transcendental
# Focus: All Virtues (L15-L21), Harmony
MYSTIC_BIAS = ArchetypeBias(
    amplifications={
        20: 3.0,  # D21: Humility (Index 20)
        12: 2.0,  # D13: Intuition (Index 12)
        6: 0.1,   # D7: Pride (Kill Ego completely)
        2: 0.1    # D3: Greed (Detach)
    },
    suppressions={
        # Suppress almost all lower instincts
        0: 0.2, 1: 0.2, 4: 0.1
    }
)
THE_MYSTIC = CognitiveArchetype(
    name="The Mystic",
    center=CognitiveCenter.HEAD, # Technically integration point
    bias=MYSTIC_BIAS,
    description="Seeks universal resonance and alignment with the North Star."
)

STANDARD_COUNCIL = [THE_LOGICIAN, THE_EMPATH, THE_GUARDIAN, THE_MYSTIC]
