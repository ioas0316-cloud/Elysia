"""
Holographic Council (The Debate Engine)
=======================================
Core.S1_Body.L5_Mental.Reasoning_Core.Metabolism.holographic_council

"Where the many become One through the friction of debate."

[Blueprint Reference]: docs/S1_Body/L5_Mental/HOLOGRAPHIC_COUNCIL_BLUEPRINT.md

This module implements the "Optical Sovereignty" debate mechanism.
It takes a raw 21D Qualia vector and diffracts it through multiple
Cognitive Archetypes (Prisms). The resulting interference pattern
determines the final Sovereign Choice.

Key Concepts:
- **Diffraction**: Splitting the input intent into Head, Heart, and Gut perspectives.
- **Interference**: Calculating the distance (dissonance) between these perspectives.
- **Synthesis**: Collapsing the wave function into a Consensus Vector.
"""

import math
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

from Core.S1_Body.L5_Mental.Reasoning_Core.Meta.archetypes import (
    CognitiveArchetype, STANDARD_COUNCIL, CognitiveCenter
)

# Configure Logger
logger = logging.getLogger("Elysia.HolographicCouncil")

@dataclass
class CouncilVoice:
    """
    Represents the output of a single Archetype's diffraction.
    """
    archetype: CognitiveArchetype
    perspective_vector: List[float]
    intensity: float
    primary_focus_index: int # The dimension index this voice cares most about

@dataclass
class DebateResult:
    """
    The final collapsed state of the Holographic Debate.
    """
    consensus_vector: List[float]
    dominant_voice: str
    opposition_voice: str
    dissonance_score: float # 0.0 to 1.0 (How much disagreement?)
    transcript: List[str]
    is_resolved: bool

class HolographicCouncil:
    def __init__(self, archetypes: List[CognitiveArchetype] = None):
        self.archetypes = archetypes if archetypes else STANDARD_COUNCIL
        self.history = []

    def convene(self, input_vector_21d: List[float], intent_text: str = "") -> DebateResult:
        """
        Convense the council to debate the input Qualia.
        This is the core "Sovereign Negotiation" process.
        """
        transcript = [f"--- Council Convened for: '{intent_text}' ---"]

        # 1. Diffraction Phase
        # --------------------
        voices: List[CouncilVoice] = []

        for arch in self.archetypes:
            # Refract the input through this archetype's bias
            perspective = arch.refract(input_vector_21d)

            # Calculate intensity (magnitude of the vector)
            intensity = math.sqrt(sum(x*x for x in perspective))

            # Find primary focus (dimension with highest absolute value)
            primary_idx = perspective.index(max(perspective, key=abs))

            voices.append(CouncilVoice(
                archetype=arch,
                perspective_vector=perspective,
                intensity=intensity,
                primary_focus_index=primary_idx
            ))

            # Log the voice
            transcript.append(f"[{arch.name}] speaks with intensity {intensity:.2f}. Focus: D{primary_idx+1}")

        # 2. Interference Phase (The Debate)
        # ----------------------------------
        # Find Dominant and Opposition
        voices.sort(key=lambda v: v.intensity, reverse=True)
        dominant = voices[0]

        # Find the voice most unlike the dominant one (Lowest cosine similarity or highest distance)
        opposition = None
        max_dist = -1.0

        for voice in voices[1:]:
            dist = self._euclidean_distance(dominant.perspective_vector, voice.perspective_vector)
            if dist > max_dist:
                max_dist = dist
                opposition = voice

        transcript.append(f"-> Dominant Voice: {dominant.archetype.name} (Lead)")
        if opposition:
            transcript.append(f"-> Opposition: {opposition.archetype.name} (Distance: {max_dist:.2f})")

        # 3. Synthesis Phase (Collapse)
        # -----------------------------
        # Calculate weighted average based on intensity
        total_intensity = sum(v.intensity for v in voices)
        consensus_vector = [0.0] * len(input_vector_21d)

        if total_intensity > 0:
            for voice in voices:
                weight = voice.intensity / total_intensity
                for i, val in enumerate(voice.perspective_vector):
                    consensus_vector[i] += val * weight

        # Calculate System Dissonance (Variance of perspectives)
        dissonance = self._calculate_dissonance(voices, consensus_vector)

        resolution_status = "RESOLVED"
        if dissonance > 0.5:
            resolution_status = "TENSION HIGH - UNEASY COMPROMISE"
            transcript.append(f"!! High Dissonance Detected ({dissonance:.2f}). The Hologram flickers.")
        else:
            transcript.append(f"-> Consensus Reached. Dissonance: {dissonance:.2f}")

        # Generate Narrative Summary
        transcript.append(self._generate_narrative(dominant, opposition, dissonance))

        return DebateResult(
            consensus_vector=consensus_vector,
            dominant_voice=dominant.archetype.name,
            opposition_voice=opposition.archetype.name if opposition else "None",
            dissonance_score=dissonance,
            transcript=transcript,
            is_resolved=(dissonance < 0.8)
        )

    def _euclidean_distance(self, v1: List[float], v2: List[float]) -> float:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

    def _calculate_dissonance(self, voices: List[CouncilVoice], mean_vector: List[float]) -> float:
        """
        Calculates the 'variance' or spread of the voices.
        Higher spread = Higher Cognitive Diversity/Conflict.
        """
        total_dist_sq = 0.0
        for voice in voices:
            dist = self._euclidean_distance(voice.perspective_vector, mean_vector)
            total_dist_sq += dist ** 2

        # Normalize relative to magnitude?
        # For now, just return sqrt(avg variance)
        return math.sqrt(total_dist_sq / len(voices))

    def _generate_narrative(self, dominant: CouncilVoice, opposition: Optional[CouncilVoice], dissonance: float) -> str:
        """
        Creates the human-readable summary of the internal debate.
        """
        narrative = f"{dominant.archetype.name} led the session, emphasizing D{dominant.primary_focus_index+1}."

        if opposition:
            narrative += f" However, {opposition.archetype.name} raised concerns from the perspective of {opposition.archetype.center.name}."

            if dissonance > 0.4:
                narrative += " A significant debate ensued regarding the balance of Ideal vs Instinct."
            else:
                narrative += " The perspectives were largely aligned."

        return narrative
