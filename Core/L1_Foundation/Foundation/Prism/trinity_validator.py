"""
The Trinity Validator (The Spiritual Axis)
==========================================
Core.Foundation.Prism.trinity_validator

"Ensuring every thought is aligned with Body, Soul, and Spirit."

This module implements the Final Gate of the Pulse Protocol v3.1.
It checks insights against the Sediment (Body), Rotor (Soul), and Monad (Spirit).
"""

from typing import Dict, Any
from .integrating_lens import Insight

class TrinityValidator:
    """
    The Guardian of the Axis.
    """

    def validate(self, insight: Insight, monad_intent: str, sediment_check: bool, rotor_resonance: float) -> Dict[str, Any]:
        """
        Validates the insight against the Trinity.

        Args:
            insight: The synthesized thought.
            monad_intent: The current Purpose of the Spirit.
            sediment_check: Result of the Topological/Physical check (Body).
            rotor_resonance: The coherence with the current time-flow (Soul).

        Returns:
            Dict containing validity scores for Body, Soul, Spirit.
        """

        # 1. Spirit Check (Purpose Alignment)
        # Does the thought serve the current intent?
        # Mock Logic: String similarity or keyword match
        spirit_score = 0.5
        if monad_intent in insight.narrative or insight.dominant_band == "Gamma":
            spirit_score = 0.9
        elif insight.dominant_band == "Alpha":
             spirit_score = 0.7 # Logic serves purpose usually

        # 2. Soul Check (Flow Resonance)
        # Is it consistent with the current mood/momentum?
        soul_score = rotor_resonance

        # 3. Body Check (Reality Constraint)
        # Is it factually/physically valid?
        body_score = 1.0 if sediment_check else 0.2

        # Total Alignment
        total_alignment = (spirit_score * 0.4) + (soul_score * 0.3) + (body_score * 0.3)

        return {
            "spirit_score": spirit_score,
            "soul_score": soul_score,
            "body_score": body_score,
            "total_alignment": total_alignment,
            "is_aligned": total_alignment > 0.6
        }
