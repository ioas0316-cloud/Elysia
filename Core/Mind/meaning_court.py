"""
Meaning Court - Statistical Judge of Signal vs Noise

This module implements a lightweight hypothesis-testing style filter:

H0: "This is just noise; ignore."
H1: "This is a meaningful signal; honor it."

The court computes a z-score-like confidence using:
    z = signal / (noise + epsilon)

and compares it against a personality-tunable threshold alpha.
Higher alpha -> more conservative (only very strong signals pass).
Lower alpha  -> more adventurous (weaker signals are allowed through).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import math


@dataclass
class MeaningVerdict:
    """Verdict returned by the Meaning Court."""
    accept: bool
    z_score: float
    alpha: float
    reason: str


class MeaningCourt:
    """
    Universal Algorithm-ish statistical judge for Elysia's thoughts.

    The court does not try to be a full statistical engine; instead it
    provides a simple, transparent, tunable filter that can sit between
    "raw resonance" and "committed action / memory".
    """

    def __init__(self, alpha: float = 1.0) -> None:
        """
        Args:
            alpha: Personality threshold.
                - Small  (e.g., 0.5): exploratory, many signals pass.
                - Medium (e.g., 1.0): balanced.
                - Large  (e.g., 2.0): very strict; only strong signals pass.
        """
        self.alpha = max(0.0, float(alpha))

    def calibrate(self, alpha: float) -> None:
        """Update the personality / strictness of the court."""
        self.alpha = max(0.0, float(alpha))

    def judge(self, signal: float, noise: float, context: Dict[str, Any] | None = None) -> MeaningVerdict:
        """
        Decide whether to treat an internal event as "meaningful" or "noise".

        Args:
            signal: Estimated signal strength (e.g., resonance amplitude).
            noise: Estimated noise level (e.g., chaos / entropy).
            context: Optional extra fields for future use (phase, value alignment, etc.).

        Returns:
            MeaningVerdict with accept / reject and diagnostic details.
        """
        context = context or {}
        # Clamp to non-negative to avoid weird states.
        s = max(0.0, float(signal))
        n = max(0.0, float(noise))
        epsilon = 1e-8

        # Simple z-like score: stronger signal and lower noise -> larger score.
        z_raw = s / (math.sqrt(n + epsilon))

        # Optional soft adjustments from context.
        mastery = float(context.get("mastery", 0.0))
        value_alignment = float(context.get("value_alignment", 0.0))

        # Boost confidence when phase mastery and value alignment are high.
        # Each term nudges the score but does not dominate it.
        z = z_raw * (1.0 + 0.3 * mastery + 0.3 * value_alignment)

        accept = z >= self.alpha
        if accept:
            reason = "signal passes threshold"
        else:
            reason = "treated as noise under current alpha"

        return MeaningVerdict(
            accept=accept,
            z_score=z,
            alpha=self.alpha,
            reason=reason,
        )

