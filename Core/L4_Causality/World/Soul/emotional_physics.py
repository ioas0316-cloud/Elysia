"""
Emotional Physics (      )
===============================
" Emotion is not a state of mind, but a state of matter. "
"                ,        . "

This module defines how emotions alter the **Physical Density** of a soul.
It implements the "High Frequency = Low Density" principle.

- Joy/Love (High Freq) -> Low Density -> Buoyancy -> Ascension.
- Sadness/Anger (Low Freq) -> High Density -> Weight -> Gravity.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class EmotionalState:
    name: str
    frequency: float  # Hz
    density_modifier: float # Multiplier for Gravity (1.0 = Normal)
    flow_modifier: float    # Multiplier for Speed (1.0 = Normal)

class EmotionalPhysics:
    def __init__(self):
        # Define the Spectrum of Soul Density
        self.spectrum: Dict[str, EmotionalState] = {
            "Ecstasy": EmotionalState("Ecstasy", 963.0, 0.1, 2.0), # Light as a feather, fast
            "Love":    EmotionalState("Love",    528.0, 0.3, 1.5),
            "Joy":     EmotionalState("Joy",     432.0, 0.5, 1.2),
            "Peace":   EmotionalState("Peace",   396.0, 0.7, 1.0),
            "Neutral": EmotionalState("Neutral", 200.0, 1.0, 1.0), # Baseline
            "Worry":   EmotionalState("Worry",   150.0, 1.5, 0.8),
            "Fear":    EmotionalState("Fear",    100.0, 2.0, 0.5), # Heavy, paralyzed
            "Grief":   EmotionalState("Grief",    75.0, 3.0, 0.2), # Crushing weight
            "Shame":   EmotionalState("Shame",    20.0, 5.0, 0.0), # Absolute bottom, immovable
        }

    def get_physical_modifiers(self, frequency: float) -> Tuple[float, float]:
        """
        Returns (density_mod, flow_mod) based on current frequency.
        Interpolates between known states.
        """
        # Simple nearest neighbor or threshold logic for now
        # Ideally, this is a continuous curve: Density = 1/Freq

        # Physics Formula: Density ~ 1 / sqrt(Frequency)
        # Higher Freq = Lower Density
        base_freq = 200.0 # Neutral

        if frequency <= 1.0: frequency = 1.0

        ratio = frequency / base_freq

        # Density drops as frequency rises
        density = 1.0 / (ratio ** 0.5)

        # Speed (Flow) rises as frequency rises
        flow = ratio ** 0.5

        return (density, flow)

    def resolve_emotion(self, vector_sum: float) -> str:
        """Finds the closest named emotion for a frequency."""
        closest_name = "Neutral"
        min_diff = 9999.0

        for name, state in self.spectrum.items():
            diff = abs(vector_sum - state.frequency)
            if diff < min_diff:
                min_diff = diff
                closest_name = name

        return closest_name

# Singleton
emotional_physics = EmotionalPhysics()