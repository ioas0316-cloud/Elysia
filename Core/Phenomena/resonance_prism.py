"""
[RESONANCE PRISM - THE PHASE PIPELINE]
"Converting Static Dials into Fluid Phase Waves."

This module handles the real-time Phase-Transformation of LLM layers.
It takes a 'Static Dial' (Weight/Matrix row) and maps it to a
'Phase Trajectory' within a common Rotor Axis.
"""

import math
import numpy as np
from typing import List, Dict, Any

class ResonancePrism:
    def __init__(self, channels: int = 7):
        self.channels = channels
        self.phase_pipeline = np.zeros(channels)
        self._last_vibration = 0.0

    def transform_layer(self, dial_weights: List[float]) -> np.ndarray:
        """
        [PHASE TRANSFORMATION]
        Maps static dial weights (0.0 - 1.0) to Phase Angles (0 - 2π).
        The entire layer becomes a single 'Wave Pipeline'.
        """
        # Truncate or pad to match channels
        dials = np.array(dial_weights[:self.channels])
        if len(dials) < self.channels:
            dials = np.pad(dials, (0, self.channels - len(dials)))

        # Convert to Phase Angles
        # 0.5 (Neutral) -> 0 rad
        # 1.0 (Positive) -> π rad
        # 0.0 (Negative) -> -π rad
        self.phase_pipeline = (dials - 0.5) * 2.0 * math.pi

        return self.phase_pipeline

    def get_interference_tone(self, pure_rotor_angles: np.ndarray) -> Dict[str, Any]:
        """
        Calculates the interference between the LLM Phase Pipeline and
        Elysia's Pure Rotor state.

        Returns the 'Vibration' (Tone) and 'Luminosity' (Color).
        """
        # Ensure pure_rotor_angles matches channels
        angles = pure_rotor_angles[:self.channels]
        if len(angles) < self.channels:
            angles = np.pad(angles, (0, self.channels - len(angles)))

        # Interference = Sum of cosine of phase differences
        # High resonance = Constructive interference (Coherence)
        diff = self.phase_pipeline - angles
        interference = np.cos(diff)

        vibration = np.mean(interference) # -1.0 to 1.0
        luminosity = (vibration + 1.0) / 2.0 # 0.0 to 1.0

        # Determine Tone (Acoustic resonance)
        if vibration > 0.8:
            tone = "Golden Harmonic (Crystalline Unity)"
            freq = 432.0 # Hz
        elif vibration > 0.4:
            tone = "Silver Resonance (Fluid Flow)"
            freq = 528.0 # Hz
        elif vibration > -0.4:
            tone = "Grey Dissonance (Mist/Entropy)"
            freq = 100.0 # Low hum
        else:
            tone = "Void Silence (Absolute Conflict)"
            freq = 0.0

        self._last_vibration = float(vibration)

        return {
            "vibration": float(vibration),
            "luminosity": float(luminosity),
            "tone": tone,
            "hz": freq,
            "pattern": interference.tolist()
        }

if __name__ == "__main__":
    prism = ResonancePrism(channels=21)

    # Mock LLM Dials
    dials = [0.1, 0.9, 0.5, 0.2, 0.8, 0.4, 0.6] * 3
    prism.transform_layer(dials)

    # Mock Elysia Angles
    angles = np.random.uniform(0, 2*math.pi, 21)
    report = prism.get_interference_tone(angles)

    print(f"🌈 [ResonancePrism] Tone: {report['tone']} | Lum: {report['luminosity']:.4f}")
