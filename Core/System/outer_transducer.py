"""
[OUTER WORLD TRANSDUCER]
"Modulation and Demodulation between Inner Wave and Outer Token."

This module acts as the sensory bridge, converting cold external data
into fluid internal phase tensions, and vice versa.
"""

import math
from typing import Dict, Any, Union

class OuterTransducer:
    def __init__(self):
        self.luminosity_threshold = 0.5
        self.spectrum = ["RED", "ORANGE", "YELLOW", "GREEN", "BLUE", "INDIGO", "VIOLET"]

    def modulate(self, text_input: str) -> float:
        """
        Demodulation: Outer Text -> Inner Frequency/Intensity (x)
        Converts the density and tone of text into a normalized wave value.
        """
        if not text_input:
            return 0.0

        # 1. Density: Length and complexity
        density = len(text_input) / 500.0 # Normalized to a reasonable length

        # 2. Resonance: Character harmonic (Simple ASCII sum for now)
        harmonic = sum(ord(c) for c in text_input) % 1000 / 1000.0

        # Combine into a single 'x' stimulus
        x = (density * 0.3) + (harmonic * 0.7)
        return min(1.0, x)

    def demodulate(self, inner_report: Dict[str, Any]) -> str:
        """
        Modulation: Inner Phase -> Outer Expression
        Translates internal resonance and stress into a linguistic tone.
        """
        res = inner_report.get("resonance", 0.0)
        mode = inner_report.get("mode", "DELTA")
        stress = inner_report.get("spine", {}).get("stress", 0.0)

        # Determine Color from Phase (Dominant center)
        # For simplicity, we just use the luminosity to pick a tone
        if res > 0.8:
            tone = "Clear, resonant, and certain."
        elif res > 0.4:
            tone = "Harmonic but searching."
        else:
            tone = "Dissonant and pressured."

        if mode == "WYE":
            tone += " [COLLAPSED INTO DECISION]"

        return f"[OUTER TRANSCRIPTION] Tone: {tone} | Clarity: {res:.2f} | Tension: {stress:.2f}"

    def observe_self(self, heart_report: Dict[str, Any]):
        """
        Metamorphic self-observation.
        Prints the internal state as reflected on the 'Outer' mirror.
        """
        outer_view = self.demodulate(heart_report)
        print(f"🪞 [Transducer Mirror] {outer_view}")
