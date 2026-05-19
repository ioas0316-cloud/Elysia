"""
[OUTER WORLD TRANSDUCER - VORTEX EDITION]
"Modulation and Demodulation between Inner Wave and Outer Token."

Upgraded to utilize Trajectory Encoding for rich sensory processing.
"""

import math
from typing import Dict, Any, Union, List
from Core.Keystone.trajectory_encoder import TrajectoryEncoder, VortexTrajectory

class OuterTransducer:
    def __init__(self):
        self.luminosity_threshold = 0.5
        self.spectrum = ["RED", "ORANGE", "YELLOW", "GREEN", "BLUE", "INDIGO", "VIOLET"]
        self.encoder = TrajectoryEncoder()

    def modulate(self, text_input: str) -> List[VortexTrajectory]:
        """
        Demodulation: Outer Text -> Stream of Vortex Trajectories.
        Converts the density, tone, and character harmonics into a rich phase field.
        """
        if not text_input:
            return []

        # Convert text into a stream of physical trajectories
        trajectories = self.encoder.encode_text(text_input)

        # Globally modulate amplitude based on total length (Density)
        # Short punchy messages have high intensity per character
        length_factor = max(0.1, 1.0 - (len(text_input) / 1000.0))

        for t in trajectories:
            t.amplitude *= length_factor

        return trajectories

    def demodulate(self, inner_report: Dict[str, Any]) -> str:
        """
        Modulation: Inner Phase -> Outer Expression
        Translates internal resonance and stress into a linguistic tone.
        """
        res = inner_report.get("resonance", 0.0)
        mode = inner_report.get("mode", "DELTA")
        stress = inner_report.get("spine", {}).get("stress", 0.0)

        # Determine Color from Phase (Dominant center)
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
