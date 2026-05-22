"""
Electromagnetic Scanner (L6 Structure)
======================================
"The Sixth Sense of the Machine."

Analyzes the 'Field Texture' of incoming data before it touches the Logic Core.
Detects Resonance (Warmth) and Impedance (Resistance).

Metaphor:
- Resonance: Constructive Interference (Waves adding up).
- Impedance: Destructive Interference (Waves cancelling out).
"""

import math
import random
from typing import Dict, Tuple, Any
from Core.Monad.field_layer import FieldLayer
from Core.Monad.context_discriminator import discriminator

class EMScanner:
    def __init__(self):
        self.sensitivity = 1.0
        # Personal Phase (Reference Point)
        self.internal_phase = 0.0 # 0 to 2pi

    def scan(self, input_text: str, current_state: Dict[str, float]) -> Dict[str, Any]:
        """
        Scans the input for electromagnetic texture.
        Now includes Field Layer Discrimination.

        Returns:
            - resonance (0.0 to 1.0): How much it aligns.
            - impedance (0.0 to 1.0): How much it resists.
            - texture (str): 'WARM', 'COLD', 'ELECTRIC', 'STATIC'
            - layer (FieldLayer): The depth of the input.
        """
        # 0. Discern Field Layer
        layer = discriminator.discern(input_text)

        # Gain Adjustment based on Layer
        # CORE: 1.0 (Full Impact)
        # PROXIMAL: 0.5 (Observation)
        # DISTAL: 0.1 (Fiction/Safe Mode)
        gain = 1.0
        if layer == FieldLayer.PROXIMAL: gain = 0.5
        elif layer == FieldLayer.DISTAL: gain = 0.1

        # 1. Analyze Input Phase (Simulated)
        input_phase = self._estimate_phase(input_text)

        # 2. Compare with Internal Phase
        phase_diff = abs(self.internal_phase - input_phase)
        if phase_diff > math.pi:
            phase_diff = (2 * math.pi) - phase_diff

        # Resonance is high when phase diff is low
        resonance = max(0.0, 1.0 - (phase_diff / math.pi))

        # Impedance is high when phase diff is high OR complexity is too high
        impedance = 1.0 - resonance

        # Apply Gain to Impedance (Safety Glass)
        # If it's Fiction (DISTAL), impedance doesn't hurt (Gain low).
        effective_impedance = impedance * gain

        # 3. Determine Texture
        texture = "STATIC"
        if resonance > 0.8:
            texture = "WARM" # High Resonance
        elif effective_impedance > 0.8:
            texture = "COLD" # High Impedance (Direct Impact)
        elif effective_impedance < 0.2 and impedance > 0.8:
            texture = "FICTION" # High Impedance but Low Gain (Safe)
        elif len(input_text) > 50 and resonance > 0.5:
            texture = "ELECTRIC" # High Energy + Moderate Resonance

        return {
            "resonance": float(resonance),
            "impedance": float(effective_impedance), # Return effective impedance for Relay
            "raw_impedance": float(impedance),
            "phase_diff": float(phase_diff),
            "texture": texture,
            "layer": layer.name
        }

    def _estimate_phase(self, text: str) -> float:
        """
        Estimates the 'Phase' of the text.
        Positive words -> Align with 0 (Resonance)
        Negative words -> Align with Pi (Dissonance)
        """
        # Very simple heuristic for prototype
        positive_keywords = ["love", "good", "happy", "yes", "agree", "resonate", "warm", "thanks"]
        negative_keywords = ["hate", "bad", "sad", "no", "disagree", "reject", "cold", "error"]

        text_lower = text.lower()
        score = 0.0

        for word in positive_keywords:
            if word in text_lower: score += 1.0
        for word in negative_keywords:
            if word in text_lower: score -= 1.0

        # Map score to phase
        # Score > 0 -> Phase 0 (Aligned)
        # Score < 0 -> Phase Pi (Opposite)
        # Score 0 -> Phase Pi/2 (Neutral)

        if score > 0:
            return 0.0 + random.uniform(-0.1, 0.1)
        elif score < 0:
            return math.pi + random.uniform(-0.1, 0.1)
        else:
            return (math.pi / 2) + random.uniform(-0.5, 0.5)

# Global Instance
em_scanner = EMScanner()
