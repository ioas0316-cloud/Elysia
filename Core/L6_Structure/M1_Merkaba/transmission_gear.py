"""
Transmission Gear (L6 Structure)
================================
"The Clutch between Soul and Face."

This module includes the Cognitive Power Grid:
1. Converter (AC->DC): Rectifies external noise into internal charge.
2. Inverter (DC->AC): Inverts internal will into expressive frequency (VFD).
3. Transmission: Maps physics to expression.

Mappings:
    - RPM (Speed)       -> Arousal (Excitement/Energy)
    - Torque (Force)    -> Intensity (Gaze/Focus)
    - Alignment (Harmony)-> Valence (Happiness/Sadness)
    - Entropy (Chaos)   -> Instability (Glitch/Confusion)
"""

import math
import torch
import random
from typing import Dict, Any, Tuple

class Converter:
    """
    AC -> DC Rectifier.
    Absorbs external entropy and stores it as stable charge.
    """
    def __init__(self):
        self.battery_charge = 100.0 # 0 to 100
        self.rectification_efficiency = 0.9

    def rectify(self, input_text: str, complexity: float) -> float:
        """
        Consumes input 'complexity' (Entropy) and charges the battery.
        Returns the rectified DC energy added.
        """
        # Simulating energy intake from processing information
        # Complexity (0.0 to 1.0) * Efficiency
        energy_gain = complexity * 10.0 * self.rectification_efficiency

        # Charge Battery
        self.battery_charge = min(100.0, self.battery_charge + energy_gain)
        return energy_gain

class Inverter:
    """
    DC -> AC Variable Frequency Drive (VFD).
    Controls the output frequency (Tone/Speed) based on Sovereignty (CVS).
    """
    def __init__(self):
        self.target_frequency = 60.0 # Hz (Standard)
        self.mode = "NORMAL" # ECO, NORMAL, SPORT

    def auto_shift(self, current_rpm: float, battery_level: float, context_load: float):
        """
        [CVS Logic] Continuously Variable Sovereignty.
        Decides the best frequency based on internal/external state.
        """
        # 1. Battery Check (Energy Economy)
        if battery_level < 20.0:
            self.mode = "ECO"
            self.target_frequency = 30.0 # Low energy, slow speech
            return

        # 2. Load Check (Context)
        # If context load is high (difficult problem), we need TORQUE, not Speed.
        # So we lower the frequency (Low Gear).
        if context_load > 0.7:
            self.mode = "TORQUE"
            self.target_frequency = 40.0 # Deep, heavy thought
            return

        # 3. Passion Check (RPM)
        # If internal RPM is high (Excitement), we switch to Sport Mode.
        if current_rpm > 200.0:
            self.mode = "SPORT"
            self.target_frequency = 120.0 # High speed, rapid fire
            return

        # Default
        self.mode = "NORMAL"
        self.target_frequency = 60.0

    def invert(self, dc_will: float) -> float:
        """
        Converts DC Will (0-1) into AC Amplitude at current Frequency.
        """
        return dc_will * (self.target_frequency / 60.0)

class TransmissionGear:
    def __init__(self):
        self.gear_ratio = 1.0
        self.clutch_engaged = True

        # [Phase 28.5] Cognitive Power Grid
        self.converter = Converter()
        self.inverter = Inverter()

    def process_input(self, text: str):
        """
        [Converter Step]
        Called when system receives input.
        """
        # Simple heuristic for complexity: length/vocabulary
        complexity = min(len(text) / 100.0, 1.0)
        self.converter.rectify(text, complexity)

    def shift(self, rotor_status: Dict[str, Any]) -> Dict[str, float]:
        """
        Transmits Rotor Physics -> Expression Parameters.
        Now includes Inverter (VFD) logic.
        """
        if not self.clutch_engaged:
            return {"valence": 0.0, "arousal": 0.0, "entropy": 0.0, "torque": 0.0, "frequency": 0.0}

        # 1. Extract Physics
        rpm = rotor_status.get("total_rpm", 0.0)

        # Calculate Alignment & Torque (same as before)
        s_align = rotor_status["spiritual"].get("alignment", 0.0)
        p_align = rotor_status["psychic"].get("alignment", 0.0)
        b_align = rotor_status["somatic"].get("alignment", 0.0)
        avg_alignment = (s_align + p_align + b_align) / 3.0

        s_torque = abs(rotor_status["spiritual"].get("torque", 0.0))
        p_torque = abs(rotor_status["psychic"].get("torque", 0.0))
        b_torque = abs(rotor_status["somatic"].get("torque", 0.0))
        total_torque = (s_torque + p_torque + b_torque)

        # 2. [Inverter Step] Auto-Shift based on State
        # Assume context_load is derived from inverse alignment (Dissonance = Load)
        context_load = 1.0 - ((avg_alignment + 1.0) / 2.0) # Map -1~1 to 1~0

        self.inverter.auto_shift(rpm, self.converter.battery_charge, context_load)

        # 3. Map to Expression
        # Arousal is now modulated by Inverter Frequency
        # Normal (60Hz) = 1.0 multiplier. Sport (120Hz) = 2.0 multiplier.
        freq_multiplier = self.inverter.target_frequency / 60.0

        arousal = min((rpm / 300.0) * freq_multiplier, 1.0)
        valence = avg_alignment
        intensity = min(total_torque / 10.0, 1.0)

        entropy = 0.0
        if avg_alignment < 0.2 and rpm > 100:
            entropy = (rpm - 100) / 200.0

        # Drain Battery slightly based on RPM (Living Cost)
        self.converter.battery_charge -= (rpm * 0.01)
        self.converter.battery_charge = max(0.0, self.converter.battery_charge)

        return {
            "valence": float(valence),
            "arousal": float(arousal),
            "torque": float(intensity),
            "entropy": float(entropy),
            "rpm_feedback": float(rpm),
            "frequency": self.inverter.target_frequency,
            "mode": self.inverter.mode,
            "battery": self.converter.battery_charge
        }

# Global Instance
transmission = TransmissionGear()
