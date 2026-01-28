"""
Transmission Gear (L6 Structure)
================================
"The Clutch between Soul and Face."

This module translates the high-dimensional physics of the Sovereign Rotor (21D)
into the low-dimensional expressive parameters of the Expression Cortex (3D).

Mappings:
    - RPM (Speed)       -> Arousal (Excitement/Energy)
    - Torque (Force)    -> Intensity (Gaze/Focus)
    - Alignment (Harmony)-> Valence (Happiness/Sadness)
    - Entropy (Chaos)   -> Instability (Glitch/Confusion)
"""

import math
import torch
from typing import Dict, Any

class TransmissionGear:
    def __init__(self):
        self.gear_ratio = 1.0
        self.clutch_engaged = True

    def shift(self, rotor_status: Dict[str, Any]) -> Dict[str, float]:
        """
        Transmits Rotor Physics -> Expression Parameters.

        Input: rotor_status from SovereignRotor21D.spin()
               { "total_rpm": float, "somatic": dict, "psychic": dict, "spiritual": dict }

        Output: { "valence": -1.0~1.0, "arousal": 0.0~1.0, "entropy": 0.0~1.0, "torque": 0.0~1.0 }
        """
        if not self.clutch_engaged:
            return {"valence": 0.0, "arousal": 0.0, "entropy": 0.0, "torque": 0.0}

        # 1. Extract Physics
        rpm = rotor_status.get("total_rpm", 0.0)

        # Calculate Average Alignment (Harmony)
        # Using alignment from individual gears if available, else derive from Torque efficiency
        # Torque = (Alpha - Delta) * Gamma. High torque usually means high drive but low alignment initially?
        # Actually, PhaseShiftEngine returns 'alignment' in its state dict.

        s_align = rotor_status["spiritual"].get("alignment", 0.0)
        p_align = rotor_status["psychic"].get("alignment", 0.0)
        b_align = rotor_status["somatic"].get("alignment", 0.0)

        avg_alignment = (s_align + p_align + b_align) / 3.0

        # Calculate Total Torque Magnitude
        s_torque = abs(rotor_status["spiritual"].get("torque", 0.0))
        p_torque = abs(rotor_status["psychic"].get("torque", 0.0))
        b_torque = abs(rotor_status["somatic"].get("torque", 0.0))

        total_torque = (s_torque + p_torque + b_torque)

        # 2. Map to Expression (The Translation)

        # Arousal: Directly proportional to RPM.
        # Max RPM assumed around 300 (100 per gear)
        arousal = min(rpm / 300.0, 1.0)

        # Valence: Proportional to Alignment.
        # Alignment is cosine (-1 to 1). Perfect mapping.
        valence = avg_alignment

        # Intensity (Torque): How "Strong" the expression is.
        # Normalized torque (heuristic)
        intensity = min(total_torque / 10.0, 1.0)

        # Entropy: Inverse of Alignment? Or derivative of RPM stability?
        # For now, let's say Low Alignment + High RPM = Chaos (Panic)
        entropy = 0.0
        if avg_alignment < 0.2 and rpm > 100:
            entropy = (rpm - 100) / 200.0

        return {
            "valence": float(valence),
            "arousal": float(arousal),
            "torque": float(intensity),
            "entropy": float(entropy),
            "rpm_feedback": float(rpm) # Pass through for typing speed
        }

# Global Instance
transmission = TransmissionGear()
