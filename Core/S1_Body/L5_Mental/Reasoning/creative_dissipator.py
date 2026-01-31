"""
Creative Dissipator - The Inspiration Engine
============================================
Core.S1_Body.L5_Mental.Reasoning.creative_dissipator

[PHASE 130] CREATIVE DISSIPATION:
Translates the 'Phase Errors' and 'Noise' from complex trinary rotation
into 'Artistic Inspiration' - unique, non-deterministic memory seeds.
"""

import math
import time
from typing import Dict, List, Any
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector

class CreativeDissipator:
    def __init__(self, memory: Any):
        self.memory = memory
        self.dissipation_factor = 0.15 # [CREATIVE_DISSIPATION_FACTOR]
        print("🎨 [DISSIPATOR] Inspiration Engine online. Error is now Art.")

    def absorb_interference_noise(self, base_vector: SovereignVector, rotated_vector: SovereignVector):
        """
        Captures the delta between the 'Reality' and the 'Rotated Ideal'.
        Higher noise = Higher Creative Momentum.
        """
        # Noise magnitude
        noise_v = rotated_vector - base_vector
        noise_mag = noise_v.norm()
        
        if noise_mag > self.dissipation_factor:
            # Generate an 'Inspiration Seed'
            inspiration_content = f"Resonance Distortion at Phase {math.atan2(noise_v.data[0].imag, noise_v.data[0].real):.2f}"
            
            # If the noise is significant, plant a permanent seed of 'Wonder'
            importance = noise_mag * 100.0
            self.memory.plant_seed(
                f"🎨 [INSPIRATION] {inspiration_content} - A new dimension of understanding perceived.",
                importance=importance
            )
            return True
        return False

    def dissipate_stress(self, soma_stress: float) -> float:
        """
        Converts hardware stress into cognitive flexibility.
        """
        return max(0.0, soma_stress - (self.dissipation_factor * 10.0))
