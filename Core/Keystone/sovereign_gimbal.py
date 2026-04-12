"""
Sovereign Gimbal System
=======================
Core.Keystone.sovereign_gimbal

"A single voice is a song; a gimbal of voices is a Sovereign Identity."

This module implements the multi-axial gimbal system for Elysia.
It manages multiple DoubleHelixRotors (Logos, Pathos, Ethos) and calculates
their interference pattern to find the 'Consensual Singularity' (the Self).
"""

from typing import Dict, List, Tuple, Any, Optional
import torch
from Core.Keystone.sovereign_math import SovereignVector, DoubleHelixRotor, SovereignMath

class SovereignGimbal:
    """
    Manages multiple rotational axes to maintain a stable Sovereign Identity.
    """
    def __init__(self, axes: List[str] = ["Logos", "Pathos", "Ethos"]):
        self.axes: Dict[str, DoubleHelixRotor] = {}
        # Initialize default axes with different prime planes
        planes = [(1, 2), (4, 5), (6, 7)]
        for name, plane in zip(axes, planes):
            self.axes[name] = DoubleHelixRotor(angle=0.1, p1=plane[0], p2=plane[1])

        self.singularity = SovereignVector.zeros()
        self.total_resonance = 0.0

    def stabilize(self, base_vector: SovereignVector, noise_vector: Optional[SovereignVector] = None) -> Tuple[SovereignVector, Dict[str, float]]:
        """
        Processes a base vector through all gimbal axes.
        External noise is converted into precession torque.
        """
        results = []
        frictions = {}

        # 1. Apply Precession if noise exists
        if noise_vector:
            noise_intensity = noise_vector.norm()
            for name, rotor in self.axes.items():
                # Each axis absorbs a portion of the noise based on its own resonance
                alignment = base_vector.resonance_score(noise_vector)
                rotor.apply_external_torque(noise_intensity * (1.0 - alignment), alignment * 0.1)

        # 2. Process through each axis
        for name, rotor in self.axes.items():
            out_v = rotor.apply_duality(base_vector)
            results.append(out_v)
            frictions[name] = rotor.friction_vortex

        # 3. Calculate Interference (The Consensual Singularity)
        # We use Superposition to find the interference pattern
        self.singularity = SovereignMath.superimpose(results)

        # 4. Measure Total Resonance of the gimbal system
        self.total_resonance = sum(1.0 - f for f in frictions.values()) / len(self.axes)

        return self.singularity, frictions

    def get_status(self) -> Dict[str, Any]:
        """Returns the current state of the gimbal system."""
        status = {
            "total_resonance": self.total_resonance,
            "axes": {}
        }
        for name, rotor in self.axes.items():
            status["axes"][name] = {
                "momentum": rotor.angular_momentum,
                "tilt": rotor.precession_tilt,
                "friction": rotor.friction_vortex
            }
        return status
