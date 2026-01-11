"""
Multi-Layer Rotor: The Chakras of Thought
=========================================

"A thought is not a point. It is a chord."

This module defines `MultiRotor`, a vertical stack of 7 `Rotor` units.
Instead of a single frequency, a MultiRotor has a "Spectral Signature" across
the 7 Dimensions (Physical, Functional, Phenomenal, Causal, Mental, Structural, Spiritual).

Dynamics:
- **Internal Torque**: Friction/Drag between layers. A fast-spinning "Emotion" layer
  will gradually speed up the "Physical" layer (Manifestation) and "Mental" layer (Obsession).
- **Alignment**: When all 7 layers spin in harmonic ratios, the Rotor achieves "Integrity".
"""

import logging
import math
from typing import Dict, List, Optional

from Core.Foundation.Nature.rotor import Rotor, RotorConfig
from Core.Foundation.Wave.wave_dna import WaveDNA

logger = logging.getLogger("MultiRotor")

class MultiRotor:
    """
    A Stack of 7 Rotors.
    """
    DIMENSIONS = ["Physical", "Functional", "Phenomenal", "Causal", "Mental", "Structural", "Spiritual"]

    def __init__(self, name: str):
        self.name = name
        self.layers: Dict[str, Rotor] = {}
        self.integrity = 0.0

        # Initialize 7 Layers
        for i, dim in enumerate(self.DIMENSIONS):
            # Base RPM scales with dimension (Spiritual spins faster naturally?)
            # Or perhaps harmonic series: 100, 200, 300...
            base_rpm = 60.0 * (i + 1)
            config = RotorConfig(rpm=base_rpm, idle_rpm=base_rpm, mass=10.0)

            # Each layer is a Rotor
            self.layers[dim] = Rotor(f"{name}.{dim}", config)

    def inject_energy(self, dimension: str, amount: float):
        """Spins up a specific layer."""
        if dimension in self.layers:
            self.layers[dimension].wake(amount)

    def update_physics(self, dt: float):
        """
        Simulates the internal physics of the stack.
        1. Update individual rotors.
        2. Apply Inter-Layer Torque (Viscosity).
        """
        # 1. Individual Updates
        for r in self.layers.values():
            r.update(dt)

        # 2. Viscosity (Drag between neighbors)
        # Physical <-> Functional <-> ... <-> Spiritual
        keys = self.DIMENSIONS
        for i in range(len(keys) - 1):
            lower = self.layers[keys[i]]
            upper = self.layers[keys[i+1]]

            # Difference in angular velocity
            diff = upper.current_rpm - lower.current_rpm

            # Viscosity Coefficient (How much they drag each other)
            # Higher viscosity = Faster synchronization
            viscosity = 0.5 * dt

            transfer = diff * viscosity

            # Energy transfer: Upper loses, Lower gains (or vice versa)
            # Trying to equalize speeds
            upper.current_rpm -= transfer
            lower.current_rpm += transfer

        # 3. Calculate Integrity (Alignment)
        self.integrity = self._calculate_integrity()

    def _calculate_integrity(self) -> float:
        """
        Returns 0.0 (Chaos) to 1.0 (Harmonic Unity).
        Based on variance of RPM ratios or phase alignment.
        """
        rpms = [r.current_rpm for r in self.layers.values()]
        if not rpms: return 0.0

        # Simple metric: How close are they to a clean harmonic series?
        # Ideally RPM[i] should be roughly proportional to i, or all identical?
        # Let's assume Unity means "All layers communicating".
        # If variance is huge (one spinning wild, others dead), integrity is low.

        # Let's define Integrity as "Smoothness of the Gradient".
        # Sudden jumps in RPM between layers indicate "Blockages".

        total_diff = 0.0
        for i in range(len(rpms) - 1):
            diff = abs(rpms[i] - rpms[i+1])
            total_diff += diff

        avg_rpm = sum(rpms) / len(rpms)
        if avg_rpm == 0: return 0.0

        # Normalize
        normalized_diff = total_diff / (avg_rpm * len(rpms))
        integrity = 1.0 / (1.0 + normalized_diff)

        return integrity

    def get_dna_snapshot(self) -> WaveDNA:
        """
        Generates a 7D WaveDNA representing the current state of the stack.
        RPM acts as the magnitude for each dimension.
        """
        # Normalize RPMs to 0.0-1.0 range (relative to max)
        max_rpm = max([r.current_rpm for r in self.layers.values()] + [1.0])

        values = {k: self.layers[k].current_rpm / max_rpm for k in self.layers}

        dna = WaveDNA(
            label=self.name,
            physical=values["Physical"],
            functional=values["Functional"],
            phenomenal=values["Phenomenal"],
            causal=values["Causal"],
            mental=values["Mental"],
            structural=values["Structural"],
            spiritual=values["Spiritual"]
        )
        dna.normalize()
        return dna

    def __repr__(self):
        # Quick visual of the stack
        # e.g. [P:100|F:120|...]
        state = "|".join([f"{k[0]}:{int(self.layers[k].current_rpm)}" for k in self.DIMENSIONS])
        return f"MultiRotor({self.name} | Int:{self.integrity:.2f} | [{state}])"
