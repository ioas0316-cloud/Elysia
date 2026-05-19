"""
Multi-Layer Rotor: The Chakras of Thought
=========================================

"A thought is not a point. It is a chord."

This module defines `MultiRotor`, a vertical stack of 7 `Rotor` units.
Updated to hold **HelixDNA** (14 Universes).

The Rotor now acts as the **Engine of Evolution** for the HelixDNA.
Spinning the rotor adds "Time" and "Experience" to the manifolds within.
"""

import logging
import math
from typing import Dict, List, Optional

from Core.System.rotor import Rotor, RotorConfig
from Core.Keystone.helix_dna import HelixDNA

logger = logging.getLogger("MultiRotor")

class MultiRotor:
    """
    A Stack of 7 Rotors wrapping a HelixDNA.
    """
    DIMENSIONS = ["Physical", "Functional", "Phenomenal", "Causal", "Mental", "Structural", "Spiritual"]

    def __init__(self, name: str):
        self.name = name
        self.layers: Dict[str, Rotor] = {}

        # [NEW] The Helix (The Universe within)
        self.dna = HelixDNA(label=name)

        self.integrity = 0.0

        # Initialize 7 Layers
        for i, dim in enumerate(self.DIMENSIONS):
            base_rpm = 60.0 * (i + 1)
            config = RotorConfig(rpm=base_rpm, idle_rpm=base_rpm, mass=10.0)
            self.layers[dim] = Rotor(f"{name}.{dim}", config)

    def inject_energy(self, dimension: str, amount: float):
        """Spins up a specific layer."""
        if dimension in self.layers:
            self.layers[dimension].wake(amount)
            # Wake up the corresponding manifold too!
            self.dna.yang_strand[dimension].evolve(dt=0.1, stimulus=amount)
            self.dna.yin_strand[dimension].evolve(dt=0.1, stimulus=amount)

    def update_physics(self, dt: float):
        """
        Simulates the internal physics of the stack.
        """
        # 1. Individual Updates
        for r in self.layers.values():
            r.update(dt)

        # 2. Viscosity (Drag between neighbors)
        keys = self.DIMENSIONS
        for i in range(len(keys) - 1):
            lower = self.layers[keys[i]]
            upper = self.layers[keys[i+1]]
            diff = upper.current_rpm - lower.current_rpm
            viscosity = 0.5 * dt
            transfer = diff * viscosity
            upper.current_rpm -= transfer
            lower.current_rpm += transfer

        # 3. Evolve the Manifolds (Time passes)
        self.dna.evolve(dt)

        # 4. Integrity
        self.integrity = self._calculate_integrity()

    def _calculate_integrity(self) -> float:
        rpms = [r.current_rpm for r in self.layers.values()]
        if not rpms: return 0.0
        total_diff = 0.0
        for i in range(len(rpms) - 1):
            total_diff += abs(rpms[i] - rpms[i+1])
        avg_rpm = sum(rpms) / len(rpms)
        if avg_rpm == 0: return 0.0
        return 1.0 / (1.0 + total_diff / (avg_rpm * len(rpms)))

    def __repr__(self):
        return f"MultiRotor({self.name} | Int:{self.integrity:.2f})"
