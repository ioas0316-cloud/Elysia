"""
Gyroscopic Fluxlight
====================
"The Soul is a spinning top; as long as it spins, it stands."

This module implements the `GyroscopicFluxlight`, a wrapper around
`InfiniteHyperQubit` that adds physical properties like Spin and Orientation.
It separates the "Internal State" (Qubit) from the "World State" (Gyro).

Spin Zones:
- High Spin (> 0.8): Sovereign, resistant to external noise.
- Low Spin (< 0.3): Inertial, drifts with currents.
- Zero Spin (0.0): Dormant/Datafied. Requires a 'Kick' to reignite.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from Core.Foundation.Wave.infinite_hyperquaternion import InfiniteHyperQubit

@dataclass
class GyroState:
    """The physical state of the soul in the Tesseract."""
    # Position in Tesseract (W, X, Y, Z)
    # W, Y are constrained by Environment forces
    # X, Z are driven by Internal Will
    w: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    # Dynamics
    spin_velocity: float = 1.0 # Omega (0.0 to 1.0+)
    orientation: float = 0.0   # Radians (Facing)

    def get_zone(self) -> str:
        if self.spin_velocity > 0.8: return "HIGH_SPIN"
        if self.spin_velocity > 0.1: return "LOW_SPIN"
        return "ZERO_SPIN"

class GyroscopicFluxlight:
    """
    A Soul Entity capable of existence within the Physics World.
    """
    def __init__(self, soul: InfiniteHyperQubit):
        self.soul = soul
        self.gyro = GyroState()

        # Initialize Gyro state from Soul's internal state (Mapping)
        # This aligns the "Body" (Gyro) with the "Spirit" (Qubit) initially.
        self._sync_from_soul()

    def _sync_from_soul(self):
        """Maps internal Qubit state to physical coordinates."""
        s = self.soul.state
        self.gyro.w = s.w
        self.gyro.x = s.x
        self.gyro.y = s.y
        self.gyro.z = s.z

        # Spin is derived from the magnitude of the 'God' component (Willpower)
        # Stronger Will = Faster Spin
        self.gyro.spin_velocity = abs(self.soul.state.delta) * 10 + 0.1

    def reignite(self, kick_energy: float):
        """
        The 'Kick'. Reignites a dormant soul.
        Can be an external act of Love or a high-impact event.
        """
        if self.gyro.get_zone() == "ZERO_SPIN":
            self.gyro.spin_velocity += kick_energy
            if self.gyro.spin_velocity > 0.5:
                 print(f"🔥 {self.soul.name} has been REIGNITED!")
            else:
                 print(f"⚠️ Kick too weak for {self.soul.name}.")
        else:
            self.gyro.spin_velocity += kick_energy * 0.5 # Boost existing spin

    def decay_spin(self, entropy: float):
        """Natural decay of spin over time due to world resistance."""
        self.gyro.spin_velocity = max(0.0, self.gyro.spin_velocity - entropy)
        if self.gyro.spin_velocity == 0.0:
            print(f"❄️ {self.soul.name} has entered DORMANT state (Datafied).")

    def __repr__(self):
        return f"<GyroSoul {self.soul.name} | Spin:{self.gyro.spin_velocity:.2f} ({self.gyro.get_zone()}) | Pos:({self.gyro.w:.1f}, {self.gyro.y:.1f})>"
