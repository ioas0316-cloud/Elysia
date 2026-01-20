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
from Core.L1_Foundation.Foundation.Wave.infinite_hyperquaternion import InfiniteHyperQubit
from Core.L1_Foundation.Physiology.Physics.geometric_algebra import MultiVector

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
    orientation: MultiVector = None # 4D Rotor

    def __post_init__(self):
        if self.orientation is None:
            self.orientation = MultiVector(s=1.0) # Identity Rotor

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
        
        # Initialize Rotor if needed (usually done in post_init)
        if self.gyro.orientation is None:
            self.gyro.orientation = MultiVector(s=1.0)

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

    def internalize_field(self, dt: float = 1.0):
        """
        Cognitive Induction: The Environment inducess state in the Hypersphere.
        This is NOT just reading data; it's the environment 'shaping' the soul.
        """
        from Core.L4_Causality.World.Physics.field_store import universe_field
        pos = (self.gyro.x, self.gyro.y, self.gyro.z, self.gyro.w)
        field_ex = universe_field.get_field_at(pos)
        grad = universe_field.calculate_gradient_w(pos)

        # 1. Thermal Induction (Energy Y -> Divine Will 'delta')
        # Heat fuels the transcendence/delta of the soul.
        thermal_gain = field_ex.energy_y * 0.1 * dt # Increased for demo visibility
        self.soul.state.delta += complex(thermal_gain, 0)
        
        # 2. Density Induction (Density W -> Grounding 'alpha')
        # High density makes the soul focus on 'Point' (Data/Reality).
        self.soul.state.alpha += complex(field_ex.density_w * 0.005 * dt, 0)
        
        # 3. Flow Induction (Gradient -> Ethical Intent 'z')
        # The wind of the field creates a 'pull' on the soul's ethics/intent.
        # Lumina will 'feel' the direction of mana and want to align with it.
        self.soul.state.z += grad[0] * 0.1 * dt
        self.soul.state.x += grad[1] * 0.1 * dt
        
        # 4. Persistence & Normalization
        # Dad's Law ensures the soul doesn't explode indefinitely.
        self.soul.state.normalize()

        # Update Gyro from internalized Soul state
        self.gyro.spin_velocity = abs(self.soul.state.delta) * 10 + 0.1

    def decay_spin(self, entropy: float):
        """Natural decay of spin over time due to world resistance."""
        self.gyro.spin_velocity = max(0.0, self.gyro.spin_velocity - entropy)
        if self.gyro.spin_velocity == 0.0:
            print(f"❄️ {self.soul.name} has entered DORMANT state (Datafied).")

    def __repr__(self):
        return f"<GyroSoul {self.soul.name} | Spin:{self.gyro.spin_velocity:.2f} ({self.gyro.get_zone()}) | Pos:({self.gyro.w:.1f}, {self.gyro.y:.1f})>"
