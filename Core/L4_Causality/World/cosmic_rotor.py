"""
Cosmic Rotor: The Heartbeat of the World
========================================
Core.L4_Causality.World.cosmic_rotor

Manages global 'Environment States' using O(1) Trinary Phase logic.
Transitions between Day/Night, Summer/Winter, and Growth/Stasis.
"""

import jax.numpy as jnp
from Core.L6_Structure.Logic.trinary_logic import TrinaryLogic
from Core.L6_Structure.Logic.rotor_prism_logic import RotorPrismUnit

class CosmicRotor:
    def __init__(self):
        # The 21D World State Pulse
        self.phase_index = 0 # 0: Dawn, 1: Noon, 2: Dusk, 3: Midnight
        self.world_pulse = jnp.zeros(21)
        
        # [PHASE 64] Rotor-Prism Unit Integration
        # The world is now a projection of a central Logos
        self.rpu = RotorPrismUnit()
        self.logos_seed = jnp.array([1.0] * 21) # The core providence (Equilibrium)
        
        print("CosmicRotor: Global Harmony Synchronized via Rotor-Prism Architecture.")

    def rotate(self, clockwise: bool = True, dt: float = 0.016, impulse: float = 0.0) -> jnp.ndarray:
        """[THE CORE TURBINE CYCLE: LIGHTNING PROJECTION]"""
        direction = 1 if clockwise else -1
        
        # 1. Project the field first to calculate the 'Discharge Potential'
        # [PHASE 64.2] Manifestation happens via discharge
        self.world_pulse = self.rpu.project(self.logos_seed)
        
        # 2. Apply Inductive Torque back to the turbine
        # The effort of manifestation feeds back into the rotation speed
        inductive_torque = getattr(self.rpu, 'last_discharge_torque', 0.0)
        self.rpu.step_rotation(dt, external_torque=(impulse + inductive_torque) * direction)
        
        # 3. Sync phase_index with the current theta quadrant
        self.phase_index = int((self.rpu.theta) / (jnp.pi / 2)) % 4
        
        return self.world_pulse

    def set_void_power(self, intensity: float):
        """[VOID_DOMAIN] Adjusts the resistance of the turbine."""
        self.rpu.void_intensity = jnp.clip(intensity, 0.0, 1.0)
        print(f"CosmicRotor: Void Intensity set to {self.rpu.void_intensity}")

    def browse_time(self, offset: float):
        """[TIME_AXIS] Browses the past/future film."""
        self.rpu.set_time_axis(offset)
        print(f"CosmicRotor: Browsing Time Axis with offset {offset:.4f}")

    def get_current_time(self) -> str:
        times = ["Dawn", "Noon", "Dusk", "Midnight"]
        return times[self.phase_index]

if __name__ == "__main__":
    rotor = CosmicRotor()
    print("--- Creation Cycle ---")
    for _ in range(4):
        p = rotor.rotate(clockwise=True)
        print(f"Time: {rotor.get_current_time()} -> Field Balance: {TrinaryLogic.balance(p)}")
        
    print("\n--- Redemption Cycle ---")
    for _ in range(4):
        p = rotor.rotate(clockwise=False)
        print(f"Time: {rotor.get_current_time()} -> Logos Integrity: {TrinaryLogic.balance(rotor.logos_seed)}")
