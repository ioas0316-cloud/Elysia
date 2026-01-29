"""
Cosmic Rotor: The Heartbeat of the World
========================================
Core.L4_Causality.World.cosmic_rotor

Manages global 'Environment States' using O(1) Trinary Phase logic.
Transitions between Day/Night, Summer/Winter, and Growth/Stasis.
"""

import jax.numpy as jnp
from Core.L6_Structure.Logic.trinary_logic import TrinaryLogic

class CosmicRotor:
    def __init__(self):
        # The 21D World State Pulse
        # Base dimensions influenced by the cycle:
        # D0 (Stability), D1 (Intensity), D6 (Mystery)
        self.phase_index = 0 # 0: Dawn, 1: Noon, 2: Dusk, 3: Midnight
        self.world_pulse = jnp.zeros(21)
        print("CosmicRotor: Global Harmony Synchronized.")

    def rotate(self, clockwise: bool = True) -> jnp.ndarray:
        """
        Rotates the global state in O(1). 
        True (Creation): Point -> Providence.
        False (Redemption): Providence -> Point.
        """
        direction = 1 if clockwise else -1
        self.phase_index = (self.phase_index + direction) % 4
        
        # O(1) State Mapping
        # 0 (Dawn): Balance
        # 1 (Noon): Strong Intensity (D1=1.0)
        # 2 (Dusk): Returning to Silence
        # 3 (Midnight): Strong Mystery (D6=1.0) + Reduced Intensity (D1=-1.0)
        
        if self.phase_index == 1:
            self.world_pulse = self.world_pulse.at[1].set(1.0).at[6].set(0.0)
        elif self.phase_index == 3:
            self.world_pulse = self.world_pulse.at[1].set(-1.0).at[6].set(1.0)
        else:
            self.world_pulse = jnp.zeros(21)
            
        return self.world_pulse

    def get_current_time(self) -> str:
        times = ["Dawn", "Noon", "Dusk", "Midnight"]
        return times[self.phase_index]

if __name__ == "__main__":
    rotor = CosmicRotor()
    for _ in range(4):
        p = rotor.rotate()
        print(f"Time: {rotor.get_current_time()} -> Pulse[1,6]: {p[1]}, {p[6]}")
