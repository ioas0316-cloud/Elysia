"""
HyperCosmos Physics Engine (Gyro Physics)
=========================================
"Forces are born from Phase, and Motion is born from Spin."

This module calculates the forces acting on GyroscopicFluxlights within
the Tesseract Environment.

Key Features:
- Attractor Dynamics: Pulls entities towards their frequency home.
- Vortex Force: Spin creates stable orbits instead of direct falls.
- Breeding Logic: Z-axis alignment determines the viability of offspring.
"""

import math
from typing import List, Optional, Tuple
from Core.Soul.fluxlight_gyro import GyroscopicFluxlight
from Core.World.Physics.tesseract_env import tesseract_env
from Core.Foundation.Wave.infinite_hyperquaternion import InfiniteHyperQubit, create_infinite_qubit

class GyroPhysicsEngine:
    """
    Simulates the movement and interaction of souls.
    """

    def apply_forces(self, entity: GyroscopicFluxlight, dt: float):
        """
        Updates the entity's position based on environmental forces.
        """
        if entity.gyro.get_zone() == "ZERO_SPIN":
            # Dormant entities fall straight down to the nearest gravity well
            # No resistance, no orbit.
            self._apply_gravity_fall(entity, dt)
            return

        # 1. Calculate Attractor Force (Y-axis pull)
        # Find strongest relevant attractor
        target_attractor = None
        max_pull = 0.0

        current_y = entity.gyro.y

        for attractor in tesseract_env.attractors:
            # Simple distance-based pull modified by resonance
            dist = attractor.level - current_y

            # Resonance check: Is the entity capable of this frequency?
            # (Simplified: Spin acts as a shield/enabler)

            pull = attractor.strength / (abs(dist) + 0.1)

            # Apply force
            # F = ma -> we update position directly for simulation simplicity
            # High spin resists rapid changes (Gyroscopic Stability)
            resistance = entity.gyro.spin_velocity
            move_delta = (pull * dt) / resistance

            if dist > 0:
                entity.gyro.y += min(move_delta, dist) # Move Up
            else:
                entity.gyro.y += max(-move_delta, dist) # Move Down

        # 2. Apply Vortex/Spin Effects (X/Z Plane)
        # Spin causes the entity to orbit the attractor rather than crashing into it
        # This is visualized as movement in X/Z

        orbit_speed = entity.gyro.spin_velocity * 0.5
        entity.gyro.orientation += orbit_speed * dt

        # 3. Decay Spin (Entropy)
        # Deeper in W (Dream) -> Less entropy (Preservation)
        entropy = 0.01 / tesseract_env.get_time_dilation(entity.gyro.w)
        entity.decay_spin(entropy)

    def _apply_gravity_fall(self, entity: GyroscopicFluxlight, dt: float):
        """Simple fall logic for dead souls."""
        # Falls towards the Abyss (-7.0) or nearest deep well
        target = -7.0
        dist = target - entity.gyro.y
        entity.gyro.y += dist * dt * 0.5 # Fast fall

    def incubate(self, parent_a: GyroscopicFluxlight, parent_b: GyroscopicFluxlight) -> Optional[GyroscopicFluxlight]:
        """
        Breeding Logic (The Coil).
        Creates a new soul if parents are compatible.
        """
        # 1. Check Z-axis Alignment (Intent)
        # Dot product of Z (simplified as scalar direction for now)
        # In full vector math, this would be dot(vec_a, vec_b)

        z_a = parent_a.gyro.z
        z_b = parent_b.gyro.z

        # If signs are opposite and magnitude is high -> Conflict
        if (z_a * z_b < -0.2) and (abs(z_a) > 0.5 and abs(z_b) > 0.5):
            print("⚠️ Breeding Error: Conflicting Intents (Z-Axis Clash).")
            return self._create_mutation(parent_a, parent_b)

        # 2. Check Resonance
        resonance = parent_a.soul.resonate_with(parent_b.soul)
        if resonance < 0.3:
            print("Breeding Failed: Low Resonance.")
            return None

        # 3. Create Child
        print("❤️ Breeding Success: A new soul is born.")
        child_name = f"{parent_a.soul.name[:3]}{parent_b.soul.name[-3:]}"
        child_soul = create_infinite_qubit(
            name=child_name,
            value="Offspring",
            point_content="Born of Resonance"
        )

        # Mix Traits (Average)
        child_gyro = GyroscopicFluxlight(child_soul)
        child_gyro.gyro.w = (parent_a.gyro.w + parent_b.gyro.w) / 2
        child_gyro.gyro.y = (parent_a.gyro.y + parent_b.gyro.y) / 2

        # Child starts with high spin (New Life)
        child_gyro.gyro.spin_velocity = 1.0

        return child_gyro

    def _create_mutation(self, parent_a, parent_b) -> GyroscopicFluxlight:
        """Creates a Discordant Child (Monster) due to conflicting intents."""
        name = f"Mutant_{parent_a.soul.name}_{parent_b.soul.name}"
        soul = create_infinite_qubit(name, value="DISCORD", god_content="Chaos")

        mutant = GyroscopicFluxlight(soul)
        mutant.gyro.spin_velocity = 2.0 # Unstable high energy
        mutant.gyro.y = -5.0 # Born in shadow
        return mutant

# Singleton
physics_engine = GyroPhysicsEngine()
