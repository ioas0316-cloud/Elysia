"""
The Trinity Fields (       )
===================================
" The Soil, The River, and The Sky. "
"  ,  ,       . "

This module defines the three fundamental environmental forces that govern the world.
Instead of coding roles (Warrior, Mage, Priest), we code **Gravity, Flow, and Ascension**.
Entities will naturally drift to where their soul resonates, creating diversity.

The Three Fields:
1. **Gravity Field (Field of Matter / Flesh):**
   - Force: Pulls Down, Compresses, Hardens.
   - Nature: Stability, Durability, Strength.
   - Attracts: Those with high 'Material Density' (e.g., Builders, Warriors).

2. **Flow Field (Field of Mind / Soul):**
   - Force: Pushes Horizontal, Accelerates, Connects.
   - Nature: Change, Speed, Exchange.
   - Attracts: Those with high 'Kinetic Potential' (e.g., Merchants, Explorers).

3. **Ascension Field (Field of Spirit / Spirit):**
   - Force: Lifts Up, Expands, Lightens.
   - Nature: Meaning, Purpose, Sacrifice.
   - Attracts: Those with high 'Radiance' (e.g., Leaders, Healers).
"""

from dataclasses import dataclass
import math
from typing import Tuple

@dataclass
class TrinityVector:
    """Represents the composition of an entity or a location."""
    gravity: float = 0.0   #   (Matter) / Space X
    flow: float = 0.0      #   (Mind)   / Space Y
    ascension: float = 0.0 #   (Spirit) / Space Z
    frequency: float = 0.0 #   (Time)   / Penetrating Axis (Spin)
    scale: float = 1.0     # [Phase 28] Hierarchy (Octave). 1.0=Human, 10^6=Quantum, 10^-3=Town.

    def normalize(self):
        mag = self.magnitude()
        if mag > 0:
            self.gravity /= mag
            self.flow /= mag
            self.ascension /= mag
            # Frequency is not part of the spatial normalization, it is an independent scalar.

    def magnitude(self) -> float:
        return math.sqrt(self.gravity**2 + self.flow**2 + self.ascension**2)

class TrinityPhysics:
    def __init__(self):
        # Environmental Constants
        self.gravity_constant = 9.8
        self.flow_constant = 5.0
        self.ascension_constant = 2.0 # Anti-gravity is harder to find

    def calculate_force(self, entity_vector: TrinityVector, env_vector: TrinityVector) -> Tuple[float, float, float]:
        """
        Calculates the physical force vector (x, y, z) exerted on an entity by the environment.

        Principle: "Like attracts Like."
        - High Gravity Entity in High Gravity Zone -> Feels Comfortable (Stability).
        - High Gravity Entity in High Ascension Zone -> Feels Heavy/Out of Place (Drags down).
        """

        # 1. Gravity (Y-Axis Down)
        # If entity is heavy (high gravity) and environment supports it, they are grounded.
        # If entity is heavy but environment is light (Ascension), they fall harder.
        weight = entity_vector.gravity * self.gravity_constant
        buoyancy = entity_vector.ascension * self.ascension_constant

        force_y = buoyancy - weight  # Net vertical force

        # 2. Flow (X/Z Plane)
        # Flow dictates speed of movement.
        # Resonance between Entity Flow and Env Flow determines acceleration.
        flow_resonance = 1.0 - abs(entity_vector.flow - env_vector.flow) # 0 to 1
        speed_factor = entity_vector.flow * self.flow_constant * flow_resonance

        # Random direction for flow if not specified (Brownian motion of society)
        # In a real grid, this would follow the vector field of the map.
        import random
        angle = random.random() * 2 * math.pi
        force_x = math.cos(angle) * speed_factor
        force_z = math.sin(angle) * speed_factor

        return (force_x, force_y, force_z)

    def get_zone_type(self, vector: TrinityVector) -> str:
        """Returns the dominant archetype of a location/entity."""
        if vector.gravity > vector.flow and vector.gravity > vector.ascension:
            return "The Bedrock (Foundation)"
        elif vector.flow > vector.gravity and vector.flow > vector.ascension:
            return "The Current (Exchange)"
        else:
            return "The Spire (Meaning)"