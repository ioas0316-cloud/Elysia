import math
from typing import Tuple, Dict
from .cognitive_terrain import CognitiveTerrain

class CausalFlowEngine:
    """
    The Physics Engine that drives Monad movement based on Structural Logos.
    It replaces 'if-else' logic with 'Gradient + Inertia'.
    """

    def __init__(self, terrain: CognitiveTerrain):
        self.terrain = terrain
        self.friction_coefficient = 0.1  # Base friction
        self.inertia_factor = 0.9        # Conservation of momentum (0.9 = 90% retained)

    def calculate_next_state(self, x: float, y: float, vx: float, vy: float) -> Dict[str, float]:
        """
        Calculates the next position and velocity of a thought-particle (Monad).

        Physics Model:
        1. Gravity: Calculate slope (Gradient). Downhill = Acceleration.
        2. Viscosity: Check density. High density = High drag.
        3. Inertia: Apply previous velocity.
        4. Update: New Pos = Old Pos + New Velocity.
        """

        # 1. Get Gradient (The "Pull" of the structure)
        grad_x, grad_y = self.terrain.get_gradient(x, y)

        # 2. Get Viscosity (The "Resistance" of the medium)
        viscosity = self.terrain.get_viscosity(x, y)

        # 3. Calculate Acceleration
        # Force = Gradient. Acceleration = Force / Mass (Assume Mass=1 for now)
        # We amplify gradient to make movement visible
        ax = grad_x * 0.5
        ay = grad_y * 0.5

        # 4. Update Velocity with Inertia and Friction/Viscosity
        # New Velocity = (Old Velocity * Inertia) + Acceleration - (Drag)
        # Drag is proportional to Velocity * Viscosity

        new_vx = (vx * self.inertia_factor) + ax
        new_vy = (vy * self.inertia_factor) + ay

        # Apply Drag (Damping)
        # High viscosity -> High damping
        damping = 1.0 / (1.0 + (viscosity * self.friction_coefficient))

        new_vx *= damping
        new_vy *= damping

        # 5. Update Position
        new_x = x + new_vx
        new_y = y + new_vy

        # 6. Apply Erosion (Feedback Loop)
        # The movement itself carves the path deeper.
        flow_intensity = math.sqrt(new_vx**2 + new_vy**2)
        if flow_intensity > 0.01:
            self.terrain.apply_erosion(x, y, flow_intensity)

        return {
            "x": new_x,
            "y": new_y,
            "vx": new_vx,
            "vy": new_vy,
            "energy": flow_intensity
        }

    def resolve_boundary(self, x: float, y: float) -> Tuple[float, float]:
        """Keeps the particle within the map bounds (Reflective or Clamping)."""
        limit = self.terrain.resolution - 1

        # Bounce effect? Or just clamp?
        # For a "Valley", clamping is safer.
        new_x = max(0, min(x, limit))
        new_y = max(0, min(y, limit))

        return new_x, new_y
