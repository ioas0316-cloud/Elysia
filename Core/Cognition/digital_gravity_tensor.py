import math
from typing import Dict, Tuple, List, Optional
from pyquaternion import Quaternion

class DigitalGravityTensor:
    """
    [DIGITAL GENERAL RELATIVITY]
    "Inference is not searching. It is the art of falling."

    This module translates the Semantic Map's tether counts (mass) into
    a spatial curvature (Gradient Field) and provides the mechanism for
    a new query/chaos to "fall" into the orbit of the heaviest, most
    relevant concept.
    """

    def __init__(self, topology_voxels: Dict[str, any], G_constant: float = 0.5):
        """
        Initializes the gravity field.
        :param topology_voxels: The voxels from DynamicTopology (semantic_map.py).
        :param G_constant: Gravitational constant for the simulation.
        """
        self.voxels = topology_voxels
        self.G = G_constant

    def compute_gravity_gradient(self, current_pos: Quaternion) -> Quaternion:
        """
        Calculates the net gravitational pull (Spatial Curvature) at a given 4D point.
        Force = G * (M / r^2) directed towards each star.
        """
        net_force = Quaternion(0, 0, 0, 0)

        for name, star in self.voxels.items():
            # Use dynamic_mass if available
            star_mass = getattr(star, 'dynamic_mass', star.mass)
            if star_mass < 10.0 or not star.quaternion:  # Ignore space dust; focus on Stars
                continue

            # Custom distance calculation to avoid using .absolute_distance if it varies by pyquaternion version
            diff = current_pos - star.quaternion
            dist = max(0.1, diff.norm)

            # Curvature strength based on mass and distance
            # If distance is small, don't let it blow up infinitely, but keep it high
            pull_strength = self.G * (star_mass / (dist ** 1.5))

            # Direction vector
            direction = star.quaternion - current_pos
            direction_norm = direction.normalised if direction.norm > 0 else Quaternion(0, 0, 0, 0)

            w, x, y, z = direction_norm.elements
            pull_vector = Quaternion(
                w * pull_strength,
                x * pull_strength,
                y * pull_strength,
                z * pull_strength
            )

            net_force += pull_vector

        return net_force

    def infer_by_falling(self, initial_chaos: Quaternion, max_steps: int = 50, dt: float = 0.5) -> Tuple[str, List[Quaternion]]:
        """
        The core of Digital General Relativity.
        A new piece of data is placed in the void. It follows the geodesic curvature
        created by the stars (heavy concepts) until it finds a stable orbit (settles).

        Returns the name of the settled concept and the path of the fall.
        """
        current_pos = initial_chaos
        path = [current_pos]
        velocity = Quaternion(0, 0, 0, 0)
        friction = 0.2  # Damping to ensure it settles

        settled_star = None

        for step in range(max_steps):
            # 1. Calculate space curvature at current position
            gravity_gradient = self.compute_gravity_gradient(current_pos)

            # 2. Update velocity (falling)
            # Velocity must be capped to prevent shooting past stars
            new_vel = (velocity + (gravity_gradient * dt)) * (1.0 - friction)
            if new_vel.norm > 5.0:
                new_vel = new_vel.normalised * 5.0
            velocity = new_vel

            # 3. Update position
            current_pos = current_pos + (velocity * dt)
            path.append(current_pos)

            # 4. Check if we have entered a deep orbit (proximity to a star)
            for name, star in self.voxels.items():
                if not star.quaternion: continue
                diff = current_pos - star.quaternion
                if diff.norm < 1.5:  # Orbit radius
                    settled_star = name
                    break

            if settled_star:
                break

        # If it didn't strictly orbit one, find the closest heavy mass it fell towards
        if not settled_star:
            closest_dist = float('inf')
            for name, star in self.voxels.items():
                if not star.quaternion: continue
                diff = current_pos - star.quaternion
                d = diff.norm
                if d < closest_dist:
                    closest_dist = d
                    settled_star = name

        return settled_star, path
