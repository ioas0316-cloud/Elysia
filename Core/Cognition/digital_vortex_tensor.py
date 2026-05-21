import math
from typing import Dict, Tuple, List, Optional
from pyquaternion import Quaternion

class DigitalVortexTensor:
    """
    [DIGITAL GENERAL RELATIVITY - VORTEX MAELSTROM DYNAMICS]
    "Inference is not searching, nor straight falling. It is the art of swirling into the Vortex."

    This module translates the Semantic Map's tether counts (vortex tension) into
    a spatial rotational field (Maelstrom) and provides the mechanism for
    a new query/chaos to "spiral" into the core of the strongest, most
    relevant concept galaxy.
    """

    def __init__(self, topology_voxels: Dict[str, any], V_constant: float = 0.5):
        """
        Initializes the vortex field.
        :param topology_voxels: The voxels from DynamicTopology (semantic_map.py).
        :param V_constant: Vortex tension constant for the simulation.
        """
        self.voxels = topology_voxels
        self.V = V_constant

    def compute_vortex_gradient(self, current_pos: Quaternion) -> Quaternion:
        """
        Calculates the net vortex pull at a given 4D point.
        It generates an inward radial pull PLUS a tangential rotational force (Spiral Arm Effect).
        """
        net_force = Quaternion(0, 0, 0, 0)

        for name, galaxy in self.voxels.items():
            # Access the new vortex tension (or fallback to legacy mass)
            tension = getattr(galaxy, 'vortex_tension', getattr(galaxy, 'mass', 0.0))
            if tension < 10.0 or not galaxy.quaternion:  # Ignore space dust; focus on Galaxies
                continue

            # Vector from query to galaxy core
            diff = current_pos - galaxy.quaternion
            dist = max(0.1, diff.norm)

            # 1. Radial Inward Pull (Gravity)
            radial_strength = self.V * (tension / (dist ** 2))
            direction = galaxy.quaternion - current_pos
            direction_norm = direction.normalised if direction.norm > 0 else Quaternion(0, 0, 0, 0)

            rw, rx, ry, rz = direction_norm.elements
            radial_vector = Quaternion(
                rw * radial_strength,
                rx * radial_strength,
                ry * radial_strength,
                rz * radial_strength
            )

            # 2. Tangential Rotational Force (The Spiral)
            # We construct an orthogonal vector by swapping components and negating one (cross-product analog for 4D)
            # This forces the query to swirl around the galaxy core
            tw, tx, ty, tz = direction_norm.elements
            tangential_vector = Quaternion(
                -tx * radial_strength * 0.8, # The swirl is 80% as strong as the pull
                tw * radial_strength * 0.8,
                -tz * radial_strength * 0.8,
                ty * radial_strength * 0.8
            )

            # The total force is the sum of the inward pull and the swirling tangent
            net_force += radial_vector + tangential_vector

        return net_force

    def infer_by_accretion(self, initial_chaos: Quaternion, max_steps: int = 100, dt: float = 0.5) -> Tuple[str, List[Quaternion]]:
        """
        The core of Maelstrom Dynamics.
        A new piece of data is placed in the void. It gets caught in the spiral arms
        created by the galaxies (heavy concepts) and spins inward until it reaches the core (settles).

        Returns the name of the accreted concept and the spiraling path.
        """
        current_pos = initial_chaos
        path = [current_pos]
        velocity = Quaternion(0, 0, 0, 0)
        friction = 0.2  # Damping to ensure it settles

        settled_star = None

        for step in range(max_steps):
            # 1. Calculate vortex maelstrom force at current position
            vortex_gradient = self.compute_vortex_gradient(current_pos)

            # 2. Update velocity (spiraling)
            # Velocity must be capped to prevent shooting past galaxies
            new_vel = (velocity + (vortex_gradient * dt)) * (1.0 - friction)
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
