"""
Field Operators (The Laws of the Ether)
=======================================

"And the Spirit of God moved upon the face of the waters."

This module defines the **Dynamics** of the Ether.
These are not methods of an object, but **Operators** that act upon the Void.
They represent the fundamental forces of the Elysian Universe.

1. Gravity (Affinity): Contextual Attraction
2. Resonance (Connection): Frequency/Phase Linking
3. Entropy (Time): Decay and Stabilization
4. Expansion (Creativity): Anti-Gravity / Divergence
"""

import math
from typing import List
from Core.Ether.ether_node import EtherNode, Quaternion
from Core.Ether.void import Void

# Constants (The Fine-Tuning of the Universe)
G_CONST = 10.0          # Gravity Strength
R_CONST = 5.0           # Resonance Strength
K_ELASTIC = 0.5         # Spring constant for connected nodes (if we add springs)
MAX_FORCE = 50.0        # Force limiter to prevent explosions
MIN_DIST = 0.5          # Minimum distance (Pauli Exclusion Principle)

class FieldOperator:
    """Base class for all physical laws."""
    def apply(self, void: Void, dt: float):
        pass

class LawOfGravity(FieldOperator):
    """
    F = G * m1 * m2 / r^2

    BUT adapted for High-Dimensional Meaning Space:
    - Nodes attract if they have High Mass (Importance).
    - But they repel if they are too close (Exclusion).
    """
    def apply(self, void: Void, dt: float):
        nodes = void.get_all()
        n = len(nodes)
        if n < 2: return

        # O(N^2) - Naive implementation.
        # Optimized: Barnes-Hut or Grid-based in future.
        for i in range(n):
            node_a = nodes[i]
            if node_a.mass <= 0: continue

            # Accumulate force for node_a
            fx, fy, fz, fw = 0.0, 0.0, 0.0, 0.0

            for j in range(n):
                if i == j: continue
                node_b = nodes[j]

                # Vector B -> A
                dx = node_b.position.x - node_a.position.x
                dy = node_b.position.y - node_a.position.y
                dz = node_b.position.z - node_a.position.z
                dw = node_b.position.w - node_a.position.w

                dist_sq = dx*dx + dy*dy + dz*dz + dw*dw
                dist = math.sqrt(dist_sq) + 1e-6

                if dist < MIN_DIST:
                    # Repulsion (Too close)
                    force_mag = -MAX_FORCE / dist
                else:
                    # Attraction (Gravity)
                    # Modify Gravity by Resonance! (Affinity Gravity)
                    # If they resonate, they attract STRONGER.
                    resonance = node_a.resonate(node_b)
                    effective_mass = node_a.mass * node_b.mass * (1.0 + resonance * 5.0)
                    force_mag = min(MAX_FORCE, (G_CONST * effective_mass) / dist_sq)

                # Apply vector component
                fx += force_mag * (dx / dist)
                fy += force_mag * (dy / dist)
                fz += force_mag * (dz / dist)
                fw += force_mag * (dw / dist)

            # Apply to Node A
            force_vector = Quaternion(fw, fx, fy, fz)
            node_a.apply_force(force_vector, dt)

class LawOfResonance(FieldOperator):
    """
    Energy Transfer based on Resonance.

    If A and B resonate:
    1. They exchange Energy (Heat).
    2. They align their Phase (Synchronization).
    3. They align their Spin (Consensus).
    """
    def apply(self, void: Void, dt: float):
        nodes = void.get_active_nodes(threshold=0.1)
        for i, node_a in enumerate(nodes):
            for node_b in nodes[i+1:]:

                coeff = node_a.resonate(node_b)
                if coeff < 0.5: continue # Too weak to matter

                # 1. Energy Transfer (Flow from High to Low)
                energy_diff = node_a.energy - node_b.energy
                flow = energy_diff * coeff * dt * 0.5

                node_a.energy -= flow
                node_b.energy += flow

                # 2. Spin Alignment (Slerp-like pull)
                # Slowly rotate towards each other's perspective
                # (Simplified linear interpolation for performance)
                rate = coeff * dt * 0.1

                # Pull A towards B
                node_a.spin.w += (node_b.spin.w - node_a.spin.w) * rate
                node_a.spin.x += (node_b.spin.x - node_a.spin.x) * rate
                node_a.spin.y += (node_b.spin.y - node_a.spin.y) * rate
                node_a.spin.z += (node_b.spin.z - node_a.spin.z) * rate

                # Pull B towards A
                node_b.spin.w += (node_a.spin.w - node_b.spin.w) * rate
                node_b.spin.x += (node_a.spin.x - node_b.spin.x) * rate
                node_b.spin.y += (node_a.spin.y - node_b.spin.y) * rate
                node_b.spin.z += (node_a.spin.z - node_b.spin.z) * rate

                # Renormalize
                node_a.spin = node_a.spin.normalize()
                node_b.spin = node_b.spin.normalize()

class LawOfMotion(FieldOperator):
    """
    Standard Newtonian Motion integration + Entropy (Friction).
    """
    def apply(self, void: Void, dt: float):
        for node in void.get_all():
            node.move(dt, friction=0.05)


class DynamicsEngine:
    """
    The Engine that runs the Laws.
    """
    def __init__(self):
        self.laws: List[FieldOperator] = [
            LawOfGravity(),
            LawOfResonance(),
            LawOfMotion()
        ]

    def step(self, void: Void, dt: float):
        """Apply all laws for one time step."""
        for law in self.laws:
            law.apply(void, dt)
