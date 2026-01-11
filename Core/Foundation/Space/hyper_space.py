"""
HyperSpace: The 7-Dimensional Void
==================================

"The Field is primary. The Particle is secondary. The Layer is Maturity."

This module defines `HyperSpace`, the coordinate system of Elysia's mind.
Updated to support **Concept Layers** (Point/Plane/Solid).
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Deque, Union
from collections import deque
from Core.Foundation.hyper_quaternion import Quaternion
from Core.Foundation.Law.law_of_resonance import LawOfResonance
from Core.Foundation.Schema.layers import ConceptLayer, PointLayer

@dataclass
class FieldKnot:
    """A concentration of energy in the field."""
    id: str
    position: List[float] # 7D
    spin: Quaternion      # 4D
    mass: float           # Amplitude

    # [NEW] Maturity Layer
    schema: ConceptLayer = field(default_factory=lambda: PointLayer(name="Seed"))

    # Trajectory
    trajectory: Deque[List[float]] = field(default_factory=lambda: deque(maxlen=50))
    velocity: List[float] = field(default_factory=lambda: [0.0]*7)

    def record_position(self):
        self.trajectory.append(list(self.position))

class HyperSpace:
    def __init__(self):
        self.knots: Dict[str, FieldKnot] = {}
        self.connections: Dict[str, List[str]] = {} # Adjacency List for Evolution
        self.time = 0.0

    def add_knot(self, id: str, pos: List[float], spin: Quaternion, mass: float):
        if len(pos) != 7: pos = pos + [0.0]*(7-len(pos))

        # Default name matches ID
        schema = PointLayer(name=id, description="Raw Input")
        self.knots[id] = FieldKnot(id, pos, spin, mass, schema=schema)
        self.connections[id] = []

    def update_field(self, dt: float) -> List[str]:
        """
        Solves Field Equations & Updates Connections.
        """
        self.time += dt
        events = []
        keys = list(self.knots.keys())

        forces = {k: [0.0]*7 for k in keys}

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                k1 = self.knots[keys[i]]
                k2 = self.knots[keys[j]]

                f_mag = LawOfResonance.calculate_force(
                    k1.position, k1.spin, k1.mass,
                    k2.position, k2.spin, k2.mass
                )

                # Connection Logic (Synapse Formation)
                # If attraction is strong and sustained, they connect
                if f_mag > 50.0:
                    if keys[j] not in self.connections[keys[i]]:
                        self.connections[keys[i]].append(keys[j])
                        self.connections[keys[j]].append(keys[i])
                        # events.append(f"🔗 Synapse: {k1.id} - {k2.id}")

                # Physics
                direction = [(k2.position[d] - k1.position[d]) for d in range(7)]
                for d in range(7):
                    forces[keys[i]][d] += direction[d] * f_mag
                    forces[keys[j]][d] -= direction[d] * f_mag

        # Integration
        for k in keys:
            knot = self.knots[k]
            f = forces[k]
            knot.record_position()
            for d in range(7):
                accel = f[d] / (knot.mass + 0.1)
                knot.velocity[d] += accel * dt
                knot.position[d] += knot.velocity[d] * dt
                knot.velocity[d] *= 0.95

        return events
