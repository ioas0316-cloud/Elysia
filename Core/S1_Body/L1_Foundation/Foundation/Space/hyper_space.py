"""
HyperSpace: The 7-Dimensional Void
==================================

"The Field is primary. The Particle is secondary. The Path is Meaning."

This module defines `HyperSpace`, the coordinate system of Elysia's mind.
Updated to track **Trajectories** (History of Movement).
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Deque
from collections import deque
from Core.S1_Body.L6_Structure.hyper_quaternion import Quaternion
from Core.S1_Body.L1_Foundation.Foundation.Law.law_of_resonance import LawOfResonance

@dataclass
class FieldKnot:
    """A concentration of energy in the field (a 'Thing')."""
    id: str
    position: List[float] # 7D: [P, F, E, C, M, S, Z]
    spin: Quaternion      # 4D Orientation
    mass: float           # Amplitude
    velocity: List[float] = field(default_factory=lambda: [0.0]*7)

    # [NEW] Trajectory History
    # We store the last N positions to analyze the "Shape of Thought"
    trajectory: Deque[List[float]] = field(default_factory=lambda: deque(maxlen=50))

    def record_position(self):
        """Snapshots the current position into history."""
        # Deep copy the list to avoid reference issues
        self.trajectory.append(list(self.position))

class HyperSpace:
    def __init__(self):
        # The Field is a collection of Knots
        self.knots: Dict[str, FieldKnot] = {}
        self.time = 0.0

    def add_knot(self, id: str, pos: List[float], spin: Quaternion, mass: float):
        if len(pos) != 7: pos = pos + [0.0]*(7-len(pos))
        self.knots[id] = FieldKnot(id, pos, spin, mass)

    def update_field(self, dt: float) -> List[str]:
        """
        Solves the Field Equations for the next time step.
        """
        self.time += dt
        events = []

        # N-Body Simulation with 7D Resonance Law
        keys = list(self.knots.keys())
        forces = {k: [0.0]*7 for k in keys}

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                k1 = self.knots[keys[i]]
                k2 = self.knots[keys[j]]

                # Calculate Force Scalar
                f_mag = LawOfResonance.calculate_force(
                    k1.position, k1.spin, k1.mass,
                    k2.position, k2.spin, k2.mass
                )

                # Direction Vector
                direction = [(k2.position[d] - k1.position[d]) for d in range(7)]

                # Apply Force Vector
                for d in range(7):
                    forces[keys[i]][d] += direction[d] * f_mag
                    forces[keys[j]][d] -= direction[d] * f_mag # Newton's 3rd

                # Event Detection (Singularity)
                if abs(f_mag) > 50.0:
                    event_type = "Resonance" if f_mag > 0 else "Dissonance"
                    # events.append(f"  Field Event: {event_type} between '{k1.id}' and '{k2.id}' (F:{f_mag:.1f})")

        # Integration (Update Position)
        for k in keys:
            knot = self.knots[k]
            f = forces[k]

            # Record before update
            knot.record_position()

            for d in range(7):
                # F = ma -> a = F/m
                accel = f[d] / (knot.mass + 0.1) # Damping
                knot.velocity[d] += accel * dt
                knot.position[d] += knot.velocity[d] * dt

                # Friction (Field Drag)
                knot.velocity[d] *= 0.95

        return events
