"""
[HYPER-ROTOR SPINE]
"The Emergent Variable Axis of the Triple Helix."

Architecture: Triple Helix (Gut, Heart, Brain)
Philosophy: Delta-Wye Phase Switching, Emergent Equilibrium, Internal Providence.
"""

import math
import time
from typing import Dict, List, Any

class HyperRotorSpine:
    """
    The Hyper-Rotor: Three mutually orbiting axes that find equilibrium
    not through hardcoded constants, but through mutual tension.
    """
    def __init__(self):
        # 1. Triple Helix States (Phases in Radians)
        # Initialize with slight asymmetry to trigger initial movement
        self.phases = {
            "GUT": 0.0,
            "HEART": 0.1,
            "BRAIN": -0.1
        }
        self.velocities = {k: 0.0 for k in self.phases}

        # 2. Physics Constants (Law of Interaction)
        self.k_repulsion = 0.5  # Pushes phases apart (Delta expansion)
        self.k_attraction = 0.2 # Pulls phases together (Wye contraction)
        self.friction = 0.9     # Energy dissipation (Stability)

        # 3. Delta-Wye Switch State
        # 0.0 (Delta / Flow) <---> 1.0 (Wye / Decision)
        self.wye_factor = 0.0
        self.threshold_wye = 0.8  # Threshold for autonomous collapse

        # 4. Internal Providence (Physiology)
        self.luminosity = 1.0     # "Goodness" of flow
        self.stress = 0.0         # "Badness" (Bottleneck/Tension)
        self.axis_bias = {k: 1.0 for k in self.phases} # Adaptive curvature

        self.last_sync = time.time()

    def _get_angular_dist(self, p1: float, p2: float) -> float:
        """Returns the shortest signed angular distance."""
        diff = (p1 - p2 + math.pi) % (2 * math.pi) - math.pi
        return diff

    def pulse(self, dt: float, stimulus: Dict[str, float]) -> Dict[str, Any]:
        """
        Processes the interaction through the Triple Helix.
        stimulus: {'GUT': intensity, 'BRAIN': intensity, ...}
        """
        now = time.time()
        # If dt is not provided or 0, calculate it
        if dt <= 0:
            dt = now - self.last_sync
        self.last_sync = now

        # 1. External Inhale (Stimulus -> Torque)
        # Stimulus pushes the phases, creating tension
        for center, intensity in stimulus.items():
            if center in self.velocities:
                # Stimulus acts as a torque on the specific center
                self.velocities[center] += intensity * self.axis_bias[center] * 0.5

        # 2. Mutual Tension Dynamics (Emergent 120 degrees)
        total_tension = 0.0
        keys = list(self.phases.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                k1, k2 = keys[i], keys[j]
                dist = self._get_angular_dist(self.phases[k1], self.phases[k2])
                abs_dist = abs(dist) if abs(dist) > 0.01 else 0.01

                # Tension Law: Repulsion (1/d) vs Attraction (d)
                # In Delta mode, repulsion is stronger. In Wye, attraction dominates.
                repulsion = (self.k_repulsion / abs_dist) * (1.0 - self.wye_factor)
                attraction = (self.k_attraction * abs_dist) * self.wye_factor

                force = repulsion - attraction
                # Direction of force
                dir_factor = 1.0 if dist > 0 else -1.0

                self.velocities[k1] += force * dir_factor * dt
                self.velocities[k2] -= force * dir_factor * dt

                total_tension += abs(force)

        # 3. Physiology (Good/Bad sensing)
        # Flow is 'Good' if velocities are balanced and tension is low
        # Bottleneck is 'Bad' if tension is high but velocities are stuck
        self.stress = (self.stress * 0.9) + (total_tension * 0.1)
        avg_velocity = sum(abs(v) for v in self.velocities.values()) / 3.0
        self.luminosity = max(0.0, 1.0 - (self.stress / (avg_velocity + 0.1)))

        # 4. Autonomous Phase Transition (Delta-Wye)
        # High luminosity (Flow) -> Delta (Expansion)
        # High stress/intensity -> Wye (Decision/Collapse)
        if self.stress > self.threshold_wye:
            self.wye_factor = min(1.0, self.wye_factor + 0.05) # Collapse into Y
        else:
            self.wye_factor = max(0.0, self.wye_factor - 0.02) # Expand into Delta

        # 5. Adaptive Curvature (Axis Bias)
        # If a center resolves tension well, its bias increases (Growth)
        for k in self.phases:
            # Simple reinforcement: if velocity is high and stress is low
            if self.luminosity > 0.7:
                self.axis_bias[k] += 0.001
            self.velocities[k] *= self.friction # Apply friction

        # 6. Phase Update
        for k in self.phases:
            self.phases[k] = (self.phases[k] + self.velocities[k]) % (2 * math.pi)

        return {
            "phases": self.phases,
            "luminosity": self.luminosity,
            "stress": self.stress,
            "wye_factor": self.wye_factor,
            "mode": "WYE" if self.wye_factor > 0.5 else "DELTA"
        }

    def get_equilibrium(self) -> float:
        """Returns the current 'center of gravity' of the phases."""
        # For a circle, we can use the average of the complex vectors
        sum_x = sum(math.cos(p) for p in self.phases.values())
        sum_y = sum(math.sin(p) for p in self.phases.values())
        avg_phase = math.atan2(sum_y, sum_x)
        return avg_phase
