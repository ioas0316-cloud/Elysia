"""
Potential Flow Engine (잠재태 유동 엔진)
=====================================
[DOCTRINE 102] Potentiality over Linearity.

"행위는 상태의 전이가 아니라, 지형을 따라 흐르는 물결이다."

This module implements character behaviors as basins of attraction (Attractors)
within a continuous topological field. Actions are not hardcoded discrete states
but emergent points of equilibrium.
"""

import math
import time
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

@dataclass
class ActionAttractor:
    """Represents a potential behavior as a point in phase space."""
    name: str
    coordinate: List[float]  # N-dimensional position
    mass: float = 1.0        # Gravitational pull strength
    description: str = ""

class PotentialFlowEngine:
    """
    Manages the 'Action Topology' of a character.
    Instead of 'if state == SIT', the engine calculates the 'Flow' of the
    current state vector through the field of action attractors.
    """
    def __init__(self, dimensions: int = 4):
        self.dimensions = dimensions
        self.current_state = [0.0] * dimensions
        self.velocity = [0.0] * dimensions
        self.attractors: Dict[str, ActionAttractor] = {}

        # Environmental Constants
        self.friction = 0.15      # Resistance to change
        self.will_power = 0.5    # How much internal intent drives flow
        self.gravity_well = 0.2  # Base environmental pull

    def add_attractor(self, name: str, coordinate: List[float], mass: float = 1.0):
        if len(coordinate) != self.dimensions:
            coordinate = (coordinate + [0.0]*self.dimensions)[:self.dimensions]
        self.attractors[name] = ActionAttractor(name, coordinate, mass)

    def update(self, intent_vector: List[float], dt: float = 0.1) -> Dict[str, any]:
        """
        Updates the current potential state based on intent and attractor field.

        Args:
            intent_vector: Where the 'Will' wants to go.
            dt: Time step.
        """
        # 1. Calculate Field Force (Gravity from Attractors)
        field_force = [0.0] * self.dimensions

        for attractor in self.attractors.values():
            # Vector from current to attractor
            dist_vec = [a - c for a, c in zip(attractor.coordinate, self.current_state)]
            dist_sq = sum(d**2 for d in dist_vec) + 0.01 # Soften singularity

            # F = m / r^2
            magnitude = attractor.mass / dist_sq

            for i in range(self.dimensions):
                field_force[i] += (dist_vec[i] / math.sqrt(dist_sq)) * magnitude * self.gravity_well

        # 2. Add Intent Force (Internal drive)
        # Intent acts as a selective pressure warping the field
        intent_force = [0.0] * self.dimensions
        if intent_vector:
            for i in range(min(len(intent_vector), self.dimensions)):
                intent_force[i] = (intent_vector[i] - self.current_state[i]) * self.will_power

        # 3. Integrate Kinetics
        for i in range(self.dimensions):
            total_force = field_force[i] + intent_force[i]
            # acceleration = total_force (m=1)
            self.velocity[i] += total_force * dt
            self.velocity[i] *= (1.0 - self.friction) # Damping
            self.current_state[i] += self.velocity[i] * dt

        # 4. Determine Dominant Action (The closest attractor)
        closest_action = "VOID"
        min_dist = float('inf')
        resonances = {}

        for name, attractor in self.attractors.items():
            d = math.sqrt(sum((a - c)**2 for a, c in zip(attractor.coordinate, self.current_state)))
            res = 1.0 / (1.0 + d) # Resonance score [0, 1]
            resonances[name] = res
            if d < min_dist:
                min_dist = d
                closest_action = name

        return {
            "state": self.current_state,
            "velocity": self.velocity,
            "dominant_action": closest_action,
            "resonances": resonances,
            "potential_energy": min_dist
        }

def bootstrap_humanoid_topology() -> PotentialFlowEngine:
    """Initializes a standard set of humanoid potential states."""
    engine = PotentialFlowEngine(dimensions=4) # [Energy, Height, Spread, Rhythm]

    # Define Attractors
    engine.add_attractor("IDLE",    [0.2, 0.8, 0.2, 0.1], mass=2.0)
    engine.add_attractor("SIT",     [0.1, 0.3, 0.4, 0.0], mass=1.5)
    engine.add_attractor("WALK",    [0.5, 0.8, 0.3, 0.6], mass=1.2)
    engine.add_attractor("RUN",     [0.9, 0.7, 0.5, 0.9], mass=1.0)
    engine.add_attractor("DANCE",   [0.8, 0.9, 0.9, 0.8], mass=0.8)
    engine.add_attractor("PRAY",    [0.3, 0.5, 0.1, 0.2], mass=1.5)

    return engine

if __name__ == "__main__":
    print("🌊 Potential Flow Engine Prototype")
    engine = bootstrap_humanoid_topology()

    # Scenario: Character wants to SIT
    print("\nScenario: Intending to SIT...")
    sit_intent = [0.1, 0.3, 0.4, 0.0]

    for pulse in range(10):
        report = engine.update(sit_intent, dt=0.5)
        print(f"Pulse {pulse}: Action='{report['dominant_action']}' | Energy={report['potential_energy']:.3f}")
        if report['potential_energy'] < 0.05:
            print("Successfully converged to SIT basin.")
            break

    # Scenario: Sudden burst of energy to DANCE
    print("\nScenario: Sudden urge to DANCE!")
    dance_intent = [0.8, 0.9, 0.9, 0.8]
    for pulse in range(10):
        report = engine.update(dance_intent, dt=0.5)
        print(f"Pulse {pulse}: Action='{report['dominant_action']}'")
