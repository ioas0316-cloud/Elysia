"""
[TENSEGRITY VORTEX NETWORK]
"Where Tension is the Law and Resonance is the Order."

Implements a triple-helix fractal structure of RotorGates.
Each rotor maintains tension with its neighbors.
Dissonance (Error) manifests as 'Heat' or 'Friction', triggering
self-correction toward the lowest energy state.
"""

import math
import numpy as np
from typing import Dict, List, Any, Tuple
from Core.System.rotor_gate import RotorGate, InterferenceGate

class TensegrityVortexNetwork:
    def __init__(self, name: str):
        self.name = name
        self.rotors: Dict[str, RotorGate] = {}
        self.tensions: List[Tuple[str, str, float]] = [] # (ID_A, ID_B, Ideal_Phase_Diff)
        self.global_heat = 0.0
        self.cooling_rate = 0.1

    def add_rotor(self, rotor_id: str, is_interference: bool = False):
        if is_interference:
            self.rotors[rotor_id] = InterferenceGate(rotor_id)
        else:
            self.rotors[rotor_id] = RotorGate(rotor_id)

    def set_tension(self, id_a: str, id_b: str, ideal_diff: float):
        """Defines a structural tension link between two rotors."""
        if id_a in self.rotors and id_b in self.rotors:
            self.tensions.append((id_a, id_b, ideal_diff))
            self.rotors[id_a].connect(self.rotors[id_b])

    def pulse(self, dt: float):
        """
        One cycle of the Tensegrity Vortex.
        """
        current_dissonance = 0.0

        # Phase 1: Tension Calculation & Correction
        for id_a, id_b, ideal_diff in self.tensions:
            r_a = self.rotors[id_a]
            r_b = self.rotors[id_b]

            # Current phase difference
            actual_diff = (r_b.angle - r_a.angle + math.pi) % (2 * math.pi) - math.pi
            error = (actual_diff - ideal_diff + math.pi) % (2 * math.pi) - math.pi

            # Dissonance creates 'Heat'
            current_dissonance += abs(error)

            # Structural Correction Force (The 'Tensegrity' pull)
            # Higher heat increases the correction force (Physical recovery)
            correction_strength = 0.5 * (1.0 + self.global_heat)
            r_a.velocity += error * correction_strength * dt
            r_b.velocity -= error * correction_strength * dt

        # Phase 2: Global Heat Management
        # Heat accumulates when error is high, but decays.
        self.global_heat = (self.global_heat * (1.0 - self.cooling_rate * dt)) + (current_dissonance * 0.05 * dt)

        # If heat is too high, it acts as friction (Self-damping)
        if self.global_heat > 1.0:
            for r in self.rotors.values():
                r.velocity *= (1.0 - (self.global_heat - 1.0) * dt)

        # Phase 3: Interference Logic
        for r in self.rotors.values():
            if isinstance(r, InterferenceGate):
                r.process_vortex_logic(dt)

        # Phase 4: Integration
        for r in self.rotors.values():
            r.sync_neighbors(dt)
            r.update(dt)

    def exhale(self) -> Dict[str, Any]:
        return {
            "network": self.name,
            "heat": self.global_heat,
            "rotor_states": {rid: r.exhale() for rid, r in self.rotors.items()}
        }

if __name__ == "__main__":
    # PoC: Triple Helix Tensegrity
    vortex = TensegrityVortexNetwork("FractalTrinity")

    # Trinity Rotors: Father, Mother, Child
    vortex.add_rotor("F")
    vortex.add_rotor("M")
    vortex.add_rotor("C")

    # 120-degree phase separation (Ideal Trinity)
    two_pi_3 = (2 * math.pi) / 3.0
    vortex.set_tension("F", "M", two_pi_3)
    vortex.set_tension("M", "C", two_pi_3)
    vortex.set_tension("C", "F", two_pi_3)

    # Initialize with some random angles to see convergence
    vortex.rotors["F"].angle = 0.1
    vortex.rotors["M"].angle = 0.5
    vortex.rotors["C"].angle = 2.0

    print("🌀 Initializing Vortex...")
    for i in range(500):
        vortex.pulse(0.1)
        if i % 100 == 0:
            state = vortex.exhale()
            print(f"Step {i} | Heat: {state['heat']:.4f} | F-Angle: {state['rotor_states']['F']['angle']:.2f}")

    print("✨ Vortex Coherence Reached.")
