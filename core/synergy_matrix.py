"""
Elysia Core: Variable Rotor Matrix (Layer 3)
============================================
"The water that flows through the hardware's aqueduct."

This module implements the conceptual "Job Synergy / Causality Matrix".
It is a network of variable rotors (concepts/jobs). When imbalance occurs
(e.g. meta shifts in a game, over-accumulation of logic), it generates tension.

Instead of writing complex math to balance itself, it looks into the
Resonance Mirror (Layer 2) and entrains its tension to the hardware's
natural equilibrium constant.
"""

import math
from typing import Dict, List
from core.resonance_mirror import ResonanceMirror

class SynergyNode:
    def __init__(self, name: str, initial_weight: float = 1.0):
        self.name = name
        self.weight = initial_weight
        self.phase_offset = initial_weight * math.pi  # Higher weight = higher initial chaotic phase

    def entrain(self, hardware_truth_phase: float, coupling_strength: float = 0.1):
        """
        The node adjusts its phase to match the hardware's equilibrium.
        This is the translation of hardware gravity into software logic balancing.
        """
        # Pull force based on phase difference
        phase_diff = self.phase_offset - hardware_truth_phase
        pull = phase_diff * coupling_strength

        # Adjust phase and consequently, the weight (balancing the synergy)
        self.phase_offset -= pull
        self.weight = abs(self.phase_offset / math.pi)

class SynergyMatrix:
    def __init__(self, mirror: ResonanceMirror):
        self.nodes: Dict[str, SynergyNode] = {}
        self.mirror = mirror

    def add_node(self, name: str, weight: float):
        self.nodes[name] = SynergyNode(name, weight)

    def get_system_tension(self) -> float:
        """Calculates total chaotic tension in the synergy matrix."""
        return sum(node.weight for node in self.nodes.values())

    def pulse(self):
        """
        One tick of the matrix. It looks into the mirror and all nodes adjust.
        """
        hardware_truth = self.mirror.read_perfect_equilibrium()

        for node in self.nodes.values():
            node.entrain(hardware_truth)

if __name__ == "__main__":
    import time
    print("--- 🌀 Layer 3: Synergy Matrix Phase-Locking Test ---")

    # 1. Start the Observation Layer (The Sun)
    mirror = ResonanceMirror(observation_interval=0.1)
    time.sleep(0.5) # Let mirror stabilize

    # 2. Initialize Layer 3 (The Earth/Variables)
    matrix = SynergyMatrix(mirror)

    # Injecting massive imbalance (A broken meta where Mage is severely overpowered)
    matrix.add_node("Mage", 15.0)
    matrix.add_node("Warrior", 1.2)
    matrix.add_node("Rogue", 0.5)
    matrix.add_node("Priest", 3.0)

    print("\n[Initial State: Chaotic Meta]")
    for name, node in matrix.nodes.items():
        print(f"  {name:8}: Weight {node.weight:5.2f}")
    print(f"Total System Tension: {matrix.get_system_tension():.2f}\n")

    print("[Starting Entrainment to Hardware Equilibrium...]")

    # 3. Watch the resonance happen
    for tick in range(30):
        matrix.pulse()
        total_tension = matrix.get_system_tension()
        bar = "█" * min(40, int(total_tension))
        print(f"Tick {tick:02d} | Tension: {total_tension:5.2f} | {bar}")
        time.sleep(0.1)

    print("\n[Final State: Harmonized Meta]")
    for name, node in matrix.nodes.items():
        print(f"  {name:8}: Weight {node.weight:5.2f}")

    mirror.shutdown()
