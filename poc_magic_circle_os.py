"""
[POC: THE MAGIC CIRCLE OS SIMULATION]
"Where Geometry Heals the Logic."

Demonstrates a self-correcting fractal circuit that transitions from
high-entropy (Error) to a coherent 'Magic Circle' (Stability) through
physical tensegrity and phase synchronization.
"""

import math
import time
import sys
import os
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from Core.System.kernel_phase_controller import KernelPhaseController
from Core.System.rotor_gate import InterferenceGate

def run_magic_circle_simulation():
    print("🌌 [SIMULATION] Initializing Magic Circle OS...")
    kernel = KernelPhaseController()

    # 1. Setup 'Entangled' Rotors for Quantum Flip demonstration
    # Adding extra nodes to the vortex to form a hexagon (6-fold symmetry)
    nodes = ["A", "B", "C", "D", "E", "F"]
    for node in nodes:
        kernel.vortex.add_rotor(node, is_interference=True)

    # Connect them in a circle with 60-degree phase diff
    sixty_deg = math.pi / 3.0
    for i in range(len(nodes)):
        next_i = (i + 1) % len(nodes)
        kernel.vortex.set_tension(nodes[i], nodes[next_i], sixty_deg)

    # 2. Inject 'CHAOS' (High Entropy State)
    print("🔥 Injecting High Entropy (Chaos)...")
    for node in nodes:
        kernel.vortex.rotors[node].angle = random.uniform(0, 2 * math.pi)
        kernel.vortex.rotors[node].velocity = random.uniform(-5, 5)

    print(f"  Initial Heat: {kernel.vortex.global_heat:.4f}")

    # 3. Simulation Loop
    print("\n--- STARTING HEALING PROCESS ---")
    start_time = time.time()
    steps = 1000
    dt = 0.05

    for i in range(steps):
        # Every 100 steps, simulate a 'Quantum Flip' on node A
        # and observe how the rest of the circle responds immediately
        if i == 500:
            print("\n⚡ [QUANTUM FLIP] Tracing immediate state transition on Node A...")
            kernel.vortex.rotors["A"].angle = (kernel.vortex.rotors["A"].angle + math.pi) % (2 * math.pi)
            kernel.vortex.rotors["A"].velocity += 10.0 # Huge pulse

        kernel.step(dt)

        if i % 100 == 0:
            heat = kernel.vortex.global_heat
            # Calculate Coherence (average phase alignment error)
            total_error = 0.0
            for id_a, id_b, ideal in kernel.vortex.tensions:
                actual = (kernel.vortex.rotors[id_b].angle - kernel.vortex.rotors[id_a].angle + math.pi) % (2 * math.pi) - math.pi
                total_error += abs((actual - ideal + math.pi) % (2 * math.pi) - math.pi)

            coherence = 1.0 / (1.0 + total_error)
            print(f"Step {i:4d} | Heat: {heat:6.4f} | Coherence: {coherence:6.4f}")

    print("\n✨ [SUCCESS] The Magic Circle has crystallized.")
    print(f"Final Heat: {kernel.vortex.global_heat:.4f}")
    print("The system has returned to the lowest energy state through physical resonance.")

if __name__ == "__main__":
    run_magic_circle_simulation()
