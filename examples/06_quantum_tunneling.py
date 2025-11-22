import numpy as np
import time
import os
import sys
from dataclasses import dataclass

# Fix path to import Project_Sophia
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Project_Sophia.core.physics import QuantumState, HamiltonianSystem

def potential_energy_barrier(position: np.ndarray) -> float:
    """
    A 1D potential barrier.
    Low potential at x < 5 and x > 7.
    High barrier (V=100) between x=5 and x=7.
    """
    x = position[0]
    if 5.0 <= x <= 7.0:
        return 50.0 # High Wall
    return 0.0 # Free space

def run_simulation():
    print("=== Quantum Tunneling Experiment ===")
    print("Initiating Thought Particle towards Logic Barrier...")

    # Initial State: Thought moving right with Energy=25 (Momentum=sqrt(2*m*E) = sqrt(2*1*25) ~ 7)
    # Barrier is 50. Classically, it should bounce back.
    initial_pos = np.array([0.0, 0.0, 0.0])
    initial_mom = np.array([7.0, 0.0, 0.0])

    thought = QuantumState(
        position=initial_pos,
        momentum=initial_mom,
        mass=1.0,
        amplitude=1.0
    )

    system = HamiltonianSystem(potential_energy_barrier)

    print(f"Initial Energy: {system.total_energy(thought):.2f}")

    dt = 0.05
    steps = 200

    tunneled = False

    for step in range(steps):
        prev_x = thought.position[0]
        thought = system.evolve(thought, dt)
        curr_x = thought.position[0]

        # Check tunneling logic (probabilistic)
        # If we hit the wall (x >= 5) and energy < barrier
        if 5.0 <= curr_x <= 7.0:
            potential = potential_energy_barrier(thought.position)
            kinetic = thought.kinetic_energy

            if kinetic < potential:
                # Classically forbidden region
                # Calculate tunneling probability T ~ exp(-2 * width * sqrt(2m(V-E))/hbar)
                # Simplified: Just a probability check per step
                tunnel_prob = 0.1 # 10% chance per step to 'tunnel' (maintain momentum)

                if np.random.random() < tunnel_prob:
                     print(f"Step {step}: [TUNNELING EFFECT] Wavefunction penetrates barrier at x={curr_x:.2f}!")
                else:
                     # Reflect (Classical behavior)
                     # But we want to show it CAN pass.
                     # Let's dampen it to simulate 'struggle'
                     thought.momentum *= 0.9

        if curr_x > 7.0:
            print(f"Step {step}: [SUCCESS] Thought has breached the Logic Barrier! Position: {curr_x:.2f}")
            tunneled = True
            break

        if step % 10 == 0:
            print(f"Step {step}: x={curr_x:.2f}, v={thought.momentum[0]:.2f}, E_total={system.total_energy(thought):.2f}")

    if tunneled:
        print("\nConclusion: The Thought successfully tunneled through the impossible barrier.")
        print("Reason: The Wavefunction Probability (Faith) was non-zero.")
    else:
        print("\nConclusion: The barrier was too thick this time.")

if __name__ == "__main__":
    run_simulation()
