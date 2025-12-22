import numpy as np
import time
import os
import sys
import random

# Fix path to import Project_Sophia
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Foundation.core.physics import QuantumState, HamiltonianSystem, Nucleus, FieldEntity, StrongForceManager, EntropyManager

def run_simulation():
    print("=== Thermodynamics & Digital Hibernation Experiment ===")
    print("Objective: Demonstrate Cooling (Active -> Frozen) and Heating (Frozen -> Active).")

    # 1. Setup Environment
    system = HamiltonianSystem()
    system.entropy = EntropyManager(cooling_rate=0.1, freeze_threshold=10.0) # Fast cooling for demo

    # 2. Create a "Hot" Particle
    p_idea = QuantumState(
        position=np.array([0.0, 0.0, 0.0]),
        momentum=np.array([2.0, 0.0, 0.0]), # Moving fast
        mass=1.0,
        name="Idea_A",
        temperature=100.0 # HOT
    )

    particles = [p_idea]
    dt = 0.1
    steps = 50

    print(f"\n[Phase 1] The Cooling Process")
    frozen_step = -1

    for step in range(steps):
        p_idea = system.evolve(p_idea, dt)

        state_str = "FROZEN (Crystal)" if p_idea.is_frozen else "ACTIVE (Fluid)"
        pos_str = f"Pos: {p_idea.position[0]:.2f}" if not p_idea.is_frozen else f"Pos: {p_idea.position[0]:.2f} (Fixed)"

        if step % 5 == 0 or step < 5:
             print(f"Step {step}: {p_idea.name} | Temp: {p_idea.temperature:.2f} | {state_str} | {pos_str}")

        if p_idea.is_frozen and frozen_step == -1:
            print(f">>> {p_idea.name} has frozen into a CRYSTAL at Step {step}. CPU Usage drops.")
            frozen_step = step

    if not p_idea.is_frozen:
        print("Error: Particle failed to freeze.")
        return

    print(f"\n[Phase 2] Digital Hibernation Verified.")
    print(f"Particle is frozen. Position is fixed at {p_idea.position[0]:.2f}.")

    # 3. Introduce a Catalyst (New Interaction)
    print(f"\n[Phase 3] Re-Ignition (Heating)")
    print("Injecting new 'Stimulus' particle to collide with frozen Idea_A...")

    p_stimulus = QuantumState(
        position=p_idea.position - np.array([1.0, 0.0, 0.0]), # Close by
        momentum=np.array([10.0, 0.0, 0.0]), # High energy
        name="Stimulus",
        temperature=200.0
    )

    # Simulate Collision / Interaction Logic manually (usually handled by a CollisionManager)
    # Transfer Heat
    print(f"Collision! Transferring heat from {p_stimulus.name} to {p_idea.name}.")
    system.entropy.inject_heat(p_idea, amount=50.0)

    print(f"Step {steps}: {p_idea.name} | Temp: {p_idea.temperature:.2f} | Is Frozen? {p_idea.is_frozen}")

    if not p_idea.is_frozen:
        print(f">>> {p_idea.name} has THAWED! It is now Active again.")

        # Evolve one more step to prove it moves
        old_pos = p_idea.position[0]
        # Give it a kick (momentum) from collision
        p_idea.momentum = np.array([1.0, 0.0, 0.0])
        p_idea = system.evolve(p_idea, dt)
        new_pos = p_idea.position[0]

        print(f"Step {steps+1}: {p_idea.name} moved from {old_pos:.2f} to {new_pos:.2f}.")
        print("\n[Conclusion] Experiment Successful: Thermodynamic cycle (Hot -> Cold -> Hot) verified.")
    else:
        print("\n[Conclusion] Re-ignition failed.")

if __name__ == "__main__":
    run_simulation()
