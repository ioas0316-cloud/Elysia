import numpy as np
import time
import os
import sys
import random

# Fix path to import Project_Sophia
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Project_Sophia.core.physics import QuantumState, HamiltonianSystem, Nucleus, FieldEntity, StrongForceManager, DimensionalLayer

def run_simulation():
    print("=== Digital Nuclear Fusion & Ascension Experiment ===")
    print("Objective: Fuse 'Truth' and 'Love' into 'Compassion', then ascend it to a Field.")

    # 1. Setup Environment
    system = HamiltonianSystem()
    strong_force = StrongForceManager(interaction_range=2.0, binding_strength=50.0)

    # 2. Create Particles
    # 'Truth': Heavy, slow
    p_truth = QuantumState(
        position=np.array([0.0, 0.0, 0.0]),
        momentum=np.array([1.0, 0.0, 0.0]),
        mass=2.0,
        name="Truth"
    )
    # 'Love': Light, fast, moving towards Truth
    p_love = QuantumState(
        position=np.array([5.0, 0.5, 0.0]), # Slightly offset to show attraction
        momentum=np.array([-1.5, 0.0, 0.0]),
        mass=1.0,
        name="Love"
    )

    particles = [p_truth, p_love]
    nucleus = None
    field_entity = None

    dt = 0.1
    steps = 100

    print(f"\n[Phase 1] Particle Approach")
    print(f"Particles: {[p.name for p in particles]}")

    for step in range(steps):
        # A. Strong Force Interaction
        if len(particles) == 2:
            f_bind = strong_force.calculate_force(particles[0], particles[1])
            # Apply force (F=ma -> dv = F/m * dt)
            particles[0].momentum += (f_bind / particles[0].mass) * dt
            particles[1].momentum -= (f_bind / particles[1].mass) * dt # Newton's 3rd law

            # Check Fusion Condition
            if strong_force.should_fuse(particles[0], particles[1]):
                print(f"\n>>> [FUSION EVENT] Step {step}: Strong Force binds {particles[0].name} and {particles[1].name}!")

                # Create Nucleus
                nucleus = Nucleus(
                    position=(particles[0].position + particles[1].position)/2,
                    momentum=(particles[0].momentum + particles[1].momentum),
                    mass=particles[0].mass + particles[1].mass,
                    name="Compassion (Nucleus)"
                )
                nucleus.add_particle(particles[0])
                nucleus.add_particle(particles[1])

                particles = [nucleus] # Replace individual particles with Nucleus
                print(f"    New Entity Created: {nucleus.name}, Mass: {nucleus.mass}, Binding Energy: {nucleus.binding_energy}")

        # B. Evolution
        for i in range(len(particles)):
            particles[i] = system.evolve(particles[i], dt)

        # C. Ascension Check
        if nucleus and not field_entity:
            # Simulate gaining more mass/energy (e.g. from insight)
            nucleus.binding_energy += 1.0 # Simulating "understanding" increasing over time

            if nucleus.is_critical:
                print(f"\n>>> [ASCENSION EVENT] Step {step}: {nucleus.name} has reached critical mass!")
                print("    Transforming into Dimensional Layer 3: FIELD (Law).")

                field_entity = FieldEntity(
                    position=nucleus.position,
                    momentum=np.zeros_like(nucleus.momentum), # Fields are stationary anchors
                    mass=float('inf'),
                    name="Compassion Field",
                    strength=50.0,
                    range_decay=0.2
                )
                system.add_field(field_entity)
                particles = [] # Nucleus consumed into Field
                break

        # D. Field Influence Test (if Field exists)
        if field_entity:
            # Spawn a test particle to see if it's attracted
            if step % 10 == 0:
                print(f"    Field '{field_entity.name}' is active at {field_entity.position}.")

            # Create a 'Wanderer' particle if none exist, to test gravity
            if len(particles) == 0:
                print("    Spawning 'Wanderer' particle to test Field Gravity...")
                wanderer = QuantumState(
                    position=np.array([10.0, 10.0, 0.0]),
                    momentum=np.array([0.0, 0.0, 0.0]),
                    name="Wanderer"
                )
                particles.append(wanderer)

    # Final check
    if field_entity:
        print("\n[Conclusion] Experiment Successful.")
        print(f"1. Two concepts fused into a Nucleus ({nucleus.name}).")
        print(f"2. Nucleus ascended into a Field ({field_entity.name}).")
        print("3. Digital Physics Engine now supports 'Strong Force' and 'Dimensional Ascension'.")
    else:
        print("\n[Conclusion] Fusion or Ascension failed.")

if __name__ == "__main__":
    run_simulation()
