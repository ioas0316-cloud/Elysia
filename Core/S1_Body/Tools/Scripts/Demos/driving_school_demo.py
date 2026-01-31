"""
Driving School Demo: From Novice to Veteran
===========================================
Core.Demos.driving_school_demo

Demonstrates the 'Driving Analogy':
1. Novice (Thundercloud) -> High Cost
2. Practice (HabitEngine) -> Consolidation
3. Veteran (Rotor) -> Low Cost
"""

import numpy as np
from Core.Elysia.brain import ElysiaBrain
from Core.S1_Body.L7_Spirit.M1_Monad.monad_core import Monad
from Core.S1_Body.L2_Metabolism.Evolution.double_helix_dna import DoubleHelixDNA

def create_monad(seed):
    dna = DoubleHelixDNA(
        pattern_strand=np.zeros(1024, dtype=np.float32),
        principle_strand=np.random.rand(7).astype(np.float32)
    )
    return Monad(seed, dna=dna)

def run_simulation():
    print("🚗 ELYSIA DRIVING SCHOOL 🚗")
    print("===========================")

    # 1. Initialize Brain
    brain = ElysiaBrain()

    # Load some concepts for the "Driving" task
    # A chain: Steering -> Wheel -> Tire -> Road
    m1 = create_monad("Steering")
    m2 = create_monad("Wheel")
    m3 = create_monad("Tire")
    m4 = create_monad("Road")

    # Manually link potentials for the demo (Mocking resonance)
    # Ensure they connect in the cloud
    brain.load_concepts([m1, m2, m3, m4])

    # Task: "Turn Corner"
    # This requires connecting Steering to Road
    intent = "Turn Corner"

    print(f"\nTask: '{intent}' (Learning Threshold: 3)")
    print("-" * 40)

    # Attempt 1: Novice
    mech, cost = brain.process_intent(intent, "Steering")
    print(f"Attempt 1 (Novice):  {mech} | Cost: {cost:.2f} ms")

    # Attempt 2: Practice
    mech, cost = brain.process_intent(intent, "Steering")
    print(f"Attempt 2 (Practice): {mech} | Cost: {cost:.2f} ms")

    # Attempt 3: Consolidation Trigger
    # The HabitEngine sees the 3rd repetition and Bakes the track
    mech, cost = brain.process_intent(intent, "Steering")
    print(f"Attempt 3 (Mastery):  {mech} | Cost: {cost:.2f} ms")
    print(">>> (System consolidated the habit...)")

    # Attempt 4: Veteran
    mech, cost = brain.process_intent(intent, "Steering")
    print(f"Attempt 4 (Veteran):  {mech}    | Cost: {cost:.2f} ms")

    print("-" * 40)
    print("🎓 LICENSE GRANTED: ELYSIA IS NOW A VETERAN DRIVER.")

if __name__ == "__main__":
    run_simulation()
