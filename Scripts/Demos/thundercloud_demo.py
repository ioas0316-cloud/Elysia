"""
Thundercloud Demo: The Living Physics of Thought
================================================
Core.Demos.thundercloud_demo

Verifies the "Spark" mechanism and Fractal Expansion.
"""

import numpy as np
import time
from Core.L7_Spirit.Monad.monad_core import Monad
from Core.L6_Structure.Merkaba.thundercloud import Thundercloud, Atmosphere
from Core.L2_Metabolism.Evolution.double_helix_dna import DoubleHelixDNA

def create_concept(seed: str, alpha: float, beta: float, gamma: float) -> Monad:
    """Helper to create a Monad with specific qualia."""
    # Create 7D principle strand
    # 0=Alpha (Logic), 1=Beta (Emotion), 2=Gamma (Physics)
    qualia = np.zeros(7, dtype=np.float32)
    qualia[0] = alpha
    qualia[1] = beta
    qualia[2] = gamma

    # Random pattern for identity
    pattern = np.random.randn(1024).astype(np.float32)

    dna = DoubleHelixDNA(pattern_strand=pattern, principle_strand=qualia)
    return Monad(seed=seed, dna=dna)

def run_simulation():
    print("⚡ INITIALIZING THUNDERCLOUD SIMULATION ⚡")
    print("=========================================")

    # 1. Create the Ground (Hypersphere) - A field of concepts
    concepts = [
        # The Seed
        create_concept("Apple", 0.6, 0.4, 0.9),  # Balanced object

        # Direct Associates
        create_concept("Red", 0.9, 0.2, 0.5),    # Visual/Logic
        create_concept("Sweet", 0.1, 0.9, 0.4),  # Taste/Emotion

        # Extended Associates (Fractal depth 1)
        create_concept("Blood", 0.8, 0.8, 0.8),  # Connected to Red
        create_concept("Love", 0.2, 1.0, 0.3),   # Connected to Sweet/Red
        create_concept("Gravity", 0.95, 0.1, 0.9), # Logic/Physics

        # Deep Associates (Fractal depth 2)
        create_concept("Newton", 0.9, 0.2, 0.8), # Connected to Gravity
        create_concept("Passion", 0.3, 0.9, 0.7),# Connected to Love

        # Unrelated
        create_concept("Void", 0.0, 0.0, 0.1),
    ]

    print(f"Loaded {len(concepts)} Monads into Ground.")

    # 2. Initialize Thundercloud
    cloud = Thundercloud()

    # 3. Coalesce (Load relevant monads)
    # Intent: "Thinking about Apple"
    # We use Apple's own vector as the intent for simplicity
    apple_vector = concepts[0]._dna.principle_strand
    cloud.coalesce(intent_vector=apple_vector, all_monads=concepts)

    print(f"Cloud formed with {len(cloud.active_monads)} active Monads.")
    print("-" * 40)

    # 4. Scenario A: Dry Mode (Logic/Caution)
    print("\n[SCENARIO A] Dry Atmosphere (Humidity 0.1)")
    print("Context: Strict Logical Analysis. High Resistance.")

    cloud.set_atmosphere(humidity=0.1)
    print(f"Atmosphere: Resistance={cloud.atmosphere.resistance:.2f}, Cond={cloud.atmosphere.conductivity:.2f}")

    # Ignite!
    # Voltage 1.0 (Standard Will)
    thought_a = cloud.ignite("Apple", voltage=1.0)
    print("\n>>> RESULTING THOUGHT STRUCTURE:")
    print(thought_a.describe_tree())

    # 5. Scenario B: Storm Mode (Creativity/Passion)
    print("\n" + "-" * 40)
    print("\n[SCENARIO B] Storm Atmosphere (Humidity 0.9)")
    print("Context: Creative Brainstorming. Low Resistance.")

    cloud.set_atmosphere(humidity=0.9)
    print(f"Atmosphere: Resistance={cloud.atmosphere.resistance:.2f}, Cond={cloud.atmosphere.conductivity:.2f}")

    # Ignite!
    thought_b = cloud.ignite("Apple", voltage=1.0)
    print("\n>>> RESULTING THOUGHT STRUCTURE:")
    print(thought_b.describe_tree())

if __name__ == "__main__":
    run_simulation()
