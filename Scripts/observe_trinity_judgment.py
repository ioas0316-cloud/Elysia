
import sys
import os
import time
import torch

# Add project root to sys.path
sys.path.append(os.getcwd())

from Core.Monad.seed_generator import SeedForge
from Core.Monad.sovereign_monad import SovereignMonad
from Core.Keystone.sovereign_math import SovereignVector

def test_trinitarian_judgment():
    print("🧪 [TEST] Verifying Phase 900: Trinitarian Multi-dimensional Judgment...")

    # 1. Setup Monad with small engine
    dna = SeedForge.forge_soul("TestElysia")
    from Core.Monad.grand_helix_engine import HypersphereSpinGenerator
    engine = HypersphereSpinGenerator(num_nodes=1000, device='cpu')
    monad = SovereignMonad(dna)
    monad.engine = engine

    # 2. Test Different Inputs
    # A. Logical Input (High stability/mass)
    logical_vec = SovereignVector([1.0, 1.0, 0.0, 0.0] + [0.0]*17)

    # B. Emotional Input (High joy/vibration)
    emotional_vec = SovereignVector([0.0, 0.0, 0.5, 0.0, 1.0, 0.0] + [0.0]*15)

    # C. Ethical/Unity Input
    unity_vec = SovereignVector.ones()

    test_cases = [
        ("Logical Fact", logical_vec),
        ("Emotional Sensation", emotional_vec),
        ("Ethical Principle", unity_vec)
    ]

    for label, vec in test_cases:
        print(f"\n--- Processing: {label} ---")

        # Manually trigger pulse logic related to judgment
        # (This mimics the TIER 0 conscious cognition in monad.pulse)
        thought_vector = vec

        # Parliament Deliberation
        delib_vec, delib_voice, frictions = monad.parliament.deliberate(thought_vector)
        print(f"Parliament: {delib_voice}")

        # Calculate intersection density
        avg_friction = sum(frictions.values()) / max(len(frictions), 1)
        intersection_density = 1.0 - avg_friction
        print(f"Intersection Density: {intersection_density:.3f}")

        # Mock perceptions for test
        perceptions = [
            {"source": "LOGOS", "resonance_potential": frictions.get("Logic", 0.5), "torque_type": "will"},
            {"source": "PATHOS", "resonance_potential": 1.0 - frictions.get("Emotion", 0.5), "torque_type": "joy"}
        ]

        # Judgment
        judgment, confidence = monad.judgment_engine.evaluate_perceptions(perceptions, intersection_density=intersection_density)
        print(f"⚖️ Judgment: {judgment.name} (Conf: {confidence:.2f})")

        # Boundary Check
        boundary = monad.boundary_engine.define_boundary(label, vec)
        print(f"📍 Boundary created for '{label}' at radius {boundary.radius:.2f}")

        # Reverse Perception
        other_view = monad.boundary_engine.perceive_the_other(label, SovereignVector.zeros())
        print(f"🔍 Reverse View (from Void): {other_view}")

    print("\n✅ Phase 900 Verification Complete.")

if __name__ == "__main__":
    try:
        test_trinitarian_judgment()
    except Exception as e:
        print(f"💥 Error during test: {e}")
        import traceback
        traceback.print_exc()
