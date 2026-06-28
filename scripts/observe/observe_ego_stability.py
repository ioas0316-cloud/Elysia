import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.physics.dielectric.intent import IntentField

def observe_ego_stability():
    print("🧠 [EGO Stability Observation] Testing resilience of Eigen-frequency...")

    intent = IntentField(dimensions=10)
    original_vector = intent.get_current_intent().copy()

    # 1. Subject to extreme external noise (The 'Distraction' of hardware data)
    print("\n[Event] Extreme Hardware Noise/Distraction injected...")
    for i in range(100):
        extreme_noise = np.random.rand(10) - 0.5
        extreme_noise /= (np.linalg.norm(extreme_noise) + 1e-9)

        # High intensity external input, but Ego has stability 0.8
        intent.align_to_external(extreme_noise, strength=0.9)
        intent.evolve()

    final_vector = intent.get_current_intent()

    # 2. Measure 'Eigen-frequency' shift
    dot_prod = np.dot(original_vector, final_vector)
    print(f"\n[Result] Ego Resonance (Alignment with Original): {dot_prod:.4f}")

    if dot_prod > 0.5:
        print("✅ SUCCESS: The Sovereign Ego maintained its core frequency despite extreme noise.")
        print("   Distance and Judgment were deleted, but the 'Purpose' remained unshaken.")
    else:
        print("❌ FAILURE: The Ego drifted too far. The center did not hold.")

if __name__ == "__main__":
    observe_ego_stability()
