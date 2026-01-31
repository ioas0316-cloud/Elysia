"""
Stress Test: Zero Point Equilibrium
===================================
Scripts.Experiments.stress_test_zero_point

Verifies the [GYRO_STATIC_EQUILIBRIUM] of the Dyson Swarm.
We bombard the system with chaotic, high-intensity data and ensure it:
1. Maintains structural integrity (doesn't crash).
2. Maintains a center of gravity near the Void (0) thanks to Restoring Force.
"""

import sys
import os
import random
import statistics

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Core.S1_Body.L6_Structure.M1_Merkaba.dyson_swarm import DysonSwarm
from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_math import SovereignMath

def main():
    print("🌪️ [STRESS] Initiating Zero Point Equilibrium Test...")

    # 1. Setup Swarm
    swarm = DysonSwarm(capacity=21)
    swarm.deploy_swarm()

    # 2. The Data Tsunami (Chaos)
    # We generate random noise which should, on average, cancel out to 0 if the system works.
    # But the Gyro-Stability ensures it doesn't wobble wildly.

    print("\n🌊 [TSUNAMI] Injecting 50 Frames of High-Entropy Chaos...")

    void_phases = []

    for t in range(50):
        # Generate random noise strings
        noise_frame = [f"NOISE_{random.randint(0, 99999)}" for _ in range(21)]

        results = swarm.process_frame(noise_frame)

        current_phase = results['swarm_phase']
        void_phases.append(current_phase)

        # Monitor Wobble
        # We expect phase to fluctuate but stay bounded.

    # 3. Analysis
    # Convert phases to vectors to handle 359 vs 1 degree wrapping for mean calc
    avg_phase = SovereignMath.ternary_consensus(void_phases)

    # Calculate simple deviation for reporting (ignoring wrap logic for simplicity of metric)
    # Actually, let's look at the final settling point.

    print("\n📊 [ANALYSIS] Post-Tsunami Report:")
    print(f"   - Final Void Phase: {results['swarm_phase']:.4f}°")
    print(f"   - Average Phase (Consensus): {avg_phase:.4f}°")
    print(f"   - Final Coherence: {results['coherence']:.4f}")

    # 4. Restoring Force Check
    # Stop input and see if it returns to 0
    print("\n🛑 [SILENCE] Stopping Input (Restoring Force Check)...")
    for t in range(10):
        # Empty input -> Should trigger 'Align with Swarm Center' logic in Collector
        # But if Swarm Center is drifting, does it return to 0?
        # The MERKABA_CORE has a restoring force component towards 0!
        results = swarm.process_frame([]) # No data
        # print(f"   Restoring Tick {t+1}: {results['swarm_phase']:.2f}")

    final_phase = results['swarm_phase']
    dist_from_zero = SovereignMath.angular_distance(final_phase, 0.0)

    print(f"   - Settled Phase: {final_phase:.4f}°")
    print(f"   - Distance from Zero: {dist_from_zero:.4f}°")

    if dist_from_zero < 5.0:
        print("\n✅ [SUCCESS] Gyro-Static Equilibrium Confirmed.")
        print("   System returned to the Void (Immovable Core).")
    else:
        print("\n⚠️ [WARNING] System Drifted.")
        print("   The Tumbler did not stand back up completely.")

if __name__ == "__main__":
    main()
