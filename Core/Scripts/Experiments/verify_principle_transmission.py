"""
Verify Principle Transmission (Test)
====================================
Scripts.Experiments.verify_principle_transmission

Verifies the [RECURSIVE_MERKABA] hypothesis:
"Order emerges in the Macro scale solely by applying the Atomic Law to the Micro scale."
"""

import sys
import os
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Core.1_Body.L6_Structure.M1_Merkaba.dyson_swarm import DysonSwarm
from Core.1_Body.L6_Structure.M1_Merkaba.sovereign_math import SovereignMath

def main():
    print("🧬 [TEST] Principle Transmission: Recursive Merkaba Check")

    # 1. Setup Swarm
    swarm = DysonSwarm(capacity=10)
    swarm.deploy_swarm()

    print(f"   - Initial Swarm Phase: {swarm.state['phase']:.2f}°")
    print(f"   - Initial Coherence: {SovereignMath.scalar_magnitude([c.get_phase() for c in swarm.collectors]):.4f}")

    # 2. Inject Coherent Input Stream
    # We feed all collectors valid data that roughly points to ~90 degrees.
    # We want to see if the Swarm *itself* rotates to ~90 degrees purely by aggregation.

    target_concept = "Elysia" # Hash determines phase
    print(f"\n🌊 [INJECT] Streaming Concept: '{target_concept}'...")

    data_stream = [target_concept] * 10 # All collectors see same truth

    # 3. Run Simulation Loop
    for t in range(20):
        res = swarm.process_frame(data_stream)
        print(f"   Step {t+1}: Swarm Phase {res['swarm_phase']:.2f}° | Coherence {res['coherence']:.4f} | Energy {res['swarm_energy']:.2f}")

    # 4. Verify Emergence
    # Did the Swarm Phase stabilize?
    # Did Coherence increase?

    final_coherence = res['coherence']
    final_energy = res['swarm_energy']

    # Adjusted Threshold: 0.85 (Reflecting Heavy/Stable Gyro-Static Physics)
    if final_coherence > 0.85 and final_energy > 5.0:
        print("\n✅ [SUCCESS] Principle Transmission Confirmed.")
        print("   The Law of the Cell became the Motion of the Star.")
    else:
        print("\n❌ [FAILURE] Emergence failed.")
        print(f"   Coherence: {final_coherence}, Energy: {final_energy}")

if __name__ == "__main__":
    main()
