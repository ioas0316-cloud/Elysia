"""
Launch Dyson Swarm (Experiment)
===============================
Scripts.Experiments.launch_dyson_swarm

Verifies the [INITIAL_ORBIT_LAUNCH] of the Dyson Cognition Swarm.
"""

import sys
import os

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Core.L6_Structure.M1_Merkaba.dyson_swarm import DysonSwarm

def main():
    print("🚀 [LAUNCH] Initiating Dyson Cognition Swarm Sequence...")

    # 1. Initialize Swarm
    swarm = DysonSwarm(capacity=7) # Small constellation for test
    swarm.deploy_swarm()

    # 2. Prepare Radiant Data (Test Vectors)
    print("\n☀️ [INPUT] Generating Radiant Data Streams...")
    # These concepts should trigger specific phase shifts
    # We repeat the stream to simulate sustained contact (Spin-Up)
    data_concepts = ["Elysia", "Sovereignty", "Monad", "Phase", "Jump", "Void", "Energy"]

    # 3. Execute Absorption Cycle (Spin-Up Loop)
    print("\n🔋 [HARVEST] Absorbing Radiance (Spin-Up Cycle)...")

    total_steps = 15
    for t in range(total_steps):
        # Create a frame from the concepts (cycle through them)
        frame_data = [data_concepts[(i + t) % len(data_concepts)] for i in range(7)]
        results = swarm.process_frame(frame_data)

        # Optional: Print progress
        # print(f"   Tick {t+1}: Energy {results['swarm_energy']:.2f} | Phase {results['swarm_phase']:.2f}")

    # 4. Report Final Status
    print("\n📊 [REPORT] Mission Status:")
    print(f"   - Active Collectors: {results['child_count']}")
    print(f"   - Void Focus Phase: {results['swarm_phase']:.2f}°")
    print(f"   - Swarm Coherence: {results['coherence']:.4f}")
    print(f"   - Total Energy Harvested: {results['swarm_energy']:.2f}")

    if results['swarm_energy'] > 0:
        print("\n✅ [SUCCESS] Energy Harvesting Confirmed. System is Online.")
    else:
        print("\n❌ [FAILURE] No Energy Harvested.")

if __name__ == "__main__":
    main()
