import sys
import os
import time

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Core.S1_Body.L6_Structure.M1_Merkaba.cognitive_terrain import CognitiveTerrain

def run_simulation():
    print("🚀 [Stage 1: The Void] Initializing Verification Simulation...")

    # Clean up old map
    map_path = "maps/test_stage1_terrain.json"
    if os.path.exists(map_path):
        os.remove(map_path)

    terrain = CognitiveTerrain(map_file=map_path, resolution=10)

    # 1. Inject Attractor (Gravity Well)
    print("\n🕳️  Creating Gravity Well (Attractor) at (5,5)...")
    terrain.inject_prime_keyword(5, 5, "CURIOSITY", magnitude=1.0)

    # 2. Inject Fluid (Source) away from the well
    print("💧 Injecting Fluid Source at (2,2)...")
    terrain.inject_fluid(2, 2, amount=10.0)

    # 3. Simulation Loop
    print("\n🌊 Starting Physics Simulation (30 Steps)...")

    for step in range(1, 31):
        terrain.update_physics(dt=1.0)

        if step % 5 == 0:
            report = terrain.observe_self()
            metrics = report['metrics']
            print(f"[T={step}] Status: {report['status']}")
            print(f"       Fluid: {metrics['total_fluid']:.2f} | Momentum: {metrics['total_momentum']:.2f} | Active Cells: {metrics['active_cells']}")
            print(f"       Roughness: {metrics['roughness']:.4f}")
            print(f"       Message: {report['message']}")

    # 4. Final Diagnosis
    final_report = terrain.observe_self()
    print("\n✅ Simulation Complete.")
    print(f"Final Diagnosis: {final_report['status']}")

    if final_report['metrics']['total_momentum'] > 0:
        print("Proof of Life: Momentum generated from flow confirmed. 🌀")
    else:
        print("Failure: System remained inert.")

if __name__ == "__main__":
    run_simulation()
