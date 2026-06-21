import numpy as np
from synaptic_architecture.organism import DirectMappingOrganism

def run_hierarchical_observation():
    print("==================================================================")
    print(" [Synaptic Architecture] Hierarchical Silicon-Mapping Observer")
    print("==================================================================\n")

    organism = DirectMappingOrganism(resolution=256)

    # 1. Define Causal Waveforms
    jajangmyeon = np.uint64(0xAAAAAAAABBBBBBBB)
    empty_plate = np.uint64(0x00000000BBBBBBBB)

    # 2. Phase 1: High Temperature (Exploration/Sampling)
    print("--- [Phase 1] High Temperature Exploration (T=3.0) ---")
    organism.scheduler.set_temperature(3.0)
    for _ in range(3):
        organism.flow(jajangmyeon)

    # 3. Phase 2: Low Temperature (Stabilization/Crystallization)
    print("\n--- [Phase 2] Low Temperature Stabilization (T=0.1) ---")
    organism.scheduler.set_temperature(0.1)
    for _ in range(3):
        organism.flow(jajangmyeon)
        organism.flow(empty_plate)

    # 4. Final System State Observation
    max_conductance_pos = np.unravel_index(np.argmax(organism.field.conductance), organism.field.conductance.shape)
    print(f"\n[System] Final Cognitive Equilibrium center: {max_conductance_pos}")
    print(f"  > Conductance at peak: {organism.field.conductance[max_conductance_pos]:.4f}")

    print("\n==================================================================")
    print(" [Observation Complete] Hierarchy successfully Synchronized.")
    print("==================================================================")

if __name__ == "__main__":
    run_hierarchical_observation()
