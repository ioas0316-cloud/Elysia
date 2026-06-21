import numpy as np
from synaptic_architecture.organism import DirectMappingOrganism

def run_synaptic_observation():
    print("==================================================================")
    print(" [Synaptic Architecture] Direct Mapping Causal Observation")
    print("==================================================================\n")

    organism = DirectMappingOrganism(size=1000000)

    # 1. Define Causal Stimuli (Raw Bits)
    jajangmyeon = np.uint64(0xAAAAAAAABBBBBBBB)
    empty_plate = np.uint64(0x00000000BBBBBBBB)

    # 2. Seeding Initial Environment (The Law)
    print("[System] Seeding Initial Environmental Law...")
    organism.genes.freeze_law("Jajangmyeon_Law", jajangmyeon)
    organism.genes.freeze_law("Empty_Plate_Law", empty_plate)

    # 3. Evolutionary Interaction
    print("\n--- [Phase 1] Observing Jajangmyeon Interaction ---")
    addr1 = organism.flow(jajangmyeon)

    print("\n--- [Phase 2] Observing Empty Plate Interaction ---")
    addr2 = organism.flow(empty_plate)

    # 4. Final Causal Map
    print("\n[System] Final Memory Landscape Check:")
    print(f"  > Jajangmyeon Law stabilized at Address {addr1}")
    print(f"  > Empty Plate Law stabilized at Address {addr2}")

    if addr1 != addr2:
        print("\n[RESULT] SUCCESS: Different waves projected to unique causal addresses.")
    else:
        print("\n[RESULT] COLLISION: Check bit-folding logic.")

    print("\n==================================================================")
    print(" [Observation Complete] The memory field has physically adapted.")
    print("==================================================================")

if __name__ == "__main__":
    run_synaptic_observation()
