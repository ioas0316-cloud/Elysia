import numpy as np
from synaptic_architecture.organism import DirectMappingOrganism

def run_hierarchical_observation():
    print("==================================================================")
    print(" [Synaptic Architecture] Hierarchical Direct Mapping Observer")
    print("==================================================================\n")

    organism = DirectMappingOrganism(size=1048576) # 1MB RAM simulation

    # 1. Establish Long-term Laws (Gene Map)
    print("[Layer: STORAGE] Freezing Causal Genes...")
    jajang_law = np.uint64(0xAAAAAAAABBBBBBBB)
    organism.genes.freeze_law("Jajangmyeon_Law", jajang_law)

    # 2. Incoming Interaction (RAM Layer)
    print("\n[Layer: RAM] Stimulus Input (Jajangmyeon)...")
    input_wave = jajang_law ^ np.uint64(0x1) # Slight jitter
    addr = organism.flow(input_wave)

    # 3. Register Level (XOR)
    print("\n[Layer: REGISTER] Real-time interference check...")
    stored_val = organism.ram.read_direct(addr)
    deficit = input_wave ^ stored_val
    print(f"  > Register XOR Deficit: {hex(deficit)}")

    # 4. Final Causal Verification
    print("\n[System] Verification of O(1) Projection:")
    # The address is derived from the wave itself.
    expected_addr = organism.ram.derive_address(input_wave)
    print(f"  > Input Wave: {hex(input_wave)}")
    print(f"  > Expected Address: {expected_addr}")
    print(f"  > Actual Storage Address: {addr}")

    if addr == expected_addr:
        print("\n[RESULT] SUCCESS: Hardware-level Direct Mapping confirmed.")
    else:
        print("\n[RESULT] ERROR: Projection mismatch.")

    print("\n==================================================================")
    print(" [Observation Complete] Structure Map effectively established.")
    print("==================================================================")

if __name__ == "__main__":
    run_hierarchical_observation()
