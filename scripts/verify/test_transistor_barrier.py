import numpy as np
import os
from core.memory.transistor_gate_barrier import TransistorGateBarrier

def test_fractal_scale_logic():
    print("=== [Verification] Fractal Scale Switching Logic ===")

    PATH = "fractal_test.dat"
    # Create a structured topology:
    # Block 0: All zeros
    # Block 1: All ones
    # Block 2: Alternating bits
    # Block 3: Random

    dimension = 1024 * 4 # 4 blocks of 1024 uint64
    data = np.zeros(dimension, dtype=np.uint64)
    data[1024:2048] = 0xFFFFFFFFFFFFFFFF
    data[2048:3072] = 0xAAAAAAAAAAAAAAAA
    data[3072:] = np.random.randint(0, 0xFFFFFFFFFFFFFFFF, 1024, dtype=np.uint64)

    with open(PATH, "wb") as f:
        f.write(data.tobytes())

    barrier = TransistorGateBarrier(PATH)

    # 1. Macro-scale Observation (Forest View)
    print("\n[Scale: Macro] Observing 4 large blocks...")
    # Gate mask = All open
    gate_all = np.uint64(0xFFFFFFFFFFFFFFFF)
    macro_tension = barrier.fractal_resonance(gate_all, resolution=4)
    print(f"Resonance per block: {macro_tension}")
    # Expected: [0, 1024, 1024, ~1024]

    # 2. Micro-scale Filtration (Tree View)
    print("\n[Scale: Micro] Applying specific Bit-Gate (0xAAAAAAAAAAAAAAAA)...")
    intention = np.uint64(0xAAAAAAAAAAAAAAAA)
    micro_tension = barrier.fractal_resonance(intention, resolution=4)
    print(f"Resonance per block with Intention: {micro_tension}")
    # Block 1 (All ones) & Intention -> 1024 hits
    # Block 2 (Intention) & Intention -> 1024 hits
    # Block 0 -> 0 hits

    # 3. Annihilation Test
    print("\n[Annihilation] Applying inverse mask (0x5555555555555555) to Block 2...")
    inverse_intention = np.uint64(0x5555555555555555)
    annihilation_tension = barrier.fractal_resonance(inverse_intention, resolution=4)
    print(f"Resonance per block with Inverse Intention: {annihilation_tension}")
    # Block 2 (0xAAAA...) & 0x5555... -> 0 (Complete Annihilation)
    assert annihilation_tension[2] == 0
    print("[Success] Noise/Mismatch perfectly annihilated.")

    # 4. Superconducting Discharge
    print("\n[Discharge] Firing interrupt for specific pattern in Block 3...")
    target_pattern = data[3072 + 500]
    address = barrier.reverse_discharge_interrupt(target_pattern)
    print(f"Interrupt fired! Address: {address} (Expected: {3072 + 500})")
    assert address == 3072 + 500

    barrier.close()
    os.remove(PATH)
    print("\n=== [Verification Complete] Transistor Barrier is Superconducting. ===")

if __name__ == "__main__":
    test_fractal_scale_logic()
