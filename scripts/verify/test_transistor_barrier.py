import numpy as np
import os
from core.memory.transistor_gate_barrier import TransistorGateBarrier

def test_transistor_barrier_logic():
    print("=== [Verification] Transistor Gate Barrier Logic ===")

    PATH = "barrier_test.dat"
    # Create a structured topology for verification
    dimension = 1024 * 4
    data = np.zeros(dimension, dtype=np.uint64)
    data[1024:2048] = 0xFFFFFFFFFFFFFFFF # Block 1: All Signal
    data[2048:3072] = 0xAAAAAAAAAAAAAAAA # Block 2: Patterned Signal
    data[3072:] = np.arange(1024, dtype=np.uint64) + 0x100 # Block 3: Linear sequence

    # Inject a unique pattern for discharge test
    unique_pattern = np.uint64(0xDEADBEEFCAFEBABE)
    data[1524] = unique_pattern

    with open(PATH, "wb") as f: f.write(data.tobytes())

    barrier = TransistorGateBarrier(PATH, chunk_size=1024)

    # Switch to DELTA mode for full-signal verification
    barrier.set_mode_DELTA()

    # 1. Full Signal Observation
    print("\n[DELTA-Mode] Observing 4 large blocks...")
    gate_all = np.uint64(0xFFFFFFFFFFFFFFFF)
    res_full = barrier.fractal_resonance(gate_all, resolution=4)
    print(f"Resonance per block (DELTA): {res_full}")

    # 2. Patterned Filtration
    print("\n[Filtration] Applying 0xAAAAAAAAAAAAAAAA mask...")
    intention = np.uint64(0xAAAAAAAAAAAAAAAA)
    res_pattern = barrier.fractal_resonance(intention, resolution=4)
    print(f"Resonance with Intention: {res_pattern}")

    # 3. Y-Mode Attenuation
    print("\n[Y-Mode] Verifying load reduction...")
    barrier.set_mode_Y()
    res_Y = barrier.fractal_resonance(gate_all, resolution=4)
    print(f"Resonance per block (Y-mode): {res_Y}")

    assert res_Y[3] < res_full[3]
    print("[Success] Cognitive load attenuated in Y-mode.")

    # 4. Superconducting Discharge
    print("\n[Discharge] Firing interrupt...")
    barrier.set_mode_DELTA()
    addr = barrier.reverse_discharge_interrupt(unique_pattern)
    print(f"Interrupt address: {addr} (Expected: 1524)")
    assert addr == 1524

    barrier.close()
    os.remove(PATH)
    print("\n=== [Verification Complete] Transistor Barrier Logic Verified. ===")

if __name__ == "__main__":
    test_transistor_barrier_logic()
