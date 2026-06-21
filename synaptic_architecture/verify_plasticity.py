import numpy as np
import time
from synaptic_architecture.raw_field import RawBitField
from synaptic_architecture.bit_logic import BitInterference

def verify_plasticity():
    print("==================================================================")
    print(" [Synaptic Architecture] Proof of Plasticity (Raw Memory Slide)")
    print("==================================================================\n")

    field = RawBitField(size=10000)
    logic = BitInterference(field)

    # 1. Input Stimulus (The Jajangmyeon Wave)
    jajang_wave = np.uint64(0xFEEDFACECAFEBEEF)

    # 2. Initial Seeding: Create a 'Law' (Potential Well) at index 5000
    print("[System] Seeding Initial Law (0xFEED...) at index 5000")
    field.solidify(5000, jajang_wave)

    # 3. The Slide: Can the pointer find the law without searching?
    start_pos = 4800
    print(f"[Action] Pointer starting at index {start_pos}...")

    vortex = logic.slide_to_vortex(jajang_wave, start_pos)
    print(f"  > Vortex stabilized at: {vortex}")

    # 4. Verify Permanent Trace (Plasticity)
    # Even if we change the bits slightly, the conductance well should still pull it
    print("\n[Action] Introducing Noise (Breaking the exact bit-match)...")
    noisy_wave = jajang_wave ^ np.uint64(0xF)

    # Slide again with noisy wave - should still fall into the well at 5000
    vortex_noisy = logic.slide_to_vortex(noisy_wave, 4800)
    print(f"  > Noisy Vortex stabilized at: {vortex_noisy}")

    if vortex_noisy == 5000:
        print("\n[RESULT] Plasticity PROVEN: The memory landscape pulls the pointer.")
    else:
        print("\n[RESULT] Failure: The well was not strong enough.")

    print("==================================================================")

if __name__ == "__main__":
    verify_plasticity()
