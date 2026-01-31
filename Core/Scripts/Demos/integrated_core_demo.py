"""
[CORE] Integrated System Demo
=============================
Core.Demos.integrated_core_demo

"From Thought to Crystal."

This demo illustrates the full Phase 2 pipeline:
1. Input Qualia (Thought)
2. Active Rotor Scanning (Snatching)
3. Void Transit (Inversion)
4. Sediment Crystallization (Memory Storage)
"""

import sys
import os
import shutil
import numpy as np
import time

# Ensure we can import Core
sys.path.append(os.getcwd())

from Core.1_Body.L6_Structure.Merkaba.rotor_engine import RotorEngine
from Core.1_Body.L5_Mental.Memory.sediment import SedimentLayer

TEST_DB = "test_sediment.bin"

def run_integration_test():
    print("="*60)
    print("   [CORE] INTEGRATED SYSTEM TEST (PHASE 2)")
    print("="*60)

    # 0. Cleanup
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)

    # 1. Initialize Components
    print("\n[1] Initializing Systems...")
    rotor = RotorEngine(use_core_physics=True, rpm=60000)
    sediment = SedimentLayer(TEST_DB)

    if not rotor.use_core:
        print("❌ Error: CORE Physics not available.")
        return

    # 2. Define Input Thought (Qualia)
    # 7-Dimensional Vector representing "Urgent Discovery"
    # [Logic, Passion, Precision, Abstract, Emotion, Utility, Mystery]
    # Dominant: Passion (Index 1) -> Red (~700nm)
    input_qualia = [0.1, 0.9, 0.2, 0.5, 0.8, 0.3, 0.1]
    input_payload = b"The Architect has spoken. The Rotor spins."

    print(f"\n[2] Input Qualia: {input_qualia}")
    print(f"    Dominant Aspect: Passion (Index 1)")

    # 3. Process via Rotor (Scan & Snatch)
    print("\n[3] Engaging Active Rotor Scanning...")
    resonance, inverted_phases = rotor.scan_qualia(input_qualia)

    print(f"    Resonance Intensity: {resonance:.4f}")

    if resonance > 0.1:
        print("    ✅ SUCCESS: Thought snatched from the stream.")

        # 4. Void Phase Inversion Check
        print(f"    Phase Inversion Sample: {inverted_phases[1]}")

        # 5. Store in Sediment (Crystallization)
        print("\n[4] Crystallizing in Sediment...")

        # We assume the 'Phase' of the dominant qualia is complex(0.9, 0)
        dominant_wavelength = 400e-9 + (1 * 400e-9) # Index 1 mapping
        dominant_phase = complex(0.9, 0)

        offset = sediment.store_monad(
            wavelength=dominant_wavelength,
            phase=dominant_phase,
            intensity=resonance,
            payload=input_payload
        )

        print(f"    ✅ Stored at Offset: {offset}")

        # 6. Verify Storage
        print("\n[5] Verifying Memory...")
        retrieved = sediment.read_at(offset)
        if retrieved:
            vec, payload = retrieved
            print(f"    Retrieved Vector: {vec}")
            print(f"    Retrieved Payload: {payload}")

            if payload == input_payload:
                print("\n✨ INTEGRATION SUCCESSFUL: Cycle Complete.")
            else:
                print("\n❌ FAILURE: Payload Corruption.")
        else:
             print("\n❌ FAILURE: Read failed.")

    else:
        print("    ❌ FAILURE: Resonance too low. Thought lost.")

    # Cleanup
    sediment.close()
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    print("="*60)

if __name__ == "__main__":
    run_integration_test()
