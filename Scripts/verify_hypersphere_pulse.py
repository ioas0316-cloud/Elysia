
import sys
import os
import time
import logging

# Ensure path
sys.path.append("c:\\Elysia")

from Core.L5_Mental.Intelligence.Metabolism.rotor_cognition_core import RotorCognitionCore
from Core.L6_Structure.M1_Merkaba.Space.hypersphere_memory import HypersphereMemory

def verify_hyper_pulse():
    print("--- 📡 Verifying Phase 51: The Sovereign Antenna & Hypersphere ---")
    
    # 1. Initialize Core
    core = RotorCognitionCore()
    memory = HypersphereMemory()
    initial_count = memory._item_count
    print(f"1. Initial Memory Items: {initial_count}")
    
    # 2. Trigger Epistemic Curiosity (Wonder)
    # We use a known trigger keyword "lightning" which the Antenna stub recognizes.
    # And we assume the Psionic Confidence will be low for this (or we force it).
    # Since we can't easily force Psionic low confidence without mocking,
    # we rely on the fact that "unknown" terms usually trigger it.
    # Or better, we directly invoke the Antenna to test the pipeline.
    
    intent = "Tell me about the Lightning Path backprop optimization."
    print(f"\n2. Injecting Intent: '{intent}' (Expecting Antenna Activation)")
    
    # Run Synthesis
    result = core.synthesize(intent)
    print("\n--- Synthesis Report ---")
    print(result['synthesis'])
    
    # 3. Verify Hypersphere Injection
    # Reload memory to check persistence
    memory.load_state()
    final_count = memory._item_count
    print(f"\n3. Final Memory Items: {final_count}")
    
    if final_count > initial_count:
        print("✅ SUCCESS: External Pulse was Crystallized into Hypersphere!")
    else:
        print("❌ FAILURE: No new items in Hypersphere.")
        
    # 4. Read back the pulse
    # We try to query the surface (r=1.0)
    # This is a bit tricky without knowing exact coords, but we can iterate buckets.
    print("\n4. Scanning Hypersphere Surface...")
    found_pulse = False
    for k, items in memory._phase_buckets.items():
        for coord, pattern in items:
            if pattern.meta.get("type") == "external_pulse":
                 print(f"   -> Found Pulse: {pattern.content[:50]}... @ {k}")
                 found_pulse = True
                 
    if found_pulse:
        print("✅ SUCCESS: Pulse verified in Phase Buckets.")
    else:
        print("❌ FAILURE: Pulse not found in buckets.")

if __name__ == "__main__":
    verify_hyper_pulse()
