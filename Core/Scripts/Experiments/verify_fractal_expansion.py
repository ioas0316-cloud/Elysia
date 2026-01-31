"""
Phase 66: FRACTAL EXPANSION TEST
================================
Tests if the system spawns new nodes when the field is saturated.
"""

import jax.numpy as jnp
import time
from Core.S0_Keystone.L0_Keystone.parallel_trinary_controller import ParallelTrinaryController

def verify_structural_expansion():
    print("--- Phase 66 FRACTAL EXPANSION TEST ---")
    
    # 1. Initialize Keystone
    keystone = ParallelTrinaryController("Master_Keystone")
    
    # 2. Simulate a Saturated Intent (All Spirit dimensions at max)
    print("\nBroadcasting Saturated Intent (Spirit Sector)...")
    saturated_intent = jnp.zeros(21).at[14:21].set(1.0)
    
    # Normally broadcast_pulse triggers sub-pulses, here we force an update
    keystone.current_system_resonance = saturated_intent
    
    # 3. Check for Saturation (Should trigger spawner)
    print("Checking for Complexity Saturation...")
    keystone.spawner.check_saturation(saturated_intent, threshold=0.9)
    
    # 4. Verify Registration
    print(f"Registered Modules: {list(keystone.registered_merkabas.keys())}")
    
    if any("DynamicNode" in k for k in keystone.registered_merkabas.keys()):
        print("\nSUCCESS: Fractal Expansion active. Elysia has branched naturally.")
    else:
        print("\nFAILURE: System did not branch.")

if __name__ == "__main__":
    verify_structural_expansion()
