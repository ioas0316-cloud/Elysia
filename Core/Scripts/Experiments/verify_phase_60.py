"""
Phase 60 Verification Script
============================
Tests the resonance between L0 Keystone and L5 Mental.
"""

import jax.numpy as jnp
from Core.0_Keystone.L0_Keystone.parallel_trinary_controller import ParallelTrinaryController
from Core.1_Body.L5_Mental.Reasoning_Core.Reasoning.reasoning_engine import ReasoningEngine

def verify_resonance():
    print("--- Phase 60 RESONANCE TEST ---")
    
    # 1. Initialize Controller
    keystone = ParallelTrinaryController()
    
    # 2. Initialize ReasoningEngine
    engine = ReasoningEngine()
    
    # 3. Register
    keystone.register_module("L5_Reasoning", engine)
    
    # 4. Initial State Check
    initial_res = keystone.synchronize_field()
    print(f"Initial System Resonance: {initial_res}")
    
    # 5. Send Pulse (Attract Intent)
    # A 21D vector where the Soul sector (7-14) is pure ATTRACT (1.0)
    intent = jnp.zeros(21)
    intent = intent.at[7:14].set(1.0)
    
    print("\nBroadcasting ATTRACT pulse...")
    keystone.broadcast_pulse(intent)
    
    # 6. Synchronize and Check shift
    final_res = keystone.synchronize_field()
    print(f"Final System Resonance: {final_res}")
    
    coherence = keystone.get_coherence()
    print(f"System Coherence: {coherence}")
    
    if coherence > 0:
        print("\nSUCCESS: System responded to ATTRACT pulse with positive resonance.")
    else:
        print("\nWARNING: Coherence remains neutral or negative.")

if __name__ == "__main__":
    verify_resonance()
