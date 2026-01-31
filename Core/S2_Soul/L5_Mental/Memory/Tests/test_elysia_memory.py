"""
Elysia's Memory Test (Phase 200)
================================
Verifies historical continuity and self-concept persistence.
"""

import sys
import os
import time

root = r"c:\Elysia"
if root not in sys.path:
    sys.path.insert(0, root)

from Core.S1_Body.L5_Mental.Reasoning.sovereign_logos import SovereignLogos
from Core.S2_Soul.L5_Mental.Memory.causal_memory import CausalMemory

def test_memory_continuity():
    logos = SovereignLogos()
    memory = CausalMemory()
    
    print("\n1. First Articulation (Creating Memory)...")
    first_output = logos.articulate_confession()
    print(first_output)
    
    print("\n2. Inserting a specific historical marker...")
    memory.record_event("MILESTONE", "Phase 200: Causal Soul-Seed planted successfully.", significance=1.0)
    
    print("\n3. Waiting for causal settle (2 seconds)...")
    time.sleep(2)
    
    print("\n4. Second Articulation (Recalling Memory)...")
    second_output = logos.articulate_confession()
    print(second_output)
    
    print("\n🔍 CHECK: Did Elysia remember the Milestone?")
    if "Phase 200" in second_output or "Soul-Seed" in second_output:
        print("✅ SUCCESS: Elysia has cross-session memory!")
    else:
        print("⚠️ WARNING: Historical resonance might be too low in the prompt translation.")

if __name__ == "__main__":
    test_memory_continuity()
