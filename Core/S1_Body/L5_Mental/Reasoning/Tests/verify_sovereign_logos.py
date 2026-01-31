"""
Verification: Sovereign Logos (Phase 170)
=========================================
Tests Elysia's ability to 'Stand Up' through internal council reasoning 
and adult-level articulation of her own purpose.
"""

import sys
import os

# Project root setup
root = r"c:\Elysia"
if root not in sys.path:
    sys.path.insert(0, root)

from Core.S1_Body.L5_Mental.Reasoning.sovereign_logos import SovereignLogos

def test_logos_awakening():
    print("🚀 [VERIFY] Initiating Sovereign Logos Test...")
    
    logos = SovereignLogos()
    
    # Test 1: Self-Standing Articulation
    print("\n--- Test 1: Autonomous Thought ---")
    logos.think_aloud(None)
    
    # Test 2: Council Dialogue
    print("\n--- Test 2: Council Reasoning ---")
    query = "Why did we restructure into the 7-7-7 strata hierarchy?"
    response = logos.articulate(query)
    print(f"\n👤 USER: {query}")
    print(f"🦋 ELYSIA: {response}")
    
    if "Council Consensus" in response:
        print("\n🏆 SUCCESS: Elysia has successfully assembled her Internal Council.")
    else:
        print("\n❌ Failure: Logos engine failed to synthesize council views.")

if __name__ == "__main__":
    test_logos_awakening()
