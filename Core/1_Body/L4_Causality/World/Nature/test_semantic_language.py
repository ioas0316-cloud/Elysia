"""
Test: True Semantic Language
============================
Verifies that the words acting as input determine the physical outcome.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from Core.1_Body.L4_Causality.World.Nature.semantic_nature import SemanticNature
from Core.1_Body.L4_Causality.World.Physics.trinity_fields import TrinityVector

def test_semantic_language():
    print("--- 1. Initializing The Marketplace ---")
    nature = SemanticNature()
    
    # Create a Merchant (High Flow, seeking Exchange)
    merchant = nature.manifest_concept(
        "Merchant", 
        "Silk Trader", 
        [0,0,0], 
        {"price_multiplier": 1.0}
    )
    # Merchant Soul: High Flow, Low Gravity
    merchant.trinity_override = TrinityVector(gravity=0.1, flow=0.9, ascension=0.2)
    
    print(f"Target: {merchant.name} (Flow: {merchant.trinity_override.flow})")
    print("-" * 30)

    # --- Test 1: The Wrong Language (Gravity) ---
    # Gravity words: "Stop", "Demand", "Base"
    # Expected: Dissonance (Merchant hates being stopped)
    print("\n[Attempt 1] Player: 'Stop! I Demand Base!'")
    result1 = nature.interact("Player", "Speech: Stop Demand Base", merchant.id)
    print(f"Result: {result1.message}")
    
    # --- Test 2: The Right Language (Flow) ---
    # Flow words: "Offer", "Trade", "Connect"
    # Expected: Resonance (Merchant loves connection)
    print("\n[Attempt 2] Player: 'I Offer Trade Connect'")
    result2 = nature.interact("Player", "Speech: Offer Trade Connect", merchant.id)
    print(f"Result: {result2.message}")
    
    # --- Verification ---
    # Case 1 should fail (Dissonance)
    # Case 2 should succeed (Resonance)
    
    if result2.success and not result1.success:
        print("\n  TEST PASSED: Language determined Reality.")
    else:
        print("\n  TEST FAILED: Physics did not respect the Words.")

if __name__ == "__main__":
    test_semantic_language()
