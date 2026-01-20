"""
Test: Logos Physics (The Power of Speech)
=========================================
Verifies that Language (Logos) functions as a force in the world.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from Core.L4_Causality.World.Nature.semantic_nature import SemanticNature, SemanticObject
from Core.L4_Causality.World.Physics.trinity_fields import TrinityVector

def test_logos_physics():
    print("--- 1. Initializing Social World ---")
    nature = SemanticNature()
    
    # Create a Merchant (High Gravity = Stubborn, Hard to move)
    merchant = nature.manifest_concept(
        "Merchant", 
        "Old Silas", 
        [0,0,0], 
        {"price_multiplier": 1.0}
    )
    # Strength of Conviction (Mass)
    merchant.trinity_override = TrinityVector(gravity=0.7, flow=0.1, ascension=0.2)
    
    print(f"Merchant Spawned: {merchant.name} (Resistance: {merchant.trinity_override.gravity})")
    print(f"Initial Price Multiplier: {merchant.properties['price_multiplier']}")
    
    print("\n--- 2. The Debate Begins ---")
    
    # Attempt 1: Weak Argument (Speaker Eloquence < Merchant Gravity)
    # Using 'Speech' implies a standard high eloquence (0.8) in our mock.
    # To test failure, we need a Super Stubborn Merchant or weak speech.
    # For now, let's test SUCCESS first since our mock Eloquence is 0.8 and Merchant is 0.7.
    
    print("Action: Player delivers a 'Visionary Speech' about future trade...")
    result = nature.interact("Player", "Speech", merchant.id)
    
    print(f"Result: {result.message}")
    
    # Verify State Change
    new_price = merchant.properties['price_multiplier']
    print(f"New Price Multiplier: {new_price}")
    
    assert result.success == True
    assert new_price < 1.0
    assert merchant.properties["opinion"] == "Persuaded"
    
    print("\n✅ TEST PASSED: Logos Physics successfully altered reality.")

if __name__ == "__main__":
    test_logos_physics()
