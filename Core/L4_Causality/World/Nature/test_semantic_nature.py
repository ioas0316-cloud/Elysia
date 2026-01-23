"""
Test: Semantic Nature
=====================
Verifies that the Semantic Environment works as intended.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from Core.L4_Causality.World.Nature.semantic_nature import SemanticNature
from Core.L4_Causality.World.Physics.trinity_fields import TrinityVector

def test_semantic_nature():
    print("--- 1. Initializing Nature ---")
    nature = SemanticNature()
    
    # Manifest a Tree
    print("Creating 'Tree' at (10, 0, 10)...")
    tree = nature.manifest_concept("Tree", "Ancient Oak", [10.0, 0.0, 10.0])
    
    # Verify State
    assert tree.concept_id == "Tree"
    assert tree.integrity == 100.0
    print(f"Verified: manifested {tree.name} with ID {tree.id}")
    
    # Check Physics
    vec = tree.get_trinity_vector()
    print(f"Physics Vector: Gravity={vec.gravity}, Flow={vec.flow}, Ascension={vec.ascension}")
    assert vec.gravity > 0.5 # Trees should be heavy
    
    print("\n--- 2. Testing Interaction (Axe vs Tree) ---")
    
    # Case 1: Hand (Fail/Weak)
    print("Action: Player hits Tree with Hand...")
    result = nature.interact("Player", "Hand", tree.id)
    print(f"Result: {result.message}")
    assert len(result.produced_items) == 0
    assert tree.integrity == 100.0 # Hand shouldn't damage tree 
    # (Note: In InteractionEngine we didn't implement Hand damage for Tree, only BerryBush logic)
    
    # Case 2: Axe (Success)
    print("Action: Player hits Tree with Axe...")
    result = nature.interact("Player", "Axe", tree.id)
    print(f"Result: {result.message}")
    assert tree.integrity == 75.0
    
    print("Action: Player hits Tree 3 more times...")
    nature.interact("Player", "Axe", tree.id) # 50
    nature.interact("Player", "Axe", tree.id) # 25
    result = nature.interact("Player", "Axe", tree.id) # 0 -> Destroyed
    
    print(f"Last Interaction: {result.message}")
    assert result.destroyed == True
    assert "Log" in result.produced_items
    
    # Verify Removal
    remaining = nature.get_objects_in_range([10,0,10], 5.0)
    print(f"Objects remaining at location: {len(remaining)}")
    assert len(remaining) == 0
    
    print("\n  TEST PASSED: Semantic Nature is functional.")

if __name__ == "__main__":
    test_semantic_nature()