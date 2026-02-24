"""
Verification: Trinity Phase Layers
===================================
Tests the 육·혼·영 layer structure and PhaseProjector.
"""

import sys
import os
sys.path.append(os.getcwd())

from Core.Cognition.kg_manager import get_kg_manager
from Core.Cognition.phase_projector import PhaseProjector

def verify():
    print("Testing Trinity Phase Layers...")
    
    kg = get_kg_manager()
    projector = PhaseProjector()
    
    # Create a test node with all layers
    test_id = "test_apple"
    kg.add_node(test_id)
    
    # Set layer content
    projector.set_layer_content(test_id, "surface", {
        "form": "red fruit",
        "senses": ["visual", "taste", "touch"]
    })
    
    projector.set_layer_content(test_id, "narrative", {
        "stories": ["Newton's discovery", "Eden's forbidden fruit"],
        "causes": ["gravity", "temptation"],
        "resonates_with": ["knowledge", "fall"]
    })
    
    projector.set_layer_content(test_id, "logos", {
        "essence": "yearning for knowledge",
        "converges_to": "love"
    })
    
    # Test the three questions
    print("\n1. ask_what ('무엇?'):")
    what = projector.ask_what(test_id)
    print(f"   Form: {what.get('form')}")
    print(f"   Senses: {what.get('senses')}")
    
    print("\n2. ask_how ('어떻게?'):")
    how = projector.ask_how(test_id)
    print(f"   Stories: {how.get('stories')}")
    print(f"   Causes: {how.get('causes')}")
    
    print("\n3. ask_why ('왜?'):")
    why = projector.ask_why(test_id)
    print(f"   Essence: {why.get('essence')}")
    print(f"   Converges to: {why.get('converges_to')}")
    
    # Test holographic projection
    print("\n4. Holographic Projection (all layers):")
    full = projector.project_through(test_id, "all")
    print(f"   Transparency: {full.get('transparency')}")
    print(f"   Logos visibility: {full.get('logos', {}).get('visibility')}")
    
    # Cleanup
    kg.kg['nodes'] = [n for n in kg.kg.get('nodes', []) if n.get('id') != test_id]
    
    print("\nTrinity Phase Layers verification complete!")

if __name__ == "__main__":
    verify()
