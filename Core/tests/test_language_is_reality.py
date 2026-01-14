"""
Language is Reality Test (The Fiat Lux Test)
============================================
tests/test_language_is_reality.py

Simulates:
1. Input: "Magma" -> Output: Hot, High Gravity Zone.
2. Input: "Ocean" -> Output: Water, High Flow Zone.
3. Input: "Light" -> Output: Plasma, High Ascension Zone.
"""

import sys
import os
sys.path.append(os.getcwd())

from Core.Foundation.hyper_sphere_core import HyperSphereCore

def test_manifestation():
    print(">>> 🗣️  Initiating Word-to-World Manifestation...")
    
    universe = HyperSphereCore()
    
    # 1. Manifest Fire (Magma)
    # Magma should be Heavy (Gravity) and Hot (Freq)
    props_magma = universe.manifest_at((0,0), "magma")
    
    assert props_magma["type"] == "Bedrock" or props_magma["type"] == "Earth", "Magma check failed (Solidity)"
    assert props_magma["climate"]["temperature"] > 50.0, "Magma should be HOT!"
    
    # 2. Manifest Water (River)
    # River should be Fluid (Flow)
    props_river = universe.manifest_at((0,1), "river")
    
    # Check if fluid
    assert "Water" in props_river["type"], "River check failed (Type)"
    assert props_river["physics"]["viscosity"] < 1.0, "River should flow!"
    
    # 3. Manifest Spirit (Love/Light)
    props_light = universe.manifest_at((1,1), "light")
    
    assert "Light" in props_light["type"] or "Air" in props_light["type"], "Light check failed"
    assert props_light["resources"]["Spirit"] > 0, "Light should yield Spirit!"
    
    print("\n>>> ✅ Language-to-Physics Bridge Verified.")
    print("Words have successfully become Terrain.")

if __name__ == "__main__":
    test_manifestation()
