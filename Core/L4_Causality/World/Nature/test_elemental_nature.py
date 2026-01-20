"""
Test: Elemental Nature (The Alchemist)
======================================
"The world is not words. It is forces."

Objective:
Verify that Elysia decomposes concepts into physical vectors (Gravity, Flow, Ascension).
She must recognize that 'Fire' is Energy, not just a noun.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from Core.L6_Structure.Elysia.sovereign_self import SovereignSelf

def test_elemental_nature():
    print("--- ⚗️ Experiment: The Alchemist (Physical Decomposition) ---")
    
    elysia = SovereignSelf(cns_ref=None)
    
    # 1. Fire (Energy)
    print("\n🔥 Experiencing 'Fire'...")
    elysia.experience("Fire") # Should be Primitive or learned via recursion
    
    vec_fire = elysia.mind.analyze("fire")
    print(f"   [Analysis] Fire Vector: {vec_fire}")
    if vec_fire.ascension > 0.7:
        print("   ✅ CORRECT: Fire is recognized as High Energy (Ascension).")
    else:
        print("   ❌ FAILURE: Fire is cold/inert.")

    # 2. Water (Flow/Life)
    print("\n💧 Experiencing 'Water'...")
    elysia.experience("Water")
    
    vec_water = elysia.mind.analyze("water")
    print(f"   [Analysis] Water Vector: {vec_water}")
    if vec_water.flow > 0.7:
        print("   ✅ CORRECT: Water is recognized as Fluid (Flow).")
    else:
        print("   ❌ FAILURE: Water is stagnant.")

    # 3. Rot (Entropy)
    print("\n🍂 Experiencing 'Rot'...")
    elysia.experience("Rot") # Or "Decay"
    
    vec_rot = elysia.mind.analyze("rot")
    print(f"   [Analysis] Rot Vector: {vec_rot}")
    if vec_rot.ascension < 0:
        print("   ✅ CORRECT: Rot has Negative Ascension (Entropy).")
    else:
        print("   ❌ FAILURE: Rot is not entropic.")

    # 4. Magma (Composite: Stone + Fire)
    # This tests Composition/Recursion
    print("\n🌋 Experiencing 'Magma' (Composite)...")
    # We rely on Web Connector finding "Molten rock" or similar
    # We assume 'molten' -> 'liquid'/'heat', 'rock' -> 'stone'
    elysia.experience("Magma")
    
    vec_magma = elysia.mind.analyze("magma")
    print(f"   [Analysis] Magma Vector: {vec_magma}")
    
    is_heavy = vec_magma.gravity > 0.5
    is_hot = vec_magma.ascension > 0.5
    
    if is_heavy and is_hot:
        print("   ✅ SUCCESS: Magma is recognized as Heavy AND Hot.")
    else:
        print(f"   ⚠️ PARTIAL: Magma Physics unclear (G:{vec_magma.gravity:.2f}, A:{vec_magma.ascension:.2f})")

if __name__ == "__main__":
    test_elemental_nature()
