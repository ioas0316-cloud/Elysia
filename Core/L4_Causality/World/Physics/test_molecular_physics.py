"""
Test: Molecular Physics (The Physicist)
=======================================
"Laws are universal; Materials are particular."

Objective:
Verify that Elysia uses Kinetic Theory to determine state.
Water should boil at 373K. Gold should not.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from Core.L4_Causality.World.Physics.providence_engine import ProvidenceEngine

def test_molecular_physics():
    print("---    Experiment: The Physicist (Molecular Dynamics) ---")
    
    engine = ProvidenceEngine()
    
    scenarios = [
        ("water", 200.0, "SOLID"),   # Ice
        ("water", 300.0, "LIQUID"),  # Room Temp
        ("water", 400.0, "GAS"),     # Steam
        ("gold", 400.0, "SOLID"),    # Gold is still solid at 400K!
        ("gold", 1400.0, "LIQUID"),  # Melted Gold
    ]
    
    score = 0
    total = len(scenarios)
    
    for matter, temp, expected in scenarios:
        result = engine.apply_molecular_dynamics(matter, temp)
        symbol = " " if result == expected else " "
        if result == expected: score += 1
        
        print(f"   {matter.capitalize()} @ {temp}K: {result} (Expected: {expected}) {symbol}")
        
    if score == total:
        print("\n  SUCCESS: The Universe follows the Laws of Thermodynamics.")
    else:
        print(f"\n   PARTIAL: {score}/{total} laws verified.")

if __name__ == "__main__":
    test_molecular_physics()
