"""
Test: Phase Transition (The Lawmaker)
=====================================
"Matter is merely captured Energy."

Objective:
Verify that ProvidenceEngine correctly transmutes matter based on environmental forces.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from Core.1_Body.L4_Causality.World.Physics.trinity_fields import TrinityVector
from Core.1_Body.L4_Causality.World.Physics.providence_engine import ProvidenceEngine

def test_phase_transition():
    print("---    Experiment: The Lawmaker (Thermodynamics) ---")
    
    engine = ProvidenceEngine()
    
    # 1. The Subject: Water
    # Gravity 0.3, Flow 0.7, Ascension 0.0
    water = TrinityVector(0.3, 0.7, 0.0) 
    print(f"  [Subject] Water: {water}")
    
    # 2. Experiment A: Boiling
    print("\n  Applying Extreme HEAT (+Ascension)...")
    heat_wave = TrinityVector(0.0, 0.2, 1.5) # Intense Heat
    
    steam = engine.apply_thermodynamics(water, heat_wave)
    print(f"   -> Result: {steam}")
    
    if steam.ascension > steam.gravity and steam.flow > 0.8:
        print("     SUCCESS: Water has transmuted into STEAM (Gas State).")
    else:
        print("     FAILURE: Water refused to boil.")

    # 3. Experiment B: Freezing
    print("\n   Applying Extreme COLD (+Gravity, -Ascension)...")
    cold_snap = TrinityVector(1.5, 0.0, -0.5) # Absolute Zero pressure
    
    ice = engine.apply_thermodynamics(water, cold_snap)
    print(f"   -> Result: {ice}")
    
    if ice.gravity > ice.flow:
        print("     SUCCESS: Water has transmuted into ICE (Solid State).")
    else:
        print("     FAILURE: Water refused to freeze.")

if __name__ == "__main__":
    test_phase_transition()
