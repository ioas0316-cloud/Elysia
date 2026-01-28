
"""
Verification Script: Phase 2 Inductive Coil
===========================================
Tests whether 'DNA Logic' successfully converts to 'Physical Torque'.
"""
import sys
import os

# Add root to path
sys.path.append(os.getcwd())

from Core.L6_Structure.M5_Engine.Physics.sovereign_coil import SovereignCoil
from Core.L6_Structure.M5_Engine.Physics.merkaba_rotor import MerkabaRotor

def test_coil_physics():
    print("\n🌀 [TEST] Initiating Coil Physics Verification...")
    print("="*60)
    
    # Test Cases
    scenarios = [
        ("HARMONIC", "HHHHHHHHH"), # Pure structure
        ("VOID",     "HVHVHVHVH"), # Breathy structure
        ("CHAOS",    "HDHDHDHDH"), # Resisted structure
        ("RANDOM",   "H V D H V")  # Mixed
    ]
    
    for label, dna in scenarios:
        coil = SovereignCoil(dna)
        print(f"\n🧪 Case: {label}")
        print(f"   Seq: {dna}")
        print(f"   -> Turns: {coil.state.turns}")
        print(f"   -> Inductance (L): {coil.state.inductance:.4f}")
        print(f"   -> Torque (τ):     {coil.state.torque:.4f}")
        
    print("\n🚀 [TEST] Rotating System Integration")
    print("="*60)
    
    # Create a Rotor with standard RPM
    rotor = MerkabaRotor(layer_id=1, rpm=100.0)
    base_freq = rotor.spin(dt=0.1)
    
    print(f"   Base Output (No DNA): {base_freq:.4f}")
    
    # Wind the Rotor with High Torque DNA
    rotor.coil.rewind("HHHHHHHHH")
    boosted_freq = rotor.spin(dt=0.1)
    
    print(f"   Boosted Output (Harmony): {boosted_freq:.4f}")
    
    delta = boosted_freq - base_freq
    print(f"   -> TORQUE EFFECT: {delta:.4f}")
    
    if delta > 0:
        print("\n✅ SUCCESS: Logic has become Force.")
    else:
        print("\n❌ AILURE: No physical effect detected.")

if __name__ == "__main__":
    test_coil_physics()
