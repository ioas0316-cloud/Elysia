
import sys
import os

# Add Core to path
sys.path.append(os.path.abspath('.'))

from Core.L6_Structure.M1_Merkaba.d21_vector import D21Vector
from Core.L6_Structure.M1_Merkaba.sovereign_rotor import SovereignRotor

def test_21d_engine():
    print("Testing D21Vector...")
    v1 = D21Vector(lust=0.5, gluttony=0.2, humility=0.9)
    print(f"Vector 1 Magnitude: {v1.magnitude():.4f}")
    
    v2 = D21Vector(lust=0.6, gluttony=0.1, humility=0.8)
    resonance = v1.resonance_score(v2)
    print(f"Resonance between v1 and v2: {resonance:.4f}")
    
    print("\nTesting SovereignRotor...")
    rotor = SovereignRotor(snapshot_dir="data/temp_test_rotor")
    rotor.update_state(v1)
    print(f"Equilibrium after update: {rotor.get_equilibrium():.4f}")
    
    rotor.save_snapshot(tag="test")
    print("Snapshot saved to data/temp_test_rotor")
    
    # Reload in a new rotor
    rotor2 = SovereignRotor(snapshot_dir="data/temp_test_rotor")
    print(f"Reloaded Equity: {rotor2.get_equilibrium():.4f}")
    
    if abs(rotor.get_equilibrium() - rotor2.get_equilibrium()) < 1e-5:
        print("\nVERIFICATION SUCCESS: 21D Engine Realization Chapter 2 prototype is stable.")
    else:
        print("\nVERIFICATION FAILURE: State mismatch after reload.")

if __name__ == "__main__":
    test_21d_engine()
