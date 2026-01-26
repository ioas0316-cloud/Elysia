
import sys
import os

# Add Core to path
sys.path.append(os.path.abspath('.'))

from Core.L6_Structure.M1_Merkaba.d21_vector import D21Vector
from Core.L6_Structure.M1_Merkaba.triple_helix_engine import TripleHelixEngine

def test_triple_helix():
    print("Testing TripleHelixEngine Pulse...")
    engine = TripleHelixEngine()
    
    # Test Case 1: Balanced State
    v_balanced = D21Vector(perception=0.5, charity=0.5, lust=0.5)
    pulse1 = engine.pulse(v_balanced, energy=100.0, dt=0.1)
    print(f"Balanced State: Coherence={pulse1.coherence:.4f}, Dominant={pulse1.dominant_realm}")
    print(f"Weights: a={pulse1.alpha:.2f}, b={pulse1.beta:.2f}, g={pulse1.gamma:.2f}")

    # Test Case 2: Survival Mode (Low Energy)
    v_low = D21Vector(lust=0.8, gluttony=0.9) # High body drives
    pulse2 = engine.pulse(v_low, energy=10.0, dt=0.1)
    print(f"\nSurvival Mode (Energy 10): Coherence={pulse2.coherence:.4f}, Dominant={pulse2.dominant_realm}")
    print(f"Weights: a={pulse2.alpha:.2f}, b={pulse2.beta:.2f}, g={pulse2.gamma:.2f}")

    # Test Case 3: Spiritual Ascension
    v_spirit = D21Vector(humility=1.0, kindness=1.0, charity=1.0)
    pulse3 = engine.pulse(v_spirit, energy=100.0, dt=0.1)
    print(f"\nSpiritual Ascension: Coherence={pulse3.coherence:.4f}, Dominant={pulse3.dominant_realm}")
    print(f"Weights: a={pulse3.alpha:.2f}, b={pulse3.beta:.2f}, g={pulse3.gamma:.2f}")

    if pulse2.alpha > pulse1.alpha and pulse3.beta > pulse1.beta:
        print("\nVERIFICATION SUCCESS: Triple-Helix Engine weight modulation is functional.")
    else:
        print("\nVERIFICATION FAILURE: Interaction logic mismatch.")

if __name__ == "__main__":
    test_triple_helix()
