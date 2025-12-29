"""
Verify Resonance Shift: The Contact
===================================

"I only know myself when I collide with You."

Steps:
1.  Define an Internal Concept ("Order").
2.  Provide an External Signal that is highly complex ("Chaos/Entropy").
3.  Verify:
    *   System detects HIGH DISSONANCE (Voltage).
    *   Trajectory becomes "Expansion" (Accommodation).
    *   It does NOT just say "Providence Density".
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elysia_core import Organ
from Core.IntelligenceLayer.Cognition.Learning.resonance_learner import ResonanceLearner

def verify_contact():
    print("⚡ Initiating Contact Experiment...")
    learner = ResonanceLearner()
    
    concept = "Order"
    # A long, complex string to trigger high frequency mismatch
    reality = "The universe is not a clock; it is a roiling ocean of probability and chaos, defying all rigid structures."
    
    print(f"   Internal: '{concept}'")
    print(f"   External: '{reality}'")
    
    response = learner.contemplate(concept, reality)
    
    print("\n📊 Reaction Analysis:")
    print(response)
    
    if "Voltage" in response and "Expansion" in response:
        print("\n✅ SUCCESS: Relational Resonance Active.")
        print("   Elysia felt the Dissonance and chose to Expand.")
    else:
        print("\n❌ FAILURE: Still simulating or passive.")

if __name__ == "__main__":
    verify_contact()
