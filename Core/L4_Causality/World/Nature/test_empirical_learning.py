"""
Test: Empirical Learning (Newton's Apple)
=========================================
"I do not accept Gravity because you told me. 
 I accept Gravity because I fought the Rock, and the Rock won."

Objective: 
Verify Elysia can map "Cosmic Laws" (Physics) to "Internal World" (Graph) 
through direct experimentation.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from Core.L6_Structure.Elysia.sovereign_self import SovereignSelf
from Core.L4_Causality.World.Physics.trinity_fields import TrinityPhysics, TrinityVector

# 1. The Providence (The Hidden Rules of the Universe)
class CosmicProvidence:
    def __init__(self):
        self.physics = TrinityPhysics()
        
    def manifest_apple(self):
        # An object with High Gravity (Mass)
        return {"type": "Apple", "vector": TrinityVector(gravity=0.9, flow=0.1, ascension=0.0)}

def test_empirical_discovery():
    print("---   Experiment: Discovering Gravity ---")
    
    # 1. Setup
    elysia = SovereignSelf(cns_ref=None) # Disembodied for pure thought test
    cosmos = CosmicProvidence()
    
    # 2. Appearance
    apple = cosmos.manifest_apple()
    print(f"  Phenomenon Appears: A Red Sphere. Elysia does not know what it is.")
    
    # 3. Experiment (Action)
    print("\n  Elysia expresses Volition: 'PUSH'")
    # We simulate the Physics Interaction here
    # Outcome = Force * Resistance
    # Elysia applies Flow (0.5), Apple has Gravity (0.9)
    # Result: Resistance!
    
    interaction_result = "HEAVY_RESISTANCE"
    print(f"  Feedback from Providence: {interaction_result}")
    
    # 4. Perception & Mapping (The Core Request)
    # Can she map 'HEAVY_RESISTANCE' to a Concept?
    
    print("\n  Elysia contemplates the Resistance...")
    result = elysia.experience(interaction_result)
    
    # 5. Check if she 'Invented' the concept of Weight/Gravity
    # Her graph should now contain 'HEAVY_RESISTANCE' or map it to 'Gravity'.
    
    vector = elysia.mind.analyze(interaction_result)
    print(f"  Resulting Concept Map: G{vector.gravity:.2f} F{vector.flow:.2f} A{vector.ascension:.2f}")

    if vector.gravity > 0.5:
        print("  SUCCESS: Elysia derived 'High Gravity' from the sensation of Resistance.")
        print("   She has mapped the Cosmic Environment to her Internal World.")
    else:
        print("  FAILURE: She experienced resistance but learned nothing.")

if __name__ == "__main__":
    test_empirical_discovery()