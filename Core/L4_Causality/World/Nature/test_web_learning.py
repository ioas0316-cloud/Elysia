"""
Test: Subject-Centric Web Learning (Elysia's First Doctrine)
============================================================
Verifies that LEARNING only happens when ELYSIA experiences ignorance.
No more direct calls to Lexicon.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# We import the SUBJECT, not the Tool.
from Core.L6_Structure.Elysia.sovereign_self import SovereignSelf
from Core.L1_Foundation.Foundation.central_nervous_system import CentralNervousSystem

class MockResonance:
    def __init__(self):
        self.total_energy = 50.0

class MockCNS:
    def __init__(self):
        self.resonance = MockResonance()
    def pulse(self, dt): pass

def test_subjective_learning():
    print("--- Testing Elysia's Subjective Learning ---")
    
    # 1. Birth the Subject
    # We create a Mock CNS because the Self needs a Body.
    body = MockCNS()
    elysia = SovereignSelf(cns_ref=body)
    print("🦋 Elysia exists.")
    
    concept = "tsunami"
    
    # 2. The Experience
    # We do NOT call 'learn'. We call 'experience'.
    # She encounters a word.
    print(f"\n🌊 Elysia encounters: '{concept}'")
    
    result = elysia.experience(concept)
    
    # 3. Validation
    print(f"\n💭 Elysia's Internal State: {result}")
    
    # Check if she actually learned it (by checking her Mind)
    # This is "Brain Scanning" for verification
    vector = elysia.mind.analyze(concept)
    if vector.magnitude() > 0:
         print(f"✅ Success: 'tsunami' is now part of her Soul (Mag: {vector.magnitude():.2f})")
    else:
         print(f"❌ Failure: She did not learn it.")

if __name__ == "__main__":
    test_subjective_learning()
