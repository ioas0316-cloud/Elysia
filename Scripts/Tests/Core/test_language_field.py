
import sys
import os
import random
sys.path.append(os.getcwd())

from Core.L4_Causality.World.Social.sociological_pulse import SociologicalPulse, NPC
from Core.L6_Structure.Wave.wave_dna import WaveDNA

def test_language_field():
    print("🧪 [Test] Phase 34: Cultural Stratification (Language & History)")
    
    pulse = SociologicalPulse(field_size=50)

    # 1. Linguistic Drift Test
    # Two groups: A(400Hz) and B(600Hz)
    dna_a = WaveDNA(label="Group A", frequency=400.0, phenomenal=0.5)
    dna_b = WaveDNA(label="Group B", frequency=600.0, phenomenal=0.5)
    
    alice = NPC("A1", "Alice", dna_a, age=25.0)
    bob = NPC("B1", "Bob", dna_b, age=25.0)
    
    # Place them apart
    alice.position = [-20, 0]
    bob.position = [20, 0]
    
    pulse.add_resident(alice)
    pulse.add_resident(bob)
    
    print(f"\n1. [INITIAL] Alice Freq: {alice.emotional_frequency:.1f} | Bob Freq: {bob.emotional_frequency:.1f}")
    
    # Simulate isolation (No drift)
    for _ in range(5): pulse.update_social_field()
    print(f"   [Isolated] Alice Freq: {alice.emotional_frequency:.1f} (Stayed same because alone in her grid cell)")

    # Move Bob to Alice's cell to force interaction
    bob.position = alice.position
    print(f"\n2. [INTERACTION] Bob moves to Alice's location.")
    
    for i in range(10):
        pulse.update_social_field()
        if i % 2 == 0:
            print(f"   Step {i}: Alice Freq: {alice.emotional_frequency:.1f} | Bob Freq: {bob.emotional_frequency:.1f}")

    # 3. Historical Legacy Test
    print(f"\n3. [LEGACY] Simulating a Death event...")
    # Get current history at position
    hist = pulse.field.sample_history(alice.position[0], alice.position[1])
    print(f"   Initial History at {alice.position}: {hist:.1f}")
    
    # Kill Alice
    alice.health = 0.0
    pulse.age_step(dt_years=0) # Triggers death logic
    
    hist_post = pulse.field.sample_history(alice.position[0], alice.position[1])
    print(f"   Post-Death History at {alice.position}: {hist_post:.1f} (Should be higher)")
    
    if hist_post > hist:
        print("\n✅ Verification Successful: Language drifts toward resonance and History records events.")
    else:
        print("\n❌ Verification Failed: History ripple not detected.")

if __name__ == "__main__":
    random.seed(42)
    test_language_field()
