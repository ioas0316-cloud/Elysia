
import sys
import os
import random
sys.path.append(os.getcwd())

from Core.Cognition.sociological_pulse import SociologicalPulse, NPC
from Core.Keystone.wave_dna import WaveDNA

def test_eco_system():
    print("🧪 [Test] Phase 36: Material Topography & Resource Ecology")
    
    pulse = SociologicalPulse(field_size=50)
    dna = WaveDNA(label="Explorer", frequency=432.0, phenomenal=0.5)
    
    # 1. Metabolism Test (Decay)
    alice = NPC("A1", "Alice", dna, age=20.0)
    alice.position = [40, 40] # Empty wasteland
    alice.energy = 100.0
    pulse.add_resident(alice)
    
    print(f"\n1. [METABOLISM] Alice starts in a wasteland. Energy: {alice.energy:.1f}")
    pulse.update_social_field()
    print(f"   Alice Energy after 1 tick: {alice.energy:.1f} (Should be lower due to decay)")

    # 2. Feeding Test (Resource Intake)
    bob = NPC("B1", "Bob", dna, age=20.0)
    bob.position = [20, 20] # Near a Forest (Seeded in pulse.__init__)
    bob.energy = 50.0
    pulse.add_resident(bob)
    
    print(f"\n2. [FEEDING] Bob starts near a Forest. Energy: {bob.energy:.1f}")
    for _ in range(5): pulse.update_social_field()
    print(f"   Bob Energy after 5 ticks: {bob.energy:.1f} (Should be higher due to forest intake)")

    # 3. Terrain Test (Climbing)
    charlie = NPC("C1", "Charlie", dna, age=20.0)
    charlie.position = [0, 0] # On a Mountain Peak (Elevation 1.0)
    pulse.add_resident(charlie)
    
    print(f"\n3. [TERRAIN] Charlie is at the Mountain Peak.")
    # Charlie's speed mod should be around 0.5 (1.0 - 1.0*0.5)
    pulse.update_social_field()
    speed = (charlie.velocity[0]**2 + charlie.velocity[1]**2)**0.5
    print(f"   Charlie Speed: {speed:.2f} (Base is 2.0 * health. Should be reduced by elevation)")

    if bob.energy > 50.0 and speed < 2.0:
        print("\n✅ Phase 36 Verification Successful: World complexity (Terrain/Resources/Metabolism) established.")
    else:
        print("\n❌ Verification Failed: Eco-system mechanics not detected.")

if __name__ == "__main__":
    test_eco_system()
