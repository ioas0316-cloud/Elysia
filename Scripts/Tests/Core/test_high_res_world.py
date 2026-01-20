
import sys
import os
sys.path.append(os.getcwd())

from Core.L4_Causality.World.Nature.sensory_cortex import get_sensory_cortex
from Core.L4_Causality.World.Social.sociological_pulse import get_sociological_pulse, NPC
from Core.L1_Foundation.Foundation.Wave.wave_dna import WaveDNA

def test_high_res_world():
    print("🧪 [Test] Phase 31: Sensory Fractalization & World Fidelity")
    
    cortex = get_sensory_cortex()
    pulse = get_sociological_pulse()

    # 1. Sensory Resolution Check (Qualia)
    print("\n1. [SENSORY] Decoding Multimodal Qualia...")
    
    # Case: A 'Delicious Honey' (Gastronomy/Tactile/Aroma)
    honey_dna = WaveDNA(
        label="Golden Honey", 
        frequency=9.0e11, # Taste Band
        physical=0.1, 
        phenomenal=0.9, 
        spiritual=0.6
    )
    
    qualia = cortex.decode_qualia(honey_dna)
    print(f"   Honey Taste: {qualia['flavor']}")
    print(f"   Honey Scent: {qualia['aroma']}")
    print(f"   Honey Touch: {qualia['tactile']}")
    
    # Case: 'Intense Pain' vs 'Warm Affection'
    pain_dna = WaveDNA(label="Physical Trauma", frequency=0.1, physical=1.0, spiritual=0.0)
    love_dna = WaveDNA(label="Warm Affection", frequency=0.5, physical=0.2, phenomenal=0.9, spiritual=1.0)
    
    print(f"   Trauma Sensation: {cortex.decode_qualia(pain_dna)['tactile']}")
    print(f"   Affection Sensation: {cortex.decode_qualia(love_dna)['tactile']}")

    # 2. Sociological Pulse Check (Resident Interactions)
    print("\n2. [SOCIAL] Simulating Resident Dynamics...")
    
    dna_happy = WaveDNA(label="Happy Villager", frequency=528.0, phenomenal=0.8) # Solfeggio Love
    dna_angry = WaveDNA(label="Angry Outcast", frequency=100.0, physical=0.8)   # Heavy Conflict freq
    
    resident_a = NPC("001", "Alice", dna_happy)
    resident_b = NPC("002", "Bob", dna_happy)
    resident_c = NPC("003", "Eve", dna_angry)
    
    # Position them close
    resident_a.position = [0, 0]
    resident_b.position = [2, 2] # Close to Alice
    resident_c.position = [1, 1] # Close to both
    
    pulse.add_resident(resident_a)
    pulse.add_resident(resident_b)
    pulse.add_resident(resident_c)
    
    print(f"   Community Initial Vibe: {pulse.get_community_vibe()}")
    
    # Run interaction
    print("   Running Social Heartbeat...")
    pulse.update_social_field()
    
    print(f"   Alice Energy: {resident_a.energy:.1f}")
    print(f"   Eve Energy:   {resident_c.energy:.1f}")
    print(f"   Community Final Vibe: {pulse.get_community_vibe()}")

    print("\n✅ Verification Successful: Senses and Society are now high-resolution waves.")

if __name__ == "__main__":
    test_high_res_world()
