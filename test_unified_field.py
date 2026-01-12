
import sys
import os
import random
sys.path.append(os.getcwd())

from Core.Foundation.hyper_sphere_core import HyperSphereCore
from Core.Foundation.Wave.wave_dna import WaveDNA
from Core.World.Physics.trinity_fields import TrinityVector
from Core.World.Nature.sensory_cortex import get_sensory_cortex

def test_unified_field():
    print("🧪 [Test] Phase 35: The Sensory Field (Unified HyperSphere Architecture)")
    
    # 1. Initialize the HyperSphere (The Medium)
    sphere = HyperSphereCore(name="Elysia.Carrier")
    cortex = get_sensory_cortex()

    # 2. Physics of Phenomena: A 'Rose' emits a scent pulse at (10, 10).
    # Scent frequency is roughly 5.0e13 Hz (decodes to Floral in SensoryCortex)
    rose_dna = WaveDNA(label="Rose", physical=0.1, phenomenal=0.9, frequency=5.0e13)
    rose_pulse = TrinityVector(rose_dna.physical, rose_dna.phenomenal, rose_dna.spiritual, rose_dna.frequency)
    
    print("\n1. [PHENOMENA] A Rose blooms at (10, 10), emitting a floral scent.")
    # Deposit into Channel 5 (Olfaction)
    sphere.field.deposit(10.0, 10.0, rose_pulse, channel_offset=5)

    # 3. Perception: An NPC 'Bob' is at (11, 11).
    bob_pos = (11, 11)
    print(f"\n2. [PERCEPTION] NPC Bob is at {bob_pos}. He sniff the air...")
    
    # Bob samples the HyperSphere's field directly
    # In a full impl, Bob would have a 'Sensor' object.
    scent_energy = sphere.field.sample(bob_pos[0], bob_pos[1], channel_offset=5)
    
    # Decode only if intensity > 0
    if scent_energy[1] > 0.1: # if phenomenal flow (intensity) is high enough
        # Create a temporary 'Experienced DNA' from the field sample
        experienced_dna = WaveDNA(
            label="Ambient Scent", 
            physical=scent_energy[0], 
            phenomenal=scent_energy[1], 
            spiritual=scent_energy[2], 
            frequency=scent_energy[3]
        )
        qualia = cortex.decode_qualia(experienced_dna)
        print(f"   Bob feels: '{qualia['aroma']}'")
    else:
        print("   Bob feels nothing.")

    # 4. Universal Clock: Time passes, the scent fades.
    print("\n3. [ENTROPY] 10 Ticks pass in the HyperSphere. The field decays...")
    for _ in range(10):
        sphere.tick(1.0)
        
    scent_after = sphere.field.sample(bob_pos[0], bob_pos[1], channel_offset=5)
    print(f"   Scent Intensity (Flow) after decay: {scent_after[1]:.4f}")
    
    if scent_after[1] < rose_pulse.flow:
        print("\n✅ Phase 35 Verification Successful: HyperSphere is the carrier of Senses.")
    else:
        print("\n❌ Verification Failed: Field did not decay.")

if __name__ == "__main__":
    test_unified_field()
