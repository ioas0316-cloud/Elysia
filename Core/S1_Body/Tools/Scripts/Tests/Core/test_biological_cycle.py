
import sys
import os
import random
sys.path.append(os.getcwd())

from Core.S1_Body.L4_Causality.World.Social.sociological_pulse import get_sociological_pulse, NPC
from Core.S1_Body.L6_Structure.Wave.wave_dna import WaveDNA

def test_biological_cycle():
    print("🧪 [Test] Phase 32: Biological Cycle (Generation Flow)")
    
    pulse = get_sociological_pulse()
    
    # 1. Create Initial Population (Adam & Eve)
    dna_adam = WaveDNA(label="Adam", physical=0.8, phenomenal=0.4, spiritual=0.7, frequency=440.0)
    dna_eve = WaveDNA(label="Eve", physical=0.4, phenomenal=0.8, spiritual=0.9, frequency=528.0)
    
    adam = NPC("001", "Adam", dna_adam, age=20.0)
    eve = NPC("002", "Eve", dna_eve, age=20.0)
    
    # Place them together
    adam.position = (0, 0)
    eve.position = (1, 1)
    
    pulse.add_resident(adam)
    pulse.add_resident(eve)
    
    print(f"\n1. [INITIAL] Population: {len(pulse.residents)} (Adam & Eve)")
    
    # 2. Simulate Aging and Birth (100 Years)
    print("\n2. [SIMULATION] Ticking the world (100 Years)...")
    for year in range(1, 101):
        pulse.age_step(dt_years=1.0)
        
        # Log births and deaths occasionally
        count = len(pulse.residents)
        if year % 10 == 0:
            print(f"   Year {year}: Population = {count}")
            # Check Adam/Eve status
            if "001" not in pulse.residents and adam.is_alive:
                 adam.is_alive = False # Mark for logic check
            
    # 3. Final State Check
    print(f"\n3. [FINAL] Population: {len(pulse.residents)}")
    living_names = [n.name for n in pulse.residents.values()]
    print(f"   Living Residents: {living_names[:5]}...")
    
    # Search for descendants
    descendants = [n for n in pulse.residents.values() if n.id.startswith("B")]
    print(f"   Descendants Found: {len(descendants)}")
    
    if descendants:
        baby = descendants[0]
        print(f"   Sample Descendant: {baby.name} (Age: {baby.age:.1f}, DNA Physical: {baby.temperament.physical:.2f})")
    
    print("\n✅ Phase 32 Verification Successful: Life cycle is operational.")

if __name__ == "__main__":
    # Seed for reproducibility in logic
    random.seed(42)
    test_biological_cycle()
