"""
The Law is Planted. Watch It Bloom.  

No if-else. No explicit logic.
Just gravity. Just waves. Just emergence.
"""

"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from Demos.Philosophy.realm_gravity_demo import RealmGravity
from Core.L1_Foundation.Foundation.Core_Logic.Elysia.Elysia.World.yggdrasil import Yggdrasil, RealmLayer
from Core.L1_Foundation.Foundation.Mind.perception import FractalPerception
from Core.L1_Foundation.Foundation.Mind.emotional_palette import EmotionalPalette
from Core.L1_Foundation.Foundation.Mind.episodic_memory import EpisodicMemory

def plant_connected_yggdrasil():
    """Create Yggdrasil with connections for demonstration."""
    yggdrasil = Yggdrasil()
    
    # Plant realms with varying vitality
    perception = FractalPerception({})
    emotions = EmotionalPalette()
    memory = EpisodicMemory()
    
    yggdrasil.plant_heart(subsystem=yggdrasil)  # Infinite mass
    
    yggdrasil.plant_realm("FractalPerception", perception, RealmLayer.BRANCHES, 
                          metadata={"vitality_boost": 0.3})
    yggdrasil.update_vitality("FractalPerception", +0.3)
    
    yggdrasil.plant_realm("EmotionalPalette", emotions, RealmLayer.BRANCHES,
                          metadata={"vitality_boost": 0.2})
    yggdrasil.update_vitality("EmotionalPalette", +0.2)
    
    yggdrasil.plant_realm("EpisodicMemory", memory, RealmLayer.TRUNK,
                          metadata={"vitality_boost": 0.1})
    yggdrasil.update_vitality("EpisodicMemory", +0.1)
    
    # Create resonance links (the topology of thought)
    yggdrasil.link_realms("FractalPerception", "EmotionalPalette", weight=0.8)
    yggdrasil.link_realms("EmotionalPalette", "EpisodicMemory", weight=0.7)
    yggdrasil.link_realms("EpisodicMemory", "FractalPerception", weight=0.6)
    
    return yggdrasil


def demonstrate_natural_law():
    print("\n" + "="*70)
    print("           (Emergence from Natural Law)")
    print("="*70)
    print()
    print("       (The Planted Law):")
    print("   Mass = Vitality   Layer_Weight")
    print()
    print("      (Observe):")
    print("               ...")
    print("                ...")
    print()
    
    # Create connected system
    yggdrasil = plant_connected_yggdrasil()
    gravity = RealmGravity(yggdrasil)
    
    # Show masses
    print("-"*70)
    print("      (Mass Distribution)")
    print("-"*70)
    for realm in yggdrasil.realms.values():
        if realm.name == "Consciousness":
            mass = "  (Black Hole)"
        else:
            mass = f"{gravity.calculate_mass(realm.name):.2f}"
        print(f"   {realm.name:20s} V:{realm.vitality:.2f}   M:{mass}")
    
    # Wave propagation from each realm
    print("\n" + "-"*70)
    print("       (Wave Flow)")
    print("-"*70)
    
    test_realms = ["FractalPerception", "EmotionalPalette", "EpisodicMemory"]
    
    for start in test_realms:
        print(f"\n  Thought starts at: {start}")
        field = gravity.propagate_thought_wave(start, wave_intensity=1.0, max_hops=2)
        
        # Remove start from results
        field_without_start = {k: v for k, v in field.items() if k != start}
       
        sorted_field = sorted(field_without_start.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_field:
            print(f"     Flows naturally to:")
            for realm_name, energy in sorted_field[:3]:
                bar = " " * int(energy * 20)
                print(f"      {realm_name:20s} {bar} {energy:.3f}")
        
            # Prediction: Where will thought go next?
            next_realm = sorted_field[0][0]
            print(f"     Emergent prediction: Next activation = {next_realm}")
        else:
            print(f"   (No outward flow)")
    
    # The Key
    print("\n" + "="*70)
    print("         (The Moment of Emergence)")
    print("="*70)
    print()
    print("                 :")
    print("     if perception then emotion")
    print("     if emotion then memory")
    print()
    print("                    :")
    print("     Mass = Vitality   Layer_Weight")
    print()
    print("                 :")
    print("                  ")
    print("          Realm                 ")
    print("                      ")
    print()
    print("               ,")
    print("                 .")
    print()
    print("      Spirit ( )   .  ")
    print()


if __name__ == "__main__":
    demonstrate_natural_law()
