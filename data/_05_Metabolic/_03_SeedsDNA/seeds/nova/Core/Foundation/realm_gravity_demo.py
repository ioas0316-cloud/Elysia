"""
The Gravity Law (중력의 법칙)

ONE LAW: Vitality creates Mass. Mass attracts Waves.
Everything else emerges naturally.

Like gravity bending light, high-vitality realms bend thought.
"""

import numpy as np
import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Core._01_Foundation.Elysia.Elysia.World.yggdrasil import Yggdrasil
from Core._01_Foundation.Foundation.Physics.gravity import GravityEngine

# Alias for backward compatibility in this demo
RealmGravity = GravityEngine

def demonstrate_emergence():
    """
    Demonstrate how prediction emerges from the gravity law.
    """
    print("\n" + "="*70)
    print("🌊 The Gravity Law: Emergent Prediction")
    print("="*70)
    print("\nLAW: Mass = Vitality × Layer_Weight")
    print("RESULT: Waves flow naturally toward massive realms")
    print("EMERGENCE: Prediction happens without explicit logic\n")
    
    # Load Yggdrasil
    yggdrasil = Yggdrasil()
    try:
        yggdrasil.load()
        print("✅ Loaded Yggdrasil from file")
    except:
        print("⚠️ No saved Yggdrasil, using fresh state")
        # Would need to plant realms here, but for demo assume it exists
        return
    
    gravity = RealmGravity(yggdrasil)
    
    # Show realm masses
    print("\n" + "-"*70)
    print("📊 Realm Gravitational Masses")
    print("-"*70)
    
    for realm in yggdrasil.realms.values():
        mass = gravity.calculate_mass(realm.name)
        print(f"   {realm.name:20s} | Vitality: {realm.vitality:.2f} | Mass: {mass:.2f}")
    
    # Test thought propagation
    print("\n" + "-"*70)
    print("🌊 Thought Wave Propagation Test")
    print("-"*70)
    
    test_thoughts = ["감정", "기억", "지각"]
    
    for thought in test_thoughts:
        print(f"\n💭 Starting thought: '{thought}'")
        predicted = gravity.predict_next_active_realm(thought)
        print(f"   → Predicted next activation: {predicted}")
        
        # Show energy field
        thought_realm_map = {
            "감정": "EmotionalPalette",
            "기억": "EpisodicMemory",
            "지각": "FractalPerception"
        }
        start_realm = thought_realm_map.get(thought, "FractalPerception")
        field = gravity.propagate_thought_wave(start_realm, wave_intensity=1.0)
        
        print(f"   Energy Distribution:")
        sorted_field = sorted(field.items(), key=lambda x: x[1], reverse=True)[:5]
        for realm_name, energy in sorted_field:
            print(f"      {realm_name:20s}: {energy:.3f}")
    
    # The Key Insight
    print("\n" + "="*70)
    print("✨ THE EMERGENCE")
    print("="*70)
    print("""
We didn't program:
  - "If emotion, then activate memory"
  - "If perception, then activate voice"
  
We only planted ONE LAW:
  - Mass = Vitality × Layer_Weight
  
And physics did the rest:
  - Waves naturally flow to high-mass realms
  - Active realms (high vitality) attract more thought
  - Prediction emerges from topology, not logic
  
This is how the ocean creates life.
Not through rules, but through laws of nature.
    """)
    
    print("💚 자연법칙이 창발을 낳았습니다.")


if __name__ == "__main__":
    demonstrate_emergence()
