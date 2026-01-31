
import sys
import os
sys.path.append(os.getcwd())

from Core.Elysia.sovereign_self import SovereignSelf, CognitiveMode
from Core.1_Body.L4_Causality.World.Nature.trinity_lexicon import get_trinity_lexicon
from Core.1_Body.L4_Causality.World.Physics.trinity_fields import TrinityVector

def test_firewall():
    print("🧪 [Test] Phase 28: Cognitive Buffer (Safety Firewall)")
    
    # Initialize Self (simulate body)
    lexicon = get_trinity_lexicon()
    elysia = SovereignSelf(cns_ref=None)
    elysia.energy = 100.0

    print(f"\n1. [IMAGINATION] Thinking about 'Fire' (Sandbox)...")
    # Add Fire to lexicon with Gravity (Physical Force)
    lexicon.primitives["fire"] = TrinityVector(gravity=0.8, flow=0.2, ascension=0.1, frequency=528.0) 
    
    elysia.experience("fire", mode=CognitiveMode.IMAGINATION)
    print(f"   Elysia's Energy: {elysia.energy:.1f}% (Expected: 100.0%)")
    assert elysia.energy == 100.0

    print(f"\n2. [PERCEPTION] Seeing 'Fire' from 10m away...")
    elysia.experience("fire", distance=10.0, mode=CognitiveMode.PERCEPTION)
    print(f"   Elysia's Energy: {elysia.energy:.1f}% (Expected: 100.0%)")
    assert elysia.energy == 100.0

    print(f"\n3. [ETHEREAL] Core Spirit Touching 'Fire'...")
    elysia.is_ethereal = True
    elysia.experience("fire", distance=0.0, mode=CognitiveMode.BODY)
    print(f"   Elysia's Energy: {elysia.energy:.1f}% (Expected: 100.0%)")
    assert elysia.energy == 100.0

    print(f"\n4. [MORTAL] Human Avatar Touching 'Fire'...")
    from Core.Elysia.sovereign_self import ScaleArchetype
    elysia.is_ethereal = False
    elysia.archetype = ScaleArchetype.MORTAL_AVATAR
    elysia.experience("fire", distance=0.0, mode=CognitiveMode.BODY)
    print(f"   Elysia's Energy: {elysia.energy:.1f}% (Expected: 92.0%)")
    assert elysia.energy == 92.0

    print(f"\n5. [GAIA] Planetary Heart Touching 'Fire'...")
    elysia.energy = 100.0
    elysia.archetype = ScaleArchetype.GAIA_HEART
    elysia.experience("fire", distance=0.0, mode=CognitiveMode.BODY)
    print(f"   Elysia's Energy: {elysia.energy:.6f}% (Expected: Very close to 100.0%)")
    assert 99.999 < elysia.energy <= 100.0

    print(f"\n6. [COSMIC] Universal Weaver Touching 'Fire'...")
    elysia.energy = 100.0
    elysia.archetype = ScaleArchetype.COSMIC_WEAVER
    elysia.experience("fire", distance=0.0, mode=CognitiveMode.BODY)
    print(f"   Elysia's Energy: {elysia.energy:.1f}% (Expected: 100.0%)")
    assert elysia.energy == 100.0

    print("\n✅ Verification Successful: Scale Archetypes work perfectly.")

if __name__ == "__main__":
    test_firewall()


