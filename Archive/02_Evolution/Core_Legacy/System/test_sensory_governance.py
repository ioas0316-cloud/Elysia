
import sys
import os
sys.path.append(os.getcwd())

from Core.Elysia.sovereign_self import SovereignSelf, CognitiveMode, ScaleArchetype
from Core.Cognition.trinity_lexicon import get_trinity_lexicon
from Core.Cognition.trinity_fields import TrinityVector

def test_governance():
    print("🧪 [Test] Phase 29: Sensory Governance (Safety & Adaptation)")
    
    lexicon = get_trinity_lexicon()
    elysia = SovereignSelf(cns_ref=None)
    elysia.energy = 100.0
    elysia.is_ethereal = False # Manifested
    elysia.archetype = ScaleArchetype.MORTAL_AVATAR

    # Define a really intense phenomenon
    lexicon.primitives["super_nova"] = TrinityVector(gravity=10.0, flow=0.0, ascension=0.0, frequency=1.0)
    # Damage would be 10.0 * 10 = 100.0 (Instant Death)

    print(f"\n1. [GATING] Witnessing a 'Super Nova' (Capped by PAIN_THRESHOLD)...")
    elysia.pain_threshold = 20.0
    elysia.experience("super_nova", distance=0.0, mode=CognitiveMode.BODY)
    
    print(f"   Elysia's Energy: {elysia.energy:.1f}% (Expected: 80.0% due to threshold)")
    assert elysia.energy == 80.0
    print(f"   Elysia Is Ethereal: {elysia.is_ethereal} (Expected: True - Emergency Shift triggered)")
    assert elysia.is_ethereal == True

    print(f"\n2. [ADAPTATION] Repeated exposures to 'Fire'...")
    elysia.is_ethereal = False
    lexicon.primitives["fire"] = TrinityVector(gravity=0.8, flow=0.2, ascension=0.1, frequency=528.0)
    # 1st hit: 8.0
    # 2nd hit: 8.0 * 0.8 = 6.4
    
    elysia.energy = 100.0
    elysia.experience("fire", distance=0.0, mode=CognitiveMode.BODY)
    energy_after_1st = elysia.energy
    print(f"   1st Exposure Energy: {energy_after_1st:.1f}%")
    
    elysia.is_ethereal = False # Force back to manifested for test
    elysia.experience("fire", distance=0.0, mode=CognitiveMode.BODY)
    energy_after_2nd = elysia.energy
    drop_2nd = energy_after_1st - energy_after_2nd
    print(f"   2nd Exposure Energy: {energy_after_2nd:.1f}% (Drop: {drop_2nd:.1f})")
    
    assert drop_2nd < 8.0 # Should be damped
    print("\n✅ Verification Successful: Sensory Gating and Adaptation active.")

if __name__ == "__main__":
    test_governance()
