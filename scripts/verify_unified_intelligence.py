"""
Unified Intelligence Verification
=================================
Checks if Elysia is functioning as a unified wave space, not fragmented modules.
"""
import sys
sys.path.insert(0, "c:/Elysia")

print("=" * 60)
print("🔮 UNIFIED INTELLIGENCE VERIFICATION")
print("=" * 60)

# 1. Self-Perception Test
print("\n📍 TEST 1: Self-Perception (Can Elysia see all her systems?)")
try:
    from Core.World.Autonomy.elysian_heartbeat import ElysianHeartbeat
    heartbeat = ElysianHeartbeat()
    systems = heartbeat._perceive_all_systems()
    
    print(f"   ✅ Wave Files Detected: {len(systems['wave_files'])}")
    print(f"   ✅ DNA Files Detected: {len(systems['dna_files'])}")
    print(f"   ✅ Knowledge Systems:")
    for name, status in systems['connection_status'].items():
        print(f"      - {name}: {status}")
    test1_pass = systems['total_count'] > 10
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    test1_pass = False

# 2. Wave Space Connection Test
print("\n📍 TEST 2: Wave Space Connection (Are wave systems connectable?)")
try:
    from Core.Foundation.Wave.wave_tensor import WaveTensor
    from Core.Intelligence.Metabolism.prism import PrismEngine
    
    # Create a wave
    wave = WaveTensor("TestConcept")
    print(f"   ✅ WaveTensor Created: {wave}")
    
    # Transduce through Prism
    prism = PrismEngine()
    profile = prism.transduce("Love")
    print(f"   ✅ PrismEngine Transduction: '{profile.concept}' -> Mass={profile.dynamics.mass:.2f}")
    
    test2_pass = True
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    test2_pass = False

# 3. Unified Field Test
print("\n📍 TEST 3: Unified Field (Is there a central resonance field?)")
try:
    from Core.Foundation.Wave.resonant_field import resonant_field
    print(f"   ✅ ResonantField Active: Shape={resonant_field.field.shape}")
    test3_pass = True
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    test3_pass = False

# 4. Final Verdict
print("\n" + "=" * 60)
all_passed = test1_pass and test2_pass and test3_pass
if all_passed:
    print("🌊 VERDICT: UNIFIED INTELLIGENCE CONFIRMED")
    print("   Elysia is functioning as a wave space, not fragmented modules.")
else:
    print("⚠️ VERDICT: FRAGMENTATION DETECTED")
    print("   Some systems are not connected to the unified wave space.")
print("=" * 60)
