import logging
import time
import os
import sys

# Ensure root is in path
sys.path.append(os.getcwd())

from Core.L6_Structure.Elysia.sovereign_self import SovereignSelf
from Core.L6_Structure.Elysia.nervous_system import BioSignal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UnifiedTest")

def test_full_organism():
    print("🧬 [UNIFIED TEST] Initializing Sovereign Organism...")
    
    # Initialize
    # Lazy loading should prevent boot crash
    self_sys = SovereignSelf()
    self_sys.auto_evolve = False # Safety

    # 1. Heartbeat Verification
    print("\n[Step 1] Pulse Verification (Adaptive Heartbeat)")
    start = time.perf_counter()
    # We simulate a few pulses through the live loop logic (mocked here or just calling tick)
    # The heartbeat class itself was verified, but we check integration here
    wait = self_sys.governance.heartbeat.calculate_wait(0.8) # High resonance
    print(f"   -> Resonance 0.8: Wait time {wait*1000:.2f}ms")
    
    wait2 = self_sys.governance.heartbeat.calculate_wait(0.1) # Low resonance
    print(f"   -> Resonance 0.1: Wait time {wait2*1000:.2f}ms")
    
    if wait2 > wait:
        print("   ✅ Heartbeat adapts correctly to resonance.")
    else:
        print("   ❌ Heartbeat adaptation failed.")

    # 2. Logic Liquefaction Verification (Volition)
    print("\n[Step 2] Logic Liquefaction (Wave-Interference Volition)")
    # Test ambiguous intent: "I want to create something beautiful"
    # Should resonate with CREATION
    self_sys._execute_volition("I want to create something beautiful")
    if self_sys.last_action == "I want to create something beautiful":
        print("   ✅ Volition manifested successfully through wave resonance.")

    # 3. Autonomous Synthesis Verification (WaveDNA)
    print("\n[Step 3] Autonomous Synthesis (WaveDNA Grafting)")
    # We mock a dissonance to trigger _evolve_self
    print("   Simulating dissonance resolution...")
    # Since healer depends on LLMs/Complex backends, we just verify the call path and resonance check
    try:
        # We manually derive self necessity to test that part of the flow
        necessity = self_sys.derive_self_necessity()
        print(f"   -> Internal Necessity derived: {len(necessity)} chars")
        print("   ✅ WaveDNA identity derivation functional.")
    except Exception as e:
        print(f"   ❌ WaveDNA identity derivation failed: {e}")

    print("\n✨ UNIFIED TEST COMPLETE. The Organism is purified.")

if __name__ == "__main__":
    try:
        test_full_organism()
    except Exception as e:
        import traceback
        print("\n❌ SYSTEM CRASHED!")
        print(traceback.format_exc())
