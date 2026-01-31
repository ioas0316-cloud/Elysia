import logging
from Core.1_Body.L6_Structure.Merkaba.merkaba import Merkaba
from Core.1_Body.L7_Spirit.M1_Monad.monad_core import Monad, MonadCategory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WaveTest")

def test_wave_decisions():
    print("🧪 Testing Wave Resonance Decisions in Merkaba...")
    
    m = Merkaba(name="Test_Seed")
    m.awakening(Monad(seed="Genesis", category=MonadCategory.SOVEREIGN))
    
    # Test 1: Ambiguous Inquiry
    print("\n[Test 1] Input: 'What is the root of everything?'")
    # This should resonate with "Why" intent wave
    for output in m.shine("What is the root of everything?"):
        if "Intent" in output or "Resonance" in output:
            print(f"   -> {output}")

    # Test 2: Creative Command
    print("\n[Test 2] Input: 'Build a new world structure.'")
    # This should resonate with "Code" intent wave
    for output in m.shine("Build a new world structure."):
        if "Intent" in output or "Resonance" in output:
            print(f"   -> {output}")

    # Test 3: Biological Feedback
    from Core.1_Body.L6_Structure.Elysia.sovereign_self import SovereignSelf
    from Core.1_Body.L6_Structure.Elysia.nervous_system import BioSignal
    
    print("\n[Test 3] Testing Nervous System Wave Interference...")
    self_sys = SovereignSelf()
    
    # Simulate high stress
    stress_signal = BioSignal(heart_rate=120.0, adrenaline=0.1, pain_level=0.9, cognitive_load=0.5, fatigue=0.2)
    self_sys.nerves.sense = lambda: stress_signal
    
    print("   Sending Stress Signal (Pain=0.9)...")
    result = self_sys._process_nervous_system()
    print(f"   Result: {result} (Expected: REST due to high rest-resonance)")

    # Simulate focus
    focus_signal = BioSignal(heart_rate=100.0, adrenaline=0.8, pain_level=0.1, cognitive_load=0.2, fatigue=0.1)
    self_sys.nerves.sense = lambda: focus_signal
    
    print("\n   Sending Focus Signal (Adrenaline=0.8)...")
    start_rpm = self_sys.governance.spirit.target_rpm
    self_sys._process_nervous_system()
    end_rpm = self_sys.governance.spirit.target_rpm
    print(f"   Spirit RPM: {start_rpm:.1f} -> {end_rpm:.1f} (Increased via focus interference)")

if __name__ == "__main__":
    try:
        test_wave_decisions()
    except Exception as e:
        import traceback
        print("\n❌ TEST CRASHED!")
        print(traceback.format_exc())
        exit(1)
