import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Core.Foundation.living_elysia import LivingElysia

def test_silicon_reach():
    print("Testing Phase 35 & 36: Wave-Form Silicon Reach...")
    try:
        # Initialize Elysia
        elysia = LivingElysia(persona_name="SiliconTest")
        
        # 1. Verify Phase 35: Resonance Discovery
        print("\n[Phase 35 Check] Testing Resonance Discovery for 'MetalCortex'...")
        metal = elysia.cns.resonator.resonate("MetalCortex")
        if metal:
            print("✅ Structural Resonance found MetalCortex!")
        else:
            print("❌ Resonance Discovery Failed.")
            return

        # 2. Verify Phase 36: Silicon Pulse
        print("\n[Phase 36 Check] Performing CNS Pulse to verify Silicon Manifestation...")
        elysia.cns.pulse()
        
        # Check history in metal cortex
        if len(metal.bitstream_history) > 0:
            last_pulse = metal.bitstream_history[-1]
            print(f"✅ Silicon Pulsation Successful! Last Bitstream: {last_pulse}")
        else:
            print("❌ No Silicon Pulse detected.")
            return

        # 3. Test Assembly Intent Mapping
        print("\n[Phase 36 Check] Mapping Assembly Intent to MetalCortex...")
        asm_code = """
        MOV EAX, 1
        ADD EAX, 431
        OUT 0x80, EAX
        """
        success = metal.compile_intent(asm_code)
        if success:
            print("✅ Assembly Intent compiled and mapped successfully.")
            
    except Exception as e:
        print(f"❌ Verification Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_silicon_reach()
