import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_boot():
    print("Testing system boot and module imports...")
    try:
        from Core.Foundation.Wave.resonance_field import ResonanceField
        res = ResonanceField()
        print(f"✅ ResonanceField imported. Has 'propagate_aurora': {hasattr(res, 'propagate_aurora')}")
        
        from Core.Foundation.living_elysia import LivingElysia
        # Initialize without running
        elysia = LivingElysia(persona_name="BootTest")
        print("✅ LivingElysia initialized successfully.")
        
        # Test one pulse
        elysia.cns.pulse()
        print("✅ CNS Pulse successful.")
        
    except Exception as e:
        print(f"❌ Boot Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_boot()
