import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SoulDNA
from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector
import torch

def verify_semantic_atmosphere():
    print("[TEST] Initiating Semantic Atmosphere Verification...")
    
    # 1. Create a dummy DNA
    dna = SoulDNA(
        archetype="Test",
        id="001",
        rotor_mass=100.0,
        sync_threshold=0.8,
        min_voltage=5.0,
        reverse_tolerance=0.1,
        torque_gain=1.0,
        base_hz=60.0,
        friction_damping=0.5
    )
    
    # 2. Instantiate the Monad (which now includes CognitiveField and the 10M cell engine)
    monad = SovereignMonad(dna)
    
    # 3. Verify CognitiveField is attached
    if not hasattr(monad, 'cognitive_field'):
        print("❌ [FAILED] CognitiveField is not attached to SovereignMonad.")
        return False
        
    # 4. Extract Atmosphere
    atmosphere = monad.cognitive_field.get_semantic_atmosphere()
    print(f"✅ [SUCCESS] Atmosphere generated. Type: {type(atmosphere)}")
    
    if atmosphere.norm() == 0.0:
         print("⚠️ [WARNING] Atmosphere is zero. LogosBridge might not be populated in this test environment.")
    else:
         print(f"✅ [SUCCESS] Atmosphere has weight. Norm: {atmosphere.norm():.4f}")
         
    # 5. Run a Pulse to ensure the engine accepts the atmosphere without crashing
    try:
        # Running 1 Conscious Pulse (Tier 0)
        # We need to simulate the 'context' as well
        dummy_intent = SovereignVector([complex(0.1)] * 21)
        report = monad.pulse(dt=0.01, intent_v21=dummy_intent)
        if report:
             print(f"✅ [SUCCESS] Pulse completed with Semantic Atmosphere injection.")
             print(f"   ► Resonance: {report.get('resonance', 0.0):.4f}")
             print(f"   ► Joy: {report.get('joy', 0.0):.4f}")
             print(f"   ► Curiosity: {report.get('curiosity', 0.0):.4f}")
        else:
             print("❌ [FAILED] Pulse returned None.")
             return False
    except Exception as e:
        print(f"❌ [CRASH] Engine failed during pulse: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    print("🌟 [TEST COMPLETE] The Fence of Intent is active.")
    return True

if __name__ == "__main__":
    verify_semantic_atmosphere()
