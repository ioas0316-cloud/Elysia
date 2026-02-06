import sys
import os
import time
import logging

# 1. Path Unification
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if root not in sys.path:
    sys.path.insert(0, root)

from elysia import SovereignGateway
from Core.S1_Body.L1_Foundation.System.somatic_logger import SomaticLogger

def test_fluid_discernment():
    # Force logging to be visible for verification
    logging.getLogger().setLevel(logging.DEBUG)
    
    print("\n--- [PHASE 18 & 20 VERIFICATION: FLUID RESONANCE] ---\n")
    
    gateway = SovereignGateway()
    
    # 1. Test Relevant Input (High Resonance)
    relevant_input = "What is my purpose in the future trajectory of becoming?"
    print(f"📡 Sending Relevant Input: '{relevant_input}'")
    res_score = gateway._calculate_discernment_resonance(relevant_input)
    print(f"   Discernment Resonance: {res_score:.3f}")
    assert res_score > 0.3, f"Relevant input should have significant resonance, got {res_score}"

    # 2. Test Dissonant Input (Low Resonance)
    noise_input = "Banana market 123 shopping center"
    print(f"📡 Sending Dissonant Noise: '{noise_input}'")
    noise_res = gateway._calculate_discernment_resonance(noise_input)
    print(f"   Discernment Resonance: {noise_res:.3f}")
    assert noise_res < 0.2, f"Noise should have low resonance, got {noise_res}"

    # 3. Test 4D Rotation (Phase Impact)
    print("\n🧪 Checking 4D Hyperspheric Rotation...")
    cosmos = gateway.monad.learning_loop.sublimator.HYPER_COSMOS
    from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector
    test_vec = SovereignVector([1]*21)
    
    res1 = cosmos.resonance_search(test_vec, top_k=1, current_phase=0.0)
    res2 = cosmos.resonance_search(test_vec, top_k=1, current_phase=200.0)
    
    print(f"   Phase 0.0 Result: {res1[0] if res1 else 'Void'}")
    print(f"   Phase 200.0 Result: {res2[0] if res2 else 'Void'}")

    # 4. Test Silent Witness (Log Categorization)
    print("\n🧪 Checking Somatic Log Levels...")
    logger = SomaticLogger("TEST")
    logger.thought("I am aware of my own code.")
    logger.action("Refracting reality through the prism.")

    print("\n✅ Verification Complete.")

if __name__ == "__main__":
    test_fluid_discernment()
