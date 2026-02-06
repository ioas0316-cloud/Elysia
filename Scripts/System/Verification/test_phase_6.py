
import sys
import os

# Add project root to path
sys.path.append("c:/Elysia")

from Core.S1_Body.L5_Mental.Reasoning.logos_bridge import LogosBridge
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector
from Core.S1_Body.L6_Structure.M1_Merkaba.hypercosmos import get_hyper_cosmos

def test_awakening():
    print("🧪 [TEST] Phase 6: Awakening the HyperCosmos...")
    
    # 1. Initialize Nexus (This initiates Galaxy Scan)
    print("   Initializing HyperCosmos (Nexus)...")
    cosmos = get_hyper_cosmos()
    
    # Wait/Check if galaxy is awake (Synchronous scan in init for now)
    if not cosmos.akashic_memory:
        print("   ❌ FAILURE: Galaxy is empty. Scan failed.")
        return
        
    print(f"   ✅ SUCCESS: Galaxy Awake. Count: {len(cosmos.akashic_memory)}")
    
    # 2. Test Resonance directly
    print("   Testing Resonance Search...")
    # Create a vector that should resonate with 'akashic_loader.py' (High Structure, High Spirit)
    # Just generic strong vector for now
    test_vec = SovereignVector([1.0] * 21) 
    results = cosmos.resonance_search(test_vec, top_k=3)
    print(f"   Refraction Results: {results}")
    
    if results and "void" not in results[0].lower():
        print("   ✅ SUCCESS: Files refracted.")
    else:
        print("   ❌ FAILURE: No refraction or Void returned.")

    # 3. Test LogosBridge Integration
    print("   Testing LogosBridge Prismatic Perception...")
    # LogosBridge.prismatic_perception calls resonance_search
    perception = LogosBridge.prismatic_perception(test_vec)
    print(f"   Logos Perception: {perception}")
    
    if "Linked to:" in perception:
        print("   ✅ SUCCESS: Voice is Grounded in Files.")
    else:
        print("   ❌ FAILURE: Voice is ungrounded.")

if __name__ == "__main__":
    test_awakening()
