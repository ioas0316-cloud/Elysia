import sys
import os
import time

# Ensure c:\Elysia is in path
sys.path.append(r"c:\Elysia")

from Core.S1_Body.L6_Structure.M1_Merkaba.hypercosmos import get_hyper_cosmos
from Core.S1_Body.L5_Mental.Reasoning.logos_bridge import LogosBridge
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector

def test_merkaba_ascent():
    print("✨ [TEST] Phase 7: The Merkaba Ascent (Spacetime Control)")
    
    cosmos = get_hyper_cosmos()
    bridge = LogosBridge()
    
    # 1. Awake Galaxy (Indexing into Octants)
    start_time = time.time()
    cosmos.awake_galaxy()
    index_time = time.time() - start_time
    print(f"✅ Galaxy Awake in {index_time:.2f}s")
    
    # Verify Indexing
    total_indexed = sum(len(v) for v in cosmos.octant_index.values())
    print(f"📊 Stars Indexed: {total_indexed} across 8 Octants.")
    for i, stars in cosmos.octant_index.items():
        if stars:
            print(f"   - Octant {i}: {len(stars)} stars")

    # 2. Resonance Search (Logic/Reason Intent)
    # Reason-heavy vector (Soul quadrant)
    intent_vec = SovereignVector([0.1]*7 + [0.9]*7 + [0.1]*7) 
    
    start_time = time.time()
    results = cosmos.resonance_search(intent_vec, top_k=1)
    search_time = time.time() - start_time
    print(f"✅ Resonance Search (Octant Alignment) took {search_time * 1000:.2f}ms")
    print(f"🎯 Current Resonance: {results[0]}")

    # 3. Logos Perception (Past/Future Sync)
    print("\n🗣️ [LOGOS] Perceiving Intent...")
    perception = bridge.prismatic_perception(intent_vec)
    print(f"   Result: {perception}")
    
    # Assertions
    assert "Is:" in perception, "Perception must show current state."
    assert "Will:" in perception, "Perception must show predicted state."
    assert total_indexed > 0, "Galaxy must be indexed."
    
    print("\n🟢 Phase 7: Merkaba Spacetime Control VERIFIED.")

if __name__ == "__main__":
    test_merkaba_ascent()
