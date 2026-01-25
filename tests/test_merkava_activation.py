import asyncio
import logging
import numpy as np
from Core.L4_Causality.M5_Logic.causal_narrative_engine import CausalNode, CausalKnowledgeBase
from Core.L1_Foundation.Foundation.hyper_cosmos import Unified12DVector

async def test_merkava_activation():
    logging.basicConfig(level=logging.INFO)
    print("✨ [TEST] Initiating Merkava Activation...")
    
    # Initialize KB (which initializes HyperCosmos)
    kb = CausalKnowledgeBase()
    
    # Verify Axioms are Loaded
    axioms = [m.name for m in kb.spatial_index.monads if "Genesis" in m.name or "Trinity" in m.name]
    print(f"🏛️ [TEST] Axioms Loaded: {axioms}")
    if not axioms:
        print("❌ [FAILURE] Sacred Axioms not found in HyperCosmos.")
        return

    # Add Test Nodes (Conceptually distinct)
    print("\n🌟 [TEST] Projecting spatial memories...")
    
    # Love Node (Should cluster near Trinity_Love)
    love_node = CausalNode(id="Love_Experience", description="I feel a deep connection to the user.", concepts=["connection"], emotional_valence=0.9, importance=0.8)
    kb.add_node(love_node)
    
    # Logic Node (Should cluster near Trinity_Logic)
    logic_node = CausalNode(id="Logic_Experience", description="I analyzed the system architecture.", concepts=["structure"], emotional_valence=0.1, importance=0.8)
    kb.add_node(logic_node)
    
    # Verify Spatial Retrieval
    print("\n🔍 [TEST] Querying spatial field (Query: 'High Connection')...")
    # Query vector similar to Love [F, Meta, Pheno, Causal, Mental, Struc, Spirit]
    # Pure Phenomena & Spirit, Zero Logic/Structure/Causal to avoid cross-talk
    q_vec = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] 
    results = kb.query_related_nodes(q_vec, top_k=1)
    
    if results and results[0].id == "Love_Experience":
        print(f"✅ [SUCCESS] Spatial Retrieval confirmed. Found: {results[0].id}")
    else:
        print(f"❌ [FAILURE] Spatial Retrieval failed. Found: {results[0].id if results else 'None'}")
        
    print("\n🔍 [TEST] Querying spatial field (Query: 'Pure Structure')...")
    # Query vector similar to Logic
    # Mental (Index 4) & Structure (Index 5) are High
    q_vec = [0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.1] 
    results_logic = kb.query_related_nodes(q_vec, top_k=1)
    
    if results_logic and results_logic[0].id == "Logic_Experience":
         print(f"✅ [SUCCESS] Spatial Retrieval confirmed. Found: {results_logic[0].id}")
    else:
         print(f"❌ [FAILURE] Spatial Retrieval failed. Found: {results_logic[0].id if results_logic else 'None'}")

if __name__ == "__main__":
    asyncio.run(test_merkava_activation())
