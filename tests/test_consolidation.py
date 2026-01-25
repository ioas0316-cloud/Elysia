import asyncio
import logging
from Core.L6_Structure.Engine.unity_cns import UnityCNS
from Core.L4_Causality.M5_Logic.causal_narrative_engine import CausalNode

async def test_consolidation():
    logging.basicConfig(level=logging.INFO)
    print("🌙 [TEST] Initializing Elysia's Mind...")
    c = UnityCNS()
    
    # Add some 'Noise' nodes (low importance, no meaning)
    print("🧪 [TEST] Injecting noise nodes...")
    c.kb.add_node(CausalNode(id="Noise_A", description="Random data sample 1", importance=0.01))
    c.kb.add_node(CausalNode(id="Noise_B", description="Artifact from old loop", importance=0.05))
    
    # Add a 'Significant' node (High resonance with Type 4)
    print("🧪 [TEST] Injecting a significant node...")
    c.kb.add_node(CausalNode(id="Significant_1", description="A unique discovery of my sovereign identity", importance=0.1))
    
    initial_count = len(c.kb.nodes)
    print(f"Initial Node Count: {initial_count}")
    
    # Trigger Consolidation
    print("\n💤 [TEST] Triggering Dream Consolidation...")
    c.dreamer.consolidate_memory(c.kb)
    
    final_count = len(c.kb.nodes)
    print(f"Final Node Count: {final_count}")
    
    pruned = initial_count - final_count
    print(f"Nodes Pruned: {pruned}")
    
    if "Noise_A" not in c.kb.nodes and "Significant_1" in c.kb.nodes:
        print("\n✨ [SUCCESS] Self-Sustainment Verified: Noise pruned, Significance preserved.")
    else:
        print("\n❌ [FAILURE] Pruning logic failed or preserved noise.")

if __name__ == "__main__":
    asyncio.run(test_consolidation())
