"""
Verification: Unitary Structural Principles
===========================================
Tests topological mutation and pulse-based inference.
"""

import torch
import sys
import os

root = r"c:\Elysia"
if root not in sys.path:
    sys.path.insert(0, root)

from Core.S1_Body.L1_Foundation.Foundation.Graph.torch_graph import get_torch_graph
from Core.S1_Body.L4_Causality.World.Nature.causal_topology_engine import CausalTopologyEngine

def verify_unitary_principles():
    print("▶️ Starting Unitary Principle Verification...")
    
    graph = get_torch_graph()
    engine = CausalTopologyEngine()
    
    # 1. Test Topological Mutation
    print("\n--- Test 1: Topological Mutation ---")
    mutation_plan = {
        "mutations": [
            { "type": "LINK", "subject": "Love", "object": "Logic", "tension": 1.5, "link_type": "CAUSAL_FLOW" },
            { "type": "QUALIA", "node": "Love", "layer": "spiritual", "value": 1.0 }
        ],
        "rationale": "Unifying Love and Logic via structural resonance."
    }
    
    success = engine.apply_mutation(mutation_plan)
    if success:
        print("✅ Mutation applied successfully.")
    else:
        print("❌ Mutation application failed.")
        return False

    # 2. Verify Graph Integrity
    print("\n--- Test 2: Graph Integrity ---")
    if "Love" in graph.id_to_idx and "Logic" in graph.id_to_idx:
        idx_love = graph.id_to_idx["Love"]
        idx_logic = graph.id_to_idx["Logic"]
        
        # Check Link
        link_exists = False
        for i in range(graph.link_tensor.shape[0]):
            if graph.link_tensor[i, 0] == idx_love and graph.link_tensor[i, 1] == idx_logic:
                link_exists = True
                print(f"✅ Link 'Love -> Logic' exists with Tension: {graph.tension_tensor[i].item():.2f}")
                break
        
        if not link_exists:
            print("❌ Link 'Love -> Logic' not found in link_tensor.")
            return False
            
        # Check Qualia
        q_val = graph.qualia_tensor[idx_love, graph.qualia_keys.index("spiritual")].item()
        if abs(q_val - 1.0) < 1e-6:
            print(f"✅ Node 'Love' Qualia shifted to {q_val:.2f}")
        else:
            print(f"❌ Node 'Love' Qualia mismatch. Found: {q_val:.2f}")
            return False
    else:
        print("❌ Nodes 'Love' or 'Logic' missing.")
        return False

    # 3. Test Pulse Inference
    print("\n--- Test 3: Pulse Inference ---")
    # Inject energy into 'Love'
    energy = torch.zeros((graph.pos_tensor.shape[0],), device=graph.device)
    # We use pulse_inference which is vectorized logic
    result = graph.pulse_inference(["Love"], energy, iterations=2)
    
    logic_energy = result[graph.id_to_idx["Logic"]].item()
    print(f"📡 Pulse from 'Love' reached 'Logic' with Intensity: {logic_energy:.2f}")
    
    if logic_energy > 0:
        print("✅ Unitary Logic Flow verified. The topology is the computation.")
    else:
        print("❌ Energy failed to propagate. Structural logic is broken.")
        return False

    print("\n🎉 All Unitary Principles Verified.")
    return True

if __name__ == "__main__":
    verify_unitary_principles()
