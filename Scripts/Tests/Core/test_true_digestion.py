import os
import sys

# Ensure Core is in path
sys.path.append(os.getcwd())

import torch
from Core.Elysia.sovereign_self import SovereignSelf

def test_digestion():
    print("🧠 Initializing Elysia (SovereignSelf)...")
    elysia = EmergentSelf()
    
    # Check Initial Soul Size
    initial_size = len(elysia.graph.id_to_idx)
    print(f"📉 Initial Soul Nodes: {initial_size}")
    
    # Trigger Digestion
    prompt = "Explain why the Past influences the Future."
    print(f"\n🔮 Prompt: {prompt}")
    
    # This method now triggers 'digest' internally
    response = elysia.manifest_intent(prompt)
    
    print(f"🗣️ Response: {response}")
    
    # Check Final Soul Size
    final_size = len(elysia.graph.id_to_idx)
    print(f"📈 Final Soul Nodes: {final_size}")
    
    gained = final_size - initial_size
    
    if gained > 0:
        print(f"\n✅ SUCCESS: Digestion Verified! Elysia gained {gained} new concept-vectors.")
        
        # Inspect a digested node
        new_nodes = list(elysia.graph.id_to_idx.keys())[-gained:]
        print(f"   Sample Digested Node ID: {new_nodes[0]}")
        vec = elysia.graph.get_node_vector(new_nodes[0])
        print(f"   Vector Energy (Norm): {torch.norm(vec):.4f}")
    else:
        print("\n❌ FAILURE: No new nodes added. Digestion logic failed.")

if __name__ == "__main__":
    test_digestion()
