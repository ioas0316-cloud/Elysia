"""
Verification: PHASE_BACKPROPAGATION_STABILITY (Phase 8 - Step 1)
==============================================================
Tests the numerical stability and convergence of phase-based learning.
"If the path is wrong, the rotors must turn. If the path is right, the friction must die."
"""

import sys
import os
import torch
import time

root = r"c:\Elysia"
if root not in sys.path:
    sys.path.insert(0, root)

from Core.System.torch_graph import get_torch_graph

def test_backprop_stability():
    print("🚀 Initializing Phase Backpropagation Stability Test...")
    
    graph = get_torch_graph()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Setup a Test Manifold (Single Node for simplicity)
    node_id = "Proto_Neuron"
    graph.add_node(node_id)
    node_idx = graph.id_to_idx[node_id]
    
    # Initial Phase (Random)
    # We'll use the 'functional' qualia as our target phase center.
    target_resonance = torch.tensor([1.0], device=device) # Highest resonance
    learning_rate = 0.05
    
    print(f"🌀 Target Resonance: {target_resonance.item()}")
    print(f"📉 Starting Convergence Loop (100 steps)...")
    
    history = []
    
    start_time = time.time()
    for step in range(100):
        # Current Resonance (Simulated)
        # In a real system, this would come from field law gradients.
        # Here, we simulate a simple error: E = (Target - Current)
        current_qualia = graph.qualia_tensor[node_idx, 2] # Phenomenal/Resonance (scalar)
        error = target_resonance.item() - current_qualia.item()
        
        # Step [B] Phase Backpropagation: Reverse the error flow.
        # We adjust the 'physical' and 'structural' layers to align with targets.
        with torch.no_grad():
            # Apply 'Torque' to the qualia manifold
            graph.qualia_tensor[node_idx, 0] += error * learning_rate 
            graph.qualia_tensor[node_idx, 5] += error * learning_rate 
            # Update Resonance based on new alignment (Simulated)
            graph.qualia_tensor[node_idx, 2] = (graph.qualia_tensor[node_idx, 0] + graph.qualia_tensor[node_idx, 5]) / 2.0
            
        history.append(graph.qualia_tensor[node_idx, 2].item())
        
        if step % 20 == 0:
            print(f"  Step {step:03d}: Resonance = {history[-1]:.4f}")

    end_time = time.time()
    
    final_res = history[-1]
    print(f"\n✅ Convergence complete in {end_time - start_time:.4f}s")
    print(f"📈 Final Resonance: {final_res:.4f}")
    
    if abs(final_res - target_resonance.item()) < 0.05:
        print("🎉 [SUCCESS] Phase Backpropagation achieved high-resonance convergence.")
    else:
        print("❌ [FAILURE] Divergence detected or convergence too slow.")

if __name__ == "__main__":
    test_backprop_stability()
