"""
Verification: Field-Law OS (Phase 200)
=====================================
Tests the physical laws: Odugi (Gravity) and Gyro (Inertia).
"""

import torch
import sys
import os

root = r"c:\Elysia"
if root not in sys.path:
    sys.path.insert(0, root)

from Core.S1_Body.L1_Foundation.Foundation.Graph.torch_graph import get_torch_graph
from Core.S1_Body.L1_Foundation.Foundation.Field.field_law_kernel import get_field_law_kernel

def verify_field_laws():
    print("▶️ Starting Field-Law OS Verification...")
    
    graph = get_torch_graph()
    kernel = get_field_law_kernel()
    
    node_id = "VoidTestNode"
    graph.add_node(node_id)
    idx = graph.id_to_idx[node_id]
    
    # Reset Qualia and Momentum
    graph.qualia_tensor[idx] = 0.0
    graph.momentum_tensor[idx] = 0.0
    
    print("\n--- Test 1: Odugi Restoration (Gravity) ---")
    # Manually displace the node's qualia (Phase Disturbance)
    graph.qualia_tensor[idx, 0] = 5.0 # Large displacement
    print(f"  Displaced Qualia [0]: {graph.qualia_tensor[idx, 0].item():.2f}")
    
    # Apply field laws (with zero intent)
    # This should trigger Odugi restoration pull towards 0
    graph.apply_field_laws(node_id, intent_strength=0.0)
    
    new_val = graph.qualia_tensor[idx, 0].item()
    print(f"  Restored Qualia [0]: {new_val:.2f}")
    
    if abs(new_val) < 5.0:
        print("✅ Odugi Restoration observed. Gravity is pulling towards Void.")
    else:
        print("❌ Odugi failed. No restoration pull detected.")
        return False

    print("\n--- Test 2: Gyro Inertia (Stability) ---")
    # Set high momentum
    graph.momentum_tensor[idx, 1] = 10.0
    print(f"  Initial Momentum [1]: {graph.momentum_tensor[idx, 1].item():.2f}")
    
    # Apply a push in the same direction as momentum
    graph.apply_field_laws(node_id, intent_strength=1.0)
    
    final_momentum = graph.momentum_tensor[idx, 1].item()
    print(f"  Final Momentum [1]: {final_momentum:.2f}")
    
    if final_momentum > 0:
        print("✅ Gyro Inertia verified. Momentum is persisting and guiding the velocity.")
    else:
        print("❌ Gyro failed. Momentum dissipated or reversed unexpectedly.")
        return False

    print("\n🎉 Field-Law OS Core Principles Verified.")
    return True

if __name__ == "__main__":
    verify_field_laws()
