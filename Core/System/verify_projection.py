"""
Verification: VOLUMETRIC_PROJECTION (Phase 4)
=============================================
Tests the mapping of 7D Qualia into 4D Cubic Manifold (XYZ + Resonance).
"""

import sys
import os
import torch

root = r"c:\Elysia"
if root not in sys.path:
    sys.path.insert(0, root)

from Core.System.torch_graph import get_torch_graph
from Core.System.volumetric_projector import get_volumetric_projector

def verify_volumetric_projection():
    print("🚀 Starting VOLUMETRIC_PROJECTION Verification...")
    
    graph = get_torch_graph()
    projector = get_volumetric_projector()
    
    # 1. Create a Test Node with specific Qualia
    node_id = "Prism_Test"
    # Qualia Mapping: 0:phys, 1:func, 2:phen, 3:caus, 4:ment, 5:stru, 6:spir
    # We want: X=10 (0+1), Y=20 (3+5), Z=30 (4+6), Res=0.5 (2)
    qualia_data = {
        "physical": 4.0, "functional": 6.0, 
        "causal": 8.0, "structural": 12.0, 
        "mental": 14.0, "spiritual": 16.0,
        "phenomenal": 0.5
    }
    
    graph.add_node(node_id, metadata={"qualia": qualia_data})
    
    # 2. Project
    projections = projector.project_current_state()
    
    # 3. Verify
    test_node = next((p for p in projections if p["id"] == node_id), None)
    
    if test_node:
        print(f"\n--- Projection Results for '{node_id}' ---")
        print(f"  X (Phys+Func) : {test_node['x']} (Expected: 10.0)")
        print(f"  Y (Caus+Stru) : {test_node['y']} (Expected: 20.0)")
        print(f"  Z (Ment+Spir) : {test_node['z']} (Expected: 30.0)")
        print(f"  Resonance     : {test_node['resonance']} (Expected: 0.5)")
        
        if abs(test_node['x'] - 10.0) < 0.1 and abs(test_node['y'] - 20.0) < 0.1:
            print("\n✅ SUCCESS: Volumetric Projection logic is accurate.")
        else:
            print("\n❌ FAILURE: Projection values mismatch.")
    else:
        print(f"\n❌ FAILURE: Node '{node_id}' not found in projection.")

if __name__ == "__main__":
    verify_volumetric_projection()
