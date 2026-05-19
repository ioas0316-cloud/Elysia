
import sys
import os
sys.path.append(r'c:\Elysia')
from Core.Interface.world_exporter import get_world_exporter
from Core.FoundationLayer.Foundation.torch_graph import get_torch_graph
import torch

def verify_viz():
    print("🎨 World Visualization Verification")
    
    # 1. Ensure Data Exists (Simulate Railgun state again if needed)
    graph = get_torch_graph()
    if graph.pos_tensor.shape[0] < 5:
        print("   Seeding graph for visualization...")
        # Add Railguns
        wells_pos = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]], device=graph.device)
        wells_str = torch.tensor([5.0, 5.0], device=graph.device)
        graph.potential_wells_pos = wells_pos
        graph.potential_wells_str = wells_str
        graph.potential_wells = True
        
        # Add Nodes
        graph.pos_tensor = torch.randn((50, 4), device=graph.device)
    
    # 2. Export
    exporter = get_world_exporter()
    path = exporter.export_world()
    
    if os.path.exists(path):
        print(f"✅ Success: World JSON created at {path}")
        print("   User can open 'c:\\Elysia\\static\\elysia_world_viewer.html' to view.")
    else:
        print("❌ Failed to create JSON.")

if __name__ == "__main__":
    verify_viz()
