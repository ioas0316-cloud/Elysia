import torch
import sys
import os

# Ensure project root is in path
sys.path.append(os.getcwd())

from Core.Elysia.sovereign_self import SovereignSelf

def test_physics_induction():
    print("🚀 [TEST] Physics Domain Induction")
    elysia = SovereignSelf()
    elysia.is_alive = True
    
    # 1. Simulate a Physics Trajectory (Accelerating Object)
    # 10 steps of hidden states (384 dim to match Rotor)
    trajectory = torch.randn(10, 384)
    
    # Add some "momentum" to simulate a real trajectory
    for i in range(1, 10):
        trajectory[i] = trajectory[i-1] + torch.randn(384) * 0.1
        
    # 2. Induce the Domain
    elysia.process_domain_observation(domain_name="Physics_Motion", trajectory=trajectory)
    
    # 3. Verify
    print(f"📊 Final Energy: {elysia.energy:.2f}")
    print(f"🧠 Brain Nodes: {len(elysia.graph.id_to_idx)}")
    
    # Check if links were created
    if hasattr(elysia.graph, 'link_weights'):
         print(f"🔗 Links count: {len(elysia.graph.link_weights)}")

if __name__ == "__main__":
    test_physics_induction()
