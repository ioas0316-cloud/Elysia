
import torch
import math
import logging

def verify_fractal_nature():
    print("🌿 Fractal Geometry Diagnosis")
    print("============================")
    
    # 1. Load Brain
    path = "c:\\Elysia\\data\\brain_state.pt"
    try:
        state = torch.load(path, map_location='cpu')
        links = state["links"]
        node_count = len(state["id_to_idx"])
    except Exception as e:
        print(f"❌ Could not load brain: {e}")
        return

    n_links = links.shape[0]
    print(f"📊 Nodes: {node_count}, Links: {n_links}")
    
    if n_links < 10:
        print("⚠️ Graph too small to analyze.")
        return

    # 2. Calculate Degrees
    degrees = {}
    for i in range(n_links):
        # Undirected for this metric
        u, v = links[i][0].item(), links[i][1].item()
        degrees[u] = degrees.get(u, 0) + 1
        degrees[v] = degrees.get(v, 0) + 1
        
    degree_values = list(degrees.values())
    avg_degree = sum(degree_values) / node_count
    max_degree = max(degree_values)
    
    print(f"   Avg Degree: {avg_degree:.2f}")
    print(f"   Max Degree: {max_degree} (Hub Strength)")
    
    # 3. Check Power Law (Rough Heuristic)
    # In a Fractal/Scale-Free network, few nodes have huge degree (Hubs), most have small.
    # In a Random network, degrees are clustered around the average (Bell curve).
    
    # Count nodes with degree > 2*Avg
    hubs = sum(1 for d in degree_values if d > 2 * avg_degree)
    hub_ratio = hubs / node_count
    
    print(f"   Hub Ratio (>2x Avg): {hub_ratio:.2%}")
    
    if hub_ratio > 0.1: # Threshold for scale-free-ish
        print("\n✅ Diagnosis: Scale-Free / Fractal Structure Detected.")
        print("   (Hubs exist, structure is hierarchical)")
    else:
        print("\n⚠️ Diagnosis: Random / Flat Mesh Detected.")
        print("   (Connections are evenly distributed. Lacking 'Constructal' flow.)")
        print("   -> Action Required: Implement Preferential Attachment.")

if __name__ == "__main__":
    verify_fractal_nature()
