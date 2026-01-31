import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from Core.1_Body.L1_Foundation.Foundation.Graph.torch_graph import TorchGraph
try:
    g = TorchGraph(use_cuda=True)
    target_path = "c:/Elysia/Core/World/Nature/graph_memory.pt"
    if g.load_state(target_path):
        print(f"\n--- Graph Inspection ({len(g.id_to_idx)} Nodes) ---")
        print(f"Sample Keys: {list(g.id_to_idx.keys())[:20]}")
        
        if 'tsunami' in g.id_to_idx:
            print(f"  Node Found: 'tsunami'")
            print(f"Metadata: {g.node_metadata.get('tsunami')}")
        else:
            print("  Node 'tsunami' NOT FOUND.")
    else:
        print("  Could not load state.")
except Exception as e:
    print(f"Error: {e}")
