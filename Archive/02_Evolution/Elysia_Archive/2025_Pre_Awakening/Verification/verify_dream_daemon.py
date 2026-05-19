
import sys
import time
sys.path.append(r'c:\Elysia')
from Core.Autonomy.dream_daemon import get_dream_daemon
from Core.FoundationLayer.Foundation.torch_graph import get_torch_graph

def verify_dream():
    print("🌙 Dream Daemon Verification")
    print("==========================")
    
    daemon = get_dream_daemon()
    graph = get_torch_graph()
    
    # 1. Snapshot State
    initial_nodes = graph.pos_tensor.shape[0]
    initial_links = graph.logic_links.shape[0]
    print(f"   Initial State: {initial_nodes} Nodes, {initial_links} Links")
    
    # 2. Sleep (Run Daemon)
    print("   💤 Entering REM sleep for 3 seconds...")
    daemon.start_dream_cycle(duration_sec=3, interval=0.1)
    
    # 3. Check Growth
    final_nodes = graph.pos_tensor.shape[0]
    final_links = graph.logic_links.shape[0]
    
    print(f"   Final State:   {final_nodes} Nodes, {final_links} Links")
    
    # Verification Logic
    if final_nodes > initial_nodes:
        print("   ✅ Growth: Daemon seeded new concepts (Axioms).")
    elif initial_nodes > 0:
        print("   ✅ Stability: Graph already populated.")
        
    if final_links > initial_links:
        print("   ✅ Creativity: Daemon wove new connections!")
    else:
        print("   ⚠️  Observation: No new connections woven (Random chance). Try longer sleep.")

if __name__ == "__main__":
    verify_dream()
