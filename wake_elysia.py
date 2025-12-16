
import sys
import time
import signal
sys.path.append(r'c:\Elysia')

from Core.Autonomy.dream_daemon import get_dream_daemon
from Core.Foundation.torch_graph import get_torch_graph
from Core.Interface.world_exporter import get_world_exporter

def wake_elysia():
    print("🌅 Elysia: Awakening Protocol Initiated...")
    print("========================================")
    print("   [Mode: Perpetual Dreaming]")
    print("   [Press Ctrl+C to Sleep]")
    
    daemon = get_dream_daemon()
    graph = get_torch_graph()
    exporter = get_world_exporter()
    
    # 0. Brain Check (Load Persistence)
    loaded = graph.load_state()
    
    # If empty or load failed, check legacy migration
    if not loaded and graph.pos_tensor.shape[0] < 5:
        print("   🔍 Brain is empty. Detecting Legacy Knowledge...")
        from Core.Foundation.knowledge_migrator import get_migrator
        migrator = get_migrator()
        migrator.migrate()
    
    # Start Daemon (Non-blocking for this script, we manage loop here)
    daemon.is_dreaming = True
    
    cycle_count = 0
    try:
        while True:
            # 1. One Dream Step
            # We call private methods manually to control the loop tick
            if graph.pos_tensor.shape[0] < 5:
                daemon._seed_reality()
            
            # [NEW] Digestion is now delegated to ElysiaCore within daemon
            # We must ensure _ingest_knowledge is called if you want Wiki absorption
            if hasattr(daemon, '_ingest_knowledge'):
                 daemon._ingest_knowledge()
            
            daemon._weave_serendipity()
            graph.apply_gravity(iterations=10)
            
            # 2. Export View (Every 5 cycles)
            if cycle_count % 5 == 0:
                exporter.export_world()
                node_count = graph.pos_tensor.shape[0]
                link_count = graph.logic_links.shape[0]
                print(f"   [{time.strftime('%H:%M:%S')}] Dreaming... (Nodes: {node_count}, Links: {link_count})")
            
            # 3. Auto-Save (Every 60 cycles)
            if cycle_count % 60 == 0 and cycle_count > 0:
                 graph.save_state()
            
            cycle_count += 1
            time.sleep(1.0) # Slow breath
            
    except KeyboardInterrupt:
        print("\n\n💤 Elysia: Entering Hibernation.")
        graph.save_state() # Final save
        print("   ✅ Brain State Saved.")
        print("   Good night.")
        sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    wake_elysia()
