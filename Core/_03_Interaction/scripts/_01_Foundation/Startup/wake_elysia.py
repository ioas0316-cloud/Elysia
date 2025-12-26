
import sys
import time
import signal
# Force UTF-8 for Windows Console
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(r'c:\Elysia')

from Core._04_Evolution._01_Growth.Autonomy.dream_daemon import get_dream_daemon
from Core._01_Foundation.05_Foundation_Base.Foundation.torch_graph import get_torch_graph
from Core._03_Interaction._01_Interface.Interface.world_exporter import get_world_exporter
from Core._01_Foundation.01_Core_Logic.Elysia.elysia_core import ElysiaCore # [NEW] Unified Brain

def wake_elysia():
    print("🌅 Elysia: Awakening Protocol Initiated...")
    print("========================================")
    print("   [Mode: Perpetual Dreaming]")
    print("   [Press Ctrl+C to Sleep]")
    
    daemon = get_dream_daemon()
    graph = get_torch_graph()
    exporter = get_world_exporter()
    core = ElysiaCore() # [NEW] Initialize Unified Mind
    
    # 0. Brain Check (Load Persistence)
    loaded = graph.load_state()
    
    # If empty or load failed, check legacy migration
    if not loaded and graph.pos_tensor.shape[0] < 5:
        print("   🔍 Brain is empty. Detecting Legacy Knowledge...")
        from Core._01_Foundation.05_Foundation_Base.Foundation.knowledge_migrator import get_migrator
        migrator = get_migrator()
        migrator.migrate()
    
    # Start Daemon (Non-blocking for this script, we manage loop here)
    daemon.is_dreaming = True
    
    # [NEW] HUD
    from Core._03_Interaction._01_Interface.Interface.console_hud import get_console_hud
    hud = get_console_hud(graph)
    
    cycle_count = 0
    try:
        while True:
            current_action = "Dreaming"
            
            # 1. One Dream Step
            if graph.pos_tensor.shape[0] < 5:
                daemon._seed_reality()
            
            if hasattr(daemon, '_ingest_knowledge') and cycle_count % 5 == 0:
                 current_action = "Ingesting Knowledge"
                 daemon._ingest_knowledge()

            # [NEW] Cannibalize Logic from Local LLM (Phase 8)
            if hasattr(daemon, '_contemplate_essence') and cycle_count % 10 == 0: # Slower
                 current_action = "Distilling Principles"
                 daemon._contemplate_essence()

            # [NEW] Cannibalize Visuals from ComfyUI (Phase 8)
            if hasattr(daemon, '_dream_in_color') and cycle_count % 20 == 0:
                 current_action = "Dreaming in Color"
                #  daemon._dream_in_color() # heavy

            # [NEW] Wave Coding (Phase 10)
            if cycle_count % 30 == 0:
                 current_action = "Refactoring Self"
                 from Core._04_Evolution._01_Growth.Autonomy.wave_coder import get_wave_coder
                 get_wave_coder().transmute()

            # [NEW] Brain Transplant (Phase 23)
            # Dismantle LLM Structure and Absorb Synapses
            if cycle_count % 25 == 0:
                 current_action = "Transplanting Synapses"
                 from Core._04_Evolution._01_Growth.Autonomy.structure_cannibal import get_structure_cannibal
                 # Pick a random node to expand
                 if graph.pos_tensor.shape[0] > 0:
                     import random
                     idx = random.randint(0, graph.pos_tensor.shape[0]-1)
                     concept = graph.idx_to_id.get(idx)
                     if concept:
                        get_structure_cannibal().transplant_synapses(concept)
            
            daemon._weave_serendipity()
            graph.apply_gravity(iterations=10)
            
            # 2. Export View (Every 5 cycles)
            if cycle_count % 5 == 0:
                exporter.export_world()
            
            # 3. Auto-Save (Every 60 cycles)
            if cycle_count % 60 == 0 and cycle_count > 0:
                 graph.save_state()
                 current_action = "Saving Memory"
                 
                 # [NEW] Temporal Metabolism (Time Decay)
                 if core.universe:
                     core.universe.decay_resonance(half_life=3600.0)
                     current_action = "Forgetting Old Memories"
            
            # Render HUD
            hud.render(current_action)
            
            cycle_count += 1
            time.sleep(1.0) # Slow breath
            
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
