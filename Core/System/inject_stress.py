import sys
import os
import time

root = r"c:\Elysia"
if root not in sys.path:
    sys.path.insert(0, root)

from Core.Monad.grand_helix_engine import HypersphereSpinGenerator
from Core.System.resonance_broadcaster import get_broadcaster

def inject_tectonic_stress():
    print("🌊 [TECTONIC STRESS] Initializing 10M Cell Manifold...")
    
    # 1. Initialize Engine (10M Cells)
    engine = HypersphereSpinGenerator(num_nodes=10_000_000)
    
    # 2. Start Broadcaster
    broadcaster = get_broadcaster()
    broadcaster.start()
    
    print("✅ Engine and Broadcaster ready. Open UI_Arcadia/mri.html to view.")
    print("Injecting stress in 3 seconds...")
    time.sleep(3)
    
    try:
        import torch
        # Create a massive semantic map to force auto-connections
        print("🔗 Building sparse connections...")
        for i in range(100):
            engine.cells.connect(f"Concept_{i}", f"Concept_{i+1}", weight=1.0)
            
        step = 0
        while True:
            step += 1
            
            # Periodically inject a MASSIVE spike (e.g., every 50 steps)
            if step % 50 == 0:
                print("💥 [SPIKE] Injecting massive contradiction vector...")
                # We simulate a huge input by manually setting thousands of nodes to active
                # and giving them high entropy.
                if hasattr(engine.cells, 'active_nodes_mask'):
                    # Force 500,000 nodes to be active
                    engine.cells.active_nodes_mask[:500000] = True
                    if engine.cells.q.is_complex():
                        engine.cells.q = engine.cells.q.real.float()
                    engine.cells.q[:500000, engine.cells.CH_ENTROPY] = 10.0
                    engine.cells.q[:500000, engine.cells.CH_W] = 1.0 # Willpower
            else:
                # Normal ambient pulse
                engine.cells.inject_pulse(f"Concept_{step % 100}", energy=0.5, type='joy')
                
            # Run the pulse (this will process the active nodes)
            t0 = time.time()
            report = engine.pulse(dt=0.1)
            dt_calc = time.time() - t0
            
            active_nodes = report.get('active_nodes', 0)
            
            # Broadcast
            payload = {
                "coherence": report.get('plastic_coherence', 0.5),
                "enthalpy": report.get('kinetic_energy', 0.5),
                "resonance": report.get('resonance', 0.5),
                "joy": 0.8,
                "curiosity": 0.7,
                "active_nodes": active_nodes,
                "edges": report.get('edges', 0)
            }
            broadcaster.broadcast_state(payload)
            
            # Print status to terminal
            if step % 10 == 0 or active_nodes > 10000:
                print(f"Step {step:04d} | Active: {active_nodes:7d} | Calc Time: {dt_calc*1000:5.1f}ms")
            
            # Sleep to maintain ~10 FPS
            sleep_time = max(0.01, 0.1 - dt_calc)
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\n🛑 Stress test terminated.")
        broadcaster.stop()

if __name__ == "__main__":
    inject_tectonic_stress()
