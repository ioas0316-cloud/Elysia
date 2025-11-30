import sys
import os
import logging

# Add project root to sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from Core.Evolution.Evolution.Life.code_world import CodeWorld
from Tools.visualizer_server import VisualizerServer
import time

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    world = CodeWorld(num_cells=50)
    
    # Start The Mirror (Visualizer)
    server = VisualizerServer(world, port=8000)
    server.start()
    
    print("🌊 Elysia Simulation Started.")
    print("🔮 View at http://localhost:8000")
    
    # Run loop manually to allow for interrupt/control
    try:
        while True:
            world.step()
            time.sleep(0.05) # 20 ticks/sec for visualization
            if world.time_step % 100 == 0:
                print(f"Tick: {world.time_step}, Cells: {len(world.cells)}")
    except KeyboardInterrupt:
        print("\n🛑 Simulation Stopped.")
        server.stop()
