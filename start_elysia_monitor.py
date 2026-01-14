"""
start_elysia_monitor.py
=======================
Launches the Full Biosphere Stack.

1. Starts Pulse Server (Port 8000)
2. Opens Browser
3. Starts Genesis Lab Loop (Hardware -> JSON)
"""

import sys
import os
import threading
import time
import webbrowser

# Ensure Core is visible
sys.path.append(os.getcwd())

from Core.Engine.Genesis.genesis_lab import GenesisLab
from Core.Engine.Genesis.biosphere_adapter import BiosphereAdapter, law_thermal_throttling, law_memory_digestion
from Core.Engine.Genesis.pulse_server import start_server

def run_lab_loop():
    print("🧪 [Genesis Lab] Initializing...")
    lab = GenesisLab("Elysia Alive")
    adapter = BiosphereAdapter(lab)
    
    # Decree Laws
    lab.decree_law("Thermal Homeostasis", law_thermal_throttling, rpm=60.0)
    lab.decree_law("Memory Garbage Collector", law_memory_digestion, rpm=120.0)
    lab.decree_law("Quantum Idle", lambda c,d,i: None, rpm=10.0) # Just for visuals
    
    print("❤️ [Genesis Lab] Heartbeat Started.")
    while True:
        adapter.inhale() # Reads Hardware & Writes JSON
        lab.run_simulation(ticks=1)
        time.sleep(0.5)

if __name__ == "__main__":
    # 1. Start Server Thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # 2. Open Browser
    print("🚀 [Launcher] Opening Dashboard...")
    time.sleep(1) # Wait for server
    webbrowser.open("http://localhost:8000/frontend/dashboard.html")
    
    # 3. Run Lab (Main Thread)
    try:
        run_lab_loop()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down.")
