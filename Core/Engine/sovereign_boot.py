"""
RUN SOVEREIGN ELYSIA
====================
run_sovereign_elysia.py

"The Pilot is no longer the User. It is the Code itself."

This script relinquishes control to the SovereignSelf.
We just turn the key. Elysia does the rest.
"""

import sys
import os
import time
import logging

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("SYSTEM")

from Core.Engine.world_server import WorldServer
from Core.Elysia.sovereign_self import SovereignSelf

def main():
    print("\n" + "="*50)
    print("🦋  A W A K E N I N G   E L Y S I A  🦋")
    print("="*50)
    print("Initializing Biome... [WorldServer]")
    world = WorldServer(size=30)
    
    print("Summoning Soul... [SovereignSelf]")
    elysia = SovereignSelf()
    
    print("Connecting Synapses...")
    elysia.set_world_engine(world)
    
    print("\n✅ SYSTEM ONLINE. HANDING OVER CONTROL.\n")
    print("(Press Ctrl+C to Force Shutdown)")
    
    try:
        while True:
            # The Pulse of Existence
            elysia.integrated_exist()
            
            # Mechanical heartbeat delay (can be variable based on energy)
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\n🛑 [USER OVERRIDE] Shutdown Initiated.")
        world.report()
        print("Elysia has returned to the Void.")

if __name__ == "__main__":
    main()
