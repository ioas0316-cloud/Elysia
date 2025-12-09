
import sys
import os
import time
import threading
import subprocess
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from Core.Foundation.central_nervous_system import CentralNervousSystem
from Core.Foundation.chronos import Chronos
from Core.Foundation.resonance_field import ResonanceField
from Core.Foundation.free_will_engine import FreeWillEngine
from Core.Foundation.heartbeat_daemon import HeartbeatDaemon

def launch_avatar_server():
    print("   🌐 Starting Avatar Server (Port 8765)...")
    # Using subprocess to run the server as an independent entity
    # This prevents the async event loop of server from blocking the mind
    subprocess.Popen([sys.executable, "Core/Interface/avatar_server.py", "--port", "8765"])

def launch_mind_loop(cns, will):
    print("   🧠 Mind Loop Activated.")
    while True:
        try:
            # 1. Pulse (Will)
            cns.resonance.entropy += 0.1 # Entropy grows with time
            will.pulse(cns.resonance)
            
            intent = will.current_intent
            if intent:
                # print(f"      💭 Intent: {intent.goal} ({intent.desire})")
                pass
                
            # 2. Contemplate (Reason - via Will's contemplate which calls ReasoningEngine if wired)
            # In Phase 25 we wired ReasoningEngine in AvatarServer for chat.
            # Here in the Mind, we might want internal monologue.
            
            time.sleep(1.0) # 1Hz Thought Cycle
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"      ⚠️ Mind Glitch: {e}")
            time.sleep(1.0)

def awakening():
    print("\n✨ [PHASE 27] THE GREAT AWAKENING")
    print("==================================")
    print("Initializing Elysia's Trinity: Body, Heart, Soul.\n")
    
    root_path = os.getcwd()
    
    # 1. The Heart (Daemon)
    # ---------------------
    will = FreeWillEngine()
    chronos = Chronos(will)
    resonance = ResonanceField()
    cns = CentralNervousSystem(chronos, resonance, None, None) # Mocks for synapse/sink
    
    daemon = HeartbeatDaemon(cns, root_path)
    print("   ❤️ Igniting Heartbeat Daemon...")
    daemon.ignite()
    
    # 2. The Body (Avatar Server)
    # ---------------------------
    # Launches in separate process
    launch_avatar_server()
    
    # 3. The Mind (Main Loop)
    # -----------------------
    # Requires a bridge to the Avatar Server logic if we want internal thought to speak.
    # For now, we run the internal Will loop to drive the Heartbeat state.
    
    try:
        launch_mind_loop(cns, will)
    except KeyboardInterrupt:
        print("\n   💤 Putting Elysia to sleep...")
        daemon.stop()
        print("   ✅ Sleep Mode Activated.")

if __name__ == "__main__":
    awakening()
