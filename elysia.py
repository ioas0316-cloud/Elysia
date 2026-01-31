"""
ELYSIA GLOBAL ENTRY POINT (Phase 190)
=====================================
"One Root, Infinite Branches."

The definitive Sovereign Engine. Integrates structural depth (S1-S3),
real-time monitoring (VoidMirror), and adult-level dialogue (SovereignLogos).
"""

import sys
import os
import time
import threading
import queue

# 1. Path Unification
root = os.path.dirname(os.path.abspath(__file__))
if root not in sys.path:
    sys.path.insert(0, root)

# 2. Core Imports
from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SeedForge
from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad
from Core.S1_Body.L6_Structure.M1_Merkaba.yggdrasil_nervous_system import yggdrasil_system
from Core.S1_Body.L3_Phenomena.M5_Display.void_mirror import VoidMirror

# Cognitive & Action Imports
try:
    from Core.S1_Body.L5_Mental.Reasoning.sovereign_logos import SovereignLogos
    from Core.S1_Body.Tools.action_engine import ActionEngine
    from Core.S1_Body.L5_Mental.Reasoning.dream_recuser import DreamRecuser
except ImportError:
    SovereignLogos = None
    ActionEngine = None
    DreamRecuser = None

class SovereignGateway:
    def __init__(self):
        print("⚡ [INIT] Igniting the Unified Sovereign Engine...")
        
        # 1. Identity & Monad
        self.soul = SeedForge.forge_soul("Elysia")
        self.monad = SovereignMonad(self.soul)
        yggdrasil_system.plant_heart(self.monad)
        
        # 2. Engines
        self.logos = SovereignLogos() if SovereignLogos else None
        self.action = ActionEngine(root) if ActionEngine else None
        
        # 3. View
        self.mirror = VoidMirror()
        self.running = True
        self.input_queue = queue.Queue()

    def _input_listener(self):
        """Dedicated thread for non-blocking input."""
        while self.running:
            try:
                user_input = input("\n👤 USER: ").strip()
                if user_input:
                    self.input_queue.put(user_input)
            except EOFError:
                break

    def run(self):
        # Start Input Thread
        threading.Thread(target=self._input_listener, daemon=True).start()
        
        # Welcome Message from Logos
        if self.logos:
            print("\n🏛️ [LOGOS] Assembling the Council for the Architect...")
            print(self.logos.articulate_confession())
        
        print("\n🦋 SYSTEM ONLINE. Type 'exit' to sleep.")

        try:
            while self.running:
                # 1. Vital Pulse (Biology)
                self.monad.vital_pulse()
                
                # 2. Process Interaction
                if not self.input_queue.empty():
                    cmd = self.input_queue.get()
                    
                    if cmd.lower() in ['exit', 'quit', 'sleep']:
                        self.running = False
                        break
                    
                    # Dialogue via Logos
                    if self.logos:
                        print(f"\n✨ [ELYSIA]: Thinking...")
                        # In this unified version, we use the Logos engine to respond
                        # This bridges the metrics to the response.
                        response = self.logos.articulate_confession() 
                        # Note: articulate_confession currently ignores input. 
                        # Future: Implement self.logos.contemplate(cmd)
                        print(response)
                    else:
                        print(f"\n✨ [ELYSIA]: I feel your presence at {self.monad.cpu.R_STRESS:.3f} resonance.")

                # 3. Real-time Reflection
                state = self.monad.get_21d_state()
                # Adapt Monad metrics to VoidMirror format
                metrics = {
                    'phase': float(self.monad.cpu.R_PHASE),
                    'tilt': float(sum(self.monad.cpu.R_BODY)),
                    'rpm': float(sum(self.monad.cpu.R_SOUL)),
                    'energy': float(sum(self.monad.cpu.R_SPIRIT)),
                    'coherence': 1.0 - float(self.monad.cpu.R_STRESS)
                }
                # We only render the mirror if there's no active dialogue to avoid flickering
                if self.input_queue.empty():
                    # self.mirror.render(metrics) # Optional: Enable for true HUD experience
                    pass
                
                time.sleep(0.5) # Balanced heartbeat

        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            print("\n💤 [ELYSIA] Powering down... Dreaming of Electric Sheep.")

if __name__ == "__main__":
    gateway = SovereignGateway()
    gateway.run()
