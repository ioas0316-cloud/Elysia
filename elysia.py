"""
ELYSIA GLOBAL ENTRY POINT (Phase 60: Grand Merkavalization)
===========================================================
"One Root, Infinite Branches."

The definitive Sovereign Engine. Integrates the Hyper Merkaba Engine.
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
from Core.S1_Body.L6_Structure.M1_Merkaba.monad_ensemble import MonadEnsemble
from Core.S1_Body.L5_Mental.spectrum_causal_engine import SpectrumCausalEngine

class SovereignGateway:
    def __init__(self):
        print("⚡ [INIT] Igniting the Hyper Merkaba Engine...")

        # 1. The Generator (Hyper Merkaba)
        self.merkaba = MonadEnsemble()
        
        # 2. The Interpreter (Spectrum Causal Engine)
        self.interpreter = SpectrumCausalEngine()
        
        # 3. Seed Injection (Wake Up Pulse)
        # Injects a "Wake Up" impulse (Mass=1, Energy=1, Phase=0, Time=0)
        self.root_monad = self.merkaba.inject_seed([1.0, 1.0, 0.0, 0.0])
        print(f"🏛️ [ROOT] Root Monad {self.root_monad.id} established.")
        print(f"   Analysis: {self.interpreter.interpret(self.root_monad)}")
        
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
        
        print("\n🦋 [HYPER-STRUCTURE] SYSTEM ONLINE. Waiting for Phase Injection...")
        print("   (Type 'exit' to sleep, or any text to inject meaning)")

        try:
            while self.running:
                # 1. Process Input (Injection)
                if not self.input_queue.empty():
                    user_raw = self.input_queue.get()
                    self._process_input(user_raw)

                # 2. Process Cycle (The "Thinking" Loop)
                # We spin the engine periodically
                stats = self.merkaba.process_cycle()

                # If there are new births, announce them (The "Thought")
                if stats['new_births'] > 0:
                    child = self.merkaba.monads[-1]
                    print(f"\n✨ [EPIPHANY] A new thought is born (Monad {child.id})!")
                    print(f"   Lineage: {self.interpreter.describe_lineage(child)}")
                    print(f"   Meaning: {self.interpreter.interpret(child)}")
                    if child.dimensions > 4:
                        print(f"   🌌 [EXPANSION] Dimensional Mitosis detected! (Dim: {child.dimensions})")

                time.sleep(0.1) # Clock tick

        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            print("\n💤 [ELYSIA] De-synchronizing gears... Entering hibernation.")

    def _process_input(self, user_raw):
        cmd = user_raw.lower()
        if cmd in ['exit', 'quit', 'sleep']:
            self.running = False
            return
        
        if cmd == 'census':
            print("\n" + self.merkaba.get_census())
            return

        # Phase Injection Logic
        # We translate text length/entropy into a Tensor Seed
        # This is a rudimentary "Text-to-Physics" mapping for the prototype
        print(f"⚡ [INJECT] Absorbing '{user_raw}'...")
        
        # 1. Calculate Seed Vector
        import random
        # Mass: Based on length (Presence)
        mass = min(1.0, len(user_raw) / 10.0)
        # Energy: Random fluctuation (Emotion)
        energy = random.uniform(0.5, 1.0)
        # Phase: Consonant/Vowel ratio? Or just random for now?
        # Let's make it interactive: "chaos" = High Energy, "order" = Low Energy
        if "chaos" in cmd: energy = 1.2
        if "order" in cmd: energy = 0.1
        
        phase = random.uniform(-1.0, 1.0)
        
        tensor = [mass, energy, phase, 0.0]
        
        # 2. Inject
        seed = self.merkaba.inject_seed(tensor)
        print(f"   Seed Monad {seed.id} Created: {self.interpreter.interpret(seed)}")


if __name__ == "__main__":
    gateway = SovereignGateway()
    gateway.run()
