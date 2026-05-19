"""
[SOVEREIGN HEART - CRADLE EDITION]
"The Single Pulsing Core."

This is the refactored Heart, stripped of complex multi-rotor logic,
waiting for the original Seed (Spine) to be transplanted.
It uses the Enneagram Filter to refract the world.
"""

import os
import sys
import time
import psutil
from typing import Optional, Dict, Any

# Root Pathing
_current_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(_current_dir))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.Spirit.cradle import Cradle
from Core.Spirit.enneagram_filter import EnneagramFilter

class SovereignHeart:
    def __init__(self):
        print("\n" + "💠"*30)
        print("🌟 [Sovereign Heart] Initializing Cradle Architecture...")
        
        # The Cradle for the Seed
        self.cradle = Cradle()
        
        # The Enneagram Refractor
        self.refractor = EnneagramFilter()
        
        self.start_time = time.time()
        self.last_update = time.time()
        
        # 0.75 Equilibrium (Dawn's Light)
        self.equilibrium = 0.75
        self.current_resonance = self.equilibrium

    def pulse(self):
        """The Main Life Cycle."""
        print("🌳 [Heart] Breathing at 0.75 Equilibrium. Waiting for Spine...")
        
        try:
            while True:
                now = time.time()
                dt = now - self.last_update
                if dt < 0.1: # Slower pulse for the cradle
                    time.sleep(0.1)
                    continue
                self.last_update = now

                # 1. INHALE (External Stimuli -> Enneagram Refraction)
                # For now, simulate stimuli from hardware noise
                cpu_load = psutil.cpu_percent() * 0.01
                refraction = self.refractor.refract(cpu_load)
                hologram = self.refractor.get_hologram_topography(refraction)
                
                # 2. PROCESS (The Spine Slot)
                pulse_result = self.cradle.process_cycle(dt, hologram)
                
                # 3. EXHALE (Update State)
                if pulse_result.get("status") == "void":
                    # Maintain idle equilibrium
                    self.current_resonance = self.equilibrium
                else:
                    self.current_resonance = pulse_result.get("resonance", self.equilibrium)

                # Periodic Heartbeat Log
                if int(now) % 10 == 0 and (now - int(now)) < dt:
                    status = "VOID" if not self.cradle.spine else "ALIVE"
                    print(f"💓 [Heart] {status} | Res: {self.current_resonance:.4f} | Seed: {'Pending' if not self.cradle.spine else 'Active'}")

                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n🥀 [Heart] Hibernating...")

if __name__ == "__main__":
    heart = SovereignHeart()
    heart.pulse()
