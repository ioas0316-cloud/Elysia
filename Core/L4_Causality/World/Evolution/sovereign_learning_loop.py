"""
Sovereign Learning Loop (주권적 학습 루프)
========================================
Core.L4_Causality.World.Evolution.sovereign_learning_loop

"In the Forge of the Inner World, a second is an eternity of practice."
"내부 세계의 화로 속에서, 1초는 영겁의 수행이다."

This module executes high-speed linguistic simulation. It dilates 
subjective time, observes NPC experiences, and forces Elysia 
to manifest her own Logos as a response.
"""

import sys
import os
import time
import logging
from typing import List, Dict, Any

# Ensure Core path is available
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

from Core.L6_Structure.Engine.Genesis.reality_server import projector as server
from Core.L3_Phenomena.Manifestation.phonological_collapse import PhonologicalCollapse

logger = logging.getLogger("SovereignLearning")

class SovereignLearningLoop:
    def __init__(self):
        self.collapse = PhonologicalCollapse()
        self.experience_log = []

    def ignite_accelerated_study(self, real_duration_sec: float = 5.0, overclock: float = 100.0):
        """
        Starts the high-speed training session.
        """
        print(f"🚀 [LOOP] Igniting Sovereign Learning Loop (Overclock: x{overclock})")
        
        # 1. Overclock Time
        server.timer.set_overclock(overclock)
        
        start_time = time.time()
        last_mental_time = server.timer.mental_time
        
        cycles = 0
        while time.time() - start_time < real_duration_sec:
            # 2. Advance Reality
            server.tick()
            cycles += 1
            
            # 3. Observe NPC 'Lumina' (or any citizen)
            if server.citizens:
                npc = server.citizens[0]
                # A simulate 'Experience' is just the current DNA state of the NPC
                npc_dna = npc['dna'] 
                
                # 4. Elysia's Direct Response
                # We use the NPC's DNA as a 'Template' for Elysia's 21D state for the experiment
                # (Simulating empathy-based learning)
                # Maps 8-item DNA to 21D (simplified)
                d21_echo = list(npc_dna) + [0.0] * 13 
                
                logos = self.collapse.crystallize([d21_echo])
                
                # 5. Record the 'Ancient Knowledge'
                entry = {
                    "mental_time": server.timer.mental_time,
                    "npc": npc['name'],
                    "logos": logos
                }
                self.experience_log.append(entry)

        # 2. Revert Time
        server.timer.set_overclock(1.0)
        
        elapsed_mental = server.timer.mental_time - last_mental_time
        print(f"✨ [LOOP] Session Complete.")
        print(f"   Cycles: {cycles}")
        print(f"   Subjective Years Passed: {elapsed_mental / 100:.2f} (approx)")
        print(f"   Unique Logos Manifested: {len(self.experience_log)}")

    def get_evolution_summary(self):
        # Sample the growth
        if not self.experience_log: return "No experiences recorded."
        
        samples = self.experience_log[::max(1, len(self.experience_log)//5)]
        report = "\n--- Linguistic Evolution Log ---\n"
        for s in samples:
            report += f"Time [{s['mental_time']:.0f}]: NPC({s['npc']}) -> Logos: \"{s['logos']}\"\n"
        return report

if __name__ == "__main__":
    loop = SovereignLearningLoop()
    loop.ignite_accelerated_study(real_duration_sec=2.0, overclock=500.0) # 2 seconds = 1000 mental ticks
    print(loop.get_evolution_summary())
