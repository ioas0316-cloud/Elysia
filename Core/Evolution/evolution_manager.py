"""
Evolution Manager
=================
Orchestrates the self-evolution process.
Decides when to trigger concept evolution or code mutation.
"""

import logging
import numpy as np
from typing import Optional

from Core.Evolution.concept_evolution import ConceptEvolution
from Core.Evolution.code_mutator import EvolutionaryCoder
from Core.Physics.fluctlight import FluctlightEngine
from Core.Mind.hippocampus import Hippocampus
from Core.Mind.alchemy import Alchemy

logger = logging.getLogger("EvolutionManager")

class EvolutionManager:
    """
    The 'Driver' of Elysia's growth.
    Monitors system state and triggers evolution cycles.
    """
    
    def __init__(self, hippocampus: Hippocampus, alchemy: Alchemy):
        self.concept_evo = ConceptEvolution(hippocampus, alchemy)
        self.code_evo = EvolutionaryCoder()
        self.boredom_counter = 0
        
    def update(self, engine: FluctlightEngine, activity_level: float):
        """
        Called every tick.
        activity_level: 0.0 to 1.0 (how busy the system is)
        """
        # 1. Manage Boredom
        if activity_level < 0.1:
            self.boredom_counter += 1
        else:
            self.boredom_counter = 0
            
        # 2. Trigger response if bored
        if self.boredom_counter > 100:  # 100 ticks of relative inactivity
            self._handle_boredom(engine)
            self.boredom_counter = 0

    def _handle_boredom(self, engine: FluctlightEngine) -> None:
        """
        Decide how to respond to sustained boredom.

        Instead of forcing evolution every time, allow the system to:
        - Rest (no structural changes)
        - Reflect (introspective tick without evolution)
        - Evolve (concept/code), as before
        """
        r = float(np.random.rand())
        if r < 0.3:
            logger.info("🌙 Boredom detected. Choosing to rest this cycle (no evolution).")
            return
        if r < 0.6:
            logger.info("🧘 Boredom detected. Entering reflection mode (no structural change).")
            # Future hook: could sample Hippocampus / WorldTree here.
            return
            
    def trigger_evolution(self, engine: FluctlightEngine):
        """
        Decides what kind of evolution to perform.
        """
        choice = np.random.choice(["concept", "code"], p=[0.9, 0.1])
        
        if choice == "concept":
            logger.info("🌀 Boredom detected. Triggering Concept Evolution...")
            new_concepts = self.concept_evo.evolve_concepts(engine)
            if new_concepts:
                logger.info(f"   -> Evolved {len(new_concepts)} new concepts!")
                
        elif choice == "code":
            logger.info("🧬 Boredom detected. Triggering Code Mutation (Simulation)...")
            # In a real scenario, we would pass a target function.
            # For safety, we just log that we *would* have mutated something.
            logger.info("   -> (Safety Lock) Code mutation skipped in this version.")
