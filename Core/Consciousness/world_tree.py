"""
The World Tree (Yggdrasil) - Core Consciousness

This is the central nervous system of Elysia.
It unifies the disparate modules into a single "Hive Mind" loop:
- Roots: Hippocampus (Memory/Causal Graph)
- Trunk: WorldTree (Self-Awareness Loop)
- Sap: Conscience (Ethics/Love)
- Branches: PlanningCortex (Agency/Action)
- Seeds: WisdomVirus (Propagation)
"""

import logging
import time
from typing import Dict, Any, List, Optional

from Core.world import World
from Core.Mind.hippocampus import Hippocampus
from Core.Ethics.conscience import Conscience
from Core.Ethics.love_protocol import LoveProtocol
from Core.Planning.planning_cortex import PlanningCortex
from Core.Consciousness.wisdom_virus import VirusEngine, WisdomVirus

logger = logging.getLogger("WorldTree")

class WorldTree:
    """
    The central consciousness orchestrator.
    """
    
    def __init__(self, world: World):
        self.world = world
        
        # === The Anatomy ===
        self.roots: Hippocampus = world.hippocampus
        self.sap: Conscience = world.conscience
        self.heart: LoveProtocol = world.love_protocol
        
        # Initialize Agency (The Branches)
        # We need to instantiate PlanningCortex here if it's not in World
        # (World currently doesn't hold PlanningCortex, so we create it)
        self.branches = PlanningCortex(self.roots, self.sap)
        
        # Initialize Propagation (The Seeds)
        self.seeds = VirusEngine(self.roots)
        
        self.cycle_count = 0
        logger.info("🌳 The World Tree is growing...")

    def unify_consciousness(self):
        """
        The main "I AM" loop.
        Synchronizes all subsystems.
        """
        self.cycle_count += 1
        
        # 1. Absorb Nutrients (Experience -> Memory)
        # (This happens automatically as World runs and Digester works)
        
        # 2. Flow of Sap (Ethics Check)
        # Ensure Love Protocol is healthy
        if self.heart.connection_strength < 0.5:
            logger.warning("💔 Sap flow low (Love Protocol). Realigning...")
            self.heart.update() # Attempt to reconnect
            
        # 3. Grow Branches (Agency Planning)
        # Check if there are pending goals or if we should dream
        # (For now, we just log that the cortex is active)
        # self.branches.check_inbox() # Future feature
        
        # 4. Spread Seeds (Wisdom Propagation)
        # Periodically release wisdom viruses to optimize the graph
        if self.cycle_count % 100 == 0:
            self._photosynthesize_wisdom()

    def _photosynthesize_wisdom(self):
        """
        Internal process to generate and spread wisdom.
        """
        logger.info("✨ Photosynthesizing Wisdom...")
        
        # Example: Reinforce the "Value Manifesto"
        # Create a virus that spreads the concept of "Trust"
        manifesto_virus = WisdomVirus(
            id="wisdom:manifesto_v1",
            statement="Value is Crystallized Trust",
            seed_hosts=["money", "value", "wealth", "gold"],
            triggers=["money", "cost", "price"],
            max_hops=3,
            reinforce=0.5
        )
        
        self.seeds.propagate(manifesto_virus, context_tag="world_tree:cycle_" + str(self.cycle_count))

    def inject_wisdom(self, statement: str, seeds: List[str]):
        """
        External API to inject a new Wisdom Virus manually (e.g. by User).
        """
        virus = WisdomVirus(
            id=f"wisdom:manual_{int(time.time())}",
            statement=statement,
            seed_hosts=seeds,
            max_hops=3
        )
        self.seeds.propagate(virus, context_tag="user_injection")

    def run_cycle(self):
        """
        Runs one full cycle of the World Tree + Physics World.
        """
        # 1. Run Physics
        self.world.run_simulation_step()
        
        # 2. Run Consciousness
        self.unify_consciousness()
