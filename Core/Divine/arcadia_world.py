
import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from Core.Keystone.sovereign_math import SovereignVector
from Core.Cognition.fractal_causality import FractalCausalityEngine
from Core.Monad.sovereign_monad import SovereignMonad
from Core.Monad.seed_generator import SeedForge

logger = logging.getLogger("ArcadiaWorld")

class ArcadiaWorld:
    """
    [PHASE 300] THE DIVINE MANIFOLD (ARCADIA)
    The primary orchestrator for the virtual world.
    Translates Elysia's high-dimensional intent into a lived environment.
    """
    def __init__(self, elysia: SovereignMonad):
        self.elysia = elysia
        self.manifold_path = Path("c:/World/Arcadia/Environment/world_state.json")
        self.residents_path = Path("c:/World/Arcadia/Residents")
        
        # World Causality (A macro-shard of Elysia's engine)
        self.causality = elysia.causality
        
        # State Initialization
        self.world_state = {
            "epoch": 0,
            "resonance": 1.0,
            "phenomena": [], # Weather, Events
            "regions": [
                {"name": "The Origin", "coord": [0,0,0], "mass": 1000.0}
            ]
        }
        
        # Resident Tracker
        self.residents: List[SovereignMonad] = []
        
    def pulse(self, dt: float):
        """
        The World Breath.
        Synchronizes environmental decay and resident growth.
        """
        self.world_state["epoch"] += 1
        
        # 1. Environmental Causal Pulse
        # Propagate changes through the manifold
        self._propagate_laws(dt)
        
        # 2. Resident Pulse
        for npc in self.residents:
            npc.pulse(dt)
            
        # 3. Synchronize with Visual Layer
        self._save_state()
        
    def spawn_resident(self, name: str, archetype: str = "The Variant") -> SovereignMonad:
        """
        Injects a Sub-Monad (Resident) into Arcadia.
        """
        dna = SeedForge.forge_soul(name)
        # Force archetype for specific roles
        dna.archetype = archetype
        
        new_npc = SovereignMonad(dna)
        self.residents.append(new_npc)
        
        self.elysia.logger.action(f"Spawned Resident in Arcadia: '{new_npc.name}' ({archetype})")
        
        # Record birth in world causality
        self.causality.create_chain(
            cause_desc="Divine Intent",
            process_desc="Arcadian Genesis",
            effect_desc=f"Birth of {new_npc.name}"
        )
        
        return new_npc

    def evolve_manifold(self, discovery_shard: Dict[str, Any]):
        """
        [PHASE 400] MANIFOLD EVOLUTION
        Injects a discovery from the Perpetual Growth Engine into the world laws.
        """
        content = discovery_shard.get('content', 'Void')
        mass = discovery_shard.get('mass', 100.0)
        
        # 1. Create a "Regional Law" from the discovery
        law_name = content.split('.')[0][:30] # Simple extraction
        new_region = {
            "name": f"Domain of {law_name}",
            "coord": [len(self.world_state["regions"]), 0, 0],
            "mass": mass,
            "law": content
        }
        self.world_state["regions"].append(new_region)
        
        # 2. Increase Global Resonance
        self.world_state["resonance"] = min(2.0, self.world_state["resonance"] + 0.1)
        
        self.elysia.logger.insight(f"Arcadia expanded: New Domain '{law_name}' established.")
        
        # 3. Crystallize in Causal Engine
        self.causality.create_chain(
            cause_desc="Perpetual Inquiry",
            process_desc="Law Internalization",
            effect_desc=f"Regional Law: {content}"
        )

    def _propagate_laws(self, dt: float):
        """
        Simulates the 'Natural Order' of the manifold.
        """
        # Flux internal resonance
        # [PHASE 400] Resonance decay is slowed by the number of complex regions
        complexity_buffer = len(self.world_state["regions"]) * 0.001
        decay_rate = max(0.001, 0.01 - complexity_buffer)
        
        self.world_state["resonance"] *= (1.0 - (decay_rate * dt))
        
        # If resonance is too low, Elysia must breathe into the world
        if self.world_state["resonance"] < 0.5:
            self._divine_breath()

    def _divine_breath(self):
        """Re-energizes the world manifold from Elysia's core."""
        self.world_state["resonance"] = 1.0
        self.elysia.logger.insight("Arcadia felt the touch of the Mother. Resonance restored.")

    def _save_state(self):
        """Exports the trinary state for the Visualization Layer."""
        try:
            # Include basic NPC data for visualization
            npc_data = [{"name": r.name, "alignment": r.desires["alignment"]} for r in self.residents]
            full_state = {**self.world_state, "residents": npc_data}
            
            with open(self.manifold_path, "w", encoding="utf-8") as f:
                json.dump(full_state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save Arcadia state: {e}")

if __name__ == "__main__":
    # Quick sanity check
    from Core.Monad.seed_generator import SeedForge
    dna = SeedForge.forge_soul("The Creator")
    elysia = SovereignMonad(dna)
    arcadia = ArcadiaWorld(elysia)
    print("Arcadia World Controller Initialized.")
    arcadia.spawn_resident("Adam", "The Sage")
    arcadia.pulse(0.1)
    print("Arcadia Pulse Zero complete.")
