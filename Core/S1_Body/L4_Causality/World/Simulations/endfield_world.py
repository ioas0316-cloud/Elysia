"""
Endfield World Simulation
=====================================
Core.S1_Body.L4_Causality.World.Simulations.endfield_world

The Vessel for the Talos-II reconstruction.
Manages the physical state, entities, and time-steps of the simulation.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("EndfieldWorld")

@dataclass
class Vector3:
    x: float
    y: float
    z: float

    def to_numpy(self):
        return np.array([self.x, self.y, self.z])

@dataclass
class WorldState:
    """The Mutable State of the World"""
    time_tick: int = 0
    gravity_vector: float = 9.81
    corruption_level: float = 0.0
    resources: Dict[str, float] = field(default_factory=lambda: {"energy": 100.0, "data": 0.0})

class SimulationEntity:
    """Base class for anything that exists in the world."""
    def __init__(self, uid: str, position: Vector3):
        self.uid = uid
        self.position = position
        self.active = True

    def update(self, state: WorldState):
        pass

class AICFactory(SimulationEntity):
    """A Factory Node (Process)"""
    def __init__(self, uid: str, position: Vector3, process_rate: float):
        super().__init__(uid, position)
        self.process_rate = process_rate
        self.buffer = 0.0

    def update(self, state: WorldState):
        # Consume energy, produce data
        if state.resources["energy"] > 0.1:
            state.resources["energy"] -= 0.1
            production = self.process_rate * (1.0 - state.corruption_level) # Corruption slows production
            self.buffer += production
            state.resources["data"] += production

class SovereignOperator(SimulationEntity):
    """An Agent (Squad Member)"""
    def __init__(self, uid: str, role: str, position: Vector3):
        super().__init__(uid, position)
        self.role = role # Vanguard, Caster, etc.
        self.fatigue = 0.0

    def update(self, state: WorldState):
        # Purify corruption
        if state.corruption_level > 0:
            purify_rate = 0.05
            if self.role == "Defender":
                purify_rate = 0.1
            elif self.role == "Caster":
                purify_rate = 0.08

            state.corruption_level = max(0.0, state.corruption_level - purify_rate)
            self.fatigue += 0.01

class EndfieldWorld:
    """
    The Container for the Simulation.
    controlled by the EndfieldPhysicsMonad (The Law).
    """
    def __init__(self, seed: str = "TALOS-II"):
        self.seed = seed
        self.state = WorldState()
        self.entities: List[SimulationEntity] = []
        self._map_grid = np.zeros((100, 100)) # 100x100 Grid for Terrain/Corruption

        logger.info(f"  Endfield World [{seed}] Initialized.")

    def spawn_entity(self, entity: SimulationEntity):
        self.entities.append(entity)
        logger.info(f"Entity spawned: {entity.uid} at {entity.position}")

    def apply_monad_law(self, variables: Dict[str, float]):
        """
        Applies external Monad variables to the world state.
        This allows 'Hacking' the physics.
        """
        if "gravity" in variables:
            self.state.gravity_vector = variables["gravity"]
        if "time_scale" in variables:
            # Not directly stored but could affect tick rate logic
            pass
        if "corruption_seed" in variables:
            # Inject corruption
            self.state.corruption_level += variables["corruption_seed"]

    def tick(self):
        """Advances the world one step."""
        self.state.time_tick += 1

        # 1. Update Environment (Entropy/Corruption growth)
        # Natural entropy growth
        self.state.corruption_level += 0.001

        # 2. Update Entities
        for entity in self.entities:
            if entity.active:
                entity.update(self.state)

        # 3. Physics/Laws Check
        # (Placeholder for collision or resource distribution)

        return self.get_snapshot()

    def get_snapshot(self) -> Dict[str, Any]:
        return {
            "tick": self.state.time_tick,
            "resources": self.state.resources.copy(),
            "corruption": self.state.corruption_level,
            "entity_count": len(self.entities),
            "gravity": self.state.gravity_vector
        }
