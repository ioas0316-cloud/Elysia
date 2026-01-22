"""
Echo Field World Simulation
=====================================
Core.World.Simulations.echo_field_world

The Vessel for Project Echo Field.
Manages the "Hybrid Genesis" simulation:
- AIC (Endfield Structure)
- Echoes & Action (Myeongjo Soul)
"""

import logging
import random
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("EchoFieldWorld")

@dataclass
class Vector3:
    x: float
    y: float
    z: float

@dataclass
class WorldState:
    """The Mutable State of the World"""
    time_tick: int = 0
    gravity_vector: float = 9.81
    entropy_level: float = 0.0
    resources: Dict[str, float] = field(default_factory=lambda: {"energy": 100.0, "data": 0.0})

    # Action Variables
    resonance_window: float = 0.2
    echo_drop_rate: float = 0.1

class SimulationEntity:
    def __init__(self, uid: str, position: Vector3):
        self.uid = uid
        self.position = position
        self.active = True

    def update(self, state: WorldState):
        pass

class CorruptionEntity(SimulationEntity):
    """The Enemy (Noise)"""
    def __init__(self, uid: str, position: Vector3, power: float):
        super().__init__(uid, position)
        self.power = power

    def defeat(self, state: WorldState) -> 'EchoEntity':
        """Calculates if an Echo is dropped upon defeat."""
        if random.random() < state.echo_drop_rate:
            return EchoEntity(
                uid=f"ECHO_{self.uid}",
                position=self.position,
                effect=f"Insight from {self.uid}"
            )
        return None

class EchoEntity(SimulationEntity):
    """The Loot (Crystallized Concept)"""
    def __init__(self, uid: str, position: Vector3, effect: str):
        super().__init__(uid, position)
        self.effect = effect
        self.absorbed = False

    def absorb(self):
        self.absorbed = True
        self.active = False
        return self.effect

class SovereignOperator(SimulationEntity):
    """The Agent (Action)"""
    def __init__(self, uid: str, role: str, position: Vector3):
        super().__init__(uid, position)
        self.role = role
        self.inventory: List[str] = []

    def parry(self, attack_time: float, current_time: float, window: float) -> bool:
        """
        Attempts to Parry an attack.
        Resonance Action: Checks if delta is within window.
        """
        delta = abs(current_time - attack_time)
        success = delta <= window
        return success

    def update(self, state: WorldState):
        # Basic patrol logic
        pass

class EchoFieldWorld:
    """
    The Container for the Hybrid Simulation.
    """
    def __init__(self, seed: str = "ECHO-GENESIS"):
        self.seed = seed
        self.state = WorldState()
        self.entities: List[SimulationEntity] = []
        self.echoes: List[EchoEntity] = []

        logger.info(f"🌌 Echo Field World [{seed}] Initialized.")

    def spawn_entity(self, entity: SimulationEntity):
        self.entities.append(entity)
        if isinstance(entity, EchoEntity):
            self.echoes.append(entity)
        logger.info(f"Entity spawned: {entity.uid}")

    def apply_monad_law(self, variables: Dict[str, float]):
        """Injects Monad Laws into the State."""
        if "resonance_window" in variables:
            self.state.resonance_window = variables["resonance_window"]
        if "echo_drop_rate" in variables:
            self.state.echo_drop_rate = variables["echo_drop_rate"]
        if "entropy_rate" in variables:
            # Apply immediate entropy
            self.state.entropy_level += variables["entropy_rate"]

    def tick(self):
        self.state.time_tick += 1
        return self.get_snapshot()

    def get_snapshot(self) -> Dict[str, Any]:
        return {
            "tick": self.state.time_tick,
            "entropy": self.state.entropy_level,
            "echoes_available": len([e for e in self.echoes if not e.absorbed]),
            "resonance_window": self.state.resonance_window
        }
