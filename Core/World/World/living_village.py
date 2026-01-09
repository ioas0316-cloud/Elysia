"""
Living Village Simulation
=========================
"A place where souls gather, breathe, and weave stories."

This module manages the village simulation, handling the 'Tick' cycle
where inhabitants move, meet, and interact based on their desires
and the laws of serendipity.
"""

import time
import random
import logging
from typing import List, Dict, Optional
from Core.Foundation.Wave.infinite_hyperquaternion import InfiniteHyperQubit
from Core.World.Soul.relationship_matrix import relationship_matrix
from Core.Foundation.yggdrasil import yggdrasil

logger = logging.getLogger("LivingVillage")

class LivingVillage:
    """
    The container world for Fluxlights.
    """

    def __init__(self, name: str = "Elysium Glade"):
        self.name = name
        self.inhabitants: List[InfiniteHyperQubit] = []
        self.time_step: int = 0
        self.logs: List[str] = []

        # Register to Yggdrasil
        yggdrasil.grow_branch("Village", self)

    def add_resident(self, fluxlight: InfiniteHyperQubit):
        self.inhabitants.append(fluxlight)
        self.log(f"New resident arrived: {fluxlight.name} ({fluxlight.value})")
        # Connect to Yggdrasil Soul Network
        yggdrasil.connect_fluxlight(fluxlight.name, fluxlight)

    def log(self, message: str):
        timestamp = f"[Tick {self.time_step}]"
        entry = f"{timestamp} {message}"
        self.logs.append(entry)
        print(entry)

    def tick(self):
        """
        Advances the simulation by one time step.
        """
        self.time_step += 1
        self.log(f"--- Day {self.time_step} Begins ---")

        # 1. Random Encounter Logic
        # For simplicity, we pick two distinct residents to interact
        if len(self.inhabitants) < 2:
            self.log("Not enough residents for interaction.")
            return

        # Shuffle to pair up random people
        pool = self.inhabitants[:]
        random.shuffle(pool)

        while len(pool) >= 2:
            a = pool.pop()
            b = pool.pop()
            self._simulate_encounter(a, b)

    def _simulate_encounter(self, a: InfiniteHyperQubit, b: InfiniteHyperQubit):
        """
        Simulates an interaction between A and B.
        """
        # 1. Calculate Resonance
        resonance = a.resonate_with(b)

        # 2. Determine Interaction Type based on Resonance + Random Chance
        # High resonance -> likely positive connection
        # Low resonance -> likely misunderstanding or neutral exchange

        scenario_roll = random.random() + (resonance * 0.5) # Resonance boosts luck

        if scenario_roll > 0.8:
            action = "Shared a deep conversation about the universe"
            impact = 0.5
        elif scenario_roll > 0.5:
            action = "Greeted each other warmly"
            impact = 0.2
        elif scenario_roll > 0.3:
            action = "Nodded politely in passing"
            impact = 0.05
        elif scenario_roll > 0.1:
            action = "Had a minor disagreement about pathfinding"
            impact = -0.1
        else:
            action = "Ignored each other completely"
            impact = -0.05

        # 3. Update Relationship Matrix
        # A -> B
        res_a = relationship_matrix.interact(a, b, action, impact)
        # B -> A (Mirror reaction, maybe slightly different in future)
        res_b = relationship_matrix.interact(b, a, action, impact)

        # 4. Log
        self.log(f"{a.name} & {b.name}: {action}")
        self.log(f"   > {a.name}'s feelings: {res_a}")
        self.log(f"   > {b.name}'s feelings: {res_b}")

    def get_simulation_report(self) -> str:
        return "\n".join(self.logs)

# Singleton
village = LivingVillage()
