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
import math
import logging
from typing import List, Dict, Optional
from Core.Foundation.Wave.infinite_hyperquaternion import InfiniteHyperQubit
from Core.World.Soul.relationship_matrix import relationship_matrix
from Core.Foundation.yggdrasil import yggdrasil
from Core.World.Physics.trinity_fields import TrinityPhysics, TrinityVector
from Core.World.Soul.emotional_physics import emotional_physics

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
        Now applies TRINITY PHYSICS (Gravity, Flow, Ascension).
        """
        self.time_step += 1
        self.log(f"--- Day {self.time_step} Begins (Physics Active) ---")

        physics = TrinityPhysics()

        # 1. Apply Physics to Each Resident
        for resident in self.inhabitants:
            # A. Get Intrinsic Nature (Seed Bias)
            # Map state (x,y,z) to Trinity Vector roughly if not set
            # Ideally, this should be set during init. We assume it exists in state.
            soul_vector = TrinityVector(
                gravity=getattr(resident.state, 'gravity', 0.5),
                flow=getattr(resident.state, 'flow', 0.5),
                ascension=getattr(resident.state, 'ascension', 0.5)
            )

            # B. Apply Emotional Buoyancy
            # Emotional state affects physical properties!
            # For simulation, we assume a fluctuating frequency or derive it from recent interactions
            current_freq = 200.0 # Neutral baseline
            # TODO: Fetch actual emotional frequency from RelationshipMatrix or Resident memory

            density_mod, flow_mod = emotional_physics.get_physical_modifiers(current_freq)

            # C. Determine Zone Affinity
            # We treat the "Village" as a neutral ground (0.5, 0.5, 0.5) for now,
            # but in reality, different coordinates would have different vectors.
            env_vector = TrinityVector(0.5, 0.5, 0.5)

            forces = physics.calculate_force(soul_vector, env_vector)

            # Apply modifications
            final_force_y = forces[1] / density_mod # Heavier = Harder to lift
            final_force_x = forces[0] * flow_mod    # Flow = Speed

            # D. Log the "Drift"
            zone = physics.get_zone_type(soul_vector)
            self.log(f"[{resident.name}] Drifting towards {zone}. ForceY: {final_force_y:.2f}, Speed: {final_force_x:.2f}")

        # 2. Interaction Logic (Simplified for now)
        if len(self.inhabitants) < 2: return

        # Shuffle to pair up random people
        pool = self.inhabitants[:]
        random.shuffle(pool)

        while len(pool) >= 2:
            a = pool.pop()
            b = pool.pop()
            self._simulate_encounter(a, b)

    def _simulate_encounter(self, a: InfiniteHyperQubit, b: InfiniteHyperQubit):
        """
        Simulates an interaction between A and B using the 'Strange Attractor' Model.
        Action = Global Rhythm (Pulse) + Personal Bias (Trinity) + Chaotic Twist (Random)
        """
        # 1. Global Rhythm (The Beat)
        # We use time_step to generate a predictable but changing 'Zeitgeist' (Spirit of the Times)
        # e.g., Even days are 'Calm', Odd days are 'Active'
        zeitgeist = (math.sin(self.time_step * 0.5) + 1) / 2 # 0.0 to 1.0

        # 2. Resonance (The Connection)
        resonance = a.resonate_with(b)

        # 3. Trinity Bias (The Improvisation)
        # If both are 'Gravity' types (Warriors), they might spar.
        # If one is 'Flow' (Merchant) and one is 'Ascension' (Priest), they might debate.

        # Determine dominant nature
        a_nature = self._get_nature(a)
        b_nature = self._get_nature(b)

        # 4. The Strange Attractor (Chaos)
        # Instead of linear random, we use the interaction of all 3 factors
        # P (Probability of Harmony) = Resonance * 0.4 + Zeitgeist * 0.3 + Compatibility * 0.3

        compatibility = 1.0 if a_nature == b_nature else 0.5
        harmony_score = (resonance * 0.4) + (zeitgeist * 0.3) + (compatibility * 0.3)

        # Add a chaotic twist (The Dance)
        chaos = (random.random() - 0.5) * 0.4
        final_score = harmony_score + chaos

        # 5. Resolve Action
        if final_score > 0.8:
            if a_nature == "Gravity": action = "Sparred in a friendly duel"
            elif a_nature == "Flow": action = "Traded secrets and items"
            else: action = "Prayed together for the village"
            impact = 0.6
        elif final_score > 0.6:
            action = f"Connected over shared {a_nature} interests"
            impact = 0.3
        elif final_score > 0.4:
            action = "Nodded politely, distracted by the rhythm"
            impact = 0.1
        elif final_score > 0.2:
            action = "Misunderstood each other's dance steps"
            impact = -0.1
        else:
            action = "Clashed due to conflicting rhythms"
            impact = -0.3

        # 3. Update Relationship Matrix
        # A -> B
        res_a = relationship_matrix.interact(a, b, action, impact)
        # B -> A
        res_b = relationship_matrix.interact(b, a, action, impact)

        # 4. Log
        self.log(f"{a.name} ({a_nature}) & {b.name} ({b_nature}): {action}")
        self.log(f"   > Score: {final_score:.2f} (R:{resonance:.2f}, Z:{zeitgeist:.2f}, C:{compatibility:.1f})")

    def _get_nature(self, entity: InfiniteHyperQubit) -> str:
        # Helper to get dominant Trinity trait
        g = getattr(entity.state, 'gravity', 0.0)
        f = getattr(entity.state, 'flow', 0.0)
        a = getattr(entity.state, 'ascension', 0.0)
        if g >= f and g >= a: return "Gravity"
        if f >= g and f >= a: return "Flow"
        return "Ascension"

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
