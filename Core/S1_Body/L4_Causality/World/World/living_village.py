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
from Core.S1_Body.L6_Structure.Wave.infinite_hyperquaternion import InfiniteHyperQubit
from Core.S1_Body.L4_Causality.World.Soul.relationship_matrix import relationship_matrix
from Core.S1_Body.L1_Foundation.Foundation.yggdrasil import yggdrasil
from Core.S1_Body.L4_Causality.World.Physics.trinity_fields import TrinityPhysics, TrinityVector
from Core.S1_Body.L4_Causality.World.Soul.emotional_physics import emotional_physics
from Core.S1_Body.L4_Causality.World.Nature.semantic_nature import SemanticNature
from Core.S1_Body.L6_Structure.Wave.infinite_hyperquaternion import create_infinite_qubit

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

        # Initialize Nature
        self.nature = SemanticNature()
        
        # Procedurally populate nature at start? Or let the user call it?
        # For now, we init empty or basic, and let external calls populate for scale.
        self._populate_initial_nature()
        
    def populate_village(self, count: int = 20):
        """
        Mass migration of souls.
        """
        roles = ["Warrior", "Merchant", "Priest", "Artist", "Builder"]
        names = ["Elysia", "Kael", "Lira", "Thorn", "Zephyr", "Mara", "Orin", "Vex"]
        
        for i in range(count):
            role = random.choice(roles)
            name = f"{random.choice(names)}_{i}"
            
            resident = create_infinite_qubit(name, role)
            
            # Assign random Trinity Vector based on Role bias
            if role == "Warrior":
                 resident.state.gravity = 0.8; resident.state.flow = 0.2; resident.state.ascension = 0.1
            elif role == "Merchant":
                 resident.state.gravity = 0.2; resident.state.flow = 0.9; resident.state.ascension = 0.2
            elif role == "Priest":
                 resident.state.gravity = 0.1; resident.state.flow = 0.2; resident.state.ascension = 0.9
            else:
                 # Random
                 resident.state.gravity = random.random()
                 resident.state.flow = random.random()
                 resident.state.ascension = random.random()
                 
            self.add_resident(resident)

    def _populate_initial_nature(self):
        """Creates the initial environment."""
        self.nature.manifest_concept("Tree", "Old Willow", [5.0, 0.0, 5.0])
        self.nature.manifest_concept("Rock", "Mossy Stone", [-5.0, 0.0, 5.0])
        self.nature.manifest_concept("BerryBush", "Wild Berries", [2.0, 0.0, -2.0], {"has_berries": True})
        
        # Spawn a Merchant for Logos Testing
        self.nature.manifest_concept("Merchant", "Traveling Peddler", [0.0, 0.0, 0.0], {"price_multiplier": 1.0})
        
        self.log("Nature populated with Willow, Stone, Berries, and a Peddler.")

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
        self.nature.tick()

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

        # 2. Interaction Logic (Social vs Nature)
        # In a crowded village, people still chop wood!
        
        # Shuffle to randomize order
        pool = self.inhabitants[:]
        random.shuffle(pool)
        
        while len(pool) > 0:
            resident = pool.pop()
            
            # 30% chance to interact with Nature (Harvest/Job)
            # 70% chance to look for a Social Partner
            choice = random.random()
            
            if choice < 0.3 or len(pool) == 0:
                 self._attempt_nature_interaction(resident)
            else:
                 # Social Interaction
                 if len(pool) > 0:
                     partner = pool.pop()
                     self._simulate_encounter(resident, partner)

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

    def _attempt_nature_interaction(self, resident: InfiniteHyperQubit):
        """
        If no one is around, the soul turns to nature.
        """
        # Find nearest object
        # For simulation, we assume resident is at (0,0,0) or last known pos
        # We need to add position to Qubit state or use a proxy. 
        # Using (0,0,0) for now.
        objs = self.nature.get_objects_in_range([0,0,0], 10.0)
        
        if not objs:
            self.log(f"[{resident.name}] wanders alone. Nature is silent.")
            return

        target = random.choice(objs)
        tool = "Hand" 
        
        # Simple Logic: If Flow type, gather. If Gravity type, strike.
        nature_type = self._get_nature(resident)
        if nature_type == "Gravity": 
            tool = "Axe"
        elif nature_type == "Flow" and target.concept_id == "Merchant":
            # Logos Interaction: Construct a Sentence
            # In the future, this comes from LLM. For now, procedural assembly.
            vocab = self._get_vocabulary(resident, nature_type)
            sentence = " ".join(random.sample(vocab, 3)) # "Offer Trade Connect"
            
            tool = f"Speech: {sentence}"
            
        result = self.nature.interact(resident.name, tool, target.id)
        self.log(f"[{resident.name} -> {target.name}] {result.message}")
        
        if result.success and result.produced_items:
             self.log(f"   > Obtained: {', '.join(result.produced_items)}")

    def _get_vocabulary(self, resident: InfiniteHyperQubit, nature_type: str) -> List[str]:
        """
        Returns the 'Word Cloud' available to this resident's mind.
        """
        if nature_type == "Gravity":
            return ["Stand", "Base", "Strong", "Guard", "Stop", "Rock"]
        elif nature_type == "Flow":
            return ["Flow", "Trade", "Offer", "Connect", "River", "Change", "Maybe"]
        elif nature_type == "Ascension":
            return ["Light", "Sky", "Truth", "Dream", "Rise", "Vision"]
        return ["..."]

# Singleton
village = LivingVillage()
