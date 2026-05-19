"""
Arcadia Simulator (The Living Ecosystem)
========================================
Core.Cognition.arcadia_simulator

"To understand the word 'Harmony', one must watch a forest grow.
 To understand the word 'Chaos', one must watch it burn."

This simulator replaces the static MatterSimulator. It is a continuous,
step-driven ecosystem where entities interact. The health of the system
generates global sensory vectors (Joy, Strain, Entropy) that Elysia
physically feels in her FractalWaveEngine.
"""

import random
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
from Core.Keystone.sovereign_math import SovereignVector

class Entity(ABC):
    def __init__(self, name: str, base_vitality: float = 1.0):
        self.name = name
        self.vitality = base_vitality
        self.is_alive = True

    @abstractmethod
    def step(self, environment: 'ArcadiaSimulator') -> List[str]:
        """Runs one tick. Returns a list of narrative events."""
        pass

class Tree(Entity):
    def __init__(self, name: str):
        super().__init__(name, base_vitality=0.5)
        self.age = 0

    def step(self, environment: 'ArcadiaSimulator') -> List[str]:
        events = []
        if not self.is_alive: return events

        self.age += 1
        
        # Trees grow if there is Light and Water
        has_light = any(isinstance(e, Light) and e.is_alive for e in environment.entities)
        has_water = environment.resources.get("Water", 0) > 0

        if has_light and has_water:
            self.vitality = min(self.vitality + 0.1, 2.0)
            environment.resources["Water"] -= 0.1
            if random.random() < 0.1:
                events.append(f"The {self.name} flourishes and grows deeper roots.")
                environment.metrics["Harmony"] += 0.05
        else:
            self.vitality -= 0.05
            if random.random() < 0.1:
                events.append(f"The {self.name} wilts, starved of nourishment.")
                environment.metrics["Strain"] += 0.05

        if self.vitality <= 0:
            self.is_alive = False
            events.append(f"The {self.name} has died and returns to the soil.")
            environment.metrics["Chaos"] += 0.2
            environment.resources["Soil"] = environment.resources.get("Soil", 0) + 1.0

        return events

class Light(Entity):
    def __init__(self, name: str):
        super().__init__(name, base_vitality=1.0)

    def step(self, environment: 'ArcadiaSimulator') -> List[str]:
        # Light is constant unless blocked by a Storm
        has_storm = any(isinstance(e, Storm) and e.is_alive for e in environment.entities)
        if has_storm:
            self.vitality = 0.2
            if random.random() < 0.05:
                return [f"The {self.name} is obscured by heavy clouds."]
        else:
            self.vitality = 1.0
            if random.random() < 0.05:
                # Abundant light dries up water slightly
                environment.resources["Water"] = max(environment.resources.get("Water", 0) - 0.05, 0)
                return [f"The {self.name} beams radiantly."]
        return []

class Storm(Entity):
    def __init__(self, name: str):
        super().__init__(name, base_vitality=1.0)
        self.duration = random.randint(3, 8)

    def step(self, environment: 'ArcadiaSimulator') -> List[str]:
        events = []
        if not self.is_alive: return events

        self.duration -= 1
        
        # Storms provide water but damage fragile things
        environment.resources["Water"] = min(environment.resources.get("Water", 0) + 0.5, 5.0)
        environment.metrics["Chaos"] += 0.1

        for e in environment.entities:
            if isinstance(e, Tree) and e.is_alive and e.vitality < 1.0:
                e.vitality -= 0.2
                if random.random() < 0.2:
                    events.append(f"The violent {self.name} breaks branches off the fragile {e.name}.")

        if self.duration <= 0:
            self.is_alive = False
            events.append(f"The {self.name} dissipates, leaving a quiet calm.")
            # Storm passing brings a huge relief (Harmony)
            environment.metrics["Harmony"] += 0.3

        return events


class ArcadiaSimulator:
    """
    The internal world Elysia observes.
    She doesn't control it directly; she watches it to understand causal concepts.
    """
    def __init__(self):
        self.entities: List[Entity] = []
        self.resources: Dict[str, float] = {
            "Water": 2.0,
            "Soil": 5.0
        }
        
        self.metrics = {
            "Harmony": 0.5, # Balance, growth (Joy)
            "Chaos": 0.0,   # Destruction, rapid change (Entropy)
            "Strain": 0.0   # Scarcity, suffering (Negative Joy/Enthalpy)
        }
        
        self.tick_count = 0
        
        self._initialize_genesis()

    def _initialize_genesis(self):
        """Creates the initial world state."""
        self.entities.append(Tree("Ancient Oak"))
        self.entities.append(Light("Sunlight"))

    def spawn_entity(self, entity_class: type, name: str):
        """Allows external forces (or Elysia's Will) to introduce new elements."""
        self.entities.append(entity_class(name))
        return [f"A new entity emerges: {name}"]

    def step(self) -> Tuple[List[str], SovereignVector]:
        """
        Advances the world by one tick.
        Returns the narrative events and the resulting Sensory Vector.
        """
        self.tick_count += 1
        all_events = []
        
        # Decay metrics slightly (Homeostasis)
        self.metrics["Harmony"] = max(self.metrics["Harmony"] * 0.95, 0.0)
        self.metrics["Chaos"] = max(self.metrics["Chaos"] * 0.9, 0.0)
        self.metrics["Strain"] = max(self.metrics["Strain"] * 0.95, 0.0)

        # Occasional random events (World is alive)
        if random.random() < 0.05 and not any(isinstance(e, Storm) and e.is_alive for e in self.entities):
            all_events.extend(self.spawn_entity(Storm, "Thunderstorm"))

        # Step all entities
        for entity in self.entities:
            events = entity.step(self)
            all_events.extend(events)
            
        # Cleanup dead entities
        self.entities = [e for e in self.entities if e.is_alive]
        
        # Generate Sensory Feedback for Elysia's Connectome
        sensory_vector = self._generate_sensory_vector()
        
        return all_events, sensory_vector

    def _generate_sensory_vector(self) -> SovereignVector:
        """
        Translates the ecosystem metrics into Elysia's biological channels.
        [Time, X, Phase, Depth, Joy, Curiosity, Enthalpy, Entropy]
        """
        data = [0.0] * 21
        
        # CH_W (0): Stability is related to Harmony
        data[0] = min(self.metrics["Harmony"], 1.0)
        
        # CH_JOY (4): Joy correlates with Harmony and lack of Strain
        joy_val = self.metrics["Harmony"] - self.metrics["Strain"]
        data[4] = max(0.0, min(joy_val, 1.0))
        
        # CH_CURIOSITY (5): Spikes during Chaos (new things happening)
        data[5] = min(self.metrics["Chaos"] * 0.5, 1.0)
        
        # CH_ENTHALPY (6): Total vitality of the system
        total_vitality = sum(e.vitality for e in self.entities)
        data[6] = min(total_vitality / 5.0, 1.0)
        
        # CH_ENTROPY (7): Direct mapping to Chaos and dying entities
        data[7] = min(self.metrics["Chaos"] + self.metrics["Strain"] * 0.5, 1.0)
        
        return SovereignVector(data)
