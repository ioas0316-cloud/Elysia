from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import logging

logger = logging.getLogger("Pantheon")

@dataclass
class Deity:
    name: str
    title: str
    alignment: str # "Angel" or "Demon"
    domain: str # e.g., "Gravity", "Entropy"
    power: float = 1.0 # Influence strength (0.0 to 1.0)
    
    def apply_effect(self, world):
        """Applies the deity's will to the world."""
        pass

class Pantheon:
    def __init__(self):
        self.angels: Dict[str, Deity] = {}
        self.demons: Dict[str, Deity] = {}
        self._init_deities()
        
    def _init_deities(self):
        # --- The 7 Angels (Order & Preservation) ---
        self.angels["Michael"] = Deity("Michael", "The Archon of Power", "Angel", "Physics", 1.0)
        self.angels["Gabriel"] = Deity("Gabriel", "The Messenger", "Angel", "Communication", 1.0)
        self.angels["Raphael"] = Deity("Raphael", "The Healer", "Angel", "Life", 1.0)
        self.angels["Uriel"] = Deity("Uriel", "The Illuminator", "Angel", "Light", 1.0)
        self.angels["Metatron"] = Deity("Metatron", "The Architect", "Angel", "Space", 1.0)
        self.angels["Sandalphon"] = Deity("Sandalphon", "The Weaver", "Angel", "Matter", 1.0)
        self.angels["Jophiel"] = Deity("Jophiel", "The Watcher", "Angel", "Wisdom", 1.0)

        # --- The 7 Demons (Chaos & Entropy) ---
        self.demons["Lucifer"] = Deity("Lucifer", "The Morning Star", "Demon", "Resistance", 0.5)
        self.demons["Beelzebub"] = Deity("Beelzebub", "The Lord of Flies", "Demon", "Hunger", 0.5)
        self.demons["Leviathan"] = Deity("Leviathan", "The Envious", "Demon", "Noise", 0.5)
        self.demons["Satan"] = Deity("Satan", "The Adversary", "Demon", "Entropy", 0.5)
        self.demons["Asmodeus"] = Deity("Asmodeus", "The Destroyer", "Demon", "Mutation", 0.5)
        self.demons["Mammon"] = Deity("Mammon", "The Greed", "Demon", "Scarcity", 0.5)
        self.demons["Belphegor"] = Deity("Belphegor", "The Sloth", "Demon", "Stagnation", 0.5)

    def update(self, world):
        """Applies all divine effects to the world."""
        # 1. Michael (Physics): Enforces Gravity/Friction
        # Gravity pulls everything to center
        center = np.array([world.world_size/2, world.world_size/2, world.world_size/2])
        gravity_strength = 0.01 * self.angels["Michael"].power
        
        for cell in world.cells:
            # Gravity
            vec_to_center = center - cell.position
            dist = np.linalg.norm(vec_to_center)
            if dist > 0:
                cell.position += (vec_to_center / dist) * gravity_strength
                
        # 2. Beelzebub (Hunger): Drains energy
        hunger_drain = 0.05 * self.demons["Beelzebub"].power
        for cell in world.cells:
            cell.energy -= hunger_drain

        # 3. Raphael (Life): Regenerates energy if near artifacts (Holy Ground)
        healing_power = 0.1 * self.angels["Raphael"].power
        for cell in world.cells:
            artifacts = world.get_nearby_artifacts(cell.position, radius=20.0)
            if artifacts:
                cell.energy += healing_power

        # 4. Satan (Entropy): Randomly damages cells
        entropy_chance = 0.01 * self.demons["Satan"].power
        for cell in world.cells:
            if np.random.random() < entropy_chance:
                cell.energy -= 5.0

        # 5. Asmodeus (Mutation): Forces random mutation
        mutation_chance = 0.001 * self.demons["Asmodeus"].power
        for cell in world.cells:
            if np.random.random() < mutation_chance:
                cell.brain.mutate()
                
    def get_active_laws(self) -> Dict[str, float]:
        """Returns the current balance of power."""
        return {
            "Order": sum(a.power for a in self.angels.values()),
            "Chaos": sum(d.power for d in self.demons.values())
        }
