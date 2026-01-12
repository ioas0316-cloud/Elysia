
import logging
import random
import math
import numpy as np
from typing import List, Dict, Tuple, Optional
from Core.Foundation.Wave.wave_dna import WaveDNA
from Core.World.Physics.trinity_fields import TrinityVector
from Core.World.Soul.emotional_physics import emotional_physics

logger = logging.getLogger("SociologicalPulse")

class SocialField:
    """
    [Phase 33] The Potential Map of Meaning.
    Replaces pairwise distance checks with a vector field.
    """
    def __init__(self, size: int = 50, resolution: float = 4.0):
        self.size = size # Grid size (e.g., 50x50)
        self.resolution = resolution # Meters per cell
        # Grid stores (Gravity, Flow, Ascension, Frequency)
        self.grid = np.zeros((size, size, 4)) 
        
        # [Phase 34] Cultural & Historical Layers
        self.language_grid = np.zeros((size, size)) # Regional Dialect/Semantic Freq
        self.history_grid = np.zeros((size, size))  # Collective Memory/Event Scars
        
        self.boundary = size * resolution / 2.0

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        gx = int((x + self.boundary) / self.resolution)
        gy = int((y + self.boundary) / self.resolution)
        return max(0, min(self.size-1, gx)), max(0, min(self.size-1, gy))

    def deposit(self, x: float, y: float, vector: TrinityVector):
        gx, gy = self.world_to_grid(x, y)
        # Additive interference in the cell
        self.grid[gx, gy, 0] += vector.gravity
        self.grid[gx, gy, 1] += vector.flow
        self.grid[gx, gy, 2] += vector.ascension
        
        # Frequency is averaged in the cell
        if self.grid[gx, gy, 3] == 0:
            self.grid[gx, gy, 3] = vector.frequency
        else:
            self.grid[gx, gy, 3] = (self.grid[gx, gy, 3] + vector.frequency) / 2.0
            
        # Social Learning: Language frequency deposit
        # Residents 'Teach' the field their way of speaking
        self.language_grid[gx, gy] = (self.language_grid[gx, gy] + vector.frequency) / 2.0

    def deposit_event(self, x: float, y: float, intensity: float):
        """[Phase 34] Records a historical ripple (Birth/Death/Event) in the field."""
        gx, gy = self.world_to_grid(x, y)
        self.history_grid[gx, gy] += intensity

    def sample(self, x: float, y: float) -> Tuple[float, float, float, float]:
        gx, gy = self.world_to_grid(x, y)
        return tuple(self.grid[gx, gy])

    def sample_culture(self, x: float, y: float) -> Tuple[float, float]:
        """Returns (Language Freq, History Intensity)."""
        gx, gy = self.world_to_grid(x, y)
        return (self.language_grid[gx, gy], self.history_grid[gx, gy])

    def decay(self, rate: float = 0.9):
        """Natural entropy - waves and memories fade."""
        self.grid *= rate
        self.language_grid *= (rate + 0.05) # Language is stickier
        self.history_grid *= (rate + 0.08)  # History is the stickiest

class NPC:
    def __init__(self, id: str, name: str, temperament: WaveDNA, age: float = 20.0):
        self.id = id
        self.name = name
        self.temperament = temperament 
        self.emotional_frequency = temperament.frequency
        self.position = [random.uniform(-90, 90), random.uniform(-90, 90)]
        self.age = age
        self.health = 1.0
        self.energy = 100.0
        self.is_alive = True
        self.velocity = [0.0, 0.0]

    def update_biology(self, dt_years: float = 0.1):
        if not self.is_alive: return
        self.age += dt_years
        if self.age < 25: vitality = 1.0
        elif self.age < 60: vitality = 1.0 - (self.age - 25) * 0.005
        else: vitality = 0.875 - (self.age - 60) * 0.015
        self.health = min(self.health, vitality)
        if self.age >= 120 or self.health <= 0:
            self.is_alive = False

    def learn_culture(self, field: SocialField):
        """[Phase 34] Absorbs the local dialect and feels the history of the land."""
        if not self.is_alive: return
        lang_freq, history_mass = field.sample_culture(self.position[0], self.position[1])
        
        # 1. Linguistic Drift: Adjust my frequency toward the local average
        if lang_freq > 0:
            drift_rate = 0.05
            self.emotional_frequency = (1.0 - drift_rate) * self.emotional_frequency + drift_rate * lang_freq
            
        # 2. Historical Resonance: History mass increases 'Respect' or 'Aura stability'
        if history_mass > 0.5:
             self.energy = min(120.0, self.energy + 0.1) # Feeling the legacy of ancestors

    def move_by_gravity(self, field: SocialField):
        """NPCs drift toward resonant frequencies (Social Gravity)."""
        if not self.is_alive: return
        
        # [Phase 34] NPCs also learn as they move
        self.learn_culture(field)
        
        x, y = self.position
        offsets = [(1,0), (-1,0), (0,1), (0,-1)]
        best_resonance = -1.0
        best_move = [0, 0]
        
        # Sample surrounding to find gradient
        for dx, dy in offsets:
            nx, ny = x + dx*field.resolution, y + dy*field.resolution
            _, _, _, nfreq = field.sample(nx, ny)
            
            if nfreq == 0: resonance = 0.5 # Empty space is neutral
            else:
                # Resonance = 1 / (1 + deltaFreq)
                resonance = 1.0 / (1.0 + abs(nfreq - self.emotional_frequency))
            
            if resonance > best_resonance:
                best_resonance = resonance
                best_move = [dx, dy]

        # Apply movement
        speed = 2.0 * self.health
        self.velocity = [best_move[0] * speed, best_move[1] * speed]
        self.position[0] += self.velocity[0]
        self.position[1] += self.velocity[1]
        
        # Constrain to boundary
        self.position[0] = max(-95, min(95, self.position[0]))
        self.position[1] = max(-95, min(95, self.position[1]))

    def radiate_aura(self) -> TrinityVector:
        return TrinityVector(
            self.temperament.physical,
            self.temperament.phenomenal * (self.energy/100.0),
            self.temperament.spiritual,
            frequency=self.emotional_frequency
        )

class SociologicalPulse:
    def __init__(self, field_size: int = 50):
        self.residents: Dict[str, NPC] = {}
        self.field = SocialField(size=field_size)

    def add_resident(self, npc: NPC):
        self.residents[npc.id] = npc

    def update_social_field(self):
        """Field-based interaction: O(N) complexity."""
        self.field.decay(rate=0.7)
        
        # 1. Deposit
        for npc in self.residents.values():
            if npc.is_alive:
                self.field.deposit(npc.position[0], npc.position[1], npc.radiate_aura())
        
        # 2. React
        for npc in self.residents.values():
            npc.move_by_gravity(self.field)
            
            # Interaction feedback
            _, _, _, local_freq = self.field.sample(npc.position[0], npc.position[1])
            diff = abs(local_freq - npc.emotional_frequency)
            if diff < 10.0: npc.energy = min(110.0, npc.energy + 1.0) # Harmonic bond
            elif diff > 400.0: npc.energy -= 2.0 # Tension

    def age_step(self, dt_years: float = 1.0):
        dead_ids = []
        new_residents = []
        for npc_id, npc in list(self.residents.items()):
            npc.update_biology(dt_years)
            if not npc.is_alive: 
                dead_ids.append(npc_id)
                # [Phase 34] Death leaves a ripple in history
                self.field.deposit_event(npc.position[0], npc.position[1], intensity=5.0)
            elif 20 < npc.age < 50 and npc.energy > 105:
                child = self.reproduce_solo(npc)
                if child: 
                    new_residents.append(child)
                    # [Phase 34] Birth leaves a ripple in history
                    self.field.deposit_event(npc.position[0], npc.position[1], intensity=3.0)
        
        for d_id in dead_ids: del self.residents[d_id]
        for baby in new_residents: self.residents[baby.id] = baby

    def reproduce_solo(self, parent: NPC) -> Optional[NPC]:
        if random.random() > 0.05: return None
        child_dna = parent.temperament
        
        # [Phase 34] Child inherits the LOCAL dialect frequency from the parent's current state
        baby = NPC(f"B{random.randint(1000, 9999)}", f"Child_{parent.name[:3]}", child_dna, age=0.0)
        baby.emotional_frequency = parent.emotional_frequency # Inherited dialect
        baby.position = [parent.position[0] + random.uniform(-2, 2), parent.position[1] + random.uniform(-2, 2)]
        return baby

    def get_community_vibe(self) -> str:
        # Global grid average frequency
        active_cells = self.field.grid[:,:,3][self.field.grid[:,:,3] > 0]
        if active_cells.size == 0: return "Void"
        avg_freq = np.mean(active_cells)
        return emotional_physics.resolve_emotion(avg_freq)

_pulse = None
def get_sociological_pulse():
    global _pulse
    if _pulse is None:
        _pulse = SociologicalPulse()
    return _pulse
