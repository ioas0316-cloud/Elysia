
import logging
import random
import math
from typing import List, Dict, Tuple, Optional
from Core.Foundation.Wave.wave_dna import WaveDNA
from Core.World.Physics.trinity_fields import TrinityVector
from Core.World.Soul.emotional_physics import emotional_physics

logger = logging.getLogger("SociologicalPulse")

class NPC:
    def __init__(self, id: str, name: str, temperament: WaveDNA, age: float = 20.0):
        self.id = id
        self.name = name
        self.temperament = temperament # Base DNA
        self.emotional_frequency = temperament.frequency
        self.position = (random.uniform(-100, 100), random.uniform(-100, 100))
        
        # [Phase 32] Biological Stats
        self.age = age # Years
        self.health = 1.0 # 0.0 to 1.0
        self.energy = 100.0
        self.is_alive = True
        self.memory_impacts = {} # Dissonance/Resonance with others

    def update_biology(self, dt_years: float = 0.1):
        """Aging and Health decay."""
        if not self.is_alive: return
        
        self.age += dt_years
        
        # Aging Curve (Vitality)
        # 0-25: Growing, Health is stable
        # 25-60: Peak, slow decay
        # 60-120: Sharp decay
        
        if self.age < 25:
            vitality = 1.0
        elif self.age < 60:
            vitality = 1.0 - (self.age - 25) * 0.005 # 25 years -> -0.125
        else:
            vitality = 0.875 - (self.age - 60) * 0.015 # 60 years -> -0.9
            
        # Actual health index
        self.health = min(self.health, vitality)
        
        # Random sickness based on health (Resistance decreases with age)
        if random.random() < (1.0 - self.health) * 0.05:
            self.health -= 0.05
            logger.warning(f"💊 {self.name} is feeling unwell (Age: {self.age:.1f}, Health: {self.health:.2f})")
            
        # Death logic
        if self.age >= 120 or self.health <= 0:
            self.is_alive = False
            logger.critical(f"💀 {self.name} has passed away at age {self.age:.1f}. (Cause: {'Old Age' if self.age >= 120 else 'Sickness'})")

    def radiate_aura(self) -> Tuple[float, float, float]:
        """Radiates (frequency, amplitude, range)."""
        if not self.is_alive: return (0, 0, 0)
        # More energy = wider range. More emotion = higher amplitude.
        # Health and Age impact the aura strength
        amplitude = self.temperament.phenomenal * (self.energy / 100.0) * self.health
        return (self.emotional_frequency, amplitude, 10.0 * amplitude)

class SociologicalPulse:
    """
    [Phase 31] Emotional Interaction Engine.
    Simulates how NPCs 'Feel' each other through wave interference.
    """
    def __init__(self):
        self.residents: Dict[str, NPC] = {}
        self.population_history = []

    def age_step(self, dt_years: float = 1.0):
        """Ticks the entire world's age."""
        dead_ids = []
        new_residents = []
        
        for npc_id, npc in list(self.residents.items()):
            npc.update_biology(dt_years)
            if not npc.is_alive:
                dead_ids.append(npc_id)
                continue
            
            # Chance to reproduce if healthy and adult
            if npc.is_alive and 20 <= npc.age <= 50 and npc.health > 0.8:
                # Find a partner nearby
                partner = self._find_nearby_partner(npc)
                if partner:
                    child = self.reproduce(npc, partner)
                    if child:
                        new_residents.append(child)
        
        # Clean up the dead
        for d_id in dead_ids:
            del self.residents[d_id]
            
        # Add the newborns
        for baby in new_residents:
            self.residents[baby.id] = baby

    def _find_nearby_partner(self, parent: NPC) -> Optional[NPC]:
        for p in self.residents.values():
            if p.id == parent.id or not p.is_alive: continue
            if not (20 <= p.age <= 50): continue
            
            dist = math.sqrt((parent.position[0]-p.position[0])**2 + (parent.position[1]-p.position[1])**2)
            if dist < 10.0: # Interaction radius
                return p
        return None

    def reproduce(self, a: NPC, b: NPC) -> Optional[NPC]:
        """Combines DNA to create a new soul."""
        if random.random() > 0.1: return None # Birth rate / Luck
        
        # Wave DNA Recombination
        child_label = f"Descendant of {a.name} & {b.name}"
        
        child_traits = {}
        for trait in ['physical', 'functional', 'phenomenal', 'causal', 'mental', 'structural', 'spiritual']:
            val = (getattr(a.temperament, trait) + getattr(b.temperament, trait)) / 2.0
            val += random.uniform(-0.05, 0.05) # Minor mutation
            child_traits[trait] = max(0.0, min(1.0, val))
            
        child_freq = (a.emotional_frequency + b.emotional_frequency) / 2.0 + random.uniform(-5, 5)
        
        child_dna = WaveDNA(label=child_label, frequency=child_freq, **child_traits)
        child_dna.normalize()
        
        baby = NPC(f"B{random.randint(1000, 9999)}", f"Baby_{a.name[:3]}{b.name[:3]}", child_dna, age=0.0)
        baby.position = (a.position[0] + random.uniform(-1, 1), a.position[1] + random.uniform(-1, 1))
        
        logger.info(f"👶 [BIRTH] A new soul is born: {baby.name}. Welcome to Elysia.")
        return baby

    def add_resident(self, npc: NPC):
        self.residents[npc.id] = npc

    def update_social_field(self):
        """Calculates interference between all residents."""
        ids = list(self.residents.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a = self.residents[ids[i]]
                b = self.residents[ids[j]]
                
                # Check distance
                dist = math.sqrt((a.position[0]-b.position[0])**2 + (a.position[1]-b.position[1])**2)
                
                aura_a = a.radiate_aura()
                aura_b = b.radiate_aura()
                
                if dist < (aura_a[2] + aura_b[2]):
                    self._interact(a, b, dist)

    def _interact(self, a: NPC, b: NPC, distance: float):
        """Wave interference between two NPCs."""
        # 1. Frequency Alignment
        # Simulating Resonance (Constructive) or Dissonance (Destructive)
        diff = abs(a.emotional_frequency - b.emotional_frequency)
        
        # If frequencies are close (Resonance)
        if diff < 50.0:
            # Constructive: They like each other
            a.energy = min(110.0, a.energy + 1.0)
            b.energy = min(110.0, b.energy + 1.0)
            logger.info(f"💕 [RESONANCE] {a.name} and {b.name} are in sync. Harmony rising.")
        elif diff > 300.0:
            # Destructive: They clash
            a.energy -= 2.0
            b.energy -= 2.0
            logger.warning(f"💢 [DISSONANCE] {a.name} and {b.name} are clashing. Conflict detected.")
        else:
            # Neutral observation
            pass

    def get_community_vibe(self) -> str:
        if not self.residents: return "Empty"
        avg_freq = sum(n.emotional_frequency for n in self.residents.values()) / len(self.residents)
        return emotional_physics.resolve_emotion(avg_freq)

_pulse = None
def get_sociological_pulse():
    global _pulse
    if _pulse is None:
        _pulse = SociologicalPulse()
    return _pulse
