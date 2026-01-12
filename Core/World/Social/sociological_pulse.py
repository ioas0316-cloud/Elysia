
import logging
import random
import math
from typing import List, Dict, Tuple
from Core.Foundation.Wave.wave_dna import WaveDNA
from Core.World.Physics.trinity_fields import TrinityVector
from Core.World.Soul.emotional_physics import emotional_physics

logger = logging.getLogger("SociologicalPulse")

class NPC:
    def __init__(self, id: str, name: str, temperament: WaveDNA):
        self.id = id
        self.name = name
        self.temperament = temperament # Base DNA
        self.emotional_frequency = temperament.frequency
        self.position = (random.uniform(-100, 100), random.uniform(-100, 100))
        self.energy = 100.0
        self.memory_impacts = {} # Dissonance/Resonance with others

    def radiate_aura(self) -> Tuple[float, float, float]:
        """Radiates (frequency, amplitude, range)."""
        # More energy = wider range. More emotion = higher amplitude.
        amplitude = self.temperament.phenomenal * (self.energy / 100.0)
        return (self.emotional_frequency, amplitude, 10.0 * amplitude)

class SociologicalPulse:
    """
    [Phase 31] Emotional Interaction Engine.
    Simulates how NPCs 'Feel' each other through wave interference.
    """
    def __init__(self):
        self.residents: Dict[str, NPC] = {}

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
