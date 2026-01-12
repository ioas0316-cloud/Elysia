
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
    [Phase 36] The OmniField: Unified Potential Map.
    Carrier for all phenomena within the HyperSphere's spatial projection.
    
    Structure (25 Channels):
    - [0-3]: Social/Carrier (G, F, A, Freq)
    - [4-7]: Tactile/Collision (G, F, A, Freq)
    - [8-11]: Acoustic/Sense Layer
    - [12-15]: Linguistic Layer (Cultural Drift)
    - [16-19]: Terrain (Hardness, Elasticity, Elevation, MaterialID)
    - [20-23]: Resource (Value, Energy, Nutrition, ResourceType)
    - [24]: History Mass (Sovereign Scars)
    """
    def __init__(self, size: int = 50, resolution: float = 4.0):
        self.size = size 
        self.resolution = resolution 
        # 25 channels: 6 Trinity Slots (24) + 1 History Slot (24)
        # Slots: 0=Social, 4=Haptic, 8=Acoustic, 12=Linguistic, 16=Terrain, 20=Resource, 24=History
        self.grid = np.zeros((size, size, 25)) 
        self.boundary = size * resolution / 2.0

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        gx = int((x + self.boundary) / self.resolution)
        gy = int((y + self.boundary) / self.resolution)
        return max(0, min(self.size-1, gx)), max(0, min(self.size-1, gy))

    def deposit(self, x: float, y: float, vector: TrinityVector, channel_offset: int = 0):
        """Deposits a 4-component Wave (G, F, A, Freq) starting at channel_offset."""
        gx, gy = self.world_to_grid(x, y)
        self.grid[gx, gy, 0 + channel_offset] += vector.gravity
        self.grid[gx, gy, 1 + channel_offset] += vector.flow
        self.grid[gx, gy, 2 + channel_offset] += vector.ascension
        
        # Frequency handling
        curr_freq = self.grid[gx, gy, 3 + channel_offset]
        if curr_freq == 0:
            self.grid[gx, gy, 3 + channel_offset] = vector.frequency
        else:
            self.grid[gx, gy, 3 + channel_offset] = (curr_freq + vector.frequency) / 2.0

    def deposit_event(self, x: float, y: float, intensity: float):
        """[Phase 34] Records a historical ripple in the field (Channel 24)."""
        gx, gy = self.world_to_grid(x, y)
        self.grid[gx, gy, 24] += intensity

    def sample(self, x: float, y: float, channel_offset: int = 0) -> Tuple[float, float, float, float]:
        gx, gy = self.world_to_grid(x, y)
        return tuple(self.grid[gx, gy, channel_offset:channel_offset+4])

    def sample_history(self, x: float, y: float) -> float:
        gx, gy = self.world_to_grid(x, y)
        return self.grid[gx, gy, 24]

    def decay(self, rate: float = 0.9):
        """Natural entropy - waves and memories fade."""
        # 0-15: Social, Haptic, Acoustic, Linguistic (Transient)
        self.grid[:, :, 0:16] *= rate 
        # 16-23: Terrain and Resources (Semi-persistent)
        self.grid[:, :, 16:24] *= min(1.0, rate + 0.15) 
        # 24: History (Stickiest)
        self.grid[:, :, 24] *= min(1.0, rate + 0.08)

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
        
        # Channel 12: Linguistic/Social
        _, _, _, lang_freq = field.sample(self.position[0], self.position[1], channel_offset=12)
        # Channel 13: History
        history_mass = field.sample_history(self.position[0], self.position[1])
        
        # 1. Linguistic Drift
        if lang_freq > 0:
            drift_rate = 0.05
            self.emotional_frequency = (1.0 - drift_rate) * self.emotional_frequency + drift_rate * lang_freq
            
        # 2. Historical Resonance
        if history_mass > 0.5:
             self.energy = min(120.0, self.energy + 0.1)

    def feel_environment(self, field: SocialField):
        """
        [Phase 36] The Act of Living.
        Samples Senses, Terrain, and Resources.
        """
        if not self.is_alive: return
        
        # 1. Metabolism: Constant effort to exists
        self.energy -= 1.5 
        
        # 2. Learn Culture (Channel 12 & 24)
        self.learn_culture(field)
        
        # 3. Terrain & Resource Sampling
        # Channel 16: Terrain (G, F, Elevation, Freq)
        ter_g, ter_f, ter_elev, ter_freq = field.sample(self.position[0], self.position[1], channel_offset=16)
        # Channel 20: Resource (Value, Energy, Nutrition, Type)
        res_g, res_f, res_nut, res_type = field.sample(self.position[0], self.position[1], channel_offset=20)
        
        # 4. Metabolic Intake: Consume local resource if available
        if res_f > 0.1:
            intake = min(res_f, 5.0) 
            self.energy = min(120.0, self.energy + intake * 10.0) # Potent resources
            
        # 5. Tactile/Social Interference
        soc_g, soc_f, soc_a, soc_freq = field.sample(self.position[0], self.position[1], channel_offset=0)
        tac_g, tac_f, tac_a, tac_freq = field.sample(self.position[0], self.position[1], channel_offset=4)
        
        for f in [soc_freq, tac_freq]:
            if f == 0: continue
            diff = abs(f - self.emotional_frequency)
            if diff < 20.0:
                self.energy = min(120.0, self.energy + 0.5)
            elif diff > 400.0:
                self.energy -= 1.0
                if random.random() < 0.01: self.health -= 0.001

        # 6. Terrain Impact: Elevation (ter_elev) affects speed/health
        if ter_elev > 0.8: # Steep slope
             self.energy -= 0.2
             
    def move_by_gravity(self, field: SocialField):
        """NPCs drift toward resonant frequencies (Social Gravity)."""
        if not self.is_alive: return
        
        self.feel_environment(field)
        
        x, y = self.position
        offsets = [(1,0), (-1,0), (0,1), (0,-1)]
        best_resonance = -1.0
        best_move = [0, 0]
        
        # Sample surrounding (Channel 0)
        for dx, dy in offsets:
            nx, ny = x + dx*field.resolution, y + dy*field.resolution
            _, _, _, nfreq = field.sample(nx, ny, channel_offset=0)
            
            if nfreq == 0: resonance = 0.5 
            else:
                resonance = 1.0 / (1.0 + abs(nfreq - self.emotional_frequency))
            
            # [Phase 36] Also avoid steep terrain? (Optional for now)
            if resonance > best_resonance:
                best_resonance = resonance
                best_move = [dx, dy]

        # Apply movement (Speed affected by health and terrain elevation)
        # Sample elevation at current spot for speed penalty
        _, _, ter_elev, _ = field.sample(x, y, channel_offset=16)
        speed_mod = 1.0 - (ter_elev * 0.5) # Up to 50% penalty
        
        speed = 2.0 * self.health * speed_mod
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
        
        # [Phase 36] Initial World Seeding
        # 1. Terrain: A 'Mountain' in the middle (Channel 16)
        mountain = TrinityVector(gravity=0.9, flow=0.1, ascension=1.0, frequency=700.0)
        self.field.deposit(0, 0, mountain, channel_offset=16)
        
        # 2. Resources: 'Forests' of energy (Channel 20)
        forest = TrinityVector(gravity=0.5, flow=1.0, ascension=0.2, frequency=432.0)
        self.field.deposit(-20, -20, forest, channel_offset=20)
        self.field.deposit(20, 20, forest, channel_offset=20)

    def add_resident(self, npc: NPC):
        self.residents[npc.id] = npc

    def update_social_field(self):
        """Field-based interaction: O(N) complexity."""
        self.field.decay(rate=0.7)
        
        # 1. Deposit (Social & Sensory)
        for npc in self.residents.values():
            if npc.is_alive:
                aura = npc.radiate_aura()
                # Social Deposit (Carrier)
                self.field.deposit(npc.position[0], npc.position[1], aura, channel_offset=0)
                # Tactile Deposit (Excitation)
                self.field.deposit(npc.position[0], npc.position[1], aura, channel_offset=4)
                # Linguistic/Symbolic
                self.field.deposit(npc.position[0], npc.position[1], aura, channel_offset=12)
        
        # 2. React (Sensing + Movement)
        for npc in self.residents.values():
            npc.move_by_gravity(self.field)

    def age_step(self, dt_years: float = 1.0):
        dead_ids = []
        new_residents = []
        for npc_id, npc in list(self.residents.items()):
            npc.update_biology(dt_years)
            if not npc.is_alive: 
                dead_ids.append(npc_id)
                # [Phase 34] Death leaves a ripple in history (Channel 24)
                self.field.deposit_event(npc.position[0], npc.position[1], intensity=5.0)
            elif 20 < npc.age < 50 and npc.energy > 105:
                child = self.reproduce_solo(npc)
                if child: 
                    new_residents.append(child)
                    # [Phase 34] Birth leaves a ripple in history (Channel 24)
                    self.field.deposit_event(npc.position[0], npc.position[1], intensity=3.0)
        
        for d_id in dead_ids: del self.residents[d_id]
        for baby in new_residents: self.residents[baby.id] = baby

    def reproduce_solo(self, parent: NPC) -> Optional[NPC]:
        if random.random() > 0.05: return None
        child_dna = parent.temperament
        
        # [Phase 34] Child inherits the LOCAL dialect frequency
        baby = NPC(f"B{random.randint(1000, 9999)}", f"Child_{parent.name[:3]}", child_dna, age=0.0)
        baby.emotional_frequency = parent.emotional_frequency 
        baby.position = [parent.position[0] + random.uniform(-2, 2), parent.position[1] + random.uniform(-2, 2)]
        return baby

    def get_community_vibe(self) -> str:
        # Global grid average frequency on Channel 0
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
