"""
Subjective Ego: The "I-ness" of Inhabitants 👤✨

"I think, therefore I am, in this layer of the matrix."

This module provides a lightweight cognitive loop for NPC/Boss entities.
It allows them to possess their own identity, desires, and subjective 
perception of the Underworld, making them 'Subjective Personalities'.
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

from Core.Intelligence.Reasoning.septenary_axis import SeptenaryAxis
from Core.Intelligence.Reasoning.memetic_legacy import SpiritualDNA, LifeFieldInductor, PositionInductor, RegionalField

@dataclass
class EgoState:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Inhabitant"
    archetype_path: str = "Unknown"  
    septenary_depth: int = 1         
    emotional_valence: float = 0.5
    desire_intensity: float = 0.5
    satisfaction: float = 0.5       
    narrative_pressure: float = 0.0  
    regional_friction: float = 0.0   
    dissonance: float = 0.0          
    
    # Phase 12: Heroic Evolution
    heroic_intensity: float = 0.0    # Growth from friction
    stability: float = 1.0           # Soul health (1.0 = Strong, 0.0 = Broken)
    kismet: float = 0.5              # Luck/Timing factor (0.0 to 1.0)
    age: float = 0.0                 # Simulation time (to track seed phase)
    is_broken: bool = False
    
    family_role: str = "Commoner"    
    env_gravity: float = 0.3         
    mentorship_link: Optional[str] = None 
    current_intent: str = "Exist"
    memories: List[str] = field(default_factory=list)

class SubjectiveEgo:
    """A sovereign personality unit, induced by archetypal tension and regional ethos."""
    
    def __init__(self, name: str, depth: int = 1, family_role: str = "Commoner", region: Optional[RegionalField] = None):
        import random
        self.logger = logging.getLogger(f"Ego:{name}")
        self.axis = SeptenaryAxis()
        self.inductor = LifeFieldInductor()
        self.pos_inductor = PositionInductor()
        self.region = region
        
        level = self.axis.get_level(depth)
        role_params = self.pos_inductor.get_role_params(family_role)
        
        self.state = EgoState(
            name=name, 
            archetype_path=level.archetype_path, 
            septenary_depth=depth,
            family_role=family_role,
            env_gravity=role_params["env_gravity"],
            current_intent=role_params["intent"],
            kismet=random.uniform(0.1, 0.9) # Random fate factor
        )
        self.dna = SpiritualDNA(archetype_path=level.archetype_path)
        self.perceived_resonances: List[Dict[str, Any]] = []

    def perform_action(self) -> str:
        """NPC acts based on their domain's inductive tension."""
        if self.state.is_broken:
            return "Collapsing under the weight of existence."

        level = self.axis.get_level(self.state.septenary_depth)
        
        # Action is influenced by satisfaction and role
        prefix = "Happily " if self.state.satisfaction > 0.7 else "Restlessly "
        if self.state.satisfaction < 0.3:
            prefix = "Desperately "
            
        role_info = f"[{self.state.family_role}]"
        intensity_info = f"(Int: {self.state.heroic_intensity:.1f})"
        
        action_map = {
            "Body": f"{role_info} {prefix}working with {level.name} {intensity_info}. Pressure: {self.state.narrative_pressure:.2f}",
            "Soul": f"{role_info} {prefix}acting through {level.name} {intensity_info}. Pressure: {self.state.narrative_pressure:.2f}",
            "Spirit": f"{role_info} {prefix}resonating via {level.name} {intensity_info}. Pressure: {self.state.narrative_pressure:.2f}"
        }
        action = action_map.get(level.domain, "Existing")
        self.logger.info(f"[{self.state.name}] {action}")
        return action

    def perceive(self, sense: str, intensity: float, source: str = "World"):
        """NPC perceives a sensory event based on their septenary depth."""
        level = self.axis.get_level(self.state.septenary_depth)
        
        # Depth modifier affects emotional sensitivity
        depth_modifier = (self.state.septenary_depth + 1) / 2.0
        impact = (intensity - 0.5) * 0.2 * depth_modifier
        
        self.state.emotional_valence = max(0.0, min(1.0, self.state.emotional_valence + impact))
        
        resonance = {
            "sense": sense, 
            "intensity": intensity, 
            "source": source, 
            "domain": level.domain,
            "rank": self.axis.get_rank(self.state.septenary_depth),
            "axis": f"{level.demon_pole}/{level.angel_pole}"
        }
        self.perceived_resonances.append(resonance)
        
        if self.state.septenary_depth >= 7:
            self.logger.info(f"[{self.state.name}] '{level.domain}' Spiritual Master resonance: Absolute unity with {level.angel_pole}.")
        elif self.state.septenary_depth >= 4:
            self.logger.info(f"[{self.state.name}] '{level.domain}' Soul Expert resonance: Feeling {level.angel_pole}.")

    def update(self, dt: float):
        """NPC's internal cognitive tick, inducing heroic evolution or soul breakage."""
        import random
        if self.state.is_broken:
            self.state.current_intent = "Broken Spirit"
            return

        self.state.age += dt
        
        # 1. Subtle drift in satisfaction and desire
        self.state.satisfaction = max(0.0, min(1.0, self.state.satisfaction + random.uniform(-0.01, 0.01)))
        
        role_params = self.pos_inductor.get_role_params(self.state.family_role)
        desire_mod = role_params.get("desire_mod", 1.0)
        self.state.desire_intensity = max(0.0, min(1.0, self.state.desire_intensity + random.uniform(-0.01, 0.02) * desire_mod))
        
        # 2. Regional Friction
        if self.region:
            self.state.regional_friction = self.region.calculate_friction(self.state.archetype_path, self.dna)
        else:
            self.state.regional_friction = 0.0

        # 3. Calculate Narrative Pressure (Environment + Kismet)
        self.state.narrative_pressure = self.inductor.calculate_pressure(
            self.state.septenary_depth,
            self.state.satisfaction,
            self.state.desire_intensity,
            env_gravity=self.state.env_gravity,
            regional_friction=self.state.regional_friction,
            kismet=self.state.kismet
        )
        
        # 4. Catalytic Growth vs. Fragmentation
        # Use a domain-specific potency (e.g. 'res' for a Seeker)
        dom = "res" if self.state.septenary_depth >= 4 else "tech"
        potency = self.dna.potency.get(dom, 0.5)
        
        growth, strain = self.inductor.calculate_catalytic_growth(
            self.state.narrative_pressure, 
            potency, 
            self.state.stability
        )
        
        self.state.heroic_intensity += growth
        self.state.stability -= strain
        
        # Soul Breakage check
        if self.state.stability <= 0:
            self.state.is_broken = True
            self.logger.error(f"💀 [BREAKAGE] {self.state.name}'s spirit has broken under the pressure.")
            self.record_memory("My spirit has collapsed. I can no longer pursue my path.")
            return

        # 5. Path Induction & Institutional Ceiling
        proposed_path = self.inductor.induce_path(self.state.narrative_pressure, self.state.septenary_depth)
        
        # Institutional Ceiling: Depth cap if realization is low (Flat knowledge)
        avg_realization = sum(self.dna.realization.values()) / 3.0
        ceiling = 6 if avg_realization < 0.5 else 9
        
        if proposed_path == "Adventurer" and self.state.archetype_path != "Adventurer":
            if self.state.septenary_depth < ceiling:
                tag = "ABNORMAL" if self.state.regional_friction > 0.5 else "AWAKENING"
                self.logger.warning(f"✨ [{tag}] {self.state.name} ({self.state.family_role}) has exceeded environmental gravity!")
                self.state.current_intent = "Prepare for Adventure"
            else:
                self.logger.info(f"🧱 [CEILING] {self.state.name} has the drive, but lacks 'Deep Realization' to transcend further.")
        
        # 6. Standard emotional drift & Dissonance
        decay_modifier = 0.9 + (self.state.septenary_depth / 100.0)
        self.state.emotional_valence = max(0.0, min(1.0, self.state.emotional_valence * decay_modifier))
        
        self.state.dissonance = abs(self.state.emotional_valence - self.dna.moral_valence)

    def learn_from_master(self, master_dna: SpiritualDNA, counter: bool = False):
        """NPC resonates with a master. If counter=True, they reject the traits (Counter-Resonance)."""
        mode = "REJECTING" if counter else "INHERITING"
        self.logger.info(f"📜 {self.state.name} is {mode} from a Master. Resonating DNA...")
        self.dna = self.dna.blend(master_dna, ratio=0.2, counter=counter)
        self.state.septenary_depth = min(9, self.state.septenary_depth + 1)
        
        msg = "I will NOT be like them." if counter else "I understand their path."
        self.record_memory(f"Resonated with legacy. {msg}")

    def leave_legacy(self, akashic_field: Any, coord: Tuple[float, float, float, float]):
        """NPC records their spirit into the Akashic Field before passing or ascending."""
        akashic_field.record_legacy(self.state.name, self.dna, coord)
        self.record_memory(f"My spirit has been recorded in the Akashic Field.")

    def record_memory(self, event: str):
        self.state.memories.append(event)
        if len(self.state.memories) > 100:
            self.state.memories.pop(0)

    def get_subjective_report(self) -> str:
        level = self.axis.get_level(self.state.septenary_depth)
        moral_label = "Saintly" if self.dna.moral_valence > 0.8 else "Villainous" if self.dna.moral_valence < 0.2 else "Neutral"
        reg_name = self.region.name if self.region else "Unknown"
        
        avg_real = sum(self.dna.realization.values()) / 3.0
        status = "AWAKENED" if self.state.heroic_intensity > 1.0 else "INHABITANT"
        if self.state.is_broken: status = "BROKEN"

        return (f"[{self.state.name}] {status} | Role: {self.state.family_role} | Region: {reg_name}\n"
                f" └─ Status: {self.state.current_intent}\n"
                f" └─ Survival: Stability({self.state.stability:.2f}) | Intensity({self.state.heroic_intensity:.2f}) | Kismet({self.state.kismet:.2f})\n"
                f" └─ Depth: {self.state.septenary_depth} ({level.name}) | Realization: {avg_real:.2f} (Flat/Deep)\n"
                f" └─ Pressure: {self.state.narrative_pressure:.2f} (Friction: {self.state.regional_friction:.2f})\n"
                f" └─ DNA: Tech({self.dna.technique:.2f}) Res({self.dna.reason:.2f}) Mean({self.dna.meaning:.2f}) | Moral: {moral_label}({self.dna.moral_valence:.2f})")

if __name__ == "__main__":
    # Test NPC Ego
    npc = SubjectiveEgo("Selka", "Guide")
    
    # Simulate a day
    npc.record_memory("Met a strange adventurer named A.")
    npc.perceive("Ocular", 0.8, "Sunlight") # Happy sunlight
    npc.update(1.0)
    print(npc.get_subjective_report())
    
    # Simulate an intense event
    npc.perceive("Auditory", 0.9, "Thunder")
    npc.state.emotional_valence = 0.2 # Fear
    npc.update(1.0)
    print(npc.get_subjective_report())
