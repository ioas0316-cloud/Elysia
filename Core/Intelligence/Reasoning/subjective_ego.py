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
from typing import Dict, List, Any, Optional

from Core.Intelligence.Reasoning.septenary_axis import SeptenaryAxis

@dataclass
class EgoState:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Inhabitant"
    archetype_path: str = "Unknown"  # Body/Soul/Spirit mapped path
    septenary_depth: int = 1         # 1 to 9
    emotional_valence: float = 0.5
    desire_intensity: float = 0.5
    current_intent: str = "Exist"
    memories: List[str] = field(default_factory=list)

class SubjectiveEgo:
    """A sovereign personality unit, induced by archetypal tension."""
    
    def __init__(self, name: str, depth: int = 1):
        self.logger = logging.getLogger(f"Ego:{name}")
        self.axis = SeptenaryAxis()
        level = self.axis.get_level(depth)
        self.state = EgoState(
            name=name, 
            archetype_path=level.archetype_path, 
            septenary_depth=depth
        )
        self.perceived_resonances: List[Dict[str, Any]] = []

    def perform_action(self) -> str:
        """NPC acts based on their domain's inductive tension."""
        level = self.axis.get_level(self.state.septenary_depth)
        action_map = {
            "Body": f"Working with {level.name} (Warrior's Path/Blacksmith). Gravity of {level.inductive_logic}",
            "Soul": f"Acting through {level.name} (Mage/Merchant/Ruler). Tension of {level.inductive_logic}",
            "Spirit": f"Resonating via {level.name} (Paladin/Priest). Light of {level.inductive_logic}"
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
        """NPC's internal cognitive tick."""
        # Simple desire decay/growth
        self.state.desire_intensity *= 0.99
        
        if self.state.emotional_valence < 0.3:
            self.state.current_intent = "Seek Comfort"
        elif self.state.emotional_valence > 0.7:
            self.state.current_intent = "Create Joy"
        else:
            self.state.current_intent = "Maintain State"

    def record_memory(self, event: str):
        self.state.memories.append(event)
        if len(self.state.memories) > 100:
            self.state.memories.pop(0) # Simple memory limit

    def get_subjective_report(self) -> str:
        return (f"[{self.state.name} ({self.state.archetype})] "
                f"Status: {self.state.current_intent} | "
                f"Mood: {self.state.emotional_valence:.2f} | "
                f"Last Memory: {self.state.memories[-1] if self.state.memories else 'None'}")

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
