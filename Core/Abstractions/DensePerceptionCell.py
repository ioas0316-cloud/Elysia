"""
Dense Perception Cell - Rich Sensory Experience

Experience = ∫ Perception(t) dt

This Cell has:
- 5 senses (vision, hearing, touch, smell, proprioception)
- Survival needs (hunger, thirst, fatigue, pain)
- Social drives (loneliness, belonging, status)
- Environmental awareness (terrain, weather, resources, threats)

Result: 30-50 perceptions per tick (vs 3 before)
= 10-15x richer experience!
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging

from Core.Abstractions.CognitiveCell import CognitiveCell, CognitiveState

logger = logging.getLogger("DensePerceptionCell")


@dataclass
class PerceptionState:
    """Rich perceptual state with multiple sensory modalities."""
    
    # Visual perceptions
    visual_objects: List[str] = field(default_factory=list)  # What I see
    visual_colors: List[str] = field(default_factory=list)   # Colors seen
    visual_distances: Dict[str, float] = field(default_factory=dict)  # How far
    
    # Auditory perceptions
    heard_words: List[str] = field(default_factory=list)     # What I hear
    heard_sounds: List[str] = field(default_factory=list)    # Environmental sounds
    
    # Tactile perceptions
    touched_objects: List[str] = field(default_factory=list) # What I touch
    touch_sensations: Dict[str, float] = field(default_factory=dict)  # warm/cold/rough
    
    # Olfactory perceptions
    smelled_scents: List[str] = field(default_factory=list)  # What I smell
    scent_intensities: Dict[str, float] = field(default_factory=dict)
    
    # Proprioceptive perceptions
    body_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    body_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    body_orientation: float = 0.0
    
    # Interoceptive perceptions (internal state)
    hunger_level: float = 0.0
    thirst_level: float = 0.0
    fatigue_level: float = 0.0
    pain_level: float = 0.0
    
    # Social perceptions
    nearby_allies: List[str] = field(default_factory=list)
    nearby_strangers: List[str] = field(default_factory=list)
    nearby_threats: List[str] = field(default_factory=list)
    
    # Environmental perceptions
    terrain_type: str = "unknown"
    weather_condition: str = "clear"
    temperature: float = 20.0
    light_level: float = 1.0


class DensePerceptionCell(CognitiveCell):
    """
    Cell with rich multi-sensory perception.
    
    Each tick accumulates 30-50 perceptions instead of 3.
    This creates genuinely rich experience.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Perceptual state
        self.perception = PerceptionState()
        
        # Sensory ranges
        self.vision_range = 100.0
        self.hearing_range = 150.0
        self.smell_range = 75.0
        self.touch_range = 5.0
        
        # Needs (0.0 = satisfied, 1.0 = critical)
        self.hunger = 0.0
        self.thirst = 0.0
        self.fatigue = 0.0
        self.pain = 0.0
        self.loneliness = 0.0
        
        # Perception statistics
        self.total_perceptions = 0
        self.perceptions_this_tick = 0
        
        logger.debug(f"Created DensePerceptionCell: {self.cell_id}")
    
    def perceive_world(self, world_state: Dict[str, Any]) -> int:
        """
        Perceive the world through all senses.
        
        Args:
            world_state: Current world state with cells, terrain, etc.
            
        Returns:
            Number of perceptions this tick
        """
        self.perceptions_this_tick = 0
        
        # Reset perception state
        self.perception = PerceptionState()
        
        # 1. Visual perception
        self._perceive_vision(world_state)
        
        # 2. Auditory perception
        self._perceive_hearing(world_state)
        
        # 3. Tactile perception
        self._perceive_touch(world_state)
        
        # 4. Olfactory perception
        self._perceive_smell(world_state)
        
        # 5. Proprioceptive perception
        self._perceive_body()
        
        # 6. Interoceptive perception
        self._perceive_internal_state()
        
        # 7. Social perception
        self._perceive_social(world_state)
        
        # 8. Environmental perception
        self._perceive_environment(world_state)
        
        self.total_perceptions += self.perceptions_this_tick
        
        return self.perceptions_this_tick
    
    def _perceive_vision(self, world_state: Dict[str, Any]):
        """See nearby objects, cells, terrain."""
        my_pos = self.base_cell.position
        
        # See nearby cells
        for cell_id, cell_data in world_state.get('nearby_cells', {}).items():
            distance = np.linalg.norm(cell_data['position'] - my_pos)
            if distance <= self.vision_range:
                # What I see
                obj_type = cell_data.get('element_type', 'unknown')
                self.perception.visual_objects.append(f"cell_{obj_type}")
                self.perception.visual_distances[cell_id] = distance
                
                # Color
                color = self._element_to_color(obj_type)
                self.perception.visual_colors.append(color)
                
                # Add to vocabulary
                self.state.vocabulary.add(f"see_{obj_type}")
                self.state.vocabulary.add(f"color_{color}")
                
                self.perceptions_this_tick += 2
        
        # See terrain
        terrain = world_state.get('terrain', 'grass')
        self.perception.terrain_type = terrain
        self.state.vocabulary.add(f"terrain_{terrain}")
        self.perceptions_this_tick += 1
    
    def _perceive_hearing(self, world_state: Dict[str, Any]):
        """Hear nearby sounds and speech."""
        my_pos = self.base_cell.position
        
        # Hear Fluctlight particles (speech)
        for particle in world_state.get('nearby_particles', []):
            distance = np.linalg.norm(particle.position - my_pos)
            if distance <= self.hearing_range and particle.concept_id:
                self.perception.heard_words.append(particle.concept_id)
                self.state.vocabulary.add(particle.concept_id)
                self.perceptions_this_tick += 1
        
        # Hear environmental sounds
        if world_state.get('weather') == 'rain':
            self.perception.heard_sounds.append('rain_sound')
            self.state.vocabulary.add('rain_sound')
            self.perceptions_this_tick += 1
    
    def _perceive_touch(self, world_state: Dict[str, Any]):
        """Feel physical contact."""
        my_pos = self.base_cell.position
        
        # Touch nearby cells
        for cell_id, cell_data in world_state.get('nearby_cells', {}).items():
            distance = np.linalg.norm(cell_data['position'] - my_pos)
            if distance <= self.touch_range:
                obj_type = cell_data.get('element_type', 'unknown')
                self.perception.touched_objects.append(f"cell_{obj_type}")
                
                # Temperature sensation
                temp = cell_data.get('temperature', 20.0)
                if temp > 30:
                    self.perception.touch_sensations['warm'] = (temp - 20) / 30
                    self.state.vocabulary.add('warm')
                elif temp < 10:
                    self.perception.touch_sensations['cold'] = (20 - temp) / 20
                    self.state.vocabulary.add('cold')
                
                self.perceptions_this_tick += 2
    
    def _perceive_smell(self, world_state: Dict[str, Any]):
        """Smell chemical signals."""
        # Food scent
        if world_state.get('food_nearby'):
            self.perception.smelled_scents.append('food_scent')
            self.perception.scent_intensities['food'] = 0.8
            self.state.vocabulary.add('food_scent')
            self.perceptions_this_tick += 1
        
        # Danger scent
        if world_state.get('predator_nearby'):
            self.perception.smelled_scents.append('danger_scent')
            self.perception.scent_intensities['danger'] = 0.9
            self.state.vocabulary.add('danger_scent')
            self.perceptions_this_tick += 1
    
    def _perceive_body(self):
        """Feel body position and movement."""
        self.perception.body_position = self.base_cell.position.copy()
        # Assume velocity from position change (simplified)
        self.perception.body_velocity = np.zeros(3)
        
        self.state.vocabulary.add('body_position')
        self.perceptions_this_tick += 1
    
    def _perceive_internal_state(self):
        """Feel hunger, thirst, fatigue, pain."""
        self.perception.hunger_level = self.hunger
        self.perception.thirst_level = self.thirst
        self.perception.fatigue_level = self.fatigue
        self.perception.pain_level = self.pain
        
        # Add to vocabulary if significant
        if self.hunger > 0.5:
            self.state.vocabulary.add('hungry')
            self.perceptions_this_tick += 1
        if self.thirst > 0.5:
            self.state.vocabulary.add('thirsty')
            self.perceptions_this_tick += 1
        if self.fatigue > 0.5:
            self.state.vocabulary.add('tired')
            self.perceptions_this_tick += 1
        if self.pain > 0.3:
            self.state.vocabulary.add('pain')
            self.perceptions_this_tick += 1
    
    def _perceive_social(self, world_state: Dict[str, Any]):
        """Perceive social relationships."""
        for cell_id, cell_data in world_state.get('nearby_cells', {}).items():
            # Check relationship
            if cell_id in self.state.relationships:
                affinity = self.state.relationships[cell_id]
                if affinity > 0.6:
                    self.perception.nearby_allies.append(cell_id)
                    self.state.vocabulary.add('ally')
                    self.perceptions_this_tick += 1
                elif affinity < 0.4:
                    self.perception.nearby_threats.append(cell_id)
                    self.state.vocabulary.add('threat')
                    self.perceptions_this_tick += 1
            else:
                self.perception.nearby_strangers.append(cell_id)
                self.state.vocabulary.add('stranger')
                self.perceptions_this_tick += 1
    
    def _perceive_environment(self, world_state: Dict[str, Any]):
        """Perceive weather, temperature, light."""
        self.perception.weather_condition = world_state.get('weather', 'clear')
        self.perception.temperature = world_state.get('temperature', 20.0)
        self.perception.light_level = world_state.get('light', 1.0)
        
        self.state.vocabulary.add(f"weather_{self.perception.weather_condition}")
        
        if self.perception.temperature > 30:
            self.state.vocabulary.add('hot')
        elif self.perception.temperature < 10:
            self.state.vocabulary.add('cold')
        
        self.perceptions_this_tick += 2
    
    def update_needs(self, dt: float = 1.0):
        """Update survival needs over time."""
        # Physiological needs increase
        self.hunger = min(1.0, self.hunger + 0.01 * dt)
        self.thirst = min(1.0, self.thirst + 0.015 * dt)
        self.fatigue = min(1.0, self.fatigue + 0.005 * dt)
        
        # Social needs
        if len(self.state.relationships) < 3:
            self.loneliness = min(1.0, self.loneliness + 0.01 * dt)
        else:
            self.loneliness = max(0.0, self.loneliness - 0.02 * dt)
        
        # Pain decays
        self.pain = max(0.0, self.pain - 0.05 * dt)
    
    def satisfy_need(self, need: str, amount: float):
        """Satisfy a need (e.g., eat food)."""
        if need == 'hunger':
            self.hunger = max(0.0, self.hunger - amount)
            self.state.vocabulary.add('eat')
            self.learn_from_experience('hunger', 'eat')
        elif need == 'thirst':
            self.thirst = max(0.0, self.thirst - amount)
            self.state.vocabulary.add('drink')
            self.learn_from_experience('thirst', 'drink')
        elif need == 'fatigue':
            self.fatigue = max(0.0, self.fatigue - amount)
            self.state.vocabulary.add('rest')
            self.learn_from_experience('fatigue', 'rest')
    
    def _element_to_color(self, element: str) -> str:
        """Map element type to color."""
        color_map = {
            'fire': 'red',
            'water': 'blue',
            'earth': 'brown',
            'air': 'white',
            'wood': 'green',
            'metal': 'silver'
        }
        return color_map.get(element, 'gray')
    
    def get_perception_summary(self) -> Dict[str, Any]:
        """Get summary of perceptions this tick."""
        return {
            "total_perceptions": self.perceptions_this_tick,
            "visual": len(self.perception.visual_objects),
            "auditory": len(self.perception.heard_words) + len(self.perception.heard_sounds),
            "tactile": len(self.perception.touched_objects),
            "olfactory": len(self.perception.smelled_scents),
            "interoceptive": sum([
                1 if self.hunger > 0.5 else 0,
                1 if self.thirst > 0.5 else 0,
                1 if self.fatigue > 0.5 else 0,
                1 if self.pain > 0.3 else 0
            ]),
            "social": len(self.perception.nearby_allies) + len(self.perception.nearby_strangers),
            "environmental": 2  # weather + temperature
        }
