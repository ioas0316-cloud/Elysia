"""
Conceptual Big Bang -       
                                                                              

"       ... '       ' ...       .
               ...        ... '   (   )'      ."

                                                                               
                                                                            
                                                                              
                                                                               
   '  '             ...                                             
        '  (Event)'          ... '       '      !     
                                                                               
   ' (Fire)'      ?                                                       
                    .                                              
   " ,    !" (   1) + " ?   ?" (   2)                               
            (Event)     ... ' '                .       
                                                                               
   '  (Mother)'     ?                                                     
   "           ." (   1) + "         ." (   2)             
           .                                                             
                                                                               

[     :        (The Conceptual Big Bang) ]

1.        (Seeding):
   -           ' (ConceptStar)'             
   -      3D         ,       (   )   (     )    

2.        (Gravity of Events):
   - '  (Event)'      ,                  
   - "  "  "  "     '  '                
   -                        '  (Constellation)'

3.          (Elysia's Journey):
   -              '  '         
   - " ?    (  )      (  ) ...          ?"
   -               ' '       

4.         (Constellation Making):
   -           '   '   
   -                          
   -     '  '  '  '    
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from collections import defaultdict
from enum import Enum
import logging
import json

logger = logging.getLogger("ConceptualBigBang")


# ============================================================================
#      
# ============================================================================

#       (                    )
SENSORY_DIMENSIONS = {
    "temperature": (-1.0, 1.0),   #    (-1) ~    (+1)
    "brightness": (0.0, 1.0),     #    (0) ~   (1)
    "softness": (0.0, 1.0),       #    (0) ~     (1)
    "size": (-1.0, 1.0),          #   (-1) ~  (+1)
    "speed": (0.0, 1.0),          #   (0) ~   (1)
    "danger": (0.0, 1.0),         #   (0) ~   (1)
    "pleasure": (-1.0, 1.0),      #   (-1) ~   (+1)
    "social": (0.0, 1.0),         #   (0) ~   (1)
}

#       (     )
EMOTIONAL_HUES = {
    "joy": 60,        #   
    "sadness": 220,   #   
    "fear": 280,      #   
    "anger": 0,       #   
    "love": 330,      #   
    "curiosity": 30,  #   
    "peace": 120,     #   
    "neutral": 0,     #    
}


# ============================================================================
# ConceptStar -      
# ============================================================================

@dataclass
class ConceptStar:
    """
          (Concept Star)
    
                      .
               ,            .
    
    " (Fire)"  "    +   "         .
    "  (Mother)"  "    +     +    "    .
    """
    
    #      
    id: str                          #    ID
    name: Optional[str] = None       #    (한국어 학습 시스템)
    
    # 3D            
    position: np.ndarray = field(default_factory=lambda: np.random.randn(3) * 100)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    #       
    mass: float = 1.0               #    (   ,      )
    radius: float = 1.0             #    (주권적 자아)
    temperature: float = 1.0        #    (주권적 자아)
    
    #          (8  )
    sensory_signature: Dict[str, float] = field(default_factory=dict)
    
    #       (     )
    emotional_hue: float = 0.0      # 0-360 (HSL)
    emotional_intensity: float = 0.5  # 0-1
    
    #        
    associated_events: List[str] = field(default_factory=list)
    
    #           (constellation    )
    connections: Dict[str, float] = field(default_factory=dict)  # star_id: bond_strength
    
    #   
    visit_count: int = 0            #      
    discovery_time: Optional[float] = None  #       
    
    def __post_init__(self):
        if isinstance(self.position, list):
            self.position = np.array(self.position, dtype=np.float32)
        if isinstance(self.velocity, list):
            self.velocity = np.array(self.velocity, dtype=np.float32)
    
    def distance_to(self, other: 'ConceptStar') -> float:
        """          """
        return float(np.linalg.norm(self.position - other.position))
    
    def sensory_similarity(self, other: 'ConceptStar') -> float:
        """        (0-1)"""
        if not self.sensory_signature or not other.sensory_signature:
            return 0.0
        
        common_dims = set(self.sensory_signature.keys()) & set(other.sensory_signature.keys())
        if not common_dims:
            return 0.0
        
        total_sim = 0.0
        for dim in common_dims:
            diff = abs(self.sensory_signature[dim] - other.sensory_signature[dim])
            total_sim += 1.0 - diff / 2.0  #    
        
        return total_sim / len(common_dims)
    
    def apply_gravity_from(self, other: 'ConceptStar', strength: float = 0.01):
        """             """
        direction = other.position - self.position
        distance = np.linalg.norm(direction)
        
        if distance < 0.1:
            return
        
        # F = G * m1 * m2 / r 
        force_magnitude = strength * self.mass * other.mass / (distance ** 2)
        force_direction = direction / distance
        
        #     = F / m
        acceleration = force_direction * force_magnitude / self.mass
        self.velocity += acceleration
    
    def update_position(self, dt: float = 1.0, damping: float = 0.99):
        """        (     )"""
        self.position += self.velocity * dt
        self.velocity *= damping  #   
    
    def connect_to(self, other_id: str, strength: float = 1.0):
        """        """
        current = self.connections.get(other_id, 0.0)
        self.connections[other_id] = current + strength
    
    def get_connection_strength(self, other_id: str) -> float:
        return self.connections.get(other_id, 0.0)


# ============================================================================
# Event -   
# ============================================================================

@dataclass
class Event:
    """
       (Event)
    
    " ,    !" + " ?   ?" = ' '
    
                              .
                         .
    """
    
    id: str
    description: str = ""
    
    #         (  ID   )
    involved_concepts: List[str] = field(default_factory=list)
    
    #        (0-1,            )
    intensity: float = 1.0
    
    #         
    timestamp: float = 0.0
    
    #        (한국어 학습 시스템)
    sensory_impression: Dict[str, float] = field(default_factory=dict)
    
    #       
    emotional_tone: str = "neutral"
    
    #       (                 )
    repetition_count: int = 1
    
    def get_binding_strength(self) -> float:
        """               """
        return self.intensity * math.log1p(self.repetition_count)


# ============================================================================
# Constellation -     (   )
# ============================================================================

@dataclass
class Constellation:
    """
        (Constellation) =    
    
              '   '   .
                             .
    
     : " " + "  " + "   " = "      "    
        " " + " " + "   " = "      "    
    """
    
    id: str
    name: str = ""
    
    #              (     !)
    star_sequence: List[str] = field(default_factory=list)
    
    #      (                 )
    connections: List[Tuple[str, str]] = field(default_factory=list)
    
    #              
    discovered_by: Optional[str] = None
    discovery_time: float = 0.0
    
    #         (   )
    emergent_meaning: str = ""
    
    #       (              "     ")
    narration_count: int = 0
    
    def add_star(self, star_id: str, connect_to_previous: bool = True):
        """         """
        if self.star_sequence and connect_to_previous:
            self.connections.append((self.star_sequence[-1], star_id))
        self.star_sequence.append(star_id)
    
    def get_narrative_length(self) -> int:
        return len(self.star_sequence)


# ============================================================================
# ConceptualUniverse -       
# ============================================================================

class ConceptualUniverse:
    """
           (Conceptual Universe)
    
                      3D   .
                         ,
                 (   )     .
    """
    
    def __init__(self, size: float = 1000.0):
        """
        Args:
            size:        (       )
        """
        self.size = size
        
        #   
        self.stars: Dict[str, ConceptStar] = {}
        
        #    
        self.events: List[Event] = []
        
        #         
        self.constellations: Dict[str, Constellation] = {}
        
        #   
        self.time = 0.0
        
        #   
        self.total_events = 0
        self.total_connections = 0
        
        logger.info(f"ConceptualUniverse created (size={size})")
    
    # ========================================================================
    #        (Seeding)
    # ========================================================================
    
    def seed_concept(
        self,
        id: str,
        name: Optional[str] = None,
        sensory_signature: Optional[Dict[str, float]] = None,
        emotional_hue: float = 0.0,
        mass: float = 1.0,
        position: Optional[np.ndarray] = None
    ) -> ConceptStar:
        """
                    
        
        Args:
            id:    ID
            name:    (  )
            sensory_signature:      
            emotional_hue:       (0-360)
            mass:    (   )
            position:    (None     )
        """
        if position is None:
            position = np.random.uniform(-self.size/2, self.size/2, 3)
        
        star = ConceptStar(
            id=id,
            name=name,
            position=position,
            mass=mass,
            sensory_signature=sensory_signature or {},
            emotional_hue=emotional_hue
        )
        
        self.stars[id] = star
        return star
    
    def seed_many(
        self,
        concept_definitions: List[Dict[str, Any]],
        scatter_radius: float = 500.0
    ) -> int:
        """
                (  !)
        
        Args:
            concept_definitions:         
            scatter_radius:        
            
        Returns:
                    
        """
        count = 0
        for concept in concept_definitions:
            position = np.random.randn(3) * scatter_radius
            
            self.seed_concept(
                id=concept.get("id", f"concept_{count}"),
                name=concept.get("name"),
                sensory_signature=concept.get("sensory", {}),
                emotional_hue=concept.get("hue", np.random.uniform(0, 360)),
                mass=concept.get("mass", 1.0),
                position=position
            )
            count += 1
        
        logger.info(f"  Big Bang! Seeded {count} concept stars")
        return count
    
    def seed_fundamental_concepts(self) -> int:
        """
                    (               )
        
                                    .
        """
        fundamentals = [
            #         
            {"id": "hot", "name": "   ", "sensory": {"temperature": 1.0, "danger": 0.5}, "hue": 0},
            {"id": "cold", "name": "   ", "sensory": {"temperature": -1.0}, "hue": 220},
            {"id": "bright", "name": "  ", "sensory": {"brightness": 1.0}, "hue": 60},
            {"id": "dark", "name": "   ", "sensory": {"brightness": 0.0, "danger": 0.3}, "hue": 240},
            {"id": "soft", "name": "    ", "sensory": {"softness": 1.0, "pleasure": 0.5}, "hue": 330},
            {"id": "hard", "name": "   ", "sensory": {"softness": 0.0}, "hue": 30},
            {"id": "big", "name": " ", "sensory": {"size": 1.0}, "hue": 180},
            {"id": "small", "name": "  ", "sensory": {"size": -1.0}, "hue": 60},
            {"id": "fast", "name": "  ", "sensory": {"speed": 1.0}, "hue": 0},
            {"id": "slow", "name": "  ", "sensory": {"speed": 0.0}, "hue": 180},
            
            #         
            {"id": "pleasure", "name": "  ", "sensory": {"pleasure": 1.0}, "hue": 60},
            {"id": "pain", "name": "  ", "sensory": {"pleasure": -1.0, "danger": 0.8}, "hue": 0},
            {"id": "fear", "name": "   ", "sensory": {"danger": 1.0}, "hue": 280},
            {"id": "safe", "name": "  ", "sensory": {"danger": 0.0, "pleasure": 0.3}, "hue": 120},
            {"id": "hunger", "name": "   ", "sensory": {"pleasure": -0.5}, "hue": 30},
            {"id": "satiety", "name": "   ", "sensory": {"pleasure": 0.7}, "hue": 120},
            
            #       
            {"id": "alone", "name": "  ", "sensory": {"social": 0.0}, "hue": 240},
            {"id": "together", "name": "  ", "sensory": {"social": 1.0, "pleasure": 0.5}, "hue": 30},
            {"id": "touch", "name": "  ", "sensory": {"social": 0.8, "softness": 0.5}, "hue": 330},
            
            #      
            {"id": "fire", "name": " ", "sensory": {"temperature": 1.0, "brightness": 0.9, "danger": 0.6}, "hue": 15},
            {"id": "water", "name": " ", "sensory": {"temperature": -0.2, "softness": 0.8}, "hue": 200},
            {"id": "sun", "name": " ", "sensory": {"brightness": 1.0, "temperature": 0.7}, "hue": 45},
            {"id": "moon", "name": " ", "sensory": {"brightness": 0.3, "temperature": -0.2}, "hue": 220},
            {"id": "rain", "name": " ", "sensory": {"temperature": -0.1, "softness": 0.4}, "hue": 210},
            {"id": "wind", "name": "  ", "sensory": {"speed": 0.6, "temperature": 0.0}, "hue": 180},
            
            #    
            {"id": "mother", "name": "  ", "sensory": {"social": 1.0, "softness": 0.9, "temperature": 0.3, "pleasure": 0.8}, "hue": 330, "mass": 3.0},
            {"id": "food", "name": "  ", "sensory": {"pleasure": 0.6}, "hue": 30, "mass": 2.0},
            {"id": "animal", "name": "  ", "sensory": {"social": 0.5, "speed": 0.5}, "hue": 90},
            {"id": "tree", "name": "  ", "sensory": {"size": 0.8, "softness": 0.3}, "hue": 120},
            {"id": "flower", "name": " ", "sensory": {"brightness": 0.6, "pleasure": 0.4, "size": -0.5}, "hue": 300},
            
            #   /   
            {"id": "stone", "name": " ", "sensory": {"softness": 0.0, "temperature": -0.1}, "hue": 45},
            {"id": "wood", "name": "  (  )", "sensory": {"softness": 0.2, "temperature": 0.1}, "hue": 30},
            {"id": "tool", "name": "  ", "sensory": {}, "hue": 45, "mass": 1.5},
            
            #       (       )
            {"id": "love", "name": "  ", "sensory": {"social": 1.0, "pleasure": 1.0}, "hue": 330, "mass": 2.5},
            {"id": "home", "name": " ", "sensory": {"danger": 0.0, "temperature": 0.3, "social": 0.7}, "hue": 30, "mass": 2.0},
            {"id": "danger", "name": "  ", "sensory": {"danger": 1.0, "pleasure": -0.8}, "hue": 0, "mass": 2.0},
            
            #   
            {"id": "eat", "name": "  ", "sensory": {"pleasure": 0.5}, "hue": 30},
            {"id": "sleep", "name": "  ", "sensory": {"pleasure": 0.4, "speed": 0.0}, "hue": 240},
            {"id": "run", "name": "   ", "sensory": {"speed": 1.0}, "hue": 0},
            {"id": "cry", "name": "  ", "sensory": {"pleasure": -0.5, "social": 0.6}, "hue": 220},
            {"id": "laugh", "name": "  ", "sensory": {"pleasure": 0.9, "social": 0.7}, "hue": 60},
            {"id": "hug", "name": "  ", "sensory": {"social": 1.0, "softness": 0.9, "pleasure": 0.8}, "hue": 330},
        ]
        
        return self.seed_many(fundamentals, scatter_radius=300.0)
    
    # ========================================================================
    #        (Gravity of Events)
    # ========================================================================
    
    def trigger_event(
        self,
        involved_concepts: List[str],
        description: str = "",
        intensity: float = 1.0,
        sensory_impression: Optional[Dict[str, float]] = None,
        emotional_tone: str = "neutral"
    ) -> Event:
        """
             !
        
                                      .
        
         : trigger_event(["fire", "hot", "bright"], " ,    !   !", intensity=1.0)
        """
        event = Event(
            id=f"event_{self.total_events}",
            description=description,
            involved_concepts=involved_concepts,
            intensity=intensity,
            timestamp=self.time,
            sensory_impression=sensory_impression or {},
            emotional_tone=emotional_tone
        )
        
        self.events.append(event)
        self.total_events += 1
        
        #                 
        binding_strength = event.get_binding_strength()
        
        for i, concept_id1 in enumerate(involved_concepts):
            if concept_id1 not in self.stars:
                continue
            star1 = self.stars[concept_id1]
            
            for concept_id2 in involved_concepts[i+1:]:
                if concept_id2 not in self.stars:
                    continue
                star2 = self.stars[concept_id2]
                
                #       
                star1.connect_to(concept_id2, binding_strength)
                star2.connect_to(concept_id1, binding_strength)
                
                #      
                star1.associated_events.append(event.id)
                star2.associated_events.append(event.id)
                
                self.total_connections += 1
        
        logger.debug(f"Event: {description} - connected {len(involved_concepts)} concepts")
        return event
    
    def trigger_sensory_event(
        self,
        sensory_impression: Dict[str, float],
        intensity: float = 1.0
    ) -> List[str]:
        """
                   
        
                 ,                     .
        
         : trigger_sensory_event({"temperature": 1.0, "brightness": 0.9})
              "fire", "hot", "bright"            
        """
        #                       
        related_stars = []
        
        for star_id, star in self.stars.items():
            if not star.sensory_signature:
                continue
            
            #          
            similarity = 0.0
            matching_dims = 0
            
            for dim, value in sensory_impression.items():
                if dim in star.sensory_signature:
                    diff = abs(star.sensory_signature[dim] - value)
                    similarity += 1.0 - diff / 2.0
                    matching_dims += 1
            
            if matching_dims > 0:
                avg_sim = similarity / matching_dims
                if avg_sim > 0.5:  #    
                    related_stars.append((star_id, avg_sim))
        
        #    5    
        related_stars.sort(key=lambda x: -x[1])
        top_concepts = [s[0] for s in related_stars[:5]]
        
        if top_concepts:
            self.trigger_event(
                involved_concepts=top_concepts,
                description=f"Sensory event: {sensory_impression}",
                intensity=intensity,
                sensory_impression=sensory_impression
            )
        
        return top_concepts
    
    # ========================================================================
    #       (Universe Physics)
    # ========================================================================
    
    def apply_gravity(self, strength: float = 0.001):
        """                """
        for star_id, star in self.stars.items():
            for other_id, bond_strength in star.connections.items():
                if other_id in self.stars:
                    other = self.stars[other_id]
                    #               
                    star.apply_gravity_from(other, strength * bond_strength)
    
    def update_positions(self, dt: float = 1.0):
        """             """
        for star in self.stars.values():
            star.update_position(dt)
    
    def step(self, dt: float = 1.0, apply_gravity: bool = True):
        """        """
        self.time += dt
        
        if apply_gravity:
            self.apply_gravity()
        
        self.update_positions(dt)
    
    # ========================================================================
    #        (Constellation Discovery)
    # ========================================================================
    
    def discover_constellation(
        self,
        star_ids: List[str],
        discoverer_id: str,
        name: str = ""
    ) -> Optional[Constellation]:
        """
               (       )
        
                             .
        """
        #               
        for star_id in star_ids:
            if star_id not in self.stars:
                return None
        
        constellation = Constellation(
            id=f"const_{len(self.constellations)}",
            name=name,
            discovered_by=discoverer_id,
            discovery_time=self.time
        )
        
        for i, star_id in enumerate(star_ids):
            constellation.add_star(star_id, connect_to_previous=(i > 0))
            self.stars[star_id].visit_count += 1
            if self.stars[star_id].discovery_time is None:
                self.stars[star_id].discovery_time = self.time
        
        self.constellations[constellation.id] = constellation
        
        logger.info(f"  New constellation discovered: {name or constellation.id}")
        return constellation
    
    def find_natural_constellations(self, min_connection_strength: float = 2.0) -> List[List[str]]:
        """
                        
        
                            .
        """
        visited = set()
        constellations = []
        
        for star_id, star in self.stars.items():
            if star_id in visited:
                continue
            
            # BFS           
            cluster = []
            queue = [star_id]
            
            while queue:
                current_id = queue.pop(0)
                if current_id in visited:
                    continue
                
                visited.add(current_id)
                cluster.append(current_id)
                
                if current_id in self.stars:
                    current = self.stars[current_id]
                    for other_id, strength in current.connections.items():
                        if strength >= min_connection_strength and other_id not in visited:
                            queue.append(other_id)
            
            if len(cluster) >= 2:
                constellations.append(cluster)
        
        return constellations
    
    # ========================================================================
    #        
    # ========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """     """
        connection_strengths = []
        for star in self.stars.values():
            connection_strengths.extend(star.connections.values())
        
        return {
            "total_stars": len(self.stars),
            "total_events": self.total_events,
            "total_connections": self.total_connections,
            "total_constellations": len(self.constellations),
            "avg_connection_strength": np.mean(connection_strengths) if connection_strengths else 0,
            "max_connection_strength": max(connection_strengths) if connection_strengths else 0,
            "time": self.time,
        }
    
    def get_most_connected_stars(self, n: int = 10) -> List[Tuple[str, int]]:
        """            """
        star_connections = [
            (star_id, len(star.connections))
            for star_id, star in self.stars.items()
        ]
        star_connections.sort(key=lambda x: -x[1])
        return star_connections[:n]
    
    def get_strongest_bonds(self, n: int = 10) -> List[Tuple[str, str, float]]:
        """         """
        bonds = []
        seen = set()
        
        for star_id, star in self.stars.items():
            for other_id, strength in star.connections.items():
                bond_key = tuple(sorted([star_id, other_id]))
                if bond_key not in seen:
                    seen.add(bond_key)
                    bonds.append((star_id, other_id, strength))
        
        bonds.sort(key=lambda x: -x[2])
        return bonds[:n]


# ============================================================================
# ConceptExplorer -        (    )
# ============================================================================

class ConceptExplorer:
    """
           (Concept Explorer)
    
                               .
    
    " ?    (  )      (  ) ...          ?"
        '  '         ... ' '       .
    """
    
    def __init__(self, name: str, universe: ConceptualUniverse):
        self.name = name
        self.universe = universe
        
        #      
        self.position = np.zeros(3)
        
        #       
        self.visited_stars: Set[str] = set()
        
        #         (    '  ')
        self.discovered_connections: Dict[Tuple[str, str], float] = {}
        
        #         (   )
        self.constellations: List[str] = []
        
        #         
        self.current_journey: List[str] = []
        
        #   
        self.total_distance_traveled = 0.0
        self.discoveries = 0
    
    def travel_to(self, star_id: str) -> bool:
        """        """
        if star_id not in self.universe.stars:
            return False
        
        target = self.universe.stars[star_id]
        distance = np.linalg.norm(target.position - self.position)
        
        self.position = target.position.copy()
        self.total_distance_traveled += distance
        
        #      
        self.visited_stars.add(star_id)
        target.visit_count += 1
        
        #          
        self.current_journey.append(star_id)
        
        #              
        self._discover_nearby_connections(star_id)
        
        return True
    
    def _discover_nearby_connections(self, current_star_id: str):
        """             """
        current = self.universe.stars[current_star_id]
        
        for other_id, strength in current.connections.items():
            if strength > 0.5:  #            
                connection_key = tuple(sorted([current_star_id, other_id]))
                
                if connection_key not in self.discovered_connections:
                    self.discovered_connections[connection_key] = strength
                    self.discoveries += 1
                    
                    current_name = current.name or current_star_id
                    if other_id in self.universe.stars:
                        other_name = self.universe.stars[other_id].name or other_id
                    else:
                        other_name = other_id
                    
                    logger.debug(f"  {self.name} discovered: {current_name}   {other_name}")
    
    def explore_randomly(self, steps: int = 10):
        """     """
        for _ in range(steps):
            if not self.universe.stars:
                break
            
            #                      
            candidates = []
            for star_id, star in self.universe.stars.items():
                distance = np.linalg.norm(star.position - self.position)
                if distance < 200:  #      
                    candidates.append((star_id, distance))
            
            if not candidates:
                #        -            
                if not hasattr(self, '_star_ids_cache') or len(self._star_ids_cache) != len(self.universe.stars):
                    self._star_ids_cache = list(self.universe.stars.keys())
                star_id = np.random.choice(self._star_ids_cache)
            else:
                #          (   )
                candidates.sort(key=lambda x: x[1])
                weights = [1.0 / (c[1] + 1) for c in candidates]
                weights = np.array(weights) / sum(weights)
                idx = np.random.choice(len(candidates), p=weights)
                star_id = candidates[idx][0]
            
            self.travel_to(star_id)
    
    def explore_by_sensory(self, sensory_preference: Dict[str, float], steps: int = 10):
        """         (            )"""
        for _ in range(steps):
            best_star = None
            best_score = -1
            
            for star_id, star in self.universe.stars.items():
                if not star.sensory_signature:
                    continue
                
                #               
                score = 0.0
                for dim, pref in sensory_preference.items():
                    if dim in star.sensory_signature:
                        score += 1.0 - abs(star.sensory_signature[dim] - pref)
                
                if score > best_score:
                    best_score = score
                    best_star = star_id
            
            if best_star:
                self.travel_to(best_star)
    
    def create_constellation(self, name: str = "") -> Optional[Constellation]:
        """                 """
        if len(self.current_journey) < 2:
            return None
        
        constellation = self.universe.discover_constellation(
            star_ids=self.current_journey.copy(),
            discoverer_id=self.name,
            name=name
        )
        
        if constellation:
            self.constellations.append(constellation.id)
            self.current_journey = []
        
        return constellation
    
    def get_vocabulary(self) -> List[Tuple[str, str, float]]:
        """    '   ' (   )"""
        vocab = []
        for (star1, star2), strength in self.discovered_connections.items():
            name1 = self.universe.stars[star1].name if star1 in self.universe.stars else star1
            name2 = self.universe.stars[star2].name if star2 in self.universe.stars else star2
            vocab.append((name1, name2, strength))
        
        vocab.sort(key=lambda x: -x[2])
        return vocab
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "visited_stars": len(self.visited_stars),
            "discovered_connections": len(self.discovered_connections),
            "constellations_created": len(self.constellations),
            "total_distance": self.total_distance_traveled,
        }


# ============================================================================
# ConceptualBigBangWorld -          
# ============================================================================

class ConceptualBigBangWorld:
    """
             
    
           ,       ,                 .
    """
    
    def __init__(
        self,
        n_explorers: int = 10,
        universe_size: float = 1000.0,
        seed_fundamentals: bool = True
    ):
        #      
        self.universe = ConceptualUniverse(size=universe_size)
        
        #           
        if seed_fundamentals:
            self.universe.seed_fundamental_concepts()
        
        #        
        self.explorers: Dict[str, ConceptExplorer] = {}
        explorer_names = ['  ', '  ', ' ', ' ', ' ', ' ', ' ', '  ', '  ', ' ']
        for i in range(n_explorers):
            name = f"{explorer_names[i % len(explorer_names)]}{i}"
            self.explorers[name] = ConceptExplorer(name, self.universe)
        
        #   
        self.total_events_triggered = 0
        self.simulation_time = 0.0
        
        logger.info(f"ConceptualBigBangWorld created: {len(self.universe.stars)} stars, {n_explorers} explorers")
    
    def trigger_life_event(self, event_type: str, intensity: float = 1.0) -> Event:
        """
                 (자기 성찰 엔진)
        
                         .
        """
        life_events = {
            "touched_fire": {
                "concepts": ["fire", "hot", "pain", "danger", "bright"],
                "sensory": {"temperature": 1.0, "brightness": 0.9, "danger": 0.8, "pleasure": -0.7},
                "description": " ,    ! (     )",
            },
            "mother_feeding": {
                "concepts": ["mother", "food", "satiety", "love", "soft", "safe"],
                "sensory": {"pleasure": 0.9, "social": 1.0, "softness": 0.9},
                "description": "          (자기 성찰 엔진)",
            },
            "mother_hugging": {
                "concepts": ["mother", "hug", "safe", "love", "soft"],
                "sensory": {"social": 1.0, "softness": 1.0, "pleasure": 0.8, "temperature": 0.3},
                "description": "        (자기 성찰 엔진)",
            },
            "saw_sunset": {
                "concepts": ["sun", "bright", "big", "fire"],
                "sensory": {"brightness": 0.8, "temperature": 0.2, "size": 0.9},
                "description": "           (          )",
            },
            "felt_rain": {
                "concepts": ["rain", "water", "cold", "soft"],
                "sensory": {"temperature": -0.2, "softness": 0.6},
                "description": "      (자기 성찰 엔진)",
            },
            "heard_thunder": {
                "concepts": ["fear", "danger", "big"],
                "sensory": {"danger": 0.7, "size": 0.8},
                "description": "         ",
            },
            "found_flower": {
                "concepts": ["flower", "bright", "pleasure", "small"],
                "sensory": {"brightness": 0.7, "pleasure": 0.5, "size": -0.5},
                "description": "         ",
            },
            "played_together": {
                "concepts": ["together", "pleasure", "laugh", "fast"],
                "sensory": {"social": 1.0, "pleasure": 0.8, "speed": 0.6},
                "description": "          ",
            },
            "felt_lonely": {
                "concepts": ["alone", "pain", "cry"],
                "sensory": {"social": 0.0, "pleasure": -0.6},
                "description": "           ",
            },
            "discovered_tool": {
                "concepts": ["stone", "tool", "hard"],
                "sensory": {"softness": 0.0},
                "description": "                     ",
            },
        }
        
        if event_type not in life_events:
            return None
        
        event_data = life_events[event_type]
        
        return self.universe.trigger_event(
            involved_concepts=event_data["concepts"],
            description=event_data["description"],
            intensity=intensity,
            sensory_impression=event_data.get("sensory", {}),
        )
    
    def simulate_childhood(self, days: int = 100):
        """
                   
        
                                 .
        """
        event_types = list([
            "touched_fire", "mother_feeding", "mother_hugging",
            "saw_sunset", "felt_rain", "heard_thunder",
            "found_flower", "played_together", "felt_lonely",
            "discovered_tool"
        ])
        
        for day in range(days):
            #     2-5        
            n_events = np.random.randint(2, 6)
            
            for _ in range(n_events):
                event_type = np.random.choice(event_types)
                intensity = np.random.uniform(0.5, 1.5)
                self.trigger_life_event(event_type, intensity)
                self.total_events_triggered += 1
            
            #             
            for explorer in self.explorers.values():
                explorer.explore_randomly(steps=2)
            
            #         
            self.universe.step(dt=1.0)
            self.simulation_time += 1.0
            
            #      
            if day > 0 and day % 20 == 0:
                stats = self.get_statistics()
                print(f"Day {day}: connections={stats['total_connections']}, "
                      f"avg_vocabulary={stats['avg_vocabulary']:.1f}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """  """
        universe_stats = self.universe.get_statistics()
        
        vocabularies = [len(e.discovered_connections) for e in self.explorers.values()]
        
        return {
            **universe_stats,
            "total_events_triggered": self.total_events_triggered,
            "simulation_time": self.simulation_time,
            "n_explorers": len(self.explorers),
            "avg_vocabulary": np.mean(vocabularies) if vocabularies else 0,
            "max_vocabulary": max(vocabularies) if vocabularies else 0,
        }
    
    def get_sample_vocabularies(self, n: int = 3) -> Dict[str, List]:
        """     """
        result = {}
        for name, explorer in list(self.explorers.items())[:n]:
            vocab = explorer.get_vocabulary()[:10]
            result[name] = vocab
        return result
    
    def get_natural_constellations(self) -> List[List[str]]:
        """              """
        return self.universe.find_natural_constellations()


# ============================================================================
# Demo
# ============================================================================

def demo():
    """         """
    print("=" * 70)
    print("Conceptual Big Bang -       ")
    print("=" * 70)
    print()
    print("'  '             ...")
    print("     '  (Event)'          ... '       '  !")
    print()
    print("       ... '       ' ...       .")
    print("           ...        ... '   (   )'      .")
    print()
    
    #      
    world = ConceptualBigBangWorld(n_explorers=10, seed_fundamentals=True)
    
    print(f"    ! {len(world.universe.stars)}                    .")
    print()
    
    #            
    print("               ... (100 )")
    print("-" * 70)
    world.simulate_childhood(days=100)
    print("-" * 70)
    print()
    
    #      
    stats = world.get_statistics()
    print("    :")
    print(f"      : {stats['total_events_triggered']}")
    print(f"      : {stats['total_connections']}")
    print(f"       : {stats['avg_vocabulary']:.1f}")
    print(f"       : {stats['max_vocabulary']}")
    print()
    
    #          (    '  ')
    print("                 (=   ):")
    strongest = world.universe.get_strongest_bonds(10)
    for star1, star2, strength in strongest:
        name1 = world.universe.stars[star1].name if star1 in world.universe.stars else star1
        name2 = world.universe.stars[star2].name if star2 in world.universe.stars else star2
        print(f"  '{name1}'   '{name2}': {strength:.2f}")
    print()
    
    #            (       )
    print("              (       ):")
    constellations = world.get_natural_constellations()
    for i, cluster in enumerate(constellations[:5]):
        names = [world.universe.stars[s].name or s for s in cluster if s in world.universe.stars]
        print(f"      {i+1}: {' - '.join(names)}")
    print()
    
    #           
    print("             :")
    samples = world.get_sample_vocabularies(3)
    for name, vocab in samples.items():
        print(f"\n  [{name}]    :")
        for word1, word2, strength in vocab[:5]:
            print(f"    '{word1}'   '{word2}' (  : {strength:.2f})")
    print()
    
    print("=" * 70)
    print("'             ' ...         !   ")
    print("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
