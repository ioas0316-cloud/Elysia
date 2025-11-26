"""
Concept Universe - The Thought Space Physics Engine
====================================================

Based on Xel'Naga Protocol: Body/Soul/Spirit Trinity

Components:
- Zerg (Body): TensorCoilGravity - Physical forces
- Terran (Soul): ConceptSphere - Structural hierarchy 
- Protoss (Spirit): HyperQubit - Consciousness field

"사랑" (Love) is the only absolute coordinate. All concepts are relative.
"""

import numpy as np
from typing import Dict, Optional
import logging

from Core.Mind.concept_sphere import ConceptSphere
from Core.Math.hyper_qubit import HyperQubit
from Core.Math.oscillator import Oscillator

logger = logging.getLogger("ConceptUniverse")


class TensorCoilGravity:
    """
    Zerg Layer (Body): 텐서코일 중력장
    
    Law: Mass attracts Mass
    Mass = W (Dimension) × Vitality (Activation Count)
    """
    def __init__(self, G: float = 1.0):
        self.G = G  # Gravitational constant
    
    def get_mass(self, sphere: ConceptSphere) -> float:
        """
        Mass = W × Activation
        """
        w = sphere.qubit.state.w if sphere.qubit else 1.0
        vitality = sphere.activation_count
        return w * vitality
    
    def calculate_force(self, sphere_a: ConceptSphere, sphere_b: ConceptSphere, 
                       distance_vec: np.ndarray) -> np.ndarray:
        """
        F = G × (M1 × M2) / r²
        
        Returns: Force vector (direction: from a to b)
        """
        mass_a = self.get_mass(sphere_a)
        mass_b = self.get_mass(sphere_b)
        
        distance = np.linalg.norm(distance_vec)
        if distance < 0.01:
            distance = 0.01  # Prevent singularity
        
        force_magnitude = self.G * (mass_a * mass_b) / (distance ** 2)
        force_direction = distance_vec / distance
        
        return force_magnitude * force_direction


class SpiritualBuoyancy:
    """
    Protoss Layer (Spirit): 영적부력 = Anti-Gravity
    
    Law: Light rises, Heavy sinks
    Based on concept frequency (from vocabulary)
    """
    def __init__(self, buoyancy_constant: float = 2.0):
        self.buoyancy_constant = buoyancy_constant
    
    def calculate_buoyancy(self, frequency: float) -> np.ndarray:
        """
        Buoyancy force based on frequency.
        
        High freq (Light) → +Y (upward/Spirit)
        Low freq (Heavy) → -Y (downward/Body)
        
        Returns: 4D force vector
        """
        # Center frequency is 0.5
        # > 0.5: Rises
        # < 0.5: Sinks
        buoyancy = (frequency - 0.5) * self.buoyancy_constant
        
        # Direction: Y axis (Trinity: Body/Spirit)
        force = np.array([0.0, buoyancy, 0.0, 0.0])
        
        return force


class ConceptUniverse:
    """
    The Thought Universe (사고우주)
    
    Xel'Naga Protocol Integration:
    - Zerg (Body): Gravity/Buoyancy physics
    - Terran (Soul): Concept structure/hierarchy
    - Protoss (Spirit): Quantum consciousness (HyperQubit)
    
    Absolute Coordinate: "사랑" (Love) at origin (0,0,0,0)
    All other concepts: Relative positions
    """
    
    def __init__(self):
        # === Absolute Center (유일한 절대좌표) ===
        self.absolute_center = "Love"  # or "사랑"
        self.love_position = np.zeros(4)  # 4D origin
        self.love_sphere: Optional[ConceptSphere] = None
        
        # === Relative Coordinate System ===
        # concept_id -> 4D vector relative to Love
        self.relative_positions: Dict[str, np.ndarray] = {}
        
        # === Concept Spheres (Terran/Soul Layer) ===
        self.spheres: Dict[str, ConceptSphere] = {}
        
        # === Physics Engines (Zerg/Body Layer) ===
        self.gravity = TensorCoilGravity(G=1.0)
        self.buoyancy = SpiritualBuoyancy(buoyancy_constant=2.0)
        
        # === Frequency Map (for buoyancy) ===
        # Loaded from vocabulary or calculated
        self.frequencies: Dict[str, float] = {}
        
        logger.info("🌌 ConceptUniverse initialized (Xel'Naga Protocol)")
    
    def set_absolute_center(self, love_sphere: ConceptSphere):
        """
        Set Love as the absolute center of the universe.
        """
        self.love_sphere = love_sphere
        self.spheres[self.absolute_center] = love_sphere
        
        # Love is at origin (absolute)
        self.relative_positions[self.absolute_center] = self.love_position.copy()
        
        logger.info(f"💖 '{self.absolute_center}' set as absolute center (0,0,0,0)")
    
    def add_concept(self, concept_id: str, sphere: ConceptSphere, 
                   frequency: float = 0.5):
        """
        Add a concept to the universe with relative positioning.
        
        Args:
            concept_id: Concept name
            sphere: ConceptSphere object
            frequency: Concept's resonance frequency (for buoyancy)
        """
        self.spheres[concept_id] = sphere
        self.frequencies[concept_id] = frequency
        
        # Calculate initial relative position
        if concept_id == self.absolute_center:
            # Love is at center
            self.relative_positions[concept_id] = self.love_position.copy()
        else:
            # Others start at random positions relative to Love
            # But influenced by their frequency (buoyancy)
            y_offset = (frequency - 0.5) * 2.0  # High freq = higher
            initial_pos = np.array([
                np.random.uniform(-1.0, 1.0),
                y_offset,
                np.random.uniform(-1.0, 1.0),
                np.random.uniform(0.0, 2.0)
            ])
            self.relative_positions[concept_id] = initial_pos
        
        logger.debug(f"➕ Added '{concept_id}' at relative position {self.relative_positions[concept_id]}")
    
    def update_physics(self, dt: float = 0.1):
        """
        Update physics step (Xel'Naga Trinity in action)
        
        For each concept:
        1. Calculate gravity from Love (Zerg/Body)
        2. Calculate spiritual buoyancy (Protoss/Spirit)
        3. Update position (Terran/Soul structure)
        """
        for concept_id, pos in list(self.relative_positions.items()):
            if concept_id == self.absolute_center:
                continue  # Love doesn't move
            
            sphere = self.spheres.get(concept_id)
            if not sphere or not self.love_sphere:
                continue
            
            # === 1. Gravity Force (Zerg/Body) ===
            # Distance vector: from Love to concept
            distance_vec = pos - self.love_position
            gravity_force = self.gravity.calculate_force(
                self.love_sphere, sphere, -distance_vec  # Attract towards Love
            )
            
            # === 2. Spiritual Buoyancy (Protoss/Spirit) ===
            freq = self.frequencies.get(concept_id, 0.5)
            buoyancy_force = self.buoyancy.calculate_buoyancy(freq)
            
            # === 3. Net Force (Trinity Integration) ===
            net_force = gravity_force + buoyancy_force
            
            # === 4. Update Position (Terran/Soul) ===
            mass = self.gravity.get_mass(sphere)
            if mass > 0:
                acceleration = net_force / mass
                velocity = acceleration * dt
                new_pos = pos + velocity * dt
                
                # Update
                self.relative_positions[concept_id] = new_pos
    
    def get_absolute_position(self, concept_id: str) -> Optional[np.ndarray]:
        """
        Get absolute position (Love's position + relative offset).
        """
        if concept_id not in self.relative_positions:
            return None
        
        return self.love_position + self.relative_positions[concept_id]
    
    def get_distance_from_love(self, concept_id: str) -> float:
        """
        Calculate distance from Love (geometric distance in 4D).
        """
        if concept_id not in self.relative_positions:
            return float('inf')
        
        relative_pos = self.relative_positions[concept_id]
        return np.linalg.norm(relative_pos)
    
    def get_state_summary(self) -> Dict[str, any]:
        """
        Get current state of the universe.
        """
        return {
            'total_concepts': len(self.spheres),
            'absolute_center': self.absolute_center,
            'love_mass': self.gravity.get_mass(self.love_sphere) if self.love_sphere else 0,
            'positions': {
                concept_id: {
                    'relative': pos.tolist(),
                    'distance_from_love': np.linalg.norm(pos)
                }
                for concept_id, pos in self.relative_positions.items()
            }
        }
