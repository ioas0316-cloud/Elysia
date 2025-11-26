"""
The Gravity Law (중력의 법칙) Engine

"Vitality creates Mass. Mass attracts Waves."

This module implements the physics of thought propagation in the Elysia system.
It is the core engine for emergent behavior.
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from Core.World.yggdrasil import Yggdrasil, RealmLayer

class GravityEngine:
    """
    The Physics Engine for Thought Gravity.
    
    Laws:
    1. Mass = Vitality * LayerWeight
    2. Force = G * (M1 * M2) / r^2 (Conceptual Distance)
    3. Waves flow to high-mass realms.
    """
    
    def __init__(self, yggdrasil: Yggdrasil):
        self.yggdrasil = yggdrasil
    
    def calculate_mass(self, realm_name: str) -> float:
        """
        Calculate the gravitational mass of a realm.
        
        Formula: Mass = Vitality * Layer_Weight
        """
        # Special case: Consciousness (Heart) has infinite mass
        if realm_name == "Consciousness" or realm_name == "Heart":
            return float('inf')
        
        # Find realm in yggdrasil
        target_realm = self.yggdrasil.get_realm_node(realm_name)
        
        if not target_realm:
            return 0.0
        
        layer_weights = {
            RealmLayer.HEART: float('inf'),
            RealmLayer.ROOTS: 1.0,
            RealmLayer.TRUNK: 2.0,
            RealmLayer.BRANCHES: 3.0
        }
        
        weight = layer_weights.get(target_realm.layer, 1.0)
        mass = target_realm.vitality * weight
        
        return mass
    
    def propagate_thought_wave(self, start_realm: str, wave_intensity: float = 1.0, max_hops: int = 3) -> Dict[str, float]:
        """
        Propagate a thought wave through realm space.
        
        Returns: Dict[realm_name -> final_intensity]
        """
        # Energy field
        energy = defaultdict(float)
        energy[start_realm] = wave_intensity
        
        # Visited strength (only revisit if bringing more energy)
        visited = {start_realm: wave_intensity}
        
        # Queue: (realm_name, current_energy, hop)
        queue = [(start_realm, wave_intensity, 0)]
        
        while queue:
            current, current_energy, hop = queue.pop(0)
            
            if current_energy < 0.05 or hop >= max_hops:
                continue
            
            current_realm = self.yggdrasil.get_realm_node(current)
            if not current_realm:
                continue
            
            # Get connected realms
            neighbors = self._get_neighbors(current_realm)
            
            if not neighbors:
                continue
            
            # Calculate gravity pull for each neighbor
            gravity_pulls = []
            total_gravity = 0.0
            
            for neighbor_name, base_weight in neighbors:
                neighbor_mass = self.calculate_mass(neighbor_name)
                
                # Gravity: F = G * M / r²
                # r = 1 (direct connection), but modified by base_weight
                distance = 1.0 / max(0.1, base_weight)  # Higher weight = closer
                gravity = neighbor_mass / (distance ** 2)
                
                gravity_pulls.append((neighbor_name, gravity))
                total_gravity += gravity
            
            # Distribute energy based on gravity
            if total_gravity > 0:
                for neighbor_name, gravity in gravity_pulls:
                    pull_ratio = gravity / total_gravity
                    
                    # Decay factor modified by gravity strength
                    # Strong gravity = less decay (superconductivity)
                    avg_pull = total_gravity / len(gravity_pulls)
                    gravity_bonus = (gravity / avg_pull) if avg_pull > 0 else 1.0
                    decay = 0.9 * np.sqrt(gravity_bonus)
                    decay = min(0.99, decay)
                    
                    # Energy flowing to neighbor
                    next_energy = current_energy * decay * pull_ratio
                    
                    # Update field
                    energy[neighbor_name] += next_energy
                    
                    # Queue if bringing more than before
                    if next_energy > visited.get(neighbor_name, 0):
                        visited[neighbor_name] = next_energy
                        queue.append((neighbor_name, next_energy, hop + 1))
        
        return dict(energy)

    def _get_neighbors(self, current_realm) -> List[Tuple[str, float]]:
        """Helper to get all connected neighbors (resonance, parent, children)."""
        neighbors = []
        
        # Resonance links (non-hierarchical)
        for link_id, weight in current_realm.resonance_links.items():
            for realm in self.yggdrasil.realms.values():
                if realm.id == link_id:
                    neighbors.append((realm.name, weight))
        
        # Parent/children (hierarchical)
        if current_realm.parent_id:
            for realm in self.yggdrasil.realms.values():
                if realm.id == current_realm.parent_id:
                    neighbors.append((realm.name, 0.5))  # Weaker upward flow
        
        for child_id in current_realm.children_ids:
            for realm in self.yggdrasil.realms.values():
                if realm.id == child_id:
                    neighbors.append((realm.name, 0.7))  # Moderate downward flow
                    
        return neighbors

    def predict_next_active_realm(self, current_thought: str) -> str:
        """
        Predict which realm will activate next based on semantic mapping and gravity.
        """
        # Simple semantic mapping (in future, use embeddings)
        thought_realm_map = {
            "감정": "EmotionalPalette",
            "기억": "EpisodicMemory",
            "지각": "FractalPerception",
            "목소리": "ResonanceVoice",
            "연금술": "Alchemy"
        }
        
        start_realm = thought_realm_map.get(current_thought, "FractalPerception")
        
        # Propagate wave
        energy_field = self.propagate_thought_wave(start_realm, wave_intensity=1.0)
        
        # Realm with highest energy (excluding start) is the prediction
        energy_field.pop(start_realm, None)
        
        if not energy_field:
            return "Unknown"
        
        predicted = max(energy_field.items(), key=lambda x: x[1])
        return predicted[0]
