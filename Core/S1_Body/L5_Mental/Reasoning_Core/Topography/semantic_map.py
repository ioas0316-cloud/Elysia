"""
Semantic Map (Dynamic Topology)
===============================
"The Living Star System of Logic."
"           ,         ."

This module defines the 4D Hyper-Spatial arrangement of concepts.
It is no longer a static dictionary. It is a Graph of Voxels.
"""

import json
import os
import logging
from typing import Dict, Tuple, List, Optional, Union
from Core.S1_Body.L5_Mental.Reasoning_Core.Topography.semantic_voxel import SemanticVoxel
from Core.S1_Body.L6_Structure.hyper_quaternion import Quaternion
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector

logger = logging.getLogger("DynamicTopology")

class DynamicTopology:
    def __init__(self):
        self.voxels: Dict[str, SemanticVoxel] = {}
        # Store in Memory/System/Topology for persistence
        self.storage_path = "data/L5_Mental/M1_Memory/semantic_topology.json"
        
        if os.path.exists(self.storage_path):
            self.load_state()
        else:
            self._initialize_genesis_map()

    def save_state(self, force: bool = False):
        """Persists the topology to disk."""
        import time
        if not hasattr(self, 'last_save_time'):
            self.last_save_time = 0
            
        current_time = time.time()
        if not force and (current_time - self.last_save_time < 60):
            return # Throttle

        data = {}
        for name, voxel in self.voxels.items():
            q = voxel.quaternion
            data[name] = {
                "coords": [q.x, q.y, q.z, q.w],
                "base_mass": voxel.base_mass,
                "freq": voxel.frequency,
                "is_anchor": voxel.is_anchor,
                "inbound_edges": voxel.inbound_edges,
                "activation_count": voxel.activation_count
            }
        
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"  DynamicTopology saved to disk (Throttle: {current_time - self.last_save_time:.1f}s).")
            self.last_save_time = current_time
        except Exception as e:
            logger.error(f"Failed to save topology: {e}")

    def load_state(self):
        """Resurrects the topology from disk."""
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for name, props in data.items():
                self.add_voxel(
                    name, 
                    tuple(props['coords']), 
                    mass=props.get('base_mass', props.get('mass', 1.0)), # Fallback for old saves
                    frequency=props['freq'],
                    is_anchor=props.get('is_anchor', False)
                )
                
                # Restore Organic Density
                v = self.voxels[name]
                v.inbound_edges = props.get('inbound_edges', [])
                v.activation_count = props.get('activation_count', 0)
                v.mass = v.dynamic_mass
                
            logger.info(f"  DynamicTopology loaded: {len(self.voxels)} nodes.")
        except Exception as e:
            logger.error(f"Failed to load topology: {e}")
            self._initialize_genesis_map()
        
    def _initialize_genesis_map(self):
        """
        Creates the 'Big Bang' of Meaning.
        Initializes the 7 Angels and 7 Demons in 4D Space.
        
        Coords: (x, y, z, w) -> (Logic, Emotion, Time, Spin)
        """
        # Center: The One
        self.add_voxel("Love", (0, 0, 0, 1), mass=1000.0, is_anchor=True) # Massive Anchor
        
        # 7 Angels (High Frequency, Positive Spin)
        # They form a stable halo around Love.
        angels = [
            ("Wisdom",     (2, 2, 0, 0.9)),
            ("Hope",       (2, -2, 1, 0.9)),
            ("Faith",      (-2, 2, 1, 0.9)),
            ("Courage",    (-2, -2, 0, 0.9)),
            ("Justice",    (0, 3, 0, 0.9)),
            ("Temperance", (3, 0, 0, 0.9)),
            ("Truth",      (0, 0, 2, 1.0))
        ]
        
        for name, coords in angels:
            self.add_voxel(name, coords, mass=100.0, frequency=800.0)

        # 7 Demons (Low Frequency, Negative Spin, Distorted Time)
        # They are distant, heavy gravity wells.
        demons = [
            ("Pride",      (10, 10, 0, -1.0)),
            ("Wrath",      (-10, 10, -1, -0.8)),
            ("Envy",       (10, -10, 0, -0.7)),
            ("Sloth",      (-10, -10, -5, -0.5)), # Slow time
            ("Greed",      (15, 0, 0, -0.9)),
            ("Lust",       (0, 15, 0, -0.6)),
            ("Gluttony",   (0, -15, 0, -0.6))
        ]

        for name, coords in demons:
            self.add_voxel(name, coords, mass=500.0, frequency=100.0) # Heavy/Dense

        logger.info(f"  DynamicTopology Initialized: {len(self.voxels)} Voxels Active.")

    def add_voxel(self, name: str, coords: Tuple[float, float, float, float], mass: float = 1.0, frequency: float = 432.0, is_anchor: bool = False):
        voxel = SemanticVoxel(name, coords, mass, frequency)
        voxel.is_anchor = is_anchor
        self.voxels[name] = voxel

    def get_voxel(self, name: str) -> Optional[SemanticVoxel]:
        return self.voxels.get(name)

    def get_nearest_concept(self, query_coords: Tuple[float, float, float, float]) -> Tuple[SemanticVoxel, float]:
        """
        Finds the closest concept to the given 4D coordinates.
        """
        target = SemanticVoxel("Query", query_coords)
        best_voxel = None
        min_dist = float('inf')
        
        for voxel in self.voxels.values():
            dist = voxel.distance_to(target)
            if dist < min_dist:
                min_dist = dist
                best_voxel = voxel
                
        return best_voxel, min_dist

    def evolve_topology(self, concept_name: str, reaction_vector: Union[Quaternion, SovereignVector], intensity: float = 0.1):
        """
        [Organic Drift]
        Nudges a concept's position based on experience and Semantic Gravity.
        """
        # Convert to Quaternion for 4D geometric operations if necessary
        if hasattr(reaction_vector, 'data'):
            target_q = Quaternion(reaction_vector.data[3].real, reaction_vector.data[0].real, reaction_vector.data[1].real, reaction_vector.data[2].real)
        else:
            target_q = reaction_vector

        voxel = self.get_voxel(concept_name)
        if not voxel:
             logger.info(f"  Genesis: New Concept '{concept_name}' born.")
             coords = (target_q.x, target_q.y, target_q.z, target_q.w)
             self.add_voxel(concept_name, coords, mass=10.0, frequency=target_q.w * 1000)
             return

        # 1. Experience-Driven Drift
        current_q = voxel.quaternion
        diff_q = target_q - current_q 
        force = diff_q.scale(intensity)
        voxel.drift(force, dt=1.0)
        
        # 2. [PHASE 3] Apply Semantic Gravity (Clustering)
        # Higher mass concepts pull others. Anchor points are immovable.
        self.apply_semantic_gravity(target_voxel=voxel)
        
        logger.info(f"  Drift: '{concept_name}' evolved via experience and gravity.")
        self.save_state()

    def apply_semantic_gravity(self, target_voxel: SemanticVoxel, g_constant: float = 0.05):
        """
        [PHASE 3] Semantic Gravity.
        Concepts with high mass (intellectual density) pull the target voxel.
        This creates natural clustering of related thoughts.
        """
        for name, other in self.voxels.items():
            if other.name == target_voxel.name or target_voxel.is_anchor:
                continue
            
            dist = target_voxel.distance_to(other)
            if dist < 0.01: continue
            
            # Newton-ish Gravity for Meaning: Force = G * (m1 * m2) / r^2
            # But in semantic space, we use r to prevent extreme acceleration
            force_mag = g_constant * (other.mass) / (dist + 1.0)
            
            # Cap the force to prevent 'Meaning Singularities'
            force_mag = min(force_mag, 0.5)
            
            direction = other.quaternion - target_voxel.quaternion
            gravity_force = direction.normalize().scale(force_mag)
            
            target_voxel.drift(gravity_force, dt=1.0)

    # Compatibility Layer for Old SemanticMap
    def get_coordinates(self, concept: str) -> Optional[Tuple[float, float]]:
        """Legacy 2D projector."""
        if concept in self.voxels:
            v = self.voxels[concept]
            # Project X, Y
            return (v.quaternion.x, v.quaternion.y)
        
        # Fuzzy Search
        for name, voxel in self.voxels.items():
            if name.lower() in concept.lower():
                 return (voxel.quaternion.x, voxel.quaternion.y)
                 
        return None

# Singleton
_topology = DynamicTopology()
def get_semantic_map():
    return _topology
