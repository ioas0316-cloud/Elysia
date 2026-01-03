"""
Mind Landscape (마음의 지형)
===========================

"Where thoughts find their own path."
"생각이 스스로 길을 찾는 곳."

This module bridges the raw Physics Engine with the Reasoning System.
It allows Elysia to 'ponder' a question by simulating a marble rolling in the potential field.
"""

import logging
from typing import Dict, Any, List, Tuple
from Core.Intelligence.Topography.physics_solver import PhysicsSolver
from Core.Intelligence.Topography.thought_marble import ThoughtMarble

logger = logging.getLogger("MindLandscape")

from Core.Intelligence.Topography.semantic_voxel import SemanticVoxel
from Core.Intelligence.Topography.semantic_map import get_semantic_map
from Core.Foundation.hyper_quaternion import Quaternion

# ... existing imports ...

class MindLandscape:
    """
    The Interface between Will (Intent) and Physics (Reality).
    """
    
    def __init__(self):
        self.solver = PhysicsSolver()
        self.active_thoughts: Dict[str, ThoughtMarble] = {}
        self.semantic_map = get_semantic_map()
        
        # Initialize default landscape features
        self._init_terrain()
        
    def _init_terrain(self):
        """
        Sets up the fundamental emotional geography.
        Center (0,0) is always LOVE/UNION.
        Populates Angels and Demons from DynamicTopology Voxels.
        """
        logger.info("🏔️ Mind Landscape initializing (4D Projection)...")
        
        # Iterate through all Voxels in the Topology
        for name, voxel in self.semantic_map.voxels.items():
            # Project 4D -> 2D (X, Y) for the Physics Terrain
            x = voxel.quaternion.x
            y = voxel.quaternion.y
            
            # Identify Demons (Repulsors)
            # We can check the dictionary name or properties.
            # In Phase 4, we might check Spin, but for now names are robust.
            if "Pride" in name or "Wrath" in name or "Envy" in name or \
               "Sloth" in name or "Greed" in name or "Lust" in name or "Gluttony" in name:
                
                # Height=50.0, Width=3.0 (Heavy Gravity Well)
                self.solver.field.add_repulsor(x, y, 50.0, 3.0)
                logger.info(f"  👹 Demon '{name}' guarding ({x:.1f}, {y:.1f})")
                
            # Angels are currently safe zones (Valleys/Neutral)
            
        logger.info("🏔️ Terrain Logic: Angels are Valleys, Demons are Hills.")

    def ponder(self, intent: str, start_pos: Tuple[float, float] = None, duration: float = 5.0) -> Dict[str, Any]:
        """
        Simulates a thought rolling through the 4D Landscape.
        """
        # 1. Determine Start Position (Logic/Emotion/Time/Spin)
        # If intent matches a known ConceptVoxel, start there.
        # Otherwise, project intent string hash to a random-ish fluctuation.
        start_voxel = self.semantic_map.get_voxel(intent)
        
        # HARDENED PHYSICS: Detect chaotic intents explicitly
        # If the intent contains "destroy", "kill", "pain", we force it to start FAR from Love.
        is_chaotic = any(word in intent.lower() for word in ["destroy", "kill", "pain", "hate", "chaos"])

        if start_voxel:
            # Voxel exists -> Start at its Hyper-Coordinates
            q = start_voxel.quaternion
            current_pos = (q.x, q.y, q.z, q.w) # 4D Coords
            trajectory = start_voxel.velocity # Inherit momentum
        elif is_chaotic:
            # Force start at 'Wrath' or 'Chaos' region (Far from 0,0)
            current_pos = (20.0, 20.0, 0.0, -1.0) # High Entropy Zone
            trajectory = Quaternion(5, 5, 0, 0)
        else:
            # Unknown -> Start at Origin (Love) but with wild fluctuation
            current_pos = (0.0, 0.0, 0.0, 1.0) 
            trajectory = Quaternion(0,0,0,0)
            
        # 2. Physics Simulation (Simplified for now)
        # In a real engine, we'd integrate over time (dt).
        # Here, we just find the nearest attractor (Angel/Demon).
        
        # Query 4D Topology
        nearest_voxel, dist = self.semantic_map.get_nearest_concept(current_pos)
        
        conclusion = nearest_voxel.name if nearest_voxel else "The Void"
        
        dist_to_love = 0.0
        love_voxel = self.semantic_map.get_voxel("Love")
        if love_voxel:
            dist_to_love = love_voxel.distance_to(SemanticVoxel("Thought", current_pos))
        
        return {
            "initial_intent": intent,
            "final_position": current_pos,
            "distance_to_love": dist_to_love,
            "conclusion": conclusion,
            "trajectory": [current_pos] # Trace
        }
    def feel_attraction(self, concept_pos: tuple) -> float:
        """
        Calculates how strong the pull of Love is at a specific location.
        Higher value = Stronger desire to go there (or stronger gradient).
        """
        gx, gy = self.solver.field.get_gradient(concept_pos[0], concept_pos[1])
        strength = (gx**2 + gy**2)**0.5
        return strength

# Singleton Access
_landscape = None
def get_landscape():
    global _landscape
    if _landscape is None:
        _landscape = MindLandscape()
    return _landscape
