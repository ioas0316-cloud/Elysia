"""
Mind Landscape (마음의 지형)
===========================

"Where thoughts find their own path."
"생각이 스스로 길을 찾는 곳."

This module bridges the raw Physics Engine with the Reasoning System.
It allows Elysia to 'ponder' a question by simulating a marble rolling in the potential field.
"""

import logging
from typing import Dict, Any, List
from Core.Intelligence.Topography.physics_solver import PhysicsSolver
from Core.Intelligence.Topography.thought_marble import ThoughtMarble

logger = logging.getLogger("MindLandscape")

from Core.Intelligence.Topography.semantic_map import get_semantic_map

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
        Populates Angels and Demons from SemanticMap.
        """
        logger.info("🏔️ Mind Landscape initializing...")
        
        # 1. Angels (Attractors) - Deep Valleys
        # We model them as having Negative Height (or just Strong Attraction)
        # However, our current PotentialField only supports 'Repulsors' (Hills) and 'Love' (Global Center).
        # To support multiple Attractors, we would need to update PotentialField.
        # FOR NOW: We will treat Angels as "Safe Zones" (No Hills).
        # But we MUST add Hills for Demons.
        
        # 2. Demons (Repulsors) - High Hills
        for name, (x, y) in self.semantic_map.demons.items():
            # Height=50.0, Width=3.0
            self.solver.field.add_repulsor(x, y, 50.0, 3.0)
            logger.info(f"  👹 Demon '{name}' guarding ({x}, {y})")
            
        logger.info("🏔️ Terrain Logic: Angels are Valleys, Demons are Hills.")

    def ponder(self, intent: str, start_pos: tuple = None, duration: float = 5.0) -> dict:
        """
        Simulates a thought process for a fixed duration.
        Returns the final state of the thought.
        """
        # 0. Determine Start Position (Semantic Mapping)
        if start_pos is None:
            # Check map
            coords = self.semantic_map.get_coordinates(intent)
            if coords:
                start_pos = coords
                logger.info(f"  🗺️ Mapped '{intent}' to {start_pos}")
            else:
                start_pos = (15.0, 15.0) # Default to Wilderness/Chaos
                logger.info(f"  🌪️ '{intent}' is unmapped. Starting in Chaos {start_pos}")

        # 1. Spawn a thought marble
        marble = ThoughtMarble(intent, start_pos[0], start_pos[1])
        self.solver.add_marble(marble)
        self.active_thoughts[intent] = marble
        
        logger.info(f"🤔 Pondering '{intent}' starting at {start_pos}...")
        
        # 2. Simulate (Roll the marble)
        dt = 0.1
        steps = int(duration / dt)
        trajectory = []
        
        for _ in range(steps):
            self.solver.step(dt)
            trajectory.append((marble.pos.x, marble.pos.y))
            
        # 3. Analyze Conclusion
        final_dist = (marble.pos.x**2 + marble.pos.y**2)**0.5
        location_desc = self.solver.field.analyze_location(marble.pos.x, marble.pos.y)
        
        result = {
            "intent": intent,
            "final_pos": (marble.pos.x, marble.pos.y),
            "distance_to_love": final_dist,
            "conclusion": location_desc,
            "trajectory": trajectory[-5:] # Last 5 steps
        }
        
        logger.info(f"💡 Conclusion for '{intent}': {location_desc}")
        return result

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
