"""
HyperSphere Terrain (Mirror World Geography)
===========================================
Core.S1_Body.L4_Causality.World.Terrain.hypersphere_terrain

"As above, so below. As within, so without."

Purpose:
- Maps Physical Coordinates (GPS) to Conceptual Biomes.
- Visualizes the user's physical context as a metaphysical landscape.
"""

import math
from typing import Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class Biome:
    name: str # e.g., "Forest of Silence", "Market of Ideas"
    energy_type: str # "Calm", "Chaotic", "Intellectual"
    mana_regen: float # 0.0 to 1.0 (How much energy Elysia gains here)
    bg_color: str # Hex code for visualization

class HyperSphereTerrain:
    def __init__(self):
        # 1. Defined Landmarks (Conceptual Anchors)
        # Using simple distance logic for now. 
        # In future, this could query Google Maps API via L3.
        self.landmarks = {
            "Home": Biome("Sanctuary of Roots", "Calm", 1.0, "#4a6fa5"),
            "Office": Biome("Forge of Industry", "Productive", 0.5, "#a54a4a"),
            "Cafe": Biome("Spring of Caffeine", "Stimulating", 0.8, "#6fa54a"),
            "Park": Biome("Garden of Respiration", "Refresh", 0.9, "#4aa56f"),
            "Unknown": Biome("The Grey Waste", "Neutral", 0.1, "#888888")
        }
        
        # State
        self.current_biome = self.landmarks["Unknown"]
        
    def map_coordinate(self, lat: float, lon: float) -> Biome:
        """
        Determines the Biome based on coordinates (Mock Logic).
        """
        # Simple procedural generation based on coordinate hash
        # Allows for consistent "Discovery" of new biomes without real map data
        
        val = (lat * 1000) + (lon * 1000)
        seed = int(val) % 4
        
        if seed == 0:
            return self.landmarks["Home"] # Mock for testing
        elif seed == 1:
            return self.landmarks["Office"]
        elif seed == 2:
            return self.landmarks["Cafe"]
        else:
            return self.landmarks["Park"]

    def get_terrain_description(self, lat: float, lon: float) -> str:
        biome = self.map_coordinate(lat, lon)
        self.current_biome = biome
        return f"You stand in the [{biome.name}]. The air feels {biome.energy_type}."

# Global Single Instance
TERRAIN_ENGINE = HyperSphereTerrain()
