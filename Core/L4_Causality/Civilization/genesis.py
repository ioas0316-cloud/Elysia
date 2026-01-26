"""
Genesis Environment (The Stage of History)
==========================================
Core.L4_Causality.Civilization.genesis

"Civilization begins where the resource map is uneven."
"""

from typing import Dict, List, Tuple
import random
from Core.L4_Causality.World.Physics.vector_math import Vector3

class Zone:
    def __init__(self, name: str, xy: Tuple[int, int], resource_type: str, richness: float):
        self.name = name
        self.coords = xy
        
        # Resource Pools
        self.resources = {
            "Food": 0.0,
            "Gold": 0.0,
            "Stone": 0.0,
            "Mana": 0.0
        }
        
        # Populate based on Type
        if resource_type == "Valley":
            self.resources["Food"] = 1000.0 * richness
        elif resource_type == "Mountain":
            self.resources["Gold"] = 500.0 * richness
            self.resources["Stone"] = 2000.0 * richness
        elif resource_type == "Forest":
            self.resources["Food"] = 500.0 * richness
            self.resources["Mana"] = 100.0 * richness
            
        self.inhabitants: List[str] = [] # List of Citizen Names

    def harvest(self, resource: str, amount: float) -> float:
        """Safe harvest respecting scarcity."""
        available = self.resources.get(resource, 0.0)
        actual = min(available, amount)
        self.resources[resource] -= actual
        return actual
    
    def regenerate(self):
        """Nature fights back."""
        # Simple regrowth logic
        if self.resources.get("Food") is not None:
            self.resources["Food"] *= 1.05 

class WorldGen:
    def __init__(self, width: int = 5, height: int = 5):
        self.grid: Dict[Tuple[int, int], Zone] = {}
        self.width = width
        self.height = height
        self._generate_world()
        
    def _generate_world(self):
        print(f"  Genesis: Forging a {self.width}x{self.height} World...")
        
        types = ["Valley", "Mountain", "Forest", "Desert"]
        
        for x in range(self.width):
            for y in range(self.height):
                r_type = random.choices(types, weights=[0.4, 0.2, 0.3, 0.1])[0]
                richness = random.uniform(0.5, 2.0)
                
                name = f"{r_type}_{x}_{y}"
                self.grid[(x,y)] = Zone(name, (x,y), r_type, richness)
                
    def get_zone(self, xy: Tuple[int, int]) -> Zone:
        return self.grid.get(xy)
