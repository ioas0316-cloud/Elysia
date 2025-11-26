from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
import random

@dataclass
class ResourceNode:
    type: str # "Wood", "Stone", "Metal"
    amount: int
    position: np.ndarray

@dataclass
class Material:
    name: str
    properties: Dict[str, float] # e.g., {"Hardness": 0.8, "Sharpness": 0.5}

@dataclass
class Item:
    name: str
    components: List[str]
    properties: Dict[str, float]
    durability: int = 100

class MaterialSystem:
    def __init__(self):
        self.base_materials = {
            "Wood": Material("Wood", {"Flexibility": 0.6, "Hardness": 0.4, "Flammability": 0.8}),
            "Stone": Material("Stone", {"Hardness": 0.9, "Sharpness": 0.2, "Weight": 0.8}),
            "Metal": Material("Metal", {"Hardness": 0.95, "Conductivity": 1.0, "Weight": 1.0}),
        }

    def get_material(self, name: str) -> Optional[Material]:
        return self.base_materials.get(name)

    def combine(self, ingredients: List[str]) -> Optional[Item]:
        """
        Dynamically synthesizes an item from ingredients based on their properties.
        This is NOT a recipe lookup. It is a physics simulation.
        """
        if not ingredients:
            return None
            
        # 1. Aggregate Properties
        combined_props = {}
        total_weight = 0.0
        
        mats = [self.base_materials.get(i) for i in ingredients if i in self.base_materials]
        if len(mats) != len(ingredients):
            return None # Unknown material
            
        for mat in mats:
            for k, v in mat.properties.items():
                combined_props[k] = combined_props.get(k, 0.0) + v
            total_weight += mat.properties.get("Weight", 0.1)

        # 2. Emergent Logic (The "Physics" of Crafting)
        # Interaction: Hardness + Sharpness -> Cutting Power
        if "Hardness" in combined_props and "Sharpness" in combined_props:
            combined_props["Cutting"] = (combined_props["Hardness"] * combined_props["Sharpness"]) / len(ingredients)
            
        # Interaction: Hardness + Weight -> Bludgeoning Power
        if "Hardness" in combined_props and "Weight" in combined_props:
            combined_props["Bludgeoning"] = (combined_props["Hardness"] * combined_props["Weight"]) / len(ingredients)

        # Interaction: Flexibility + Hardness -> Structure (Tensile Strength)
        if "Flexibility" in combined_props and "Hardness" in combined_props:
            combined_props["Structure"] = (combined_props["Flexibility"] + combined_props["Hardness"]) / 2.0

        # 3. Determine Item Type & Name
        # The name is emergent based on dominant property
        dominant_prop = max(combined_props, key=combined_props.get)
        
        name_suffix = f"{random.randint(100,999)}"
        if dominant_prop == "Cutting":
            name = f"Cutter_{name_suffix}"
        elif dominant_prop == "Bludgeoning":
            name = f"Hammer_{name_suffix}"
        elif dominant_prop == "Structure":
            name = f"Shelter_{name_suffix}"
        else:
            name = f"Thing_{name_suffix}"
            
        return Item(name, ingredients, combined_props)
