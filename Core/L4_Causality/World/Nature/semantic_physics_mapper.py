"""
Semantic Physics Mapper (The Word-to-World Bridge)
==================================================
Core.L4_Causality.World.Nature.semantic_physics_mapper

"And the Word became Flesh (Terrain)."
"""

from typing import Dict, Any, Tuple
from Core.L4_Causality.World.Physics.trinity_fields import TrinityVector

class SemanticPhysicsMapper:
    """
    Translates Trinity Vectors into Physical Properties.
    """
    
    @staticmethod
    def map_vector_to_physics(vector: TrinityVector) -> Dict[str, Any]:
        """
        Input: TrinityVector (Gravity, Flow, Ascension)
        Output: Physical Properties (TerrainType, Density, Temperature, etc.)
        """
        # Normalize contributions
        g = vector.gravity
        f = vector.flow
        a = vector.ascension
        
        # 1. Determine State of Matter (Terrain Type)
        terrain_type = "Void"
        visual_color = (0.5, 0.5, 0.5) # Grey default
        
        if g > f and g > a:
            # Gravity Dominant -> SOLID
            if g > 0.8:
                terrain_type = "Bedrock"
                visual_color = (0.2, 0.2, 0.2) # Dark Grey
            else:
                terrain_type = "Earth"
                visual_color = (0.4, 0.3, 0.1) # Brown
        elif f > g and f > a:
            # Flow Dominant -> FLUID
            if f > 0.8:
                terrain_type = "DeepWater"
                visual_color = (0.0, 0.0, 0.8) # Blue
            else:
                terrain_type = "Water"
                visual_color = (0.0, 0.5, 1.0) # Light Blue
        elif a > g and a > f:
            # Ascension Dominant -> GAS/ENERGY
            if a > 0.8:
                terrain_type = "Light/Plasma"
                visual_color = (1.0, 1.0, 0.8) # Glowing
            else:
                terrain_type = "Air"
                visual_color = (0.8, 0.9, 1.0) # Sky Blue
                
        # 2. Resource Mapping
        # High Gravity = Ore
        # High Flow = Mana/Energy
        # High Ascension = Spirit/Buffs
        
        resources = {
            "Ore": max(0, g * 100),
            "Mana": max(0, f * 100),
            "Spirit": max(0, a * 100)
        }
        
        # 3. Frequency -> Temperature
        # High Freq = Hot
        # Low Freq = Cold
        temp = 20.0 + (vector.frequency / 10.0) # Base 20C + modifier
        
        return {
            "type": terrain_type,
            "visual": {"color": visual_color},
            "physics": {"density": g * 10.0, "viscosity": 1.0/max(0.1, f)},
            "resources": resources,
            "climate": {"temperature": temp}
        }