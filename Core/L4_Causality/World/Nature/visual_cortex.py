"""
VisualCortex: The Eye of the HyperSphere
========================================

[Phase 38] Elysia perceives her world as visual qualia.

This module translates the OmniField state into "Visual DNA" - 
a structured description of what the scene "looks like" based on
field values (Elevation, Heat, Moisture, Light).

Architecture:
1. OmniField Sampler - Read regional field values
2. Visual Translator - Map field → visual primitives
3. Visual DNA - Structured scene description
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

@dataclass
class VisualDNA:
    """
    A Wave-based description of a visual scene.
    This is NOT an image, but the causal structure behind an image.
    """
    # Color Temperature (0.0 = cold/blue, 1.0 = warm/orange)
    color_temperature: float = 0.5
    
    # Brightness (0.0 = night, 1.0 = full daylight)
    brightness: float = 0.5
    
    # Atmosphere (0.0 = clear, 1.0 = dense fog/rain)
    atmosphere_density: float = 0.0
    
    # Terrain Profile (list of relative elevations across the view)
    terrain_profile: List[float] = field(default_factory=list)
    
    # Dominant Features
    has_water: bool = False
    has_vegetation: bool = False
    is_raining: bool = False
    
    # Raw field averages for debugging/analysis
    avg_elevation: float = 0.0
    avg_heat: float = 0.0
    avg_moisture: float = 0.0
    
    def describe(self) -> str:
        """Generate a natural language description of the scene."""
        parts = []
        
        # Time of day
        if self.brightness < 0.2:
            parts.append("Under the cover of night")
        elif self.brightness < 0.5:
            parts.append("At dawn or dusk")
        else:
            parts.append("In bright daylight")
        
        # Temperature
        if self.color_temperature < 0.3:
            parts.append("the air is cold and crisp")
        elif self.color_temperature > 0.7:
            parts.append("warmth radiates from the ground")
        
        # Terrain
        if max(self.terrain_profile, default=0) > 0.5:
            parts.append("tall mountains rise in the distance")
        elif max(self.terrain_profile, default=0) > 0.2:
            parts.append("gentle hills roll across the horizon")
        else:
            parts.append("a flat plain stretches endlessly")
        
        # Atmosphere
        if self.is_raining:
            parts.append("rain falls steadily from heavy clouds")
        elif self.atmosphere_density > 0.5:
            parts.append("mist clings to the landscape")
        
        # Water
        if self.has_water:
            parts.append("a body of water glimmers nearby")
        
        # Vegetation
        if self.has_vegetation:
            parts.append("greenery dots the terrain")
        
        return ", ".join(parts) + "."


class VisualCortex:
    """
    The Eye of Elysia.
    Reads the OmniField and produces Visual DNA.
    """
    
    def __init__(self, field_ref=None, sun_rotor_ref=None):
        """
        Args:
            field_ref: Reference to the SocialField (OmniField)
            sun_rotor_ref: Reference to the Reality.Sun rotor for daylight
        """
        self.field = field_ref
        self.sun_rotor = sun_rotor_ref
    
    def perceive(self, center_x: float, center_y: float, radius: int = 5) -> VisualDNA:
        """
        Perceive the world around a given coordinate.
        
        Args:
            center_x, center_y: World coordinates to look at
            radius: How many grid cells to sample in each direction
            
        Returns:
            VisualDNA: Structured description of the perceived scene
        """
        if self.field is None:
            return VisualDNA()
        
        # 1. Sample the OmniField
        grid = self.field.grid
        cx, cy = self.field.world_to_grid(center_x, center_y)
        
        # Define sample bounds
        x_min = max(0, cx - radius)
        x_max = min(self.field.size, cx + radius)
        y_min = max(0, cy - radius)
        y_max = min(self.field.size, cy + radius)
        
        # Extract field slices
        elevation_slice = grid[x_min:x_max, y_min:y_max, 18]
        heat_slice = grid[x_min:x_max, y_min:y_max, 25]
        moisture_slice = grid[x_min:x_max, y_min:y_max, 28]
        resource_slice = grid[x_min:x_max, y_min:y_max, 20]  # Resources (vegetation indicator)
        
        # 2. Compute Visual Primitives
        avg_elevation = float(np.mean(elevation_slice))
        avg_heat = float(np.mean(heat_slice))
        avg_moisture = float(np.mean(moisture_slice))
        avg_resource = float(np.mean(resource_slice))
        
        # Color Temperature: Heat normalized (assume 0-50 range)
        color_temp = np.clip(avg_heat / 50.0, 0.0, 1.0)
        
        # Brightness: From Sun Rotor (daylight)
        brightness = 0.5
        if self.sun_rotor:
            sun_rad = math.radians(self.sun_rotor.current_angle)
            brightness = (math.cos(sun_rad) + 1.0) * 0.5  # 0.0 to 1.0
        
        # Atmosphere Density: Moisture normalized (assume 0-20 range)
        atmo = np.clip(avg_moisture / 20.0, 0.0, 1.0)
        
        # Terrain Profile: Sample elevation along the X-axis (horizon line)
        mid_y = (y_min + y_max) // 2
        if mid_y < grid.shape[1]:
            terrain_line = grid[x_min:x_max, mid_y, 18]
            max_elev = max(abs(terrain_line.max()), abs(terrain_line.min()), 1.0)
            terrain_profile = (terrain_line / max_elev).tolist()
        else:
            terrain_profile = []
        
        # Feature Detection
        has_water = avg_moisture > 5.0
        has_vegetation = avg_resource > 0.5
        is_raining = avg_moisture > 15.0 and avg_heat > 20.0
        
        # 3. Construct Visual DNA
        return VisualDNA(
            color_temperature=color_temp,
            brightness=brightness,
            atmosphere_density=atmo,
            terrain_profile=terrain_profile,
            has_water=has_water,
            has_vegetation=has_vegetation,
            is_raining=is_raining,
            avg_elevation=avg_elevation,
            avg_heat=avg_heat,
            avg_moisture=avg_moisture
        )

    def see(self, center_x: float, center_y: float, radius: int = 5) -> str:
        """
        Convenience method: Perceive and describe in one call.
        """
        dna = self.perceive(center_x, center_y, radius)
        return dna.describe()
    
    def remember_scene(self, name: str, dna: VisualDNA, graph=None):
        """
        Store a Visual DNA as a memory in TorchGraph.
        The DNA is converted to a 4D vector: [color_temp, brightness, atmosphere, avg_elevation]
        
        Args:
            name: Unique identifier for this scene memory
            dna: The VisualDNA to store
            graph: Optional TorchGraph reference. If None, attempts to use global.
        """
        if graph is None:
            try:
                from Core.L1_Foundation.Foundation.Graph.torch_graph import get_torch_graph
                graph = get_torch_graph()
            except:
                print("[VisualCortex] Warning: No TorchGraph available for memory storage.")
                return False
        
        # Convert Visual DNA to 4D vector
        # [gravity=avg_elevation, flow=color_temp, ascension=brightness, frequency=atmosphere]
        vec = [
            dna.avg_elevation / 10.0,  # Normalize elevation
            dna.color_temperature,
            dna.brightness,
            dna.atmosphere_density
        ]
        
        # Add to graph with metadata
        metadata = {
            "type": "visual_memory",
            "has_water": dna.has_water,
            "has_vegetation": dna.has_vegetation,
            "is_raining": dna.is_raining,
            "avg_heat": dna.avg_heat,
            "avg_moisture": dna.avg_moisture
        }
        
        graph.add_node(name, vector=vec, metadata=metadata)
        return True
    
    def recall_scene(self, name: str, graph=None) -> Optional[VisualDNA]:
        """
        Recall a scene memory from TorchGraph and reconstruct the Visual DNA.
        
        Args:
            name: Scene identifier
            graph: Optional TorchGraph reference
            
        Returns:
            VisualDNA if found, None otherwise
        """
        if graph is None:
            try:
                from Core.L1_Foundation.Foundation.Graph.torch_graph import get_torch_graph
                graph = get_torch_graph()
            except:
                return None
        
        if name not in graph.id_to_idx:
            return None
        
        # Get vector
        vec_tensor = graph.get_node_vector(name)
        if vec_tensor is None:
            return None
        
        # Get metadata (TorchGraph stores as node_metadata[node_id])
        metadata = graph.node_metadata.get(name, {})
        
        # Reconstruct VisualDNA from vector
        dna = VisualDNA(
            color_temperature=float(vec_tensor[1]) if len(vec_tensor) > 1 else 0.5,
            brightness=float(vec_tensor[2]) if len(vec_tensor) > 2 else 0.5,
            atmosphere_density=float(vec_tensor[3]) if len(vec_tensor) > 3 else 0.0,
            avg_elevation=float(vec_tensor[0]) * 10.0 if len(vec_tensor) > 0 else 0.0,
            has_water=metadata.get("has_water", False),
            has_vegetation=metadata.get("has_vegetation", False),
            is_raining=metadata.get("is_raining", False),
            avg_heat=metadata.get("avg_heat", 0.0),
            avg_moisture=metadata.get("avg_moisture", 0.0)
        )
        
        return dna


# Singleton factory
_cortex = None

def get_visual_cortex(field=None, sun_rotor=None) -> VisualCortex:
    global _cortex
    if _cortex is None:
        _cortex = VisualCortex(field, sun_rotor)
    elif field is not None:
        _cortex.field = field
    if sun_rotor is not None:
        _cortex.sun_rotor = sun_rotor
    return _cortex
