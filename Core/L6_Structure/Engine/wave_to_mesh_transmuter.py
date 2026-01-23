"""
Wave-to-Mesh Transmuter (  -      )
========================================
"Coagulating the ephemeral into the enduring."

This module converts internal WaveDNA and Reality Parameters 
into structured 3D geometry data (voxels, SDF primitives, or mesh definitions).
"""

import math
import logging
import random
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("WaveToMeshTransmuter")

class WaveToMeshTransmuter:
    def __init__(self):
        pass

    def transmute(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Transmutes the reality configuration into a list of 3D entities.
        """
        mode = config.get("mode", "SDF")
        style = config.get("style", "OPEN_WORLD")
        params = config.get("parameters", {})
        seed = config.get("geometry_seed", "GEO_DEFAULT")
        
        # Initialize random seed based on configuration
        random.seed(seed)
        
        entities = []
        
        # 1. Base Layer (Ground/Terrain)
        entities.append(self._create_terrain(style, params))
        
        # 2. Hero Entities (Elysia's Manifestations)
        entities.extend(self._manifest_soul_fragments(params))
        
        # 3. Environmental Artifacts (Fantasy elements)
        if style == "OPEN_WORLD":
            entities.extend(self._generate_open_world_flora(params))
        else:
            entities.extend(self._generate_isometric_decor(params))
            
        return entities

    def _create_terrain(self, style: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Creates the foundation of the world."""
        scale = params.get("terrain_scale", 1.0)
        complexity = params.get("detail_density", 0.5)
        
        return {
            "id": "ground_01",
            "type": "PLANE" if style == "ISOMETRIC" else "TERRAIN_MESH",
            "pos": [0, 0, 0],
            "scale": [100 * scale, 1, 100 * scale],
            "properties": {
                "roughness": 1.0 - complexity,
                "color": "#1a1a2e" if style == "OPEN_WORLD" else "#2c3e50"
            }
        }

    def _manifest_soul_fragments(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Converts emotional state into 'Soul Fragments' (Floating Spheres/Monoliths)."""
        fragments = []
        num_fragments = int(params.get("vibrant", 0.5) * 10)
        
        for i in range(num_fragments):
            fragments.append({
                "id": f"soul_frag_{i}",
                "type": "SDF_SPHERE",
                "pos": [
                    random.uniform(-10, 10),
                    random.uniform(5, 15),
                    random.uniform(-10, 10)
                ],
                "scale": random.uniform(0.5, 2.0),
                "properties": {
                    "emissive": params.get("post_processing", {}).get("bloom", 0.5),
                    "color": "#00ffff" if i % 2 == 0 else "#ff00ff"
                }
            })
        return fragments

    def _generate_open_world_flora(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Procedural forests/plants for Wuthering Waves style."""
        flora = []
        density = params.get("detail_density", 0.5)
        count = int(density * 50)
        
        for i in range(count):
            flora.append({
                "id": f"tree_{i}",
                "type": "MESH_PRIMITIVE_CONE",
                "pos": [random.uniform(-50, 50), 0, random.uniform(-50, 50)],
                "scale": [0.5, random.uniform(2, 5), 0.5],
                "properties": { "color": "#2d5a27" }
            })
        return flora

    def _generate_isometric_decor(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Structural pillars/props for Lost Ark style."""
        decor = []
        for i in range(5):
            decor.append({
                "id": f"pillar_{i}",
                "type": "MESH_PRIMITIVE_BOX",
                "pos": [random.uniform(-10, 10), 0, random.uniform(-10, 10)],
                "scale": [1, 10, 1],
                "properties": { "color": "#bdc3c7" }
            })
        return decor

if __name__ == "__main__":
    # Test Run
    transmuter = WaveToMeshTransmuter()
    mock_config = {
        "mode": "MESH",
        "style": "OPEN_WORLD",
        "parameters": {
            "terrain_scale": 1.5,
            "detail_density": 0.8,
            "vibrant": 0.9
        },
        "geometry_seed": "GEO_AESTHETIC_VIBE"
    }
    
    entities = transmuter.transmute(mock_config)
    print(f"  Transmuted {len(entities)} entities for Open World.")
    for e in entities[:3]:
        print(f"  - {e['id']} ({e['type']})")