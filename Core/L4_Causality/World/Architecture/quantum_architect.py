"""
Quantum Architect (The WFC Engine)
==================================
Core.L4_Causality.World.Architecture.quantum_architect

"The observer decides the form."

This module implements the Wave Function Collapse logic to extrude 2D intent into 3D reality.
"""

import numpy as np
import logging
from Core.L4_Causality.World.Architecture.spatial_memory import SpatialMemory
from Core.L4_Causality.World.Architecture.blueprint_analyzer import BlueprintGenerator

logger = logging.getLogger("QuantumArchitect")

class QuantumArchitect:
    """
    The Engine that collapses probability into geometry.
    """

    VOXEL_AIR = 0
    VOXEL_WALL = 1
    VOXEL_FLOOR = 2
    VOXEL_CEILING = 3
    VOXEL_DOOR = 4
    VOXEL_WINDOW = 5

    def __init__(self):
        self.memory = SpatialMemory()

    def collapse_space(self, blueprint: np.ndarray, height: int = 3) -> dict:
        """
        Extrudes the 2D blueprint into 3D voxels based on Quantum Rules.
        Returns a dictionary with 'voxels', 'obj_data', and 'memory_address'.
        """
        rows, cols = blueprint.shape
        voxels = np.zeros((height, rows, cols), dtype=int)

        # Apply Wave Function Constraints (Simplified WFC)
        for r in range(rows):
            for c in range(cols):
                tile_type = blueprint[r, c]

                if tile_type == BlueprintGenerator.WALL:
                    # Wall is solid from bottom to top
                    voxels[:, r, c] = self.VOXEL_WALL

                elif tile_type == BlueprintGenerator.DOOR:
                    # Door is empty space at bottom, wall/lintel above
                    voxels[0, r, c] = self.VOXEL_FLOOR # Threshold
                    voxels[1, r, c] = self.VOXEL_DOOR  # Open space
                    voxels[2:, r, c] = self.VOXEL_WALL # Lintel

                elif tile_type == BlueprintGenerator.WINDOW:
                    # Window is wall below/above, glass in middle
                    voxels[0, r, c] = self.VOXEL_WALL
                    voxels[1, r, c] = self.VOXEL_WINDOW
                    voxels[2:, r, c] = self.VOXEL_WALL

                else: # EMPTY
                    # Floor at bottom, Ceiling at top, Air in between
                    voxels[0, r, c] = self.VOXEL_FLOOR
                    voxels[-1, r, c] = self.VOXEL_CEILING
                    voxels[1:-1, r, c] = self.VOXEL_AIR

        # Generate OBJ Representation
        obj_data = self._generate_obj(voxels)

        # Crystallize in Akashic Records
        meta = {
            "type": "Architecture",
            "grid_size": list(blueprint.shape),
            "height": height
        }
        address = self.memory.crystallize(obj_data, meta)

        return {
            "voxels": voxels,
            "obj": obj_data,
            "address": address
        }

    def _generate_obj(self, voxels: np.ndarray) -> str:
        """Converts Voxel Grid to Wavefront OBJ format string."""
        vertices = []
        faces = []

        h, r, c = voxels.shape

        # Simple Cube Meshing (Greedy-ish)
        # For prototype, just emit cubes for non-air blocks

        def add_cube(x, y, z):
            base_idx = len(vertices) + 1
            # 8 vertices for a cube
            # (Simplification: Just storing center points or generating proper cubes?
            #  Let's generate unit cubes)
            vs = [
                (x, y, z), (x+1, y, z), (x+1, y+1, z), (x, y+1, z),     # Bottom
                (x, y, z+1), (x+1, y, z+1), (x+1, y+1, z+1), (x, y+1, z+1) # Top
            ]
            vertices.extend(vs)

            # 6 Faces (Quad)
            # f v1 v2 v3 v4
            fs = [
                (base_idx, base_idx+1, base_idx+2, base_idx+3), # Bottom
                (base_idx+4, base_idx+5, base_idx+6, base_idx+7), # Top
                (base_idx, base_idx+1, base_idx+5, base_idx+4), # Front
                (base_idx+1, base_idx+2, base_idx+6, base_idx+5), # Right
                (base_idx+2, base_idx+3, base_idx+7, base_idx+6), # Back
                (base_idx+3, base_idx, base_idx+4, base_idx+7)  # Left
            ]
            faces.extend(fs)

        for z in range(h):
            for y in range(r):
                for x in range(c):
                    v = voxels[z, y, x]
                    if v != self.VOXEL_AIR and v != self.VOXEL_DOOR:
                        # Treating DOOR as Air for visual mesh (it's a hole)
                        # Treating WINDOW as solid for now (or maybe separate material)
                        add_cube(x, y, z)

        lines = ["# Quantum Pop-Up Generated OBJ"]
        for v in vertices:
            lines.append(f"v {v[0]} {v[1]} {v[2]}")
        for f in faces:
            lines.append(f"f {f[0]} {f[1]} {f[2]} {f[3]}")

        return "\n".join(lines)