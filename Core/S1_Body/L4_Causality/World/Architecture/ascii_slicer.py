"""
ASCII Slicing Viewer (The Cyberpunk Retina)
===========================================
Core.S1_Body.L4_Causality.World.Architecture.ascii_slicer

"Rendering is just organizing text."

This module visualizes 3D voxel data in the terminal.
"""

import numpy as np

class AsciiSlicer:
    """
    Renders 3D voxels as 2D text slices.
    """

    # CHAR_MAP: Maps Voxel ID to ASCII char
    CHAR_MAP = {
        0: "  ",  # AIR
        1: "##",  # WALL
        2: "..",  # FLOOR
        3: "^^",  # CEILING
        4: "[]",  # DOOR (Empty but marked)
        5: "==",  # WINDOW
    }

    def render_slice(self, voxels: np.ndarray, z_index: int) -> str:
        """
        Renders a specific Z-layer.
        """
        h, rows, cols = voxels.shape

        if z_index < 0 or z_index >= h:
            return f"[Error] Z-Index {z_index} out of bounds (0-{h-1})"

        layer = voxels[z_index]
        lines = []
        lines.append(f"--- Layer Z={z_index} ---")

        for r in range(rows):
            line = ""
            for c in range(cols):
                val = layer[r, c]
                char = self.CHAR_MAP.get(val, "✨")
                line += char
            lines.append(line)

        return "\n".join(lines)

    def render_all(self, voxels: np.ndarray) -> str:
        """
        Renders all layers.
        """
        h, _, _ = voxels.shape
        outputs = []
        # Render from bottom (Z=0) to top
        for z in range(h):
            outputs.append(self.render_slice(voxels, z))
        return "\n\n".join(outputs)
