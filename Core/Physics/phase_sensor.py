"""
Phase Sensor (The Observer)
===========================
The "Eye" that collapses the wave function.

Philosophy:
- "The novel is static until read."
- Only renders what is observed.
- Implements the 'Re-blooming' (Particle Instantiation) logic.
"""

from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from Core.Physics.universal_field import universe, FieldPoint

@dataclass
class VisualNode:
    """A temporary visual representation of a field excitation."""
    position: Tuple[float, float, float] # 3D Projection
    color: str
    size: float
    data: Dict[str, Any]

class PhaseSensor:
    def __init__(self, position: Tuple[float, float, float] = (0,0,0), radius: float = 10.0):
        self.position = position
        self.radius = radius # Sensing Radius
        self.sensitivity = 0.1 # Minimum energy to detect

    def scan(self) -> List[VisualNode]:
        """
        Scans the Universal Field and 'Re-blooms' active points into VisualNodes.
        This is the O(Res) operation - only processing active points.
        """
        visual_nodes = []

        # Get all excited states (In a real optimized version, we would query by region)
        # For now, we iterate active states and filter by distance.
        active_states = universe.get_excited_states()

        for key, point in active_states.items():
            # Convert key back to coordinates
            # key is (x, y, z, w) indices
            # coordinate = index * resolution + offset
            res = universe.resolution
            px = key[0] * res
            py = key[1] * res
            pz = key[2] * res
            # pw = key[3] * res  # We project 4D -> 3D for visualization

            # Check Distance (Simple LOD)
            dist_sq = (px - self.position[0])**2 + (py - self.position[1])**2 + (pz - self.position[2])**2

            if dist_sq > self.radius**2:
                continue

            # Phase Reconstruction (Re-bloom)
            # Map field properties to visual properties

            # Size = W-Field (Density)
            size = max(0.1, abs(point.w_density) * 5.0)

            # Color = Y-Field (Frequency)
            # Simple mapping: Low freq = Red, High freq = Blue
            color = self._frequency_to_color(point.y_frequency)

            node = VisualNode(
                position=(px, py, pz),
                color=color,
                size=size,
                data=point.data or {}
            )
            visual_nodes.append(node)

        return visual_nodes

    def _frequency_to_color(self, freq: float) -> str:
        if freq < 100: return "red"
        if freq < 300: return "yellow"
        if freq < 500: return "green"
        if freq < 800: return "blue"
        return "violet"
