"""
Universal Field (The Void)
==========================
The central state container for the "Universal Law Engine".
It manages the 4 fundamental fields (W, X, Y, Z) using a sparse storage mechanism.

Philosophy:
- The Universe is mostly empty (Void).
- We only store 'Excitations' (Non-default values).
- O(Res) complexity instead of O(N).
"""

from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import math

@dataclass
class FieldPoint:
    """Represents the state of a single point in the 4D field."""
    w_density: float = 0.0     # Scale / Density
    x_texture: float = 0.0     # Perception / Texture
    y_frequency: float = 0.0   # Spectrum / Energy
    z_torque: float = 0.0      # Spin / Intention

    # Optional metadata (only created if needed)
    data: Optional[Dict[str, Any]] = None

    @property
    def is_active(self) -> bool:
        """Returns True if the point has significant energy."""
        return (abs(self.w_density) > 0.001 or
                abs(self.x_texture) > 0.001 or
                abs(self.y_frequency) > 0.001 or
                abs(self.z_torque) > 0.001)

class UniversalField:
    """
    The Sparse 4D Field.
    """
    def __init__(self):
        # Sparse Storage: Map coordinates (x,y,z,w_int) to FieldPoint
        # We quantize coordinates to integer keys for spatial hashing.
        self._field: Dict[Tuple[int, int, int, int], FieldPoint] = {}

        # Grid Resolution (The 'Planck Length' of this universe)
        self.resolution = 0.1

    def _quantize(self, val: float) -> int:
        return int(math.floor(val / self.resolution))

    def _get_key(self, x: float, y: float, z: float, w: float) -> Tuple[int, int, int, int]:
        return (
            self._quantize(x),
            self._quantize(y),
            self._quantize(z),
            self._quantize(w)
        )

    def inject_signal(self, position: Tuple[float, float, float, float],
                      w: float = 0, x: float = 0, y: float = 0, z: float = 0,
                      data: Optional[Dict] = None):
        """
        Injects energy into the field at a specific location.
        This "Excites" the vacuum at that coordinate.
        """
        key = self._get_key(*position)

        if key not in self._field:
            self._field[key] = FieldPoint()

        point = self._field[key]
        point.w_density += w
        point.x_texture += x
        point.y_frequency += y
        point.z_torque += z

        if data:
            if point.data is None: point.data = {}
            point.data.update(data)

    def query_field(self, position: Tuple[float, float, float, float]) -> FieldPoint:
        """
        Reads the field state at a location.
        Returns a default (empty) point if the location is in the Void.
        """
        key = self._get_key(*position)
        return self._field.get(key, FieldPoint())

    def get_excited_states(self) -> Dict[Tuple[int, int, int, int], FieldPoint]:
        """
        Returns all currently active points in the universe.
        Used by the PhaseSensor to "Re-bloom" reality.
        """
        return {k: v for k, v in self._field.items() if v.is_active}

    def entropy_tick(self, decay_rate: float = 0.01):
        """
        Applies universal entropy.
        Unsustained energy fades back into the Void.
        """
        keys_to_remove = []
        for key, point in self._field.items():
            point.w_density *= (1.0 - decay_rate)
            point.x_texture *= (1.0 - decay_rate)
            point.y_frequency *= (1.0 - decay_rate)
            point.z_torque *= (1.0 - decay_rate)

            if not point.is_active:
                keys_to_remove.append(key)

        for k in keys_to_remove:
            del self._field[k]

# Global Instance
universe = UniversalField()