"""Warp Layer         (  )   

         `World`            (     )        
Quaternion                      .

       :
-        `world.positions` (N 3) numpy          .
- `Quaternion`         3 D          .
- 2 D   (`value_mass_field`, `will_field`  )          
  `scipy.ndimage.rotate`            ,         
  **      **      .              .

     :
```python
from warp_layer import WarpLayer
from pyquaternion import Quaternion

warp = WarpLayer()
q = Quaternion(axis=[0, 0, 1], angle=1.57)  # 90  Z     
warp.apply(world, q)
```
"""

import logging
import numpy as np
from pyquaternion import Quaternion
from scipy.ndimage import rotate as nd_rotate


class WarpLayer:
    """         (  )       .

           **     **       .      2 D       
    `apply_to_fields=True`              .
    """

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger("WarpLayer")

    def _rotate_vectors(self, vectors: np.ndarray, quat: Quaternion) -> np.ndarray:
        """Nx3     Quaternion     .
        `vectors`   (x, y, z)     ,           .
        """
        # Quaternion expects (w, x, y, z) internally; we use * operator.
        # Convert each vector to quaternion with zero scalar part.
        # Efficient batch rotation via broadcasting.
        # Create quaternion array of shape (N, 4)
        zeros = np.zeros((vectors.shape[0], 1), dtype=np.float32)
        vec_quat = np.concatenate([zeros, vectors], axis=1)  # (w, x, y, z)
        # Perform q * v * q^{-1}
        q = quat.normalised
        q_conj = q.conjugate
        # Use quaternion multiplication formula via numpy broadcasting.
        # Helper for quaternion multiplication of arrays.
        def mul(a, b):
            w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
            w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
            return np.stack([w, x, y, z], axis=1)

        if vec_quat.size == 0:
            return vectors

        q_arr = np.broadcast_to(q.elements, vec_quat.shape)
        q_conj_arr = np.broadcast_to(q_conj.elements, vec_quat.shape)
        rotated = mul(mul(q_arr, vec_quat), q_conj_arr)
        # Result quaternion has zero scalar part; extract vector part.
        return rotated[:, 1:]

    def apply(self, world, quat: Quaternion, apply_to_fields: bool = False):
        """World           .

        Parameters
        ----------
        world: Project_Sophia.core.world.World
                       .
        quat: Quaternion
               quaternion.
        apply_to_fields: bool, optional
            True        2 D   (`value_mass_field`, `will_field`,
            `intentional_field`  )       .     False.
        """
        self.logger.info("Applying warp rotation to world (Quaternion: %s)", quat)
        # 1           
        if hasattr(world, "positions") and world.positions.size > 0:
            original = world.positions.copy()
            world.positions = self._rotate_vectors(world.positions, quat)
            self.logger.debug(
                "Rotated %d particle positions (first before/after: %s -> %s)",
                world.positions.shape[0],
                original[0] if original.shape[0] > 0 else None,
                world.positions[0] if world.positions.shape[0] > 0 else None,
            )
        else:
            self.logger.warning("World has no particle positions to rotate.")

        # 2             (2 D       )
        if apply_to_fields:
            # Simple nearest neighbor rotation using scipy.ndimage.rotate
            fields = [
                "value_mass_field",
                "will_field",
                "intentional_field",
                "coherence_field",
                "threat_field",
            ]
            for fname in fields:
                if hasattr(world, fname):
                    arr = getattr(world, fname)
                    if arr.ndim == 2:
                        # Rotate around centre, preserve shape
                        rotated = nd_rotate(arr, angle=quat.angle * 180 / np.pi, axes=(1, 0), reshape=False, order=1)
                        setattr(world, fname, rotated)
                        self.logger.debug("Rotated field %s", fname)
        self.logger.info("Warp rotation completed.")

# Helper to create a quaternion from axis angle (degrees)
def quaternion_from_axis_angle(axis, angle_deg):
    """axis: iterable of 3 floats, angle in degrees"""
    angle_rad = np.deg2rad(angle_deg)
    return Quaternion(axis=axis, angle=angle_rad)
"""Example usage (not executed automatically):
from warp_layer import WarpLayer, quaternion_from_axis_angle
warp = WarpLayer()
q = quaternion_from_axis_angle([0, 0, 1], 90)  # Z   90    
warp.apply(world, q)
"""
