"""
Octonion Mathematics - 8D Extension of Quaternions

Octonions are 8-dimensional hypercomplex numbers that extend quaternions.
Used for hyperquaternion time rotation in the Fluctlight system.

Structure: o = w + xi + yj + zk + e₁ + e₂ + e₃ + e₄
where i,j,k are quaternion imaginaries and e₁,e₂,e₃,e₄ are additional dimensions.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple
import logging

logger = logging.getLogger("Octonion")


@dataclass
class Octonion:
    """
    8-dimensional hypercomplex number for time axis manipulation.
    
    Components:
        w: Real part (scalar, like quaternion w)
        x, y, z: Quaternion imaginary parts (3D rotation)
        e, i, o, k: Additional octonion dimensions (4D extension)
    
    Properties:
        - Non-associative (unlike quaternions)
        - Non-commutative (like quaternions)
        - Normed division algebra
        - Used for 8D time rotation in Fluctlight system
    """
    
    w: float = 1.0  # Real part
    x: float = 0.0  # i component
    y: float = 0.0  # j component
    z: float = 0.0  # k component
    e: float = 0.0  # e₁ component
    i: float = 0.0  # e₂ component  
    o: float = 0.0  # e₃ component
    k: float = 0.0  # e₄ component
    
    def __post_init__(self):
        """Ensure all components are floats."""
        self.w = float(self.w)
        self.x = float(self.x)
        self.y = float(self.y)
        self.z = float(self.z)
        self.e = float(self.e)
        self.i = float(self.i)
        self.o = float(self.o)
        self.k = float(self.k)
    
    @property
    def components(self) -> np.ndarray:
        """Return all 8 components as numpy array."""
        return np.array([self.w, self.x, self.y, self.z, self.e, self.i, self.o, self.k], dtype=np.float64)
    
    @property
    def norm(self) -> float:
        """Calculate norm (magnitude) of the octonion."""
        return np.sqrt(np.sum(self.components ** 2))
    
    @property
    def norm_squared(self) -> float:
        """Calculate squared norm (faster than norm)."""
        return np.sum(self.components ** 2)
    
    def normalize(self) -> Octonion:
        """Return normalized octonion (unit norm)."""
        n = self.norm
        if n < 1e-10:
            return Octonion(1, 0, 0, 0, 0, 0, 0, 0)
        return Octonion(
            self.w / n, self.x / n, self.y / n, self.z / n,
            self.e / n, self.i / n, self.o / n, self.k / n
        )
    
    def conjugate(self) -> Octonion:
        """Return conjugate (negate all imaginary parts)."""
        return Octonion(self.w, -self.x, -self.y, -self.z, -self.e, -self.i, -self.o, -self.k)
    
    def inverse(self) -> Octonion:
        """Return multiplicative inverse."""
        norm_sq = self.norm_squared
        if norm_sq < 1e-10:
            raise ValueError("Cannot invert zero octonion")
        conj = self.conjugate()
        return Octonion(
            conj.w / norm_sq, conj.x / norm_sq, conj.y / norm_sq, conj.z / norm_sq,
            conj.e / norm_sq, conj.i / norm_sq, conj.o / norm_sq, conj.k / norm_sq
        )
    
    def __add__(self, other: Octonion) -> Octonion:
        """Add two octonions."""
        return Octonion(
            self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z,
            self.e + other.e, self.i + other.i, self.o + other.o, self.k + other.k
        )
    
    def __sub__(self, other: Octonion) -> Octonion:
        """Subtract two octonions."""
        return Octonion(
            self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z,
            self.e - other.e, self.i - other.i, self.o - other.o, self.k - other.k
        )
    
    def __mul__(self, other: Octonion | float) -> Octonion:
        """
        Multiply two octonions (non-associative!).
        
        Uses Cayley-Dickson construction for octonion multiplication.
        WARNING: (a * b) * c ≠ a * (b * c) in general!
        """
        if isinstance(other, (int, float)):
            # Scalar multiplication
            return Octonion(
                self.w * other, self.x * other, self.y * other, self.z * other,
                self.e * other, self.i * other, self.o * other, self.k * other
            )
        
        # Octonion multiplication using Cayley-Dickson construction
        # Split into two quaternions: (a, b) * (c, d) = (ac - d*b, da + bc*)
        # where * is quaternion conjugate
        
        a = np.array([self.w, self.x, self.y, self.z])
        b = np.array([self.e, self.i, self.o, self.k])
        c = np.array([other.w, other.x, other.y, other.z])
        d = np.array([other.e, other.i, other.o, other.k])
        
        # Quaternion multiplication helper
        def quat_mul(q1, q2):
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            return np.array([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2
            ])
        
        def quat_conj(q):
            return np.array([q[0], -q[1], -q[2], -q[3]])
        
        # ac - d*b
        part1 = quat_mul(a, c) - quat_mul(quat_conj(d), b)
        
        # da + bc*
        part2 = quat_mul(d, a) + quat_mul(b, quat_conj(c))
        
        return Octonion(
            part1[0], part1[1], part1[2], part1[3],
            part2[0], part2[1], part2[2], part2[3]
        )
    
    def __rmul__(self, scalar: float) -> Octonion:
        """Right scalar multiplication."""
        return self * scalar
    
    def __truediv__(self, other: Octonion | float) -> Octonion:
        """Divide by octonion or scalar."""
        if isinstance(other, (int, float)):
            if abs(other) < 1e-10:
                raise ValueError("Division by zero")
            return self * (1.0 / other)
        return self * other.inverse()
    
    def rotate_time_vector(self, time_vec: np.ndarray) -> np.ndarray:
        """
        Rotate a 4D time vector using octonion rotation.
        
        This is the core operation for time axis manipulation in Fluctlight.
        
        Args:
            time_vec: 4D time vector [t, tx, ty, tz]
                t: Temporal coordinate (like w in quaternion)
                tx, ty, tz: Spatial time components (time flowing in different directions)
        
        Returns:
            Rotated 4D time vector
        """
        # Embed time vector in octonion space
        # Use first 4 components (w, x, y, z) for time vector
        time_oct = Octonion(time_vec[0], time_vec[1], time_vec[2], time_vec[3], 0, 0, 0, 0)
        
        # Rotation: v' = o * v * o^(-1)
        # (Similar to quaternion rotation but in 8D)
        rotated = self * time_oct * self.inverse()
        
        # Extract first 4 components
        return np.array([rotated.w, rotated.x, rotated.y, rotated.z])
    
    def compress_time_axis(self, compression_factor: float) -> Octonion:
        """
        Create an octonion that compresses the time axis.
        
        Args:
            compression_factor: How much to compress (>1 = faster time)
        
        Returns:
            Octonion representing time compression transformation
        """
        # Compression is represented as scaling in the w (temporal) direction
        # while preserving spatial components
        
        # Use exponential map: exp(θ * axis)
        # For time compression, we scale the real part
        theta = np.log(compression_factor)
        
        # Create compression octonion
        # This is a simplified version - full implementation would use
        # proper octonion exponential map
        return Octonion(
            w=np.cosh(theta),  # Hyperbolic cosine for real part
            x=0, y=0, z=0,
            e=np.sinh(theta),  # Hyperbolic sine for time-like imaginary part
            i=0, o=0, k=0
        ).normalize()
    
    @classmethod
    def from_quaternion(cls, w: float, x: float, y: float, z: float) -> Octonion:
        """Create octonion from quaternion (embedding in 8D)."""
        return cls(w, x, y, z, 0, 0, 0, 0)
    
    @classmethod
    def identity(cls) -> Octonion:
        """Return identity octonion (1 + 0i + 0j + ...)."""
        return cls(1, 0, 0, 0, 0, 0, 0, 0)
    
    @classmethod
    def random(cls, scale: float = 1.0) -> Octonion:
        """Generate random octonion with given scale."""
        components = np.random.randn(8) * scale
        return cls(*components).normalize()
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Octonion({self.w:.3f} + {self.x:.3f}i + {self.y:.3f}j + {self.z:.3f}k + "
            f"{self.e:.3f}e₁ + {self.i:.3f}e₂ + {self.o:.3f}e₃ + {self.k:.3f}e₄)"
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "w": self.w, "x": self.x, "y": self.y, "z": self.z,
            "e": self.e, "i": self.i, "o": self.o, "k": self.k,
            "norm": self.norm
        }


def slerp(o1: Octonion, o2: Octonion, t: float) -> Octonion:
    """
    Spherical linear interpolation between two octonions.
    
    Args:
        o1: Start octonion
        o2: End octonion
        t: Interpolation parameter (0 = o1, 1 = o2)
    
    Returns:
        Interpolated octonion
    """
    # Normalize inputs
    o1_norm = o1.normalize()
    o2_norm = o2.normalize()
    
    # Calculate angle between octonions
    dot = np.dot(o1_norm.components, o2_norm.components)
    dot = np.clip(dot, -1.0, 1.0)
    
    # If very close, use linear interpolation
    if abs(dot) > 0.9995:
        result_components = (1 - t) * o1_norm.components + t * o2_norm.components
        return Octonion(*result_components).normalize()
    
    # Spherical interpolation
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    
    if abs(sin_theta) < 1e-6:
        # Fallback to linear
        result_components = (1 - t) * o1_norm.components + t * o2_norm.components
        return Octonion(*result_components).normalize()
    
    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta
    
    result_components = w1 * o1_norm.components + w2 * o2_norm.components
    return Octonion(*result_components)


# Example usage and tests
if __name__ == "__main__":
    # Test basic operations
    o1 = Octonion(1, 0, 0, 0, 0, 0, 0, 0)  # Identity
    o2 = Octonion(0, 1, 0, 0, 0, 0, 0, 0)  # Pure imaginary
    
    print(f"o1 = {o1}")
    print(f"o2 = {o2}")
    print(f"o1 + o2 = {o1 + o2}")
    print(f"o1 * o2 = {o1 * o2}")
    print(f"o1.norm = {o1.norm}")
    
    # Test time compression
    compression = o1.compress_time_axis(1000.0)
    print(f"\n1000x compression octonion: {compression}")
    
    # Test time vector rotation
    time_vec = np.array([1.0, 0.0, 0.0, 0.0])  # Pure temporal
    rotated = compression.rotate_time_vector(time_vec)
    print(f"Original time vector: {time_vec}")
    print(f"After 1000x compression: {rotated}")
