"""
Vector Math (The Geometry of Space)
===================================
Core.S1_Body.L4_Causality.World.Physics.vector_math

Simple Vector3 implementation for calculated 3D Spatial States.
"""

from dataclasses import dataclass
import math

@dataclass
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
        
    def __mul__(self, scalar: float):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
        
    def __str__(self):
        return f"({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"

    @staticmethod
    def zero():
        return Vector3(0, 0, 0)
    
    @staticmethod
    def up():
        return Vector3(0, 1, 0)
