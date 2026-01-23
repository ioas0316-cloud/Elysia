"""
GENESIS MATH
============
Phase 110: The Genesis Engine

"To create a world, one must first define Space."

This module implements the Fundamental Laws of Geometry (Linear Algebra) 
from first principles, removing reliance on external engines like Unity/Unreal.

Capabilities:
- Vector4 (Homogeneous Coordinates)
- Matrix4x4 (Transformation: Translation, Rotation, Scale)
- Projection (World Space -> Clip Space)
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Vector3:
    x: float
    y: float
    z: float

    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def dot(self, other) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def normalize(self):
        mag = math.sqrt(self.x**2 + self.y**2 + self.z**2)
        if mag == 0: return Vector3(0,0,0)
        return Vector3(self.x/mag, self.y/mag, self.z/mag)

    def to_vec4(self, w: float = 1.0):
        return Vector4(self.x, self.y, self.z, w)

    def __repr__(self):
        return f"Vec3({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"


@dataclass
class Vector4:
    x: float
    y: float
    z: float
    w: float

    def to_vec3(self):
        # Perspective Division (Normalization)
        if self.w != 0 and self.w != 1:
            return Vector3(self.x/self.w, self.y/self.w, self.z/self.w)
        return Vector3(self.x, self.y, self.z)

    def __repr__(self):
        return f"Vec4({self.x:.2f}, {self.y:.2f}, {self.z:.2f}, {self.w:.2f})"


class Matrix4x4:
    """
    Row-major 4x4 Matrix for 3D Transformations.
    """
    def __init__(self, data: List[List[float]] = None):
        if data:
            self.m = data
        else:
            self.m = [[0]*4 for _ in range(4)] # Zero matrix

    @staticmethod
    def identity():
        return Matrix4x4([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    @staticmethod
    def translation(x: float, y: float, z: float):
        mat = Matrix4x4.identity()
        mat.m[0][3] = x
        mat.m[1][3] = y
        mat.m[2][3] = z
        return mat

    @staticmethod
    def scaling(x: float, y: float, z: float):
        mat = Matrix4x4.identity()
        mat.m[0][0] = x
        mat.m[1][1] = y
        mat.m[2][2] = z
        return mat

    @staticmethod
    def rotation_y(angle_deg: float):
        rad = math.radians(angle_deg)
        c = math.cos(rad)
        s = math.sin(rad)
        mat = Matrix4x4.identity()
        mat.m[0][0] = c
        mat.m[0][2] = s
        mat.m[2][0] = -s
        mat.m[2][2] = c
        return mat

    @staticmethod
    def perspective(fov: float, aspect: float, near: float, far: float):
        """Standard Projection Matrix construction."""
        f = 1.0 / math.tan(math.radians(fov) / 2)
        range_inv = 1.0 / (near - far)
        
        mat = Matrix4x4()
        mat.m[0][0] = f / aspect
        mat.m[1][1] = f
        mat.m[2][2] = (near + far) * range_inv
        mat.m[2][3] = near * far * range_inv * 2
        mat.m[3][2] = -1.0
        mat.m[3][3] = 0.0
        return mat

    def __mul__(self, other):
        # Matrix * Vector4
        if isinstance(other, Vector4):
            res = [0]*4
            for i in range(4):
                res[i] = (self.m[i][0] * other.x + 
                          self.m[i][1] * other.y + 
                          self.m[i][2] * other.z + 
                          self.m[i][3] * other.w)
            return Vector4(res[0], res[1], res[2], res[3])
        
        # Matrix * Matrix
        elif isinstance(other, Matrix4x4):
            res = Matrix4x4()
            for i in range(4):
                for j in range(4):
                    res.m[i][j] = sum(self.m[i][k] * other.m[k][j] for k in range(4))
            return res
        return NotImplemented

if __name__ == "__main__":
    # Test: Project a point
    print("  Genesis Math Test: Projection")
    point = Vector3(10, 5, 20) # A point in 3D world
    print(f"Original Point: {point}")
    
    # 1. Transform: Move it back 10 units
    model_matrix = Matrix4x4.translation(0, 0, -10)
    
    # 2. Project: Perspective Camera
    proj_matrix = Matrix4x4.perspective(fov=60, aspect=1.77, near=0.1, far=100.0)
    
    # 3. Apply MVP
    transformed = model_matrix * point.to_vec4()
    projected = proj_matrix * transformed
    
    print(f"Transformed (View): {transformed}")
    print(f"Projected (Clip): {projected}")
    print(f"Screen Normalized: {projected.to_vec3()}")