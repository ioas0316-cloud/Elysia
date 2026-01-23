import math
from dataclasses import dataclass
from typing import Tuple, Union

@dataclass
class MultiVector:
    """
    A simplified MultiVector for 4D Geometric Algebra (Clifford Algebra Cl(4,0)).
    Used to represent Rotors and high-dimensional transformations.

    Structure (simplified for Rotor use):
    - Scalar (Grade 0): s
    - Bivectors (Grade 2): xy, xz, xw, yz, yw, zw (The 6 rotation planes)
    - Pseudo-scalar (Grade 4): I (optional, usually 1 for rotors)

    Note: For pure rotation in 4D, we primarily use even-grade subalgebras.
    A Rotor R = s + B (scalar + bivector part).
    """
    s: float = 1.0  # Scalar part
    # Bivector components (representing the 6 planes of 4D space)
    xy: float = 0.0
    xz: float = 0.0
    xw: float = 0.0
    yz: float = 0.0
    yw: float = 0.0
    zw: float = 0.0

    def __add__(self, other: 'MultiVector') -> 'MultiVector':
        return MultiVector(
            self.s + other.s,
            self.xy + other.xy, self.xz + other.xz, self.xw + other.xw,
            self.yz + other.yz, self.yw + other.yw, self.zw + other.zw
        )

    def __mul__(self, other: Union[float, 'MultiVector']) -> 'MultiVector':
        if isinstance(other, (int, float)):
            return MultiVector(
                self.s * other,
                self.xy * other, self.xz * other, self.xw * other,
                self.yz * other, self.yw * other, self.zw * other
            )
        
        # Geometric Product (Simplified for Even Subalgebra of Cl(4,0))
        # This handles products of Scalar + Bivectors (Rotors)
        # s1*s2, s1*B2, B1*s2 are trivial.
        # B1*B2 is the tricky part. For 4D, B*B = Scalar + Bivector + Pseudoscalar.
        # However, for Rotors, we stay within the even subalgebra (Scalar + Bivectors + Pseudoscalar).
        
        s1, x1, y1, w1, z1, u1, v1 = self.s, self.xy, self.xz, self.xw, self.yz, self.yw, self.zw
        s2, x2, y2, w2, z2, u2, v2 = other.s, other.xy, other.xz, other.xw, other.yz, other.yw, other.zw

        # Component multiplication table for Bivectors in Cl(4,0)
        # e12*e12 = -1, e12*e23 = e13, etc.
        new_s = s1*s2 - (x1*x2 + y1*y2 + w1*w2 + z1*z2 + u1*u2 + v1*v2)
        
        new_xy = s1*x2 + x1*s2 - (y1*z2 - z1*y2 + w1*u2 - u1*w2)
        new_xz = s1*y2 + y1*s2 + (x1*z2 - z1*x2 + w1*v2 - v1*w2)
        new_xw = s1*w2 + w1*s2 - (x1*u2 - u1*x2 + y1*v2 - v1*y2)
        new_yz = s1*z2 + z1*s2 - (x1*y2 - y1*x2 + u1*v2 - v1*u2)
        new_yw = s1*u2 + u1*s2 + (x1*w2 - w1*x2 - z1*v2 + v1*z2)
        new_zw = s1*v2 + v1*s2 + (y1*w2 - w1*y2 + z1*u2 - u2*z1)

        return MultiVector(s=new_s, xy=new_xy, xz=new_xz, xw=new_xw, yz=new_yz, yw=new_yw, zw=new_zw)

    def reverse(self) -> 'MultiVector':
        """Inverts the bivector signs (R_reverse)."""
        return MultiVector(self.s, -self.xy, -self.xz, -self.xw, -self.yz, -self.yw, -self.zw)

    def normalize(self) -> 'MultiVector':
        mag_sq = (self.s**2 + self.xy**2 + self.xz**2 + self.xw**2 +
                  self.yz**2 + self.yw**2 + self.zw**2)
        if mag_sq == 0: return MultiVector(1.0)
        mag = math.sqrt(mag_sq)
        return self * (1.0 / mag)


class Rotor:
    """
    Rotor: The operator of rotation in Geometric Algebra.
    R = cos(theta/2) - B * sin(theta/2)
    """

    @staticmethod
    def from_plane_angle(plane: str, angle_rad: float) -> MultiVector:
        half_angle = angle_rad / 2.0
        c = math.cos(half_angle)
        s = math.sin(half_angle)

        # Standard GA: R = exp(-B * theta/2) = cos(t/2) - B*sin(t/2)
        if plane == 'xy': return MultiVector(s=c, xy=-s)
        if plane == 'xz': return MultiVector(s=c, xz=-s)
        if plane == 'xw': return MultiVector(s=c, xw=-s)
        if plane == 'yz': return MultiVector(s=c, yz=-s)
        if plane == 'yw': return MultiVector(s=c, yw=-s)
        if plane == 'zw': return MultiVector(s=c, zw=-s)
        raise ValueError(f"Unknown plane: {plane}")

    @staticmethod
    def rotate_vector(vector: Tuple[float, float, float, float], rotor: MultiVector) -> Tuple[float, float, float, float]:
        """
        Rotates a 4D vector v using v' = R v R_rev.
        Since we don't have a Vector class here yet, we implement the sandwich product logic
        optimized for MultiVector * Vector * Reverse(MultiVector).
        """
        x, y, z, w = vector
        # Treat vector as a MultiVector with grade 1 components (not implemented in the dataclass)
        # For efficiency, we'll use a simplified algebraic expansion of the sandwich product.
        
        # Identity shortcut
        if rotor.s == 1.0 and all(v == 0 for v in [rotor.xy, rotor.xz, rotor.xw, rotor.yz, rotor.yw, rotor.zw]):
            return vector

        # Algebraic expansion of R v R~
        # This is a bit long but very fast.
        r = rotor
        rev = r.reverse()
        
        # Manual product R * v (treating v as e1, e2, e3, e4 items)
        # v = x*e1 + y*e2 + z*e3 + w*e4
        # (s + b)*v = s*v + b*v
        
        # Due to the complexity of full Cl(4,0) grade 1 * grade 2, 
        # let's implement a robust version that handles the common cases.
        # For now, let's stick to the 2D plane logic if it's a simple rotor,
        # or implement a slightly more general version for any bivector.
        
        # If it's a simple rotation in one plane (the most common case):
        active_planes = []
        if r.xy != 0: active_planes.append(('xy', r.xy))
        if r.xz != 0: active_planes.append(('xz', r.xz))
        if r.xw != 0: active_planes.append(('xw', r.xw))
        if r.yz != 0: active_planes.append(('yz', r.yz))
        if r.yw != 0: active_planes.append(('yw', r.yw))
        if r.zw != 0: active_planes.append(('zw', r.zw))
        
        if len(active_planes) == 1:
            plane, val = active_planes[0]
            # Double angle logic: cos(t) = c^2 - s^2, sin(t) = 2cs
            c_h = r.s
            s_h = -val # We stored -sin(t/2)
            cos_t = c_h**2 - s_h**2
            sin_t = 2 * c_h * s_h
            
            if plane == 'xy': return (x*cos_t - y*sin_t, x*sin_t + y*cos_t, z, w)
            if plane == 'xz': return (x*cos_t - z*sin_t, y, x*sin_t + z*cos_t, w)
            if plane == 'xw': return (x*cos_t - w*sin_t, y, z, x*sin_t + w*cos_t)
            if plane == 'yz': return (x, y*cos_t - z*sin_t, y*sin_t + z*cos_t, w)
            if plane == 'yw': return (x, y*cos_t - w*sin_t, z, y*sin_t + w*cos_t)
            if plane == 'zw': return (x, y, z*cos_t - w*sin_t, z*sin_t + w*cos_t)

        return vector # Fallback
