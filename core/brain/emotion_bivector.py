import math
from core.utils.math_utils import Multivector, Quaternion

class EmotionBivector:
    """
    Represents emotional states as 3D Bivectors in Clifford Algebra Cl(3,0).
    Basis bivectors map directly to psychological states:
    - e12 (mask 3): Approach-Avoidance (Love vs Fear)
    - e23 (mask 6): Activation-Deactivation (Excitement vs Lethargy)
    - e31 (mask 5): Dominance-Submission (Anger vs Acceptance)
    """
    SIGNATURE = (3, 0)

    def __init__(self, e12: float = 0.0, e23: float = 0.0, e31: float = 0.0):
        self.multivector = Multivector({
            3: float(e12),
            6: float(e23),
            5: float(e31)
        }, self.SIGNATURE)

    @property
    def e12(self) -> float:
        return self.multivector.data.get(3, 0.0)

    @e12.setter
    def e12(self, val: float):
        self.multivector.data[3] = float(val)

    @property
    def e23(self) -> float:
        return self.multivector.data.get(6, 0.0)

    @e23.setter
    def e23(self, val: float):
        self.multivector.data[6] = float(val)

    @property
    def e31(self) -> float:
        return self.multivector.data.get(5, 0.0)

    @e31.setter
    def e31(self, val: float):
        self.multivector.data[5] = float(val)

    def norm(self) -> float:
        """Calculates the magnitude of the emotion bivector."""
        return math.sqrt(self.e12**2 + self.e23**2 + self.e31**2)

    def to_rotor(self) -> Quaternion:
        """
        Maps the emotion bivector to a 4D Quaternion rotor.
        E = e23*i + e31*j + e12*k
        Rotor = exp(E/2) = cos(|E|/2) + sin(|E|/2) * (E / |E|)
        """
        mag = self.norm()
        if mag < 1e-9:
            return Quaternion(1.0, 0.0, 0.0, 0.0)
        
        half_mag = mag / 2.0
        c = math.cos(half_mag)
        s = math.sin(half_mag)
        
        # Normalize bivector components to form the unit axis
        # e23 maps to x (i), e31 maps to y (j), e12 maps to z (k)
        ux = self.e23 / mag
        uy = self.e31 / mag
        uz = self.e12 / mag
        
        return Quaternion(c, s * ux, s * uy, s * uz).normalize()

    def decay(self, rate: float):
        """Naturally decays the emotional intensity toward homeostasis (zero curvature)."""
        self.e12 *= (1.0 - rate)
        self.e23 *= (1.0 - rate)
        self.e31 *= (1.0 - rate)

    def add_stimulus(self, de12: float, de23: float, de31: float):
        """Adds emotional stimulus to the bivector state."""
        self.e12 += de12
        self.e23 += de23
        self.e31 += de31

    def __repr__(self):
        return f"EmotionBivector(Love/Fear: {self.e12:.3f}e12, Excitement/Lethargy: {self.e23:.3f}e23, Anger/Acceptance: {self.e31:.3f}e31)"
