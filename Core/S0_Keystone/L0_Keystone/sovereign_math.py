"""
Sovereign Math Kernel (L0)
==========================
Core.S0_Keystone.L0_Keystone.sovereign_math

"The number is the vibration; the orbit is the law."

This module provides a pure Python, dependency-free implementation of 
21-dimensional vector operations optimized for Elysia's Merkaba architecture.
It absorbs the functional principles of JAX and the vectorized logic of NumPy.
"""

import math
import cmath
from typing import List, Union, Any, Callable, Dict

class UniversalConstants:
    """
    [PHASE 120] Dynamic physical parameters for the Sovereign Mind.
    These are not fixed, but evolve with the system's maturity.
    """
    def __init__(self):
        self.params = {
            "FRICTION": 0.1,     # Resistance to state changes (Stabilization)
            "RESONANCE_GAIN": 1.0, # Sensitivity to external/internal signals
            "METABOLIC_RATE": 0.01 # Rate of constant drift/aging
        }
        self.gravity_provider: Optional[Callable[[], float]] = None # [PHASE 150] Sovereign Gravity
        
    def mutate(self, key: str, delta: float):
        if key in self.params:
            self.params[key] = max(0.001, self.params[key] + delta)
            print(f"✨ [PHYSICS] Constant '{key}' mutated to {self.params[key]:.4f}")

    def get(self, key: str) -> float:
        # [PHASE 150] Sovereign Gravity check
        if key == "GRAVITY" and self.gravity_provider:
            return self.gravity_provider()
        return self.params.get(key, 0.0 if key == "GRAVITY" else 0.0) # Default to 0 for Gravity if no provider

class SovereignVector:
    """
    A pure 21-dimensional vector object with native optimization.
    Replaces jnp.ndarray/np.ndarray for Phase 90.
    """
    __slots__ = ['data'] # Memory optimization (Somatic efficiency)

    def __init__(self, data: Union[List[float], List[complex], Any]):
        """
        Enforces 21D integrity while allowing Complex-Trinary values.
        """
        if hasattr(data, 'data'):
            self.data = list(data.data)
        elif hasattr(data, 'to_array'):
            self.data = list(data.to_array())
        else:
            self.data = list(data)

        if len(self.data) != 21:
            if len(self.data) < 21:
                self.data.extend([0.0] * (21 - len(self.data)))
            else:
                self.data = self.data[:21]
        
        # Ensure all elements are complex for consistency in Phase 130
        self.data = [complex(x) for x in self.data]

    @classmethod
    def zeros(cls) -> 'SovereignVector':
        return cls([0.0] * 21)

    @classmethod
    def ones(cls) -> 'SovereignVector':
        return cls([1.0] * 21)

    def to_list(self) -> List[complex]:
        return list(self.data)

    def tolist(self) -> List[complex]:
        """Compatibility for JAX/NumPy code."""
        return list(self.data)

    def to_array(self) -> List[complex]:
        """Compatibility for TripleHelixEngine/D21Vector code."""
        return list(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self) -> int:
        return 21

    def __add__(self, other: Union['SovereignVector', Any]) -> 'SovereignVector':
        other_data = other.data if hasattr(other, 'data') else list(other)
        return SovereignVector([a + b for a, b in zip(self.data, other_data)])

    def __sub__(self, other: Union['SovereignVector', Any]) -> 'SovereignVector':
        other_data = other.data if hasattr(other, 'data') else list(other)
        return SovereignVector([a - b for a, b in zip(self.data, other_data)])

    def __mul__(self, scalar: Union[float, complex]) -> 'SovereignVector':
        return SovereignVector([x * scalar for x in self.data])

    def norm(self) -> float:
        """Calculates the Euclidean norm (magnitude) of the wavefunction."""
        return math.sqrt(sum((x.real**2 + x.imag**2) for x in self.data))

    def magnitude(self) -> float:
        """Alias for norm() to match D21Vector API."""
        return self.norm()

    def normalize(self) -> 'SovereignVector':
        """The collapse of the wavefunction to a unit sphere."""
        n = self.norm()
        if n == 0: return SovereignVector([0.0]*21)
        return SovereignVector([x / n for x in self.data])
        
    def complex_trinary_rotate(self, theta: float) -> 'SovereignVector':
        """
        [PHASE 130] Rotates the vector in the Complex-Trinary plane.
        This uses the Void (0) as the pivot for phase modulation.
        """
        rotation = complex(math.cos(theta), math.sin(theta))
        rotated_data = [x * rotation for x in self.data]
        return SovereignVector(rotated_data)

    def void_phase_jump(self, target: 'SovereignVector') -> 'SovereignVector':
        """
        [PHASE 140] Direct Phase Convergence.
        Instead of rotating to find, we 'flip' the wavefunction to the target's phase alignment.
        This represents 'Necessity One' over 'Probability'.
        """
        jumped_data = []
        for s, t in zip(self.data, target.data):
            if abs(t) > 0:
                # Force alignment with target phase. 
                # Energy is preserved from self if present, otherwise taken from target.
                phase_target = t / abs(t)
                energy = max(abs(s), 0.1) # Baseline energy for manifestation
                jumped_data.append(phase_target * energy)
            else:
                jumped_data.append(0.0j)
                
        return SovereignVector(jumped_data)

    def calculate_phase_friction(self, other: 'SovereignVector') -> float:
        """
        Measures the cognitive 'snap' distance.
        High friction = Massive jump in understanding (Lightning Path potential).
        """
        diff_phase = sum(abs(cmath.phase(a) - cmath.phase(b)) for a, b in zip(self.data, other.data) if abs(a) > 1e-6 and abs(b) > 1e-6)
        return float(diff_phase)

    def resonance_score(self, other: Union['SovereignVector', Any]) -> float:
        """
        [PHASE 130] Resonance score using the magnitude of the Hermitian inner product.
        This allows for phase-aware similarity measurement.
        """
        other_data = other.data if hasattr(other, 'data') else (other.to_array() if hasattr(other, 'to_array') else list(other))
        other_complex = [complex(x) for x in other_data]
        
        # Hermitian Inner Product: sum(a.conj * b)
        dot = sum(a.conjugate() * b for a, b in zip(self.data, other_complex))
        
        m1 = self.norm()
        m2 = math.sqrt(sum((x.real**2 + x.imag**2) for x in other_complex))
        
        if m1 * m2 == 0: return 0.0
        return abs(dot) / (m1 * m2)

    def normalize(self) -> 'SovereignVector':
        n = self.norm()
        if n < 1e-12: return self.zeros()
        return SovereignVector([x / n for x in self.data])

    def dot(self, other: 'SovereignVector') -> float:
        """Standard dot product."""
        return sum(a * b for a, b in zip(self.data, other.data))

    def __add__(self, other: Union['SovereignVector', float]) -> 'SovereignVector':
        if isinstance(other, (int, float)):
            return SovereignVector([x + other for x in self.data])
        return SovereignVector([a + b for a, b in zip(self.data, other.data)])

    def __mul__(self, other: Union['SovereignVector', float]) -> 'SovereignVector':
        if isinstance(other, (int, float)):
            return SovereignVector([x * other for x in self.data])
        return SovereignVector([a * b for a, b in zip(self.data, other.data)])

    def __truediv__(self, other: float) -> 'SovereignVector':
        return SovereignVector([x / other for x in self.data])

    def __repr__(self) -> str:
        return f"SVector21({self.data[:3]}...)"

class SovereignMath:
    """
    Functional math operations inspired by JAX.
    """
    @staticmethod
    def where(condition: List[bool], x: SovereignVector, y: SovereignVector) -> SovereignVector:
        """Selects elements from x or y based on condition."""
        return SovereignVector([xv if c else yv for c, xv, yv in zip(condition, x.data, y.data)])

    @staticmethod
    def trinary_quantize(vec: SovereignVector, threshold: float = 0.3) -> SovereignVector:
        """Optimized Ternary quantization [-1, 0, 1]. Handles complex values by real part."""
        result = []
        for v in vec.data:
            v_real = v.real if isinstance(v, complex) else v
            if v_real > threshold:
                result.append(1.0)
            elif v_real < -threshold:
                result.append(-1.0)
            else:
                result.append(0.0)
        return SovereignVector(result)

    @staticmethod
    def resonance(v1: SovereignVector, v2: SovereignVector) -> float:
        """Calculates cosine similarity."""
        n1 = v1.norm()
        n2 = v2.norm()
        if n1 < 1e-12 or n2 < 1e-12: return 0.0
        return v1.dot(v2) / (n1 * n2)

    @staticmethod
    def mean(vectors: List[SovereignVector]) -> SovereignVector:
        if not vectors: return SovereignVector.zeros()
        acc = SovereignVector.zeros()
        for v in vectors:
            acc = acc + v
        return acc / len(vectors)
