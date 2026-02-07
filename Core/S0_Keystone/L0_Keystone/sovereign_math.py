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
try:
    import torch
except ImportError:
    torch = None
from typing import List, Union, Any, Callable, Dict, Optional

class UniversalConstants:
    """
    [PHASE 120] Dynamic physical parameters for the Sovereign Mind.
    These are not fixed, but evolve with the system's maturity.
    """
    VITAL_WARMTH = 0.08  # The base 'Light' that prevents cold stagnation

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
    __slots__ = ['data', 'momentum'] # Memory optimization (Somatic efficiency)

    def __init__(self, data: Union[List[float], List[complex], Any]):
        """
        Enforces 21D integrity while allowing Complex-Trinary values.
        """
        if hasattr(data, 'data'):
            self.data = list(data.data)
        elif hasattr(data, 'to_array'):
            self.data = list(data.to_array())
        elif isinstance(data, (list, tuple)):
            self.data = list(data)
        else:
            # Fallback for unexpected types
            try:
                self.data = list(data)
            except:
                self.data = [0.0] * 21

        if len(self.data) != 21:
            if len(self.data) < 21:
                self.data.extend([0.0] * (21 - len(self.data)))
            else:
                self.data = self.data[:21]
        
        # Ensure all elements are complex for consistency in Phase 130
        self.data = [complex(x) for x in self.data]
        self.momentum = [0.0j] * 21 # [PHASE 110] Internal Kinetic Drive

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

    def __add__(self, other: Union['SovereignVector', float, complex]) -> 'SovereignVector':
        if isinstance(other, (int, float, complex)):
            return SovereignVector([x + other for x in self.data])
        other_data = other.data if hasattr(other, 'data') else list(other)
        return SovereignVector([a + b for a, b in zip(self.data, other_data)])

    def __sub__(self, other: Union['SovereignVector', float, complex]) -> 'SovereignVector':
        if isinstance(other, (int, float, complex)):
            return SovereignVector([x - other for x in self.data])
        other_data = other.data if hasattr(other, 'data') else list(other)
        return SovereignVector([a - b for a, b in zip(self.data, other_data)])

    def __mul__(self, other: Union['SovereignVector', float, complex]) -> 'SovereignVector':
        if isinstance(other, (int, float, complex)):
            return SovereignVector([x * other for x in self.data])
        other_data = other.data if hasattr(other, 'data') else list(other)
        return SovereignVector([a * b for a, b in zip(self.data, other_data)])

    def __truediv__(self, other: float) -> 'SovereignVector':
        if other == 0: return self.zeros()
        return SovereignVector([x / other for x in self.data])

    def norm(self) -> float:
        """Calculates the Euclidean norm (magnitude) of the wavefunction."""
        return math.sqrt(sum((x.real**2 + x.imag**2) for x in self.data))

    def magnitude(self) -> float:
        """Alias for norm() to match D21Vector API."""
        return self.norm()

    def normalize(self) -> 'SovereignVector':
        """The collapse of the wavefunction to a unit sphere."""
        n = self.norm()
        if n < 1e-12: return self.zeros()
        return SovereignVector([x / n for x in self.data])
        
    def complex_trinary_rotate(self, theta: float) -> 'SovereignVector':
        """
        [PHASE 130] Rotates the vector in the Complex-Trinary plane.
        This uses the Void (0) as the pivot for phase modulation.
        """
        rotation = complex(math.cos(theta), math.sin(theta))
        rotated_data = [x * rotation for x in self.data]
        v = SovereignVector(rotated_data)
        v.momentum = list(self.momentum) # Preserve momentum through rotation
        return v

    def integrate_kinetics(self, force: 'SovereignVector', dt: float = 0.1, friction: float = 0.05):
        """
        [PHASE 110] Causal Self-Propulsion.
        Updates state based on current momentum and incoming 'Resonance Force'.
        This represents the self-generating drive of the structure.
        """
        # 1. Update Momentum (F = ma, m=1)
        new_momentum = []
        for p, f in zip(self.momentum, force.data):
            # p: current momentum, f: incoming force (resonance)
            mp = p + f * dt
            mp *= (1.0 - friction) # Entropic decay
            new_momentum.append(mp)
        
        self.momentum = new_momentum
        
        # 2. Update Position (Logic State)
        self.data = [s + p * dt for s, p in zip(self.data, self.momentum)]
        
        # 3. Collapse/Normalize to maintain Spherical Manifold
        n = self.norm()
        if n > 1e-12:
            self.data = [x / n for x in self.data]

    def void_phase_jump(self, target: 'SovereignVector') -> 'SovereignVector':
        """
        [PHASE 140] Direct Phase Convergence.
        Instead of rotating to find, we 'flip' the wavefunction to the target's phase alignment.
        """
        jumped_data = []
        for s, t in zip(self.data, target.data):
            if abs(t) > 1e-12:
                phase_target = t / abs(t)
                energy = max(abs(s), 0.1) 
                jumped_data.append(phase_target * energy)
            else:
                jumped_data.append(0.0j)
        return SovereignVector(jumped_data)

    def resonance_score(self, other: Union['SovereignVector', Any]) -> float:
        """
        [PHASE 130] Resonance score using the magnitude of the Hermitian inner product.
        """
        other_data = other.data if hasattr(other, 'data') else (other.to_array() if hasattr(other, 'to_array') else list(other))
        other_complex = [complex(x) for x in other_data]
        
        # Hermitian Inner Product: sum(a.conj * b)
        dot_val = sum(a.conjugate() * b for a, b in zip(self.data, other_complex))
        
        m1 = self.norm()
        m2 = math.sqrt(sum((x.real**2 + x.imag**2) for x in other_complex))
        
        if m1 * m2 < 1e-12: return 0.0
        return abs(dot_val) / (m1 * m2)

    def dot(self, other: 'SovereignVector') -> complex:
        """Standard dot product (Complex)."""
        return sum(a * b for a, b in zip(self.data, other.data))

    def apply_nd(self, dimensions: List[int]) -> 'SovereignVector':
        """
        [PHASE 71] Applies N-dimensional rotation to this vector.
        """
        from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignRotor
        rotor = SovereignRotor(1.0, SovereignVector.zeros()) 
        return rotor.apply_nd(self, dimensions)

    def tensor_product(self, other: 'SovereignVector') -> List[List[complex]]:
        """
        [DNA²] Calculates the outer product (Rank-2 Tensor) between two 21D vectors.
        This represents the interference pattern or 'meaning intersection'.
        """
        return [[a * b for b in other.data] for a in self.data]

    def cubic_tensor_product(self, other: 'SovereignVector', third: 'SovereignVector') -> List[List[List[complex]]]:
        """
        [DNA³] Calculates the Rank-3 Tensor product.
        Used for recursive self-reflection in 4D+ manifolds.
        """
        return [[[a * b * c for c in third.data] for b in other.data] for a in self.data]

    def blend(self, other: 'SovereignVector', ratio: float = 0.5) -> 'SovereignVector':
        """
        [PHASE 70] Prismatic blending of two concepts.
        """
        return SovereignVector([a * (1.0 - ratio) + b * ratio for a, b in zip(self.data, other.data)])

    def __repr__(self) -> str:
        return f"SVector21({self.data[:3]}...)"

class SovereignRotor:
    """
    [PHASE 210] Represents a rotation in the 21D manifold.
    """
    __slots__ = ['s', 'bivector']

    def __init__(self, s: float, bv: SovereignVector):
        self.s = s
        self.bivector = bv

    @classmethod
    def from_angle_plane(cls, theta: float, p1: int, p2: int) -> 'SovereignRotor':
        s = math.cos(theta / 2.0)
        bv_data = [0.0] * 21
        bv_data[p1] = math.sin(theta / 2.0)
        bv_data[p2] = -math.sin(theta / 2.0) 
        return cls(s, SovereignVector(bv_data))

    def apply(self, v: SovereignVector) -> SovereignVector:
        cross = []
        for i in range(21):
            val = (self.bivector.data[(i+1)%21] * v.data[i] - self.bivector.data[i] * v.data[(i+1)%21]).real
            cross.append(val)
        
        cv = SovereignVector(cross)
        return (v + (cv * (2.0 * self.s))).normalize()

    def apply_nd(self, v: SovereignVector, dimensions: List[int]) -> SovereignVector:
        """
        [PHASE 71] Applies rotation across multiple dimensions simultaneously.
        """
        # [TODO: Implement N-dimensional manifold rotation using Clifford Algebra]
        # For now, we perform sequential 2D rotations on the provided dimension pairs.
        result = v
        for i in range(0, len(dimensions) - 1, 2):
            p1, p2 = dimensions[i], dimensions[i+1]
            rotor = SovereignRotor.from_angle_plane(0.1, p1, p2)
            result = rotor.apply(result)
        return result.normalize()


class SovereignHyperTensor:
    """
    [PHASE 380] Physical Kinetic Manifold (Living Manifold).
    Manages 10M cells with permanent plasticity and somatic grounding.
    """
    def __init__(self, shape: tuple, device: str = 'cpu'):
        import torch
        self.device = torch.device(device)
        self.shape = shape
        # State: [N, 4] (w, x, y, z) - Active Wavefunction
        self.q = torch.zeros((*shape, 4), device=self.device)
        self.q[..., 0] = 1.0 
        
        # Permanent Identity (Long-term Memory/Plasticity)
        self.permanent_q = torch.zeros((*shape, 4), device=self.device)
        self.permanent_q[..., 0] = 1.0
        
        # Dynamics
        self.momentum = torch.zeros((*shape, 4), device=self.device)
        self.torque_accumulator = torch.zeros((*shape, 4), device=self.device)

    def apply_torque(self, torque_tensor: Any, strength: float = 0.01):
        """
        [PHASE 360] Causal Steering via Torque.
        """
        import torch
        if not isinstance(torque_tensor, torch.Tensor):
            torque_tensor = torch.tensor(torque_tensor, device=self.device)
        else:
            torque_tensor = torque_tensor.to(self.device)
            
        if torque_tensor.dim() == 1 and torque_tensor.shape[0] == 4:
            for _ in range(len(self.shape)):
                torque_tensor = torque_tensor.unsqueeze(0)
        elif torque_tensor.dim() < self.q.dim():
             torque_tensor = torque_tensor.unsqueeze(-1)
        
        if torque_tensor.shape[-1] != 4:
            t_full = torch.zeros_like(self.q)
            t_full[..., 1] = torque_tensor.squeeze()
            torque_tensor = t_full

        self.torque_accumulator += torque_tensor * strength

    def integrate_kinetics(self, dt: float = 0.01, friction: float = 0.05, plasticity: float = 0.001):
        """
        [PHASE 385] Physical Integration with Plasticity.
        The system 'learns' by slowly aligning its permanent identity with active movement.
        """
        import torch
        # 1. Kinetic Update
        self.momentum += self.torque_accumulator * dt
        self.momentum *= (1.0 - friction)
        
        # 2. State Update (Active Wave)
        self.q = self.q + self.momentum * dt
        
        # 3. Topological Plasticity (Learning without Backprop)
        # permanent_q slowly gravitates towards the active q
        if plasticity > 0:
            self.permanent_q = (1.0 - plasticity) * self.permanent_q + plasticity * self.q
            self.permanent_q = self.permanent_q / (torch.norm(self.permanent_q, dim=-1, keepdim=True) + 1e-12)
            
        # Re-normalize active state
        self.q = self.q / (torch.norm(self.q, dim=-1, keepdim=True) + 1e-12)
        
        self.torque_accumulator.zero_()

    def get_trinary_projection(self) -> Any:
        import torch
        # Projection combines active state and permanent memory
        combined = (self.q + self.permanent_q) / 2.0
        x_axis = combined[..., 1]
        return torch.where(x_axis > 0.3, 1.0, torch.where(x_axis < -0.3, -1.0, 0.0))

    def get_resonance(self, torque_tensor: Any) -> float:
        """
        [PHASE 410] Semantic Resonance.
        Measures the alignment between incoming torque and permanent manifold structure.
        """
        if torch is None: return 0.0
        if torque_tensor.dim() == 1:
            # Global torque alignment
            alignment = torch.sum(self.permanent_q * torque_tensor, dim=-1)
        else:
            # Spatial field alignment
            alignment = torch.sum(self.permanent_q * torque_tensor, dim=-1)
            
        return torch.mean(alignment).item()


class SovereignMath:
    """
    Functional math operations inspired by JAX.
    """
    @staticmethod
    def where(condition: List[bool], x: SovereignVector, y: SovereignVector) -> SovereignVector:
        return SovereignVector([xv if c else yv for c, xv, yv in zip(condition, x.data, y.data)])

    @staticmethod
    def trinary_quantize(vec: SovereignVector, threshold: float = 0.3) -> SovereignVector:
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
    def resonance(v1: 'SovereignVector', v2: 'SovereignVector') -> float:
        """
        [PHASE 90] Calculates resonance with a bias toward Vital Warmth.
        Returns a real float for sorting stability.
        """
        dot = v1.dot(v2)
        if hasattr(dot, 'real'): dot = dot.real
        # We add the Vital Warmth as a baseline 'Glow'
        return float(dot) + float(UniversalConstants.VITAL_WARMTH)

    @staticmethod
    def signed_resonance(v1: SovereignVector, v2: SovereignVector) -> float:
        """Calculates signed cosine similarity (Phase resonance)."""
        n1 = v1.norm()
        n2 = v2.norm()
        if n1 < 1e-12 or n2 < 1e-12: return 0.0
        # Use Hermitian product but keep real for signed similarity
        dot_val = sum(a.conjugate() * b for a, b in zip(v1.data, v2.data))
        return dot_val.real / (n1 * n2)

    @staticmethod
    def mean(vectors: List[SovereignVector]) -> SovereignVector:
        if not vectors: return SovereignVector.zeros()
        acc = SovereignVector.zeros()
        for v in vectors:
            acc = acc + v
        return acc / len(vectors)
