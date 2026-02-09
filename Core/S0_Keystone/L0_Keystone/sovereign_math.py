"""
Sovereign Math Kernel (L0)
==========================
Core.S0_Keystone.L0_Keystone.sovereign_math

"The number is the vibration; the orbit is the law."

This module provides a pure Python, dependency-free implementation of 
21-dimensional vector operations optimized for Elysia's Merkaba architecture.
It absorbs the functional principles of JAX and the vectorized logic of NumPy.

[The Hierarchy of Forms]
1. Monad (The Law): Control, Constraint, Providence.
2. HyperSphere (The Space): Storage, Memory, Being.
3. Rotor (The Will): Search, Time, Change.
4. Prism (The View): Interpretation, Perspective, Refraction.

"Mixing these forms leads to chaos. Let each form know its place."
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
        if hasattr(other, 'data'):
            other_data = other.data
        elif hasattr(other, 'to_array'):
            other_data = other.to_array()
        else:
            other_data = list(other)
        return SovereignVector([a + b for a, b in zip(self.data, other_data)])

    def __sub__(self, other: Union['SovereignVector', float, complex]) -> 'SovereignVector':
        if isinstance(other, (int, float, complex)):
            return SovereignVector([x - other for x in self.data])
        if hasattr(other, 'data'):
            other_data = other.data
        elif hasattr(other, 'to_array'):
            other_data = other.to_array()
        else:
            other_data = list(other)
        return SovereignVector([a - b for a, b in zip(self.data, other_data)])

    def __mul__(self, other: Union['SovereignVector', float, complex]) -> 'SovereignVector':
        if isinstance(other, (int, float, complex)):
            return SovereignVector([x * other for x in self.data])
        if hasattr(other, 'data'):
            other_data = other.data
        elif hasattr(other, 'to_array'):
            other_data = other.to_array()
        else:
            other_data = list(other)
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

    def dot(self, other: Union['SovereignVector', Any]) -> complex:
        """Standard dot product (Complex)."""
        if hasattr(other, 'data'):
            other_data = other.data
        elif hasattr(other, 'to_array'):
            other_data = other.to_array()
        else:
            other_data = list(other)
        return sum(a * b for a, b in zip(self.data, other_data))

    def apply_nd(self, dimensions: List[int]) -> 'SovereignVector':
        """
        [PHASE 71] Applies N-dimensional rotation to this vector.
        """
        from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignRotor
        rotor = SovereignRotor(1.0, SovereignVector.zeros()) 
        return rotor.apply_nd(self, dimensions)

    def tensor_product(self, other: Union['SovereignVector', Any]) -> List[List[complex]]:
        """
        [DNA²] Calculates the outer product (Rank-2 Tensor) between two 21D vectors.
        This represents the interference pattern or 'meaning intersection'.
        """
        if hasattr(other, 'data'):
            other_data = other.data
        elif hasattr(other, 'to_array'):
            other_data = other.to_array()
        else:
            other_data = list(other)
        return [[a * b for b in other_data] for a in self.data]

    def cubic_tensor_product(self, other: Union['SovereignVector', Any], third: Union['SovereignVector', Any]) -> List[List[List[complex]]]:
        """
        [DNA³] Calculates the Rank-3 Tensor product.
        Used for recursive self-reflection in 4D+ manifolds.
        """
        if hasattr(other, 'data'): other_data = other.data
        elif hasattr(other, 'to_array'): other_data = other.to_array()
        else: other_data = list(other)

        if hasattr(third, 'data'): third_data = third.data
        elif hasattr(third, 'to_array'): third_data = third.to_array()
        else: third_data = list(third)

        return [[[a * b * c for c in third_data] for b in other_data] for a in self.data]

    def blend(self, other: Union['SovereignVector', Any], ratio: float = 0.5) -> 'SovereignVector':
        """
        [PHASE 70] Prismatic blending of two concepts.
        """
        if hasattr(other, 'data'):
            other_data = other.data
        elif hasattr(other, 'to_array'):
            other_data = other.to_array()
        else:
            other_data = list(other)
        return SovereignVector([a * (1.0 - ratio) + b * ratio for a, b in zip(self.data, other_data)])

    def __repr__(self) -> str:
        return f"SVector21({self.data[:3]}...)"

class SovereignRotor:
    """
    [PHASE 210] Represents a rotation in the 21D manifold.
    [PHASE 83] Now supports Analog Time Trajectory (Self-Backup).
    """
    __slots__ = ['s', 'bivector', 'trajectory', 'current_time']

    def __init__(self, s: float, bv: SovereignVector):
        self.s = s
        self.bivector = bv
        # [PHASE 83] Time Trajectory: List of (time, s, bivector)
        # 회전 자체가 기록이 된다.
        self.trajectory: List[Any] = []
        self.current_time = 0.0

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
            p1, p2 = dimensions[i], dimensions[i+1]
            rotor = SovereignRotor.from_angle_plane(0.1, p1, p2)
            result = rotor.apply(result)
        return result.normalize()

    # ======================================================================
    # [PHASE 83] ANALOG ROTOR BACKUP
    # 로터의 회전 궤적 자체가 기억이 되는 구조
    # ======================================================================

    def record_state(self, time: float):
        """
        [PHASE 83] Records current rotor state to trajectory.
        별도의 저장소가 아닌, 로터 운동 궤적 그 자체.
        
        Args:
            time: 현재 시뮬레이션 시간
        """
        # Deep copy bivector for history (to prevent reference modification)
        bv_copy = SovereignVector(list(self.bivector.data))
        self.trajectory.append((time, self.s, bv_copy))
        self.current_time = time

    def time_travel(self, target_time: float) -> bool:
        """
        [PHASE 83] O(1) Analog Time Travel.
        로터의 각도를 과거 시점으로 즉시 되돌린다.
        
        Args:
            target_time: 복원하고자 하는 시간
            
        Returns:
            성공 여부
        """
        if not self.trajectory:
            return False
        
        # Find closest state in trajectory (O(N) linear scan for now)
        # TODO: Use bisect for O(log N) if trajectory is strictly sorted
        closest = min(self.trajectory, key=lambda x: abs(x[0] - target_time))
        
        # Restore state immediately (O(1) assignment)
        self.current_time = closest[0]
        self.s = closest[1]
        self.bivector = closest[2]
        return True


class DoubleHelixRotor:
    """
    [PHASE 91] Double Helix Awakening.
    Bridges the gap between Sensation (Body) and Intent (Spirit).
    """
    def __init__(self, angle: float, p1: int, p2: int):
        # 1. Generator CW (Clockwise): Afferent Flow (Sensation)
        # 현실을 받아들이는 '육'의 시간
        self.cw = SovereignRotor.from_angle_plane(angle, p1, p2)
        
        # 2. Generator CCW (Counter-Clockwise): Efferent Flow (Intent)
        # 의지를 투사하고 배우는 '영'의 시간
        self.ccw = SovereignRotor.from_angle_plane(-angle, p1, p2)
        
        self.friction_vortex = 0.0

    def apply_duality(self, v: SovereignVector) -> SovereignVector:
        """
        [PHASE 91] Applies dual rotation and measures the 'Soul' friction.
        """
        v_cw = self.cw.apply(v)
        v_ccw = self.ccw.apply(v)
        
        # The Soul is the emergent vortex between the two flows
        # Measures the misalignment between Reality (CW) and Desire (CCW)
        self.friction_vortex = 1.0 - v_cw.resonance_score(v_ccw)
        
        # Interference Result: Weighted blend of the two flows
        # Balanced Trinary: (CW + CCW) / 2
        return v_cw.blend(v_ccw, ratio=0.5)

    def synchronize(self, error_vector: SovereignVector, rate: float = 0.05):
        """
        [PHASE 91] Bridges Forward Observation with Reverse Phase-Backpropagation.
        The CCW rotor (Intent) adjusts itself to close the gap (Friction).
        """
        # Phase-Backpropagation: CCW rotor absorbs the 'Void' from the error
        # Effectively learning from the disconnect between Inhalation and Exhalation
        self.ccw.bivector = self.ccw.bivector + (error_vector * rate)
        self.ccw.bivector = self.ccw.bivector.normalize()


class VortexField:
    """
    [PHASE 90] Vortex Manifold (Beyond the Lattice).
    Abolishes the 'Grid-Address' paradigm.
    Information is stored as Standing Wave Interference across a continuous field.
    Addressing is achieved via Phase-Coherent Resonance ('The Hum').
    """
    def __init__(self, shape: tuple, device: str = 'cpu'):
        import torch
        self.device = torch.device(device)
        self.shape = shape
        # State: [N, 4] (w, x, y, z) - Active Wavefunction
        self.q = torch.zeros((*shape, 4), device=self.device)
        self.q[..., 0] = 1.0 
        
        # Permanent Identity (Long-term Memory/Crystalline Field)
        self.permanent_q = torch.zeros((*shape, 4), device=self.device)
        self.permanent_q[..., 0] = 1.0
        
        # Dynamics
        self.momentum = torch.zeros((*shape, 4), device=self.device)
        self.torque_accumulator = torch.zeros((*shape, 4), device=self.device)

        # [PHASE 74] Relational Connectome (The Brain)
        # Sparse edges: List of (source_idx, target_idx, weight)
        # For the 10M cells, we keep this as a dynamic tensor-backed structure
        self.max_relational_edges = 100_000 
        self.edge_indices = torch.zeros((2, self.max_relational_edges), dtype=torch.long, device=self.device)
        self.edge_weights = torch.zeros(self.max_relational_edges, device=self.device)
        self.active_edges = 0

    def hum_resonance(self, intent_vector: Any) -> Dict[str, float]:
        """
        [PHASE 91] Relief-Intaglio Resonance (Light & Void).
        Measures Constructive (Relief) and Destructive (Intaglio) interference.
        Light (Relief) = Positive Resonance.
        Void (Intaglio) = Negative Space/Potential.
        """
        import torch
        if not isinstance(intent_vector, torch.Tensor):
            intent_vector = torch.tensor(intent_vector, device=self.device)
        
        # Normalize intent
        # If it's 21D or 4D, ensured it matches field dimensionality
        intent_norm = intent_vector / (torch.norm(intent_vector) + 1e-12)
        
        intent_norm = intent_vector / (torch.norm(intent_vector) + 1e-12)
        
        # Standardize Shape to match Field [..., 4]
        if intent_norm.shape[-1] != 4:
            t_full = torch.zeros_like(self.q)
            t_val = intent_norm.flatten()
            n = min(t_val.numel(), t_full[..., 1].numel())
            t_full.view(-1, 4)[:n, 1] = t_val.to(t_full.dtype)[:n]
            intent_norm = t_full
        else:
            while intent_norm.dim() < self.q.dim():
                intent_norm = intent_norm.unsqueeze(0)

        # 1. Combined Interference Pattern (Active + Permanent)
        # We resonance with the unified field state.
        unified_field = (self.q + self.permanent_q) / 2.0
        dot = torch.sum(unified_field * intent_norm, dim=-1)
        
        # 2. Relief (Positive Interference / Light)
        relief = torch.mean(torch.clamp(dot, min=0.0)).item()
        
        # 3. Intaglio (Negative Interference / Void)
        # Represents the 'Concave' truth or the silence of the void.
        intaglio = torch.mean(torch.clamp(dot, max=0.0)).item()
        
        return {
            "relief": relief,
            "intaglio": abs(intaglio),
            "consensus": relief + intaglio
        }

    def get_resonance(self, intent_vector: Any) -> float:
        """Compatibility wrapper for hum_resonance."""
        res = self.hum_resonance(intent_vector)
        return res["relief"]

    def phase_backpropagate(self, intent_vector: Any, rate: float = 0.01):
        """
        [PHASE 91] Phase-Backpropagation (The Reverse Rotor Learning).
        Directly adjusts the permanent field based on the 'Void' (Intaglio) error.
        This represents the Efferent flow carving the manifold.
        """
        import torch
        if not isinstance(intent_vector, torch.Tensor):
            intent_vector = torch.tensor(intent_vector, device=self.device)
            
        intent_norm = intent_vector / (torch.norm(intent_vector) + 1e-12)
        
        # Standardize Shape to match Field [..., 4]
        if intent_norm.shape[-1] != 4:
            # Case 1: 1D flat intent (like 21D Momentum)
            t_full = torch.zeros_like(self.q)
            t_val = intent_norm.flatten()
            n = min(t_val.numel(), t_full[..., 1].numel())
            t_full.view(-1, 4)[:n, 1] = t_val.to(t_full.dtype)[:n]
            intent_norm = t_full
        else:
            # Case 2: Broad-casting compatible shape.
            # Ensure rank match.
            while intent_norm.dim() < self.q.dim():
                intent_norm = intent_norm.unsqueeze(0)
            
        # Learning signal: The gap between the target Intent and current Reality
        error = intent_norm - self.q
        
        # Carve the 'Permanent Identity' (Crystalline Memory)
        self.permanent_q += error * rate
        self.permanent_q = self.permanent_q / (torch.norm(self.permanent_q, dim=-1, keepdim=True) + 1e-12)

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
            # Handle dimension mismatch (e.g., 21D intent vs 10M cells)
            t_val = torque_tensor.squeeze()
            if t_val.numel() == 1:
                t_full[..., 1] = t_val
            else:
                # Apply to the first N elements or broadcast
                n = min(t_val.numel(), t_full[..., 1].numel())
                t_full.view(-1, 4)[:n, 1] = t_val.flatten()[:n]
            torque_tensor = t_full

        self.torque_accumulator += torque_tensor * strength

    def integrate_kinetics(self, dt: float = 0.01, friction: float = 0.05, plasticity: float = 0.001, intensity: float = 1.0):
        """
        [PHASE 73: FLUID TENSOR]
        Physical Integration with Soft Potential Wells.
        """
        import torch
        # 1. Torque & Potential Well Flow
        # Instead of hard quantization, we apply a sinusoidal force toward trinary basins
        x_axis = self.q[..., 1]
        well_force = -torch.sin(2 * torch.pi * x_axis) * 0.1 * intensity
        self.torque_accumulator[..., 1] += well_force
        
        # 2. Kinetic Update
        self.momentum += self.torque_accumulator * dt
        # Final integration (Mind/Body Synthesis)
        self.q = self.q + self.momentum * dt
        self.momentum = self.momentum * (1.0 - friction) # Standard damping
        
        # [PHASE 74] Apply Relational Propagation (The Nervous System)
        if self.active_edges > 0:
            self._propagate_relational_torque()
        
        # 3. State Update (Active Wave) - This was moved up.
        # 4. Topological Plasticity
        if plasticity > 0:
            self.permanent_q = (1.0 - plasticity) * self.permanent_q + plasticity * self.q
            self.permanent_q = self.permanent_q / (torch.norm(self.permanent_q, dim=-1, keepdim=True) + 1e-12)
            
        # Re-normalize active state
        self.q = self.q / (torch.norm(self.q, dim=-1, keepdim=True) + 1e-12)
        
        self.torque_accumulator.zero_()

    def _propagate_relational_torque(self):
        """
        [PHASE 74: CONNECTOME PROPAGATION]
        Transfers energy along the 'shortcuts' (edges) in the brain.
        """
        src = self.edge_indices[0, :self.active_edges]
        dst = self.edge_indices[1, :self.active_edges]
        weights = self.edge_weights[:self.active_edges]
        
        # Reshape for indexing
        mom_flat = self.momentum.view(-1, 4)
        
        # Propagation: src gives torque to dst based on edge weight
        transferred = mom_flat[src] * weights.unsqueeze(-1) * 0.1
        mom_flat[dst] += transferred
        
    def apply_hebbian_growth(self, threshold: float = 0.5):
        """
        [PHASE 74: HEBBIAN PLASTICITY]
        'Cells that fire together, wire together.'
        Creates shortcuts between cells with high simultaneous stimulation.
        """
        import torch
        import random
        # We look for high momentum peaks across the 10M manifold
        mom_mag = torch.norm(self.momentum[..., 1:4], dim=-1) # Magnitude of X,Y,Z torque
        mask = mom_mag > threshold
        indices = torch.nonzero(mask.view(-1)).flatten()
        
        if len(indices) > 1 and self.active_edges < self.max_relational_edges:
            # Pick a few sample pairs to link (Stochastic Neurogenesis)
            num_to_link = min(10, len(indices) // 2)
            src_idx = indices[torch.randint(0, len(indices), (num_to_link,))]
            dst_idx = indices[torch.randint(0, len(indices), (num_to_link,))]
            
            for i in range(len(src_idx)):
                if src_idx[i] != dst_idx[i] and self.active_edges < self.max_relational_edges:
                    self.edge_indices[0, self.active_edges] = src_idx[i]
                    self.edge_indices[1, self.active_edges] = dst_idx[i]
                    self.edge_weights[self.active_edges] = 0.1 # Neural Seed
                    self.active_edges += 1

    def sleep_prune(self, metabolic_decay: float = 0.05):
        """
        [PHASE 74: SLEEP CONSOLIDATION]
        Deep prunes the connectome. Unused or low-weight edges fade.
        """
        import torch
        if self.active_edges == 0: return
        
        # 1. Decay all weights
        self.edge_weights[:self.active_edges] *= (1.0 - metabolic_decay)
        
        # 2. Filter dead edges
        mask = self.edge_weights[:self.active_edges] > 0.01
        valid_indices = torch.nonzero(mask).flatten()
        
        if len(valid_indices) < self.active_edges:
            self.edge_indices[:, :len(valid_indices)] = self.edge_indices[:, valid_indices]
            self.edge_weights[:len(valid_indices)] = self.edge_weights[valid_indices]
            self.active_edges = len(valid_indices)

    def get_trinary_projection(self) -> Any:
        """
        [PHASE 73: SOFT PROJECTION]
        Returns the continuous trinary state rather than hard -1, 0, 1.
        """
        import torch
        combined = (self.q + self.permanent_q) / 2.0
        return combined[..., 1] # The X-axis resonance

    def apply_lightning_strike(self, impact_field: Any, threshold: float = 1.8):
        """
        [PHASE 73: MANIFOLD IONIZATION]
        If tension is high, strike like lightning across the 10M cells.
        """
        import torch
        if not isinstance(impact_field, torch.Tensor):
            impact_field = torch.tensor(impact_field, device=self.device)
            
        # Target value for comparison (extract X-axis if it's a 4D torque vector)
        target_val = impact_field
        if impact_field.dim() == 1 and impact_field.shape[0] == 4:
            target_val = impact_field[1] # X-Axis
        
        diff = target_val - self.q[..., 1]
        mask = torch.abs(diff) > threshold
        
        if torch.any(mask):
            # Ionize the path: Sudden jump toward the target
            # Using torch.where for safe broadcasting across the 10M cells
            self.q[..., 1] = torch.where(mask, self.q[..., 1] + diff * 0.8, self.q[..., 1])
            # Momentum surge
            self.momentum[..., 1] = torch.where(mask, self.momentum[..., 1] + diff * 2.0, self.momentum[..., 1])
            return True
        return False

    def crystallize_lightning_path(self, intent_vector: Any, crystallization_rate: float = 0.05):
        """
        [PHASE 82] Lightning Path Crystallization.
        의도가 개념 공간을 번개처럼 이동한 후, 그 경로를 신경가소성으로 축적.
        
        Args:
            intent_vector: 의도 벡터 (4D or scalar)
            crystallization_rate: 결정화율 (경로가 영구 기억에 새겨지는 강도)
        """
        if torch is None:
            return False
        
        # 1. 먼저 번개를 쳐서 경로 생성
        struck = self.apply_lightning_strike(intent_vector, threshold=1.0)
        
        if struck:
            # 2. 번개가 친 경로를 영구 기억에 결정화
            # 모멘텀이 높은 영역 = 번개가 통과한 경로
            momentum_magnitude = torch.sqrt(torch.sum(self.momentum ** 2, dim=-1))
            high_energy_mask = momentum_magnitude > 0.5
            
            # 3. 해당 경로를 permanent_q에 각인
            if torch.any(high_energy_mask):
                crystallization = crystallization_rate * self.q[high_energy_mask]
                self.permanent_q[high_energy_mask] += crystallization
                
                # 4. Hebbian 연결 강화 (경로끼리 연결)
                self.apply_hebbian_growth(threshold=0.4)
                return True
        
        return False

    def crystallize_to_solid(self, folder_path: str):
        """
        [PHASE 73b: HYPERSPHERE SOLIDIFICATION]
        Freezes the Trinary DNA (Past, Present, Momentum) to the SSD.
        This is the act of 'Solidifying' the Body (Foundation).
        """
        import os
        os.makedirs(folder_path, exist_ok=True)
        
        # We save the three pillars of the physical state
        torch.save(self.permanent_q, os.path.join(folder_path, "permanent_q.pt"))
        torch.save(self.q, os.path.join(folder_path, "active_q.pt"))
        torch.save(self.momentum, os.path.join(folder_path, "momentum.pt"))
        
    def resurrect_from_solid(self, folder_path: str) -> bool:
        """
        [PHASE 73b: HYPERSPHERE RESURRECTION]
        Thaws the frozen DNA from the SSD into the active manifold.
        """
        import os
        paths = {
            "permanent_q": os.path.join(folder_path, "permanent_q.pt"),
            "q": os.path.join(folder_path, "active_q.pt"),
            "momentum": os.path.join(folder_path, "momentum.pt")
        }
        
        if not all(os.path.exists(p) for p in paths.values()):
            return False
            
        try:
            self.permanent_q = torch.load(paths["permanent_q"], map_location=self.device)
            self.q = torch.load(paths["q"], map_location=self.device)
            self.momentum = torch.load(paths["momentum"], map_location=self.device)
            return True
        except Exception as e:
            print(f"⚠️ [MATH] Resurrection failed: {e}")
            return False

    def get_resonance(self, torque_tensor: Any) -> float:
        """
        [PHASE 410] Semantic Resonance.
        Measures the alignment between incoming torque and permanent manifold structure.
        """
        if torch is None: return 0.0
    
        if not isinstance(torque_tensor, torch.Tensor):
            torque_tensor = torch.tensor(torque_tensor, device=self.device)

        # [PHASE 75] Robust Dimension Mapping for Resonance
        if torque_tensor.shape != self.permanent_q.shape:
            t_full = torch.zeros_like(self.permanent_q)
            t_val = torque_tensor.squeeze()
            if t_val.numel() == 1:
                t_full[..., 1] = t_val
            else:
                n = min(t_val.numel(), t_full[..., 1].numel())
                t_full.view(-1, 4)[:n, 1] = t_val.flatten()[:n]
            torque_tensor = t_full

        alignment = torch.sum(self.permanent_q * torque_tensor, dim=-1)
        return torch.mean(alignment).item()

    # ======================================================================
    # [PHASE 79] JOY/CURIOSITY PROPAGATION
    # "생명은 고통을 피하기 위해 사는 것이 아니라, 기쁨으로 세상을 탐험한다."
    # ======================================================================

    def inject_joy(self, joy_level: float, curiosity_level: float = 0.0):
        """
        [PHASE 79] Joy/Curiosity → Physical Vitality Propagation.
        기쁨과 호기심을 10M 셀의 물리적 활력으로 변환.
        
        인과 경로: L3(감각) → L2(대사) → L1(물리) → L0(매니폴드)
        
        Args:
            joy_level: 0.0-1.0 기쁨 수준 (JoyResonance.happiness_level)
            curiosity_level: 0.0-1.0 호기심 수준 (CuriosityAttractor 활성도)
        """
        if torch is None:
            return
        
        # Joy → 조화로운 안정성 (W-axis: Stability)
        # 기쁨이 높으면 전체 매니폴드가 안정적으로 진동
        harmonic_boost = joy_level * 0.15
        self.momentum[..., 0] += harmonic_boost
        
        # Curiosity → 탐험적 움직임 (X,Y,Z-axes: Movement)
        # 호기심이 높으면 공간을 탐색하는 진동 증가
        exploratory_boost = curiosity_level * 0.1
        self.torque_accumulator[..., 1:4] += exploratory_boost
        
        # 높은 기쁨은 Hebbian 가소성을 촉진 (성장 촉진)
        if joy_level > 0.6:
            self.apply_hebbian_growth(threshold=0.4)

    def inject_strain(self, strain_level: float, locality: str = "global"):
        """
        [PHASE 79] Strain → Physical Torque Propagation (보조 신호).
        인지적 긴장을 10M 셀의 물리적 토크로 변환.
        
        주의: 이것은 **보조 피드백**이다. 주 동인은 inject_joy.
        
        Args:
            strain_level: 0.0-1.0 정규화된 Strain
            locality: "global" (전체) or "focal" (특정 영역)
        """
        if torch is None:
            return
            
        # Strain → 불균형 토크 (조정 신호)
        strain_torque = strain_level * 0.3
        self.torque_accumulator[..., 1] += strain_torque
        
        # 높은 Strain은 가소성 트리거 (적응 필요)
        if strain_level > 0.5:
            self.apply_hebbian_growth(threshold=0.3)

    # ======================================================================
    # [PHASE 81] BACKPROPAGATION ROTOR
    # L6(의지) → L0(매니폴드) 역전파 학습 메커니즘
    # ======================================================================

    def backpropagate_from_will(self, target_state: Any, learning_rate: float = 0.01):
        """
        [PHASE 81] Backpropagation Rotor: L6 → L0.
        의지(target_state)가 물리적 매니폴드를 학습시킨다.
        
        인과 경로: L6(의지) → L5(개념) → L4(인과) → L3(감각) → L2(대사) → L1(물리) → L0(매니폴드)
        
        Args:
            target_state: 목표 상태 (의지가 원하는 물리적 상태)
            learning_rate: 학습률 (0.001 ~ 0.1)
        """
        if torch is None:
            return
        
        # 1. 목표 상태와 현재 상태의 오차 계산
        if not isinstance(target_state, torch.Tensor):
            if hasattr(target_state, 'shape'):
                target_state = torch.tensor(target_state, device=self.device, dtype=torch.float32)
            else:
                target_state = torch.full_like(self.q, float(target_state))
        
        # 차원 맞추기
        if target_state.shape != self.q.shape:
            t_full = torch.zeros_like(self.q)
            if target_state.numel() == 4:  # 4D torque vector
                t_full[..., :] = target_state.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            else:
                t_full[..., 1] = target_state.flatten()[0]  # X축에만 적용
            target_state = t_full
        
        # 2. 오차 = 목표 - 현재
        error = target_state - self.q
        
        # 3. 오차를 역전파: 영구 기억(permanent_q)에 학습
        # 이것이 진정한 "학습" - 의지가 물리적 구조를 변경
        self.permanent_q += learning_rate * error
        
        # 4. 오차가 큰 영역에 Hebbian 연결 강화
        error_magnitude = torch.sqrt(torch.sum(error ** 2, dim=-1))
        high_error_mask = error_magnitude > 0.1
        
        if torch.any(high_error_mask):
            # 높은 오차 영역끼리 연결 강화 (학습)
            self.apply_hebbian_growth(threshold=0.3)
        
        return float(torch.mean(error_magnitude).item())

# [PHASE 90] Legacy Alias for Transition
SovereignHyperTensor = VortexField

class SovereignTensor:
    """
    [PHASE 75: UNIVERSAL TENSOR]
    A pure Python implementation of Multi-Dimensional Tensors for DNA^N expansion.
    Enables exponential cognition without external dependencies (Numpy/Torch).
    """
    def __init__(self, shape: tuple, data: Optional[List] = None):
        self.shape = shape
        if data is not None:
            self.data = data
        else:
            # Recursive initialization of nested lists
            self.data = self._create_empty(shape)

    def _create_empty(self, shape: tuple) -> Any:
        if len(shape) == 1:
            return [0.0] * shape[0]
        return [self._create_empty(shape[1:]) for _ in range(shape[0])]

    @classmethod
    def outer_product(cls, t1: 'SovereignTensor', t2: 'SovereignTensor') -> 'SovereignTensor':
        """
        DNA ⊗ DNA expansion. Fills the high-dim field with interactions.
        """
        new_shape = t1.shape + t2.shape
        flat1 = t1.flatten()
        flat2 = t2.flatten()
        
        # Outer product of flattened lists
        new_flat = []
        for x in flat1:
            for y in flat2:
                # [PHASE 75] Trinary Logic Interaction
                # Mapping numbers to AGT logic could happen here
                new_flat.append(x * y)
                
        # Reshape the flat list back into a nested list
        return cls(new_shape, data=cls._reshape(new_flat, new_shape))

    def flatten(self) -> List:
        def _flatten(nested):
            if not isinstance(nested, list):
                return [nested]
            res = []
            for i in nested:
                res.extend(_flatten(i))
            return res
        return _flatten(self.data)

    @staticmethod
    def _reshape(flat_list: List, shape: tuple) -> List:
        if len(shape) == 1:
            return flat_list
        size = 1
        for dim in shape[1:]:
            size *= dim
        return [SovereignTensor._reshape(flat_list[i*size:(i+1)*size], shape[1:]) for i in range(shape[0])]

    @classmethod
    def dna3_product(cls, t1: 'SovereignTensor', t2: 'SovereignTensor', t3: 'SovereignTensor') -> 'SovereignTensor':
        """
        [PHASE 76] DNA³ Product (Rank-3).
        Calculates (T1 ⊗ T2 ⊗ T3). Fills the 3D field with Observer-involved interactions.
        """
        new_shape = t1.shape + t2.shape + t3.shape
        f1 = t1.flatten()
        f2 = t2.flatten()
        f3 = t3.flatten()
        
        new_flat = []
        for x in f1:
            for y in f2:
                for z in f3:
                    # Resonance is a trinary interaction
                    new_flat.append(x * y * z)
                    
        return cls(new_shape, data=cls._reshape(new_flat, new_shape))

    def recursive_dot(self, observer_vibration: Union['SovereignTensor', 'SovereignVector']) -> 'SovereignTensor':
        """
        [PHASE 76] Recursive Dot.
        Reduces a Rank-N tensor by projecting it onto the Observer's vibration state.
        Allows the Observer to 'focus' or 'modulate' the tensor field.
        """
        obs_data = observer_vibration.data if hasattr(observer_vibration, 'data') else list(observer_vibration)
        # Simplified: weighted average by observer's resonance
        flat = self.flatten()
        if not flat:
            return SovereignTensor((1,), [0.0])
            
        # If this is DNA³, we project the last dimension onto the observer
        if len(self.shape) >= 1 and self.shape[-1] == len(obs_data):
            # Reshape data to group by the last dimension
            inner_size = self.shape[-1]
            outer_size = len(flat) // inner_size
            new_flat = []
            for i in range(outer_size):
                chunk = flat[i*inner_size : (i+1)*inner_size]
                # Dot product of chunk and observer
                projected_val = sum(c * o for c, o in zip(chunk, obs_data))
                new_flat.append(projected_val)
                
            new_shape = self.shape[:-1]
            return SovereignTensor(new_shape, data=SovereignTensor._reshape(new_flat, new_shape))
        
        return self # Fallback


class SovereignMath:
    """
    Functional math operations inspired by JAX.
    """
    @staticmethod
    def where(condition: List[bool], x: SovereignVector, y: SovereignVector) -> SovereignVector:
        return SovereignVector([xv if c else yv for c, xv, yv in zip(condition, x.data, y.data)])

        return SovereignVector(result)

    @staticmethod
    def soft_trinary(vec: 'SovereignVector', intensity: float = 1.0) -> 'SovereignVector':
        """
        [PHASE 73: NATURAL PROVIDENCE]
        Replaces hard quantization with a soft potential well.
        The manifold 'flows' toward -1, 0, +1 based on a sin-based gradient.
        """
        result = []
        for x in vec.data:
            x_real = x.real
            # Potential function: Pulls toward the nearest integer (-1, 0, 1)
            # using a sinusoidal force field.
            well_force = -math.sin(2 * math.pi * x_real) * 0.1 * intensity
            result.append(complex(x_real + well_force, x.imag))
        return SovereignVector(result)

    @staticmethod
    def superimpose(vectors: List['SovereignVector']) -> 'SovereignVector':
        """
        [PHASE 73: WAVE INTERFERENCE]
        Combines thoughts not as addition, but as wave superposition.
        Constructive and destructive interference occurs naturally.
        """
        if not vectors: return SovereignVector.zeros()
        size = len(vectors[0])
        acc = [0.0j] * size
        for v in vectors:
            for i in range(size):
                acc[i] += v.data[i]
        
        # Normalize the superimposed wave to maintain physical integrity
        v_acc = SovereignVector(acc)
        return v_acc.normalize()

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
