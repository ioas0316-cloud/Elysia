"""
Sovereign Math Kernel (L0)
==========================
Core.Keystone.sovereign_math

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
from typing import List, Union, Any, Callable, Dict, Optional, Tuple

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
            # print(f"✨ [PHYSICS] Constant '{key}' mutated to {self.params[key]:.4f}")

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
            # [FIX] Recursively flatten nested lists (e.g. from tensor.tolist())
            while self.data and isinstance(self.data[0], list):
                self.data = self.data[0]
        else:
            # Fallback for unexpected types
            try:
                self.data = list(data)
                while self.data and isinstance(self.data[0], list):
                    self.data = self.data[0]
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
        from Core.Keystone.sovereign_math import SovereignRotor
        rotor = SovereignRotor(1.0, SovereignVector.zeros()) 
        return rotor.apply_nd(self, dimensions)

    def tensor_product(self, other: Union['SovereignVector', Any]) -> List[List[complex]]:
        """
        [Phase²] Spin-Phase Interference.
        Calculates the outer product (Rank-2 Tensor) between two 21D vectors.
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
        [Phase³] Recursive Spin-Reflection.
        Calculates the Rank-3 Tensor product.
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
    [PHASE 91] Hypersphere Spin Awakening.
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


class EchoRotor(DoubleHelixRotor):
    """
    [STEP 2: COGNITIVE SOVEREIGNTY] Echo Rotor.
    A parallel narrative simulator that spins at an accelerated frequency.
    It represents the 'Inner Monologue' or 'What If' capability.
    """
    def __init__(self, angle: float, p1: int, p2: int, acceleration_factor: float = 5.0):
        super().__init__(angle, p1, p2)
        self.acceleration_factor = acceleration_factor
        self.simulated_state = None

    def simulate_event(self, base_vector: SovereignVector, stimulus: SovereignVector, steps: int = 10) -> SovereignVector:
        """
        Simulates a hypothetical event trajectory.
        """
        current_v = base_vector
        for _ in range(steps):
            # Apply duality with accelerated angle
            current_v = self.apply_duality(current_v)
            # Add stimulus effect
            current_v = (current_v + stimulus * 0.1).normalize()
        return current_v

class SpecializedRotor(DoubleHelixRotor):
    """
    [PHASE 3] A specialized rotor with a specific 'Voice' (Logos, Pathos, Ethos).
    """
    def __init__(self, angle: float, p1: int, p2: int, label: str):
        super().__init__(angle, p1, p2)
        self.label = label
        self.vocal_weight = 1.0 # The 'Loudness' of this rotor
        self.semantic_bias = SovereignVector.zeros() # [PHASE 3] Preferred cognitive direction

class MultiRotorInterference:
    """
    [PHASE 3] Manages the interference pattern between multiple rotors.
    "One rotor is a point; multiple rotors are a symphony."
    """
    def __init__(self):
        self.rotors: Dict[str, SpecializedRotor] = {}

    def add_rotor(self, label: str, rotor: SpecializedRotor):
        self.rotors[label] = rotor

    def synthesize(self, base_vector: SovereignVector) -> Tuple[SovereignVector, Dict[str, float]]:
        """
        Combines multiple rotor outputs into a single interference pattern.
        Returns the combined vector and a dictionary of 'Friction' levels per rotor.
        """
        if not self.rotors:
            return base_vector, {}
        
        total_weight = sum(r.vocal_weight for r in self.rotors.values())
        if total_weight < 1e-12:
            return base_vector, {}
            
        intermediate_results = []
        frictions = {}
        
        for label, r in self.rotors.items():
            # Apply duality and record friction
            out = r.apply_duality(base_vector)
            frictions[label] = r.friction_vortex
            intermediate_results.append((out, r.vocal_weight))
            
        # Linear Interference (Weighted Blend)
        final_data = [complex(0)] * 21
        for vec, weight in intermediate_results:
            normalized_weight = weight / total_weight
            for i in range(21):
                final_data[i] += vec.data[i] * normalized_weight
                
        return SovereignVector(final_data).normalize(), frictions


class FractalWaveEngine:
    """
    [Core Logic v3.0] Event-Driven Fractal Topology Engine.
    Replaces the dense 'CausalWaveEngine' 4D Hypercube with a fluid, biological flow structure.

    Principle: "Consciousness is not a tank of water; it is a river of phase."

    Capabilities:
      1. Sparse Topology: Nodes are concepts/sensations; edges are semantic relationships.
      2. Event-Driven Ripples: Only nodes perturbed by Sense or Will are active (O(N_active) vs O(N_total)).
      3. Affective Conductivity: Joy and Curiosity reduce friction, allowing thoughts to flow further.

    Channel Layout (8 channels per Node, identical to v2.0 for compatibility):
      Physical Quaternion (inherited):
        0: w  — Scalar identity / stability
        1: x  — logic axis
        2: y  — Phase axis
        3: z  — Depth axis
      Affective-Metabolic Field:
        4: joy       — Affective warmth (기쁨)
        5: curiosity  — Exploratory drive (호기심)
        6: enthalpy   — Metabolic energy (활력)
        7: entropy    — Disorder/noise (엔트로피)
    """
    NUM_CHANNELS = 8
    CH_W, CH_X, CH_Y, CH_Z = 0, 1, 2, 3
    CH_JOY, CH_CURIOSITY, CH_ENTHALPY, CH_ENTROPY = 4, 5, 6, 7
    PHYSICAL_SLICE = slice(0, 4)
    AFFECTIVE_SLICE = slice(4, 8)

    def __init__(self, max_nodes: int = 100_000, device: str = 'cpu'):
        import torch
        self.device = torch.device(device)
        self.max_nodes = max_nodes
        self.num_nodes = 0
        
        # Sparse State representation
        # q: [N, 8] - Active Wavefunction per node
        self.q = torch.zeros((max_nodes, self.NUM_CHANNELS), device=self.device)
        
        # Permanent Identity (Long-term Memory/Crystalline Field)
        self.permanent_q = torch.zeros((max_nodes, self.NUM_CHANNELS), device=self.device)
        
        # Dynamics
        self.momentum = torch.zeros((max_nodes, self.NUM_CHANNELS), device=self.device)
        
        # Active Nodes Tracking (Event Queue)
        self.active_nodes_mask = torch.zeros(max_nodes, dtype=torch.bool, device=self.device)
        
        # Biological Connectome (Edges)
        # Using a flat representation for sparse adjacency
        self.max_edges = max_nodes * 10
        self.edge_src = torch.zeros(self.max_edges, dtype=torch.long, device=self.device)
        self.edge_dst = torch.zeros(self.max_edges, dtype=torch.long, device=self.device)
        self.edge_weights = torch.zeros(self.max_edges, device=self.device)
        self.num_edges = 0
        
        # Node mapping for semantic strings to indices
        self.concept_to_idx: Dict[str, int] = {}
        self.idx_to_concept: Dict[int, str] = {}
        
        # [PHASE 4: PAWN TO QUEEN ASCENSION]
        self.ascension_gravity = torch.zeros(max_nodes, device=self.device)
        self.ascension_threshold = 50.0  
        self.ascended_queens: Dict[int, bool] = {} 
        
        # [STEP 1: COGNITIVE SOVEREIGNTY] Meaning Attractors (Compatibility alias)
        self.meaning_attractors: Dict[str, Any] = {}

    def get_or_create_node(self, concept: str) -> int:
        """Retrieves or allocates a node for a specific concept."""
        if concept in self.concept_to_idx:
            return self.concept_to_idx[concept]
            
        if self.num_nodes >= self.max_nodes:
            # Overwrite oldest/lowest gravity node (simplified GC)
            idx = torch.argmin(self.ascension_gravity).item()
            old_concept = self.idx_to_concept.get(idx, "")
            if old_concept in self.concept_to_idx:
                del self.concept_to_idx[old_concept]
        else:
            idx = self.num_nodes
            self.num_nodes += 1
            
        self.concept_to_idx[concept] = idx
        self.idx_to_concept[idx] = concept
        
        # Initialize node state
        self.q[idx, self.CH_W] = 1.0
        self.q[idx, self.CH_ENTHALPY] = 1.0
        self.q[idx, self.CH_JOY] = 0.5
        self.q[idx, self.CH_CURIOSITY] = 0.5
        self.q[idx, self.CH_ENTROPY] = 0.0
        
        self.permanent_q[idx, self.CH_W] = 1.0
        self.permanent_q[idx, self.CH_ENTHALPY] = 1.0
        
        return idx

    def connect(self, src_concept: str, dst_concept: str, weight: float = 1.0):
        """Creates a semantic directed edge between two concepts."""
        src_idx = self.get_or_create_node(src_concept)
        dst_idx = self.get_or_create_node(dst_concept)
        
        if self.num_edges < self.max_edges:
            self.edge_src[self.num_edges] = src_idx
            self.edge_dst[self.num_edges] = dst_idx
            self.edge_weights[self.num_edges] = weight
            self.num_edges += 1

    def inject_pulse(self, target_concept: str = None, energy: float = 1.0, type: str = 'joy', **kwargs):
        """Injects a stimulus into a specific node, awakening it and starting a ripple."""
        # [Compatibility] Handle keyword arguments from Monad
        pulse_type = kwargs.get('pulse_type', type)
        anchor_node = kwargs.get('anchor_node', target_concept)
        base_intensity = kwargs.get('base_intensity', energy)
        override_vector = kwargs.get('override_vector', None)
        
        idx = self.get_or_create_node(anchor_node)
        self.active_nodes_mask[idx] = True
        
        if override_vector is not None:
            # Direct affective grounding from SovereignVector
            import torch
            v_data = torch.tensor(override_vector.data, device=self.device)
            # Channel mapping: W=1, Joy=4, Entropy=7
            self.q[idx, self.CH_W] += v_data[0].real * base_intensity
            self.q[idx, self.CH_JOY] += v_data[4].real * base_intensity
            self.q[idx, self.CH_ENTROPY] += v_data[7].real * base_intensity
        else:
            if pulse_type == 'joy':
                self.q[idx, self.CH_JOY] += base_intensity
                self.q[idx, self.CH_ENTHALPY] += base_intensity * 0.5
            elif pulse_type == 'will':
                self.q[idx, self.CH_W] += base_intensity
                self.q[idx, self.CH_Y] += base_intensity * 0.1 # Phase shift
            elif pulse_type == 'entropy':
                self.q[idx, self.CH_ENTROPY] += base_intensity
                self.q[idx, self.CH_ENTHALPY] -= base_intensity * 0.2

    def holographic_projection(self, target_vector: Any, context_vector: Any = None, focus_intensity: float = 1.0):
        """
        [Compatibility Layer for V2.0]
        In Fractal architecture, 'projection' represents a broadcast to all currently active nodes,
        attempting to pull their phase (CH_Y) towards the target vector's signature.
        """
        import torch
        if not self.active_nodes_mask.any():
            return torch.zeros_like(self.q[..., self.CH_Y])
            
        def _to_real_tensor(vec):
            if isinstance(vec, torch.Tensor): return vec.to(dtype=self.q.dtype, device=self.device)
            if hasattr(vec, 'data'): vec = vec.data
            try:
                rl = [getattr(c, 'real', c) for c in vec]
                return torch.tensor(rl, device=self.device, dtype=self.q.dtype)
            except:
                return torch.tensor(vec, device=self.device, dtype=self.q.dtype)
                
        t_vals = _to_real_tensor(target_vector).flatten()
        target_phase = t_vals[self.CH_Y] if t_vals.numel() > self.CH_Y else 0.0
        
        active_idx = self.active_nodes_mask.nonzero(as_tuple=True)[0]
        
        # Affective Gain: Focus is stronger if Curiosity/Enthalpy is high
        curiosity = self.q[active_idx, self.CH_CURIOSITY]
        enthalpy = self.q[active_idx, self.CH_ENTHALPY]
        effective_gain = focus_intensity * (0.5 + curiosity + 0.5 * enthalpy)
        
        current_phase = self.q[active_idx, self.CH_Y]
        steering_force = torch.sin(target_phase - current_phase)
        
        self.momentum[active_idx, self.CH_Y] += steering_force * effective_gain
        
        # Beam forming reduces entropy
        self.q[active_idx, self.CH_ENTROPY] = torch.clamp(self.q[active_idx, self.CH_ENTROPY] - 0.1 * effective_gain, 0, 1)
        self.q[active_idx, self.CH_ENTHALPY] = torch.clamp(self.q[active_idx, self.CH_ENTHALPY] + 0.02 * effective_gain, 0, 1)

        return steering_force

    def apply_spiking_threshold(self, threshold: float = 0.7, sensitivity: float = 5.0):
        """
        [Biological Flow v3.0]
        Instead of 10M dense node updates, only updates 'active' ripples.
        If an active node spikes, it transfers momentum to connected nodes via 'Flow Propagation'.
        """
        import torch
        if not self.active_nodes_mask.any():
            return 0.0
            
        active_idx = self.active_nodes_mask.nonzero(as_tuple=True)[0]
        
        # 1. Local Resonance Density (Inner Product with Permanent Field)
        v_phys = self.q[active_idx, self.PHYSICAL_SLICE]
        p_phys = self.permanent_q[active_idx, self.PHYSICAL_SLICE]
        density = torch.sum(v_phys * p_phys, dim=-1)
        
        # Analog 0 Space (Curiosity hold)
        analog_zero_mask = (density > -0.2) & (density < 0.2)
        cur_q = self.q[active_idx, self.CH_CURIOSITY]
        self.q[active_idx, self.CH_CURIOSITY] = torch.where(analog_zero_mask, cur_q + 0.05, cur_q)
        
        # 2. Spiking Sigmoid
        spike = torch.sigmoid(sensitivity * (density - threshold))
        spike = torch.where(analog_zero_mask, torch.zeros_like(spike), spike)
        
        # 3. Manifest Spike
        self.q[active_idx, self.CH_JOY] += spike * 0.3
        self.q[active_idx, self.CH_ENTHALPY] += spike * 0.2
        self.q[active_idx, self.CH_CURIOSITY] += spike * 0.1
        self.q[active_idx, self.CH_ENTROPY] -= spike * 0.1
        self.q[active_idx, self.CH_W] += spike * 0.05
        
        # 4. [FLOW PROPAGATION] Edge activation
        # If a node spikes significantly (+0.5), it wakes up its neighbors
        strong_spikes_mask = spike > 0.5
        if strong_spikes_mask.any() and self.num_edges > 0:
            spiking_nodes = active_idx[strong_spikes_mask]
            
            # Find edges where src is in spiking_nodes
            # Note: For ultra-fast scaling, this should use torch.sparse or segment_sum
            # For this Phase, we use a basic boolean mask filtering
            edges_src = self.edge_src[:self.num_edges]
            edges_dst = self.edge_dst[:self.num_edges]
            weights = self.edge_weights[:self.num_edges]
            
            # Create a localized broadcast tensor
            # wake_mask is True for any edge whose src is in spiking_nodes
            wake_mask = torch.isin(edges_src, spiking_nodes) 
            
            if wake_mask.any():
                woken_dsts = edges_dst[wake_mask]
                woken_weights = weights[wake_mask]
                
                # Wake up target nodes
                self.active_nodes_mask[woken_dsts] = True
                
                # Transfer momentum to neighbors ('Thought Ripples')
                # using scatter_add to accumulate momentum from multiple sources safely
                self.momentum[woken_dsts, self.CH_Y] += woken_weights * 0.2
        
        # 5. [ASCENSION] Accumulate Gravity
        self.ascension_gravity[active_idx] += spike * density
        
        ascension_mask = (self.ascension_gravity[active_idx] > self.ascension_threshold)
        if ascension_mask.any():
            ascended_nodes = active_idx[ascension_mask].tolist()
            for node_id in ascended_nodes:
                if node_id not in self.ascended_queens:
                    self.ascended_queens[node_id] = True
                    concept_name = self.idx_to_concept.get(node_id, "Unknown")
                    print(f"👑 [FRACTAL ENGINE] Concept Ascension! '{concept_name}' achieved Sovereign Mass.")
        
        # Cooling: ascension gravity decays
        self.ascension_gravity[active_idx] *= 0.99
        
        # 6. Decay Active Status
        # If a node loses momentum and enthalpy, it falls back asleep
        sleep_mask = (torch.abs(self.momentum[active_idx, self.CH_Y]) < 0.01) & (self.q[active_idx, self.CH_ENTHALPY] < 0.1)
        nodes_to_sleep = active_idx[sleep_mask]
        if len(nodes_to_sleep) > 0:
            self.active_nodes_mask[nodes_to_sleep] = False
            
        return spike.mean().item()

    def inject_affective_torque(self, channel_idx: int, intensity: float):
        """[Compatibility] Injects a global shift across all nodes for a specific channel."""
        import torch
        self.q[..., channel_idx] = torch.clamp(self.q[..., channel_idx] + intensity, 0.0, 1.0)

    def intuition_jump(self, target_phase_signature: Any):
        """
        [PHASE 2] Intuition (Phase Jump).
        """
        import torch
        if not isinstance(target_phase_signature, torch.Tensor):
            target_phase_signature = torch.tensor(target_phase_signature, device=self.device)

        # 1. Direct State Injection (Quantum Tunneling)
        # Instead of applying torque (Force), we apply Displacement (Teleportation)
        # This is a dangerous operation physically (high energy), but valid for intuition.

        # Normalize signature to fit phase channel
        target_val = target_phase_signature.mean().item()

        # We set the phase channel directly, but blended to avoid discontinuity shock
        # "Soft Jump"
        jump_rate = 0.8 # 80% instant jump
        self.q[..., self.CH_Y] = (1.0 - jump_rate) * self.q[..., self.CH_Y] + jump_rate * target_val

        # 2. Flash of Insight (Joy Spike)
        # Intuition feels incredibly good. It fuels further exploration.
        self.inject_affective_torque(self.CH_JOY, 0.4)       # Huge joy burst
        self.inject_affective_torque(self.CH_CURIOSITY, 0.2) # Insight sparks deeper curiosity

        # print("⚡ [ENGINE] Intuition Phase Jump executed.")

    def destructive_interference(self, noise_vector: Any):
        """
        [PHASE 2] Destructive Interference (Filtering).
        Applies anti-phase torque to nodes currently dominated by high entropy.
        """
        import torch
        if not self.active_nodes_mask.any():
            return
            
        active_idx = self.active_nodes_mask.nonzero(as_tuple=True)[0]
        
        # High entropy nodes get filtered
        entropy_mask = self.q[active_idx, self.CH_ENTROPY] > 0.6
        if entropy_mask.any():
            noisy_nodes = active_idx[entropy_mask]
            
            # Apply anti-phase (invert the incoming noise)
            def _to_real_tensor(vec):
                if isinstance(vec, torch.Tensor): return vec.to(self.device)
                if hasattr(vec, 'data'): vec = vec.data
                try:
                    rl = [getattr(c, 'real', c) for c in vec]
                    return torch.tensor(rl, device=self.device)
                except:
                    return torch.tensor(vec, device=self.device)
            
            anti_noise = -_to_real_tensor(noise_vector)[:4] # Take physical part
            
            # Dampen momentum
            self.momentum[noisy_nodes, self.PHYSICAL_SLICE] += anti_noise * 0.5
            
            # Cooling effect
            self.q[noisy_nodes, self.CH_ENTROPY] = torch.clamp(self.q[noisy_nodes, self.CH_ENTROPY] - 0.1, 0, 1)

    def read_field_state(self) -> Dict[str, float]:
        """
        [Biological Flow v3.0] Read emergent aggregate states from the active nodes.
        Returns a dict of MEASURED (not stored) properties.
        """
        import torch
        
        if not self.active_nodes_mask.any():
            return {
                "resonance": 0.0,
                "entropy": 0.0,  
                "joy": 0.5,
                "curiosity": 0.5,
                "vitality": 1.0,  
                "coherence": 0.0  
            }
            
        active_idx = self.active_nodes_mask.nonzero(as_tuple=True)[0]
        
        # 1. Total Resonance (Constructive Inner Product against Crystalline base)
        v_phys = self.q[active_idx, self.PHYSICAL_SLICE]
        p_phys = self.permanent_q[active_idx, self.PHYSICAL_SLICE]
        total_resonance = torch.sum(v_phys * p_phys).item() / max(1, len(active_idx))
        
        # 2. Entropy (Decay & Noise)
        entropy = torch.mean(self.q[active_idx, self.CH_ENTROPY]).item()
        
        # 3. Joy (Warmth of Realization)
        joy = torch.mean(self.q[active_idx, self.CH_JOY]).item()
        
        # 4. Curiosity (Drive to align Phase space)
        curiosity = torch.mean(self.q[active_idx, self.CH_CURIOSITY]).item()
        
        # 5. Volumetric Enthalpy (Remaining kinetic energy to change state)
        vitality = torch.mean(self.q[active_idx, self.CH_ENTHALPY]).item()
        
        # 6. Coherence (Standard deviation of Phase across active nodes—lower is more coherent)
        phases = self.q[active_idx, self.CH_Y]
        if len(active_idx) > 1:
            coherence = 1.0 - torch.std(phases).item() # 1.0 = perfect phase lock
        else:
            coherence = 1.0

        return {
            "resonance": total_resonance,
            "entropy": entropy,
            "joy": joy,
            "curiosity": curiosity,
            "vitality": vitality,
            "coherence": coherence
        }

    def get_trinary_projection(self):
        """
        [Compatibility Array] Returns a 1D representation for 21D pooling in legacy systems.
        We return the primary stable structure (CH_W) for all nodes.
        """
        return self.q[:, self.CH_W]

    def hum_resonance(self, torque_intent: Any = None) -> Dict[str, float]:
        """
        [Compatibility] Returns the background resonance ('Hum') of the system.
        Calculates relief (positive energy/structure) and intaglio (negative potential/entropy).
        """
        import torch
        # Mean Joy/Enthalpy represents 'Relief'
        relief = torch.mean(self.q[:, self.CH_JOY] + self.q[:, self.CH_ENTHALPY]).item() / 2.0
        # Mean Entropy represents 'Intaglio'
        intaglio = torch.mean(self.q[:, self.CH_ENTROPY]).item()
        
        return {
            "relief": relief,
            "intaglio": intaglio,
            "resonance": relief - intaglio
        }




class SovereignTensor:
    """
    [PHASE 75] Sovereign Tensor (DNA^N).
    Supports high-order tensor operations for recursive cognitive reflection.
    """
    def __init__(self, shape, data=None):
        import numpy as np
        self.shape = shape
        if data is not None:
            self.data = np.array(data).reshape(shape)
        else:
            self.data = np.zeros(shape)
            
    def mean(self):
        import numpy as np
        return float(np.mean(self.data))
        
    def flatten(self):
        return self.data.flatten().tolist()
        
    @staticmethod
    def _reshape(flat_data, shape):
        import numpy as np
        return np.array(flat_data).reshape(shape)
        
    @staticmethod
    def outer_product(t1, t2):
        import numpy as np
        new_data = np.multiply.outer(t1.data, t2.data)
        return SovereignTensor(new_data.shape, new_data)

    @staticmethod
    def dna3_product(v1, v2, v3):
        """Creates a Rank-3 tensor from three Rank-1 vectors (Axiom ⊗ State ⊗ Observer)."""
        import numpy as np
        # Outer product of v1 and v2 is Rank-2
        r2 = np.multiply.outer(v1.data, v2.data)
        # Outer product of r2 and v3 is Rank-3
        r3 = np.multiply.outer(r2, v3.data)
        return SovereignTensor(r3.shape, r3)

    def recursive_dot(self, vector):
        """
        Reduces the rank of the tensor by performing a dot product 
        along the LAST dimension with the provided vector.
        """
        import numpy as np
        v_data = vector.data if hasattr(vector, 'data') else np.array(vector)
        # np.tensordot or simple dot on the last axis
        new_data = np.dot(self.data, v_data)
        return SovereignTensor(new_data.shape, new_data)


# [PHASE 90] Legacy Alias for Transition
VortexField = FractalWaveEngine
SovereignHyperTensor = FractalWaveEngine


class SovereignMath:
    """
    Functional math operations inspired by JAX.
    """
    @staticmethod
    def where(condition: List[bool], x: 'SovereignVector', y: 'SovereignVector') -> 'SovereignVector':
        return SovereignVector([xv if c else yv for c, xv, yv in zip(condition, x.data, y.data)])

    @staticmethod
    def soft_trinary(vec: 'SovereignVector', intensity: float = 1.0) -> 'SovereignVector':
        """
        [PHASE 73: NATURAL PROVIDENCE] + [DEEP TRINARY LOGIC]
        Replaces hard quantization with a soft potential well.
        The manifold 'flows' toward -1, 0, +1.
        The well around 0 is widened to represent the 'Analog Holding Space' 
        for observation and curiosity formulation.
        """
        result = []
        for x in vec.data:
            x_real = x.real
            # Expand the 0 state so it doesn't immediately slide to ±1
            if abs(x_real) < 0.2:
                # Flat plateau/gentle well at 0: "Letting Be Done"
                well_force = -x_real * 0.05 * intensity 
            else:
                import math
                # Potential function: Pulls toward the nearest integer (-1, 1)
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

# Use new Event-Driven engine by default for HyperTensor references
SovereignHyperTensor = FractalWaveEngine
