r"""
State-Delta Knowledge Graph, Bit Modulation, and Fluid Phase Transition Engine
================================================================================

Implements:
1. Decoupled Execution & State-Delta Knowledge Graph (\Delta S = S_{t+1} \ominus S_t):
   - Nodes store state data.
   - Edges record What, Where, How, and Why metadata for full inverse causal reasoning.
2. Bit Modulation & Demodulation with Dynamic Resolution:
   - Encodes high-dimensional attributes into 1D primitive byte layouts.
   - Demodulates based on observer position and causal accessibility.
3. Fluid Information Dynamics & 4-Phase Transition Pipeline:
   - Fluid Phase (continuous density field \rho, velocity \vec{v})
   - Crystallization \hat{\mathbf{C}} (phase-lock \theta_{lock} -> node/edge coagulation)
   - Causal Lattice Phase (discrete topological memory bound to 1D addresses)
   - Melting Phase \hat{\mathbf{M}} (thermal activation \Delta E -> continuous fluid field)
"""

import numpy as np
import dataclasses
from typing import Dict, List, Tuple, Optional, Any


@dataclasses.dataclass
class CausalDeltaEdge:
    r"""
    Edge metadata for State-Delta Knowledge Graph (\Delta S = S_{t+1} \ominus S_t).
    Records:
    - What: Tensor delta and affected bytes
    - Where: 1D memory offset and graph topological coordinates
    - How: Applied operator parameters (quaternion rotation, Clifford product, operator ID)
    - Why: Constraint / goal trajectory trajectory ID
    """
    source_node_id: str
    target_node_id: str
    what_delta: np.ndarray
    where_memory_offset: int
    how_operator_id: str
    how_operator_params: Dict[str, Any]
    why_intent_id: str


class StateDeltaKnowledgeGraph:
    """
    Decoupled State-Delta Knowledge Graph for inverse causal tracing.
    """
    def __init__(self):
        self.nodes: Dict[str, np.ndarray] = {}
        self.edges: List[CausalDeltaEdge] = []

    def add_node(self, node_id: str, state_data: np.ndarray):
        self.nodes[node_id] = np.array(state_data, dtype=np.float64)

    def apply_operator_and_record_edge(
        self,
        source_id: str,
        target_id: str,
        operator_fn,
        operator_id: str,
        operator_params: Dict[str, Any],
        memory_offset: int,
        intent_id: str
    ) -> np.ndarray:
        r"""
        Executes operator_fn on source state, stores result in target_id,
        and records State-Delta edge (\Delta S = S_{t+1} \ominus S_t).
        """
        source_state = self.nodes[source_id]
        target_state = operator_fn(source_state, **operator_params)
        self.nodes[target_id] = target_state

        delta = target_state - source_state
        edge = CausalDeltaEdge(
            source_node_id=source_id,
            target_node_id=target_id,
            what_delta=delta,
            where_memory_offset=memory_offset,
            how_operator_id=operator_id,
            how_operator_params=operator_params,
            why_intent_id=intent_id
        )
        self.edges.append(edge)
        return target_state

    def inverse_trace(self, target_id: str) -> List[CausalDeltaEdge]:
        """
        Traces back all causal delta edges leading to target_id.
        """
        path = [e for e in self.edges if e.target_node_id == target_id]
        return path


class BitModulatorDemodulator:
    """
    Encodes high-dimensional attributes into primitive byte arrays (Modulation)
    and procedurally reconstructs/emerges phenomena based on dynamic observer resolution (Demodulation).
    """
    def __init__(self, buffer_size: int = 1024):
        self.buffer = bytearray(buffer_size)

    def modulate(self, tensor_data: np.ndarray, offset: int = 0) -> int:
        """
        Modulates float tensor data into primitive byte array layout.
        Returns bytes written.
        """
        raw_bytes = tensor_data.astype(np.float32).tobytes()
        end_idx = offset + len(raw_bytes)
        self.buffer[offset:end_idx] = raw_bytes
        return len(raw_bytes)

    def demodulate(
        self,
        offset: int,
        shape: Tuple[int, ...],
        observer_position: np.ndarray,
        target_position: np.ndarray,
        max_distance: float = 10.0
    ) -> np.ndarray:
        """
        Dynamic Resolution Demodulation:
        Evaluates distance to observer. If within accessible reach, demodulates full precision.
        If distant, demodulates low-resolution compressed potential.
        """
        num_floats = int(np.prod(shape))
        raw_bytes = bytes(self.buffer[offset : offset + num_floats * 4])
        full_tensor = np.frombuffer(raw_bytes, dtype=np.float32).reshape(shape)

        distance = float(np.linalg.norm(observer_position - target_position))
        if distance > max_distance:
            # Low resolution compressed potential (scaled/smoothed)
            return full_tensor * 0.1
        return full_tensor


class FluidPhaseTransitionEngine:
    """
    4-Phase Transition Engine:
    Continuous Fluid Phase <-> Crystallization (hat{C}) <-> Causal Lattice Phase <-> Melting Phase (hat{M})
    """
    def __init__(self, grid_size: int = 16):
        self.grid_size = grid_size
        self.phase = "FLUID"
        # Continuous density field rho and velocity field v (grid_size x grid_size)
        self.rho = np.random.uniform(0.1, 1.0, (grid_size, grid_size))
        self.v = np.random.normal(0, 0.5, (grid_size, grid_size, 2))

        # Lattice storage
        self.lattice_nodes: List[Tuple[int, int]] = []
        self.lattice_edges: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []

    def step_fluid_dynamics(self, source_s: float = 0.1):
        """
        Advection-Diffusion step in Fluid Phase:
        d(rho)/dt + div(rho * v) = S
        """
        if self.phase != "FLUID":
            return

        # Simple finite difference advection step
        grad_rho_x, grad_rho_y = np.gradient(self.rho)
        advection = self.v[:, :, 0] * grad_rho_x + self.v[:, :, 1] * grad_rho_y
        self.rho = np.clip(self.rho - 0.1 * advection + source_s, 0.0, 10.0)

    def crystallize(self, entropy_threshold: float = 0.5) -> bool:
        """
        Crystallization Operator hat{C}:
        When entropy drops or vorticity reaches local maxima, coagulates fluid points into discrete lattice nodes and edges.
        """
        # Calculate vorticity curl(v) = dvy/dx - dvx/dy
        # In 2D grid: axis=0 corresponds to y (rows), axis=1 corresponds to x (cols)
        dvx_dy = np.gradient(self.v[:, :, 0], axis=0)
        dvy_dx = np.gradient(self.v[:, :, 1], axis=1)
        vorticity = np.abs(dvy_dx - dvx_dy)

        # Local maxima become lattice nodes
        nodes = np.argwhere(vorticity > np.percentile(vorticity, 90))
        self.lattice_nodes = [tuple(n) for n in nodes]

        # Connect adjacent nodes as lattice edges
        edges = []
        for i, n1 in enumerate(self.lattice_nodes):
            for j, n2 in enumerate(self.lattice_nodes[i + 1 :]):
                dist = np.linalg.norm(np.array(n1) - np.array(n2))
                if dist <= 2.5:
                    edges.append((n1, n2))
        self.lattice_edges = edges

        self.phase = "LATTICE"
        return True

    def melt(self, thermal_energy_delta_e: float) -> bool:
        """
        Melting Operator hat{M}:
        Applies energy delta_e to dissolve discrete lattice back into continuous fluid velocity/density fields.
        """
        if self.phase != "LATTICE":
            return False

        # Convert lattice nodes back to fluid vorticity & density diffusion
        for node in self.lattice_nodes:
            r, c = node
            self.rho[r, c] += thermal_energy_delta_e
            self.v[r, c] += np.random.normal(0, thermal_energy_delta_e, 2)

        self.lattice_nodes = []
        self.lattice_edges = []
        self.phase = "FLUID"
        return True
