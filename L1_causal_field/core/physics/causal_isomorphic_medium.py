"""
Causal Isomorphic Medium & Cognitive Isomorphism Engine
======================================================
This module provides:
1. Matrix-based Isomorphic Field Relaxation (`ConstraintTensorMatrix`, `CausalIsomorphicMedium`, `IsomorphicDomainFactory`)
   Demonstrating that electrical current, fluid flow, and logical inference follow the exact same field relaxation dynamics.
2. Continuous-to-Discrete Cognitive Isomorphism Engine (`CausalIsomorphicMediumEngine`, `QuaternionUtil`, `RotorTile`)
   Grounded in continuous potential tensor fields, quaternion/rotor phase dynamics, Phase-Lock Loop (PLL) order parameters,
   Gibbs-Boltzmann entropy-guided Wave Function Collapse (WFC), AC-3 constraint wave propagation, and thermal relaxation reflection loops.

Core Philosophy: "Do not calculate, let it flow."
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


# ============================================================================
# PART 1: MATRIX-BASED ISOMORPHIC FIELD RELAXATION (Physical, Fluid, Logical)
# ============================================================================

class MediumDomain(Enum):
    PHYSICAL = "physical_circuit"
    FLUID = "fluid_medium"
    LOGICAL = "logical_structure"


@dataclass
class IsomorphicNode:
    """Represents a discrete or spatial node in a constraint field."""
    node_id: str
    domain: MediumDomain
    potential: float         # Voltage V, Pressure P, or Truth Potential Phi
    capacity: float = 1.0    # Capacitance C, Volume Rho, or Logical Capacity K
    strain: float = 0.0      # Mechanical strain, thermal noise, or contradiction score


@dataclass
class IsomorphicEdge:
    """Represents a coupling edge between two constraint field nodes."""
    source_id: str
    target_id: str
    coupling_weight: float = 1.0  # Conductance G, Pipe cross-section A, or Rule Strength W
    impedance: float = 1.0        # Resistance R, Viscosity Eta, or Inference Barrier Z
    tension: float = 0.0          # Field strain / potential gradient magnitude


class ConstraintTensorMatrix:
    """
    [ConstraintTensorMatrix - 통합 제약 텐서 행렬]
    Encapsulates N-node field coupling into a 3D Tensor:
    Tensor Shape: (N, N, 4)
      - Layer 0: Potential difference matrix Phi_diff[i, j] = Phi_i - Phi_j
      - Layer 1: Impedance matrix Z[i, j]
      - Layer 2: Coupling weight matrix W[i, j]
      - Layer 3: Field tension / strain energy T[i, j]
    """

    def __init__(self, nodes: List[IsomorphicNode], edges: List[IsomorphicEdge]):
        self.nodes = nodes
        self.edges = edges
        self.num_nodes = len(nodes)
        self.id_to_idx = {node.node_id: i for i, node in enumerate(nodes)}

        # State vectors
        self.potential_vector = np.array([node.potential for node in nodes], dtype=np.float64)
        self.capacity_vector = np.array([node.capacity for node in nodes], dtype=np.float64)
        self.strain_vector = np.array([node.strain for node in nodes], dtype=np.float64)

        # 3D Tensor Initialization: (N, N, 4)
        self.tensor = np.zeros((self.num_nodes, self.num_nodes, 4), dtype=np.float64)

        # Populate tensor layers
        for edge in edges:
            if edge.source_id in self.id_to_idx and edge.target_id in self.id_to_idx:
                u = self.id_to_idx[edge.source_id]
                v = self.id_to_idx[edge.target_id]

                # Symmetrical coupling
                self.tensor[u, v, 1] = max(1e-6, edge.impedance)        # Z_uv
                self.tensor[v, u, 1] = max(1e-6, edge.impedance)        # Z_vu
                self.tensor[u, v, 2] = edge.coupling_weight             # W_uv
                self.tensor[v, u, 2] = edge.coupling_weight             # W_vu
                self.tensor[u, v, 3] = edge.tension                     # T_uv
                self.tensor[v, u, 3] = edge.tension                     # T_vu

        self._update_potential_difference_matrix()

    def _update_potential_difference_matrix(self):
        """Updates potential difference layer in tensor: Phi_diff[i, j] = Phi_i - Phi_j."""
        phi = self.potential_vector
        self.tensor[:, :, 0] = np.subtract.outer(phi, phi)

    def compute_field_flow_matrix(self) -> np.ndarray:
        """
        Calculates field flux/flow matrix J_ij = (Phi_i - Phi_j) * W_ij / (1 + Z_ij).
        Works identically for electric current I, fluid flux Q, or logical inference propagation.
        """
        phi_diff = self.tensor[:, :, 0]
        impedance = self.tensor[:, :, 1]
        weight = self.tensor[:, :, 2]

        flow_matrix = (phi_diff * weight) / (1.0 + impedance)
        return flow_matrix

    def compute_total_field_tension(self) -> float:
        """
        Computes total system tension / strain energy:
        E_tension = sum_{i,j} (Phi_i - Phi_j)^2 * W_ij / (1 + Z_ij).
        """
        phi_diff = self.tensor[:, :, 0]
        impedance = self.tensor[:, :, 1]
        weight = self.tensor[:, :, 2]

        strain_energy = np.sum((phi_diff ** 2) * weight / (1.0 + impedance))
        return float(strain_energy)


@dataclass
class RelaxationStepLog:
    """Log entry for an individual relaxation step during field convergence."""
    step: int
    total_tension: float
    max_gradient: float
    potential_delta: float
    emergent_flow_magnitude: float


class CausalIsomorphicMedium:
    """
    [CausalIsomorphicMedium - 동형 제약장 이완 및 창발 엔진]
    Performs field relaxation across Physical, Fluid, or Logical constraint matrices.
    Demonstrates that pressure equilibrium, electrical potential balance, and logical
    contradiction resolution follow the exact same isomorphic tensor operation.
    """

    def __init__(self, matrix: ConstraintTensorMatrix):
        self.matrix = matrix
        self.relaxation_history: List[RelaxationStepLog] = []
        self.causal_trajectory_graph: List[Dict[str, Any]] = []

    def relax_step(self, dt: float = 0.05, damping: float = 0.95) -> RelaxationStepLog:
        """
        Performs a single step of topological field relaxation.
        1. Calculates net field flow J into each node: Net_Flow_i = sum_j J_ji
        2. Adjusts potentials: Phi_i <- Phi_i + dt * Net_Flow_i / capacity_i
        3. Damps residual tension to simulate structural dissipation.
        """
        flow_matrix = self.matrix.compute_field_flow_matrix()
        net_flow_in = np.sum(flow_matrix, axis=0)  # Incoming flow vector

        capacity = np.maximum(1e-3, self.matrix.capacity_vector)
        potential_delta = (net_flow_in / capacity) * dt

        self.matrix.potential_vector += potential_delta
        self.matrix._update_potential_difference_matrix()

        # Update edge tensions
        self.matrix.tensor[:, :, 3] = (self.matrix.tensor[:, :, 3] + np.abs(flow_matrix) * 0.1) * damping

        total_tension = self.matrix.compute_total_field_tension()
        max_grad = float(np.max(np.abs(flow_matrix)))
        max_p_delta = float(np.max(np.abs(potential_delta)))
        flow_mag = float(np.sum(np.abs(flow_matrix)))

        log = RelaxationStepLog(
            step=len(self.relaxation_history) + 1,
            total_tension=total_tension,
            max_gradient=max_grad,
            potential_delta=max_p_delta,
            emergent_flow_magnitude=flow_mag
        )
        self.relaxation_history.append(log)

        self.causal_trajectory_graph.append({
            "step": log.step,
            "potentials": self.matrix.potential_vector.copy().tolist(),
            "tension": total_tension,
            "max_gradient": max_grad
        })

        return log

    def relax_to_equilibrium(
        self,
        max_steps: int = 100,
        dt: float = 0.05,
        tolerance: float = 1e-4
    ) -> List[RelaxationStepLog]:
        """Relaxes the constraint field until tension convergence (< tolerance) or max_steps."""
        logs = []
        for s in range(max_steps):
            log = self.relax_step(dt=dt)
            logs.append(log)

            if log.potential_delta < tolerance and log.max_gradient < tolerance:
                break

        return logs

    def get_emergent_scalar_properties(self) -> Dict[str, Any]:
        """Extracts emergent scalar properties resulting from relaxation."""
        flow = self.matrix.compute_field_flow_matrix()
        return {
            "relaxed_potentials": self.matrix.potential_vector.tolist(),
            "emergent_flux_matrix": flow.tolist(),
            "total_residual_tension": self.matrix.compute_total_field_tension(),
            "max_emergent_flow": float(np.max(np.abs(flow))),
            "relaxation_steps_count": len(self.relaxation_history)
        }


class IsomorphicDomainFactory:
    """Factory for constructing equivalent ConstraintTensorMatrix instances."""

    @staticmethod
    def create_physical_circuit(num_nodes: int = 5) -> ConstraintTensorMatrix:
        nodes = []
        for i in range(num_nodes):
            pot = 12.0 if i == 0 else (0.0 if i == num_nodes - 1 else 6.0 + np.random.uniform(-1, 1))
            nodes.append(IsomorphicNode(
                node_id=f"elec_node_{i}",
                domain=MediumDomain.PHYSICAL,
                potential=pot,
                capacity=1.0,
                strain=0.1
            ))

        edges = []
        for i in range(num_nodes - 1):
            edges.append(IsomorphicEdge(
                source_id=f"elec_node_{i}",
                target_id=f"elec_node_{i+1}",
                coupling_weight=1.0,
                impedance=0.5 * (i + 1),
                tension=0.0
            ))

        return ConstraintTensorMatrix(nodes, edges)

    @staticmethod
    def create_fluid_medium(num_nodes: int = 5) -> ConstraintTensorMatrix:
        nodes = []
        for i in range(num_nodes):
            pot = 12.0 if i == 0 else (0.0 if i == num_nodes - 1 else 6.0 + np.random.uniform(-1, 1))
            nodes.append(IsomorphicNode(
                node_id=f"fluid_node_{i}",
                domain=MediumDomain.FLUID,
                potential=pot,
                capacity=1.0,
                strain=0.1
            ))

        edges = []
        for i in range(num_nodes - 1):
            edges.append(IsomorphicEdge(
                source_id=f"fluid_node_{i}",
                target_id=f"fluid_node_{i+1}",
                coupling_weight=1.0,
                impedance=0.5 * (i + 1),
                tension=0.0
            ))

        return ConstraintTensorMatrix(nodes, edges)

    @staticmethod
    def create_logical_structure(num_nodes: int = 5) -> ConstraintTensorMatrix:
        nodes = []
        for i in range(num_nodes):
            pot = 12.0 if i == 0 else (0.0 if i == num_nodes - 1 else 6.0 + np.random.uniform(-1, 1))
            nodes.append(IsomorphicNode(
                node_id=f"logic_node_{i}",
                domain=MediumDomain.LOGICAL,
                potential=pot,
                capacity=1.0,
                strain=0.1
            ))

        edges = []
        for i in range(num_nodes - 1):
            edges.append(IsomorphicEdge(
                source_id=f"logic_node_{i}",
                target_id=f"logic_node_{i+1}",
                coupling_weight=1.0,
                impedance=0.5 * (i + 1),
                tension=0.0
            ))

        return ConstraintTensorMatrix(nodes, edges)


# ============================================================================
# PART 2: ROTOR-PHASE WFC COGNITIVE ISOMORPHISM ENGINE
# ============================================================================

class QuaternionUtil:
    """Utility class for 2D/3D Rotors and 4D Quaternions math in Causal Fields."""

    @staticmethod
    def normalize(q: np.ndarray) -> np.ndarray:
        """Normalizes a quaternion or 2D rotor vector."""
        norm = np.linalg.norm(q)
        if norm < 1e-12:
            q_out = np.zeros_like(q)
            q_out[0] = 1.0
            return q_out
        return q / norm

    @staticmethod
    def inner_product_sq(q1: np.ndarray, q2: np.ndarray) -> float:
        """Calculates squared inner product, respecting double-cover symmetry (q == -q)."""
        dot = float(np.dot(q1, q2))
        return dot * dot

    @staticmethod
    def compute_rotor_alignment_energy(field_q: np.ndarray, tile_q: np.ndarray) -> float:
        """Rotor alignment energy: E_rotor = 1 - (q_field . q_tile)^2"""
        q_f = QuaternionUtil.normalize(field_q)
        q_t = QuaternionUtil.normalize(tile_q)
        dot_sq = QuaternionUtil.inner_product_sq(q_f, q_t)
        return float(1.0 - dot_sq)

    @staticmethod
    def generate_random_rotor_noise(ndim: int = 4, scale: float = 0.2) -> np.ndarray:
        """Generates random rotor/quaternion thermal fluctuation noise."""
        noise = np.random.normal(0, scale, size=ndim)
        noise[0] += 1.0
        return QuaternionUtil.normalize(noise)

    @staticmethod
    def multiply_quaternions(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiplies two 4D quaternions q1 * q2."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ])


class RotorTile:
    """Discrete symbol/tile state with a geometric phase rotor."""

    def __init__(
        self,
        tile_id: int,
        name: str,
        rotor_q: np.ndarray,
        direction_vector: Optional[np.ndarray] = None
    ):
        self.tile_id = tile_id
        self.name = name
        self.rotor = QuaternionUtil.normalize(rotor_q)

        if direction_vector is not None:
            norm_d = np.linalg.norm(direction_vector)
            self.dir_vec = direction_vector / norm_d if norm_d > 1e-12 else direction_vector
        else:
            if len(self.rotor) == 2:
                angle = 2.0 * np.arctan2(self.rotor[1], self.rotor[0])
                self.dir_vec = np.array([np.cos(angle), np.sin(angle)])
            elif len(self.rotor) == 4:
                w, x, y, z = self.rotor
                self.dir_vec = np.array([
                    1 - 2 * (y**2 + z**2),
                    2 * (x * y + w * z),
                    2 * (x * z - w * y)
                ])
            else:
                self.dir_vec = np.array([1.0, 0.0])


class CausalIsomorphicMediumEngine:
    """
    Unified Causal Isomorphic Medium Engine.
    Combines Continuous Tensor/Rotor Potential Fields, Phase-Locking (PLL),
    Gibbs-Boltzmann Entropy Reduction, WFC Collapse, AC-3 Wave Propagation,
    and Reflection/Thermal Relaxation Loops.
    """

    def __init__(
        self,
        width: int,
        height: int,
        tiles: List[RotorTile],
        compatibility_matrix: np.ndarray,
        beta_0: float = 2.0,
        gamma_lock: float = 5.0,
        lock_threshold: float = 0.80,
        w_field: float = 1.0,
        w_causal: float = 2.0
    ):
        self.W = width
        self.H = height
        self.tiles = tiles
        self.num_tiles = len(tiles)

        self.M = compatibility_matrix.copy()

        self.beta_0 = beta_0
        self.beta = beta_0
        self.gamma_lock = gamma_lock
        self.lock_threshold = lock_threshold
        self.w_field = w_field
        self.w_causal = w_causal

        self.P = np.full((width, height, self.num_tiles), 1.0 / self.num_tiles)
        self.collapsed = np.zeros((width, height), dtype=bool)
        self.phase_locked = np.zeros((width, height), dtype=bool)

        self.rotor_dim = len(tiles[0].rotor)
        self.tile_rotors = np.array([t.rotor for t in tiles])

        self.tensor_field = np.zeros((width, height, self.rotor_dim))
        identity_rotor = np.zeros(self.rotor_dim)
        identity_rotor[0] = 1.0
        for x in range(width):
            for y in range(height):
                self.tensor_field[x, y] = identity_rotor.copy()

        self.reflection_count = 0
        self.decision_steps = 0

    def inject_sensory_stimulus(
        self,
        x: int,
        y: int,
        target_tile_id: int,
        intensity: float = 1.0
    ):
        """Step 1: Sensory Transformation into tensor field potential hill."""
        if 0 <= x < self.W and 0 <= y < self.H and 0 <= target_tile_id < self.num_tiles:
            target_rotor = self.tiles[target_tile_id].rotor
            intensity = np.clip(intensity, 0.0, 1.0)
            self.tensor_field[x, y] = (1.0 - intensity) * self.tensor_field[x, y] + intensity * target_rotor
            self.tensor_field[x, y] = QuaternionUtil.normalize(self.tensor_field[x, y])

    def compute_field_energy(self, field_q: np.ndarray) -> np.ndarray:
        """Calculates rotor alignment energy for all tiles at a cell."""
        q_f = QuaternionUtil.normalize(field_q)
        dots = np.dot(self.tile_rotors, q_f)
        return 1.0 - (dots ** 2)

    def compute_neighbor_energy(self, x: int, y: int) -> np.ndarray:
        """Calculates causal neighbor compatibility violation energy for all tiles at (x, y)."""
        e_causal = np.zeros(self.num_tiles)
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.W and 0 <= ny < self.H:
                neighbors.append((nx, ny))

        if not neighbors:
            return e_causal

        for nx, ny in neighbors:
            p_j = self.P[nx, ny]
            compat_sum = np.dot(self.M, p_j)
            e_causal += (1.0 - compat_sum)

        return e_causal

    def compute_phase_lock_index(self, x: int, y: int) -> float:
        """Phase Lock Index (PLI) measuring local field coherence order parameter Psi_i."""
        neighbors_rotors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.W and 0 <= ny < self.H:
                neighbors_rotors.append(self.tensor_field[nx, ny])

        if not neighbors_rotors:
            return 0.0

        mean_rotor = np.mean(neighbors_rotors, axis=0)
        pli = float(np.linalg.norm(mean_rotor))
        return float(np.clip(pli, 0.0, 1.0))

    def update_phase_and_probabilities(self) -> np.ndarray:
        """Step 2: Causal Field Alignment & Entropy Calculation."""
        entropy = np.full((self.W, self.H), np.inf)

        for x in range(self.W):
            for y in range(self.H):
                if self.collapsed[x, y]:
                    continue

                e_r = self.compute_field_energy(self.tensor_field[x, y])
                e_c = self.compute_neighbor_energy(x, y)
                e_total = self.w_field * e_r + self.w_causal * e_c

                pli = self.compute_phase_lock_index(x, y)
                if pli >= self.lock_threshold:
                    self.phase_locked[x, y] = True

                beta_eff = self.beta * (1.0 + self.gamma_lock * pli)

                unnorm_p = np.exp(-beta_eff * e_total)
                z = np.sum(unnorm_p)

                if z < 1e-12 or np.isnan(z):
                    self.P[x, y] = np.zeros(self.num_tiles)
                else:
                    self.P[x, y] = unnorm_p / z

                p_curr = self.P[x, y]
                valid_p = p_curr[p_curr > 1e-12]
                if len(valid_p) > 0:
                    entropy[x, y] = -np.sum(valid_p * np.log(valid_p))
                else:
                    entropy[x, y] = 0.0

        return entropy

    def make_decision_step(self) -> Tuple[bool, bool]:
        """Step 3: Decision & Observation with WFC collapse and AC-3 propagation."""
        self.decision_steps += 1
        entropy = self.update_phase_and_probabilities()

        for x in range(self.W):
            for y in range(self.H):
                if not self.collapsed[x, y] and np.sum(self.P[x, y]) < 1e-6:
                    return False, True

        if np.all(self.collapsed):
            return True, False

        selection_metric = np.where(self.collapsed, np.inf, entropy)
        selection_metric = np.where(self.phase_locked, selection_metric * 0.1, selection_metric)

        min_idx = np.argmin(selection_metric)
        cx, cy = np.unravel_index(min_idx, (self.W, self.H))

        if np.isinf(selection_metric[cx, cy]):
            return True, False

        probabilities = self.P[cx, cy]
        sum_p = np.sum(probabilities)
        if sum_p < 1e-12:
            return False, True

        probabilities = probabilities / sum_p
        chosen_k = np.random.choice(self.num_tiles, p=probabilities)

        self.P[cx, cy] = 0.0
        self.P[cx, cy, chosen_k] = 1.0
        self.collapsed[cx, cy] = True

        has_conflict = not self._propagate_constraints(cx, cy)
        if has_conflict:
            return False, True

        return np.all(self.collapsed), False

    def _propagate_constraints(self, start_x: int, start_y: int) -> bool:
        """AC-3 constraint propagation."""
        queue = [(start_x, start_y)]

        while queue:
            cx, cy = queue.pop(0)
            p_c = self.P[cx, cy]

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.W and 0 <= ny < self.H and not self.collapsed[nx, ny]:
                    changed = False

                    for m in range(self.num_tiles):
                        if self.P[nx, ny, m] > 0:
                            has_compatible = np.any((self.M[:, m] > 0) & (p_c > 0))
                            if not has_compatible:
                                self.P[nx, ny, m] = 0.0
                                changed = True

                    sum_p = np.sum(self.P[nx, ny])
                    if sum_p < 1e-12:
                        return False

                    if changed:
                        self.P[nx, ny] /= sum_p
                        queue.append((nx, ny))

        return True

    def reflect_and_relax(self, radius: int = 1, thermal_factor: float = 0.3):
        """Step 4: Reflection & Thermal Relaxation."""
        self.reflection_count += 1
        self.beta *= thermal_factor

        for x in range(self.W):
            for y in range(self.H):
                self.collapsed[x, y] = False
                self.phase_locked[x, y] = False
                self.P[x, y] = np.full(self.num_tiles, 1.0 / self.num_tiles)

                noise_rotor = QuaternionUtil.generate_random_rotor_noise(
                    ndim=self.rotor_dim, scale=0.3
                )
                if self.rotor_dim == 4:
                    self.tensor_field[x, y] = QuaternionUtil.multiply_quaternions(
                        self.tensor_field[x, y], noise_rotor
                    )
                else:
                    self.tensor_field[x, y] = self.tensor_field[x, y] + noise_rotor[:self.rotor_dim]

                self.tensor_field[x, y] = QuaternionUtil.normalize(self.tensor_field[x, y])

        self.w_field *= 0.9
        self.update_phase_and_probabilities()

    def get_grid_state_summary(self) -> List[List[str]]:
        """Returns string grid representation of current collapsed or superposition states."""
        grid = []
        for y in range(self.H):
            row = []
            for x in range(self.W):
                if self.collapsed[x, y]:
                    k = int(np.argmax(self.P[x, y]))
                    row.append(self.tiles[k].name)
                else:
                    row.append("?")
            grid.append(row)
        return grid
