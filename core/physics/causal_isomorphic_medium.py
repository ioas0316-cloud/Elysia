"""
Elysia Causal Isomorphic Medium Engine
=====================================
Core module unifying Physical, Fluid, and Logical Constraint Fields into a
single Isomorphic Constraint Tensor Matrix (ConstraintTensorMatrix).

Key Architectural Principles:
1. Relational Medium Emergence (관계적 매질의 자발적 발현):
   - Scalar flow, velocity, potential equilibrium, and current emerge naturally
     from topological field relaxation without hardcoded 1D differential equations.
2. HW-SW Constraint Isomorphism (HW-SW 제약장 동형성):
   - Physical domain (voltage, resistance, clock timing), Fluid domain (pressure, viscosity, density),
     and Logical domain (dependency graph, state barriers, contradiction strain) are mapped
     to the exact same ConstraintTensorMatrix specification.
   - Pressure equilibrium in fluid, voltage-current balance in circuits, and contradiction
     resolution in logic execute through identical tensor relaxation operations.
3. Introspective Causal Awareness (인과적 자각):
   - Records backward trajectories of energy/tension relaxation and invariant topological stems.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class MediumDomain(Enum):
    PHYSICAL = "physical"    # Voltage, Resistance, Clock Timing
    FLUID = "fluid"          # Pressure, Viscosity, Density
    LOGICAL = "logical"      # Dependency, State Barrier, Contradiction Strain


@dataclass
class IsomorphicNode:
    """Represents a node (voxel / circuit node / logical proposition) in the constraint matrix."""
    node_id: str
    domain: MediumDomain
    potential: float  # Voltage V / Pressure P / Logical Truth Potential L
    capacity: float   # Capacitance C / Density rho / Information Capacity K
    strain: float     # Temperature / Viscosity eta / Contradiction Strain S


@dataclass
class IsomorphicEdge:
    """Represents a constraint interaction coupling between two nodes."""
    source_id: str
    target_id: str
    coupling_weight: float  # Conductance / Flow Permeability / Logical Dependency
    impedance: float        # Electrical Resistance R / Fluid Friction eta / Logical Barrier B
    tension: float          # Field strain tension


class ConstraintTensorMatrix:
    """
    [ConstraintTensorMatrix - 단일 위상 제약장 텐서]
    Encapsulates N nodes and their pairwise couplings in an N x N x K tensor representation.
    Dimension 0: Potential field matrix [Phi_i - Phi_j]
    Dimension 1: Impedance matrix Z_ij
    Dimension 2: Coupling weight matrix W_ij
    Dimension 3: Tension / Strain matrix T_ij
    """

    def __init__(self, nodes: List[IsomorphicNode], edges: List[IsomorphicEdge]):
        self.nodes = {n.node_id: n for n in nodes}
        self.node_ids = [n.node_id for n in nodes]
        self.id_to_idx = {n_id: i for i, n_id in enumerate(self.node_ids)}
        self.num_nodes = len(nodes)

        # 4D tensor matrix shape: (N, N, 4)
        self.tensor = np.zeros((self.num_nodes, self.num_nodes, 4), dtype=np.float32)

        # Initialize node potential vector
        self.potential_vector = np.array([n.potential for n in nodes], dtype=np.float32)
        self.capacity_vector = np.array([n.capacity for n in nodes], dtype=np.float32)

        # Populate initial edges
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
        # Outer subtract: Phi_diff[i, j] = Phi[i] - Phi[j]
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

        # Net flow entering node i: sum over column j of J_ji
        net_flow_in = np.sum(flow_matrix, axis=0)  # Incoming flow vector

        # Update node potentials according to local capacity/density
        capacity = np.maximum(1e-3, self.matrix.capacity_vector)
        potential_delta = (net_flow_in / capacity) * dt

        # Apply update
        old_potentials = self.matrix.potential_vector.copy()
        self.matrix.potential_vector += potential_delta
        self.matrix._update_potential_difference_matrix()

        # Update edge tensions (dampened strain accumulation)
        self.matrix.tensor[:, :, 3] = (self.matrix.tensor[:, :, 3] + np.abs(flow_matrix) * 0.1) * damping

        # Calculate metrics
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

        # Record trajectory snapshot for self-awareness back-tracing
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
        """
        Relaxes the constraint field until tension convergence (< tolerance) or max_steps.
        Returns full relaxation step history.
        """
        logs = []
        for s in range(max_steps):
            log = self.relax_step(dt=dt)
            logs.append(log)

            if log.potential_delta < tolerance and log.max_gradient < tolerance:
                break

        return logs

    def get_emergent_scalar_properties(self) -> Dict[str, Any]:
        """
        Extracts emergent scalar properties (e.g. flow magnitude, relaxed potentials,
        impedance profile) resulting from relaxation without using 1D formulas.
        """
        flow = self.matrix.compute_field_flow_matrix()
        return {
            "relaxed_potentials": self.matrix.potential_vector.tolist(),
            "emergent_flux_matrix": flow.tolist(),
            "total_residual_tension": self.matrix.compute_total_field_tension(),
            "max_emergent_flow": float(np.max(np.abs(flow))),
            "relaxation_steps_count": len(self.relaxation_history)
        }


class IsomorphicDomainFactory:
    """
    Factory for constructing equivalent ConstraintTensorMatrix instances for
    Physical (Circuit), Fluid (Water Medium), and Logical (Knowledge Graph) domains.
    """

    @staticmethod
    def create_physical_circuit(num_nodes: int = 5) -> ConstraintTensorMatrix:
        """
        Creates an electrical circuit domain with voltage source, resistor network, and ground.
        """
        nodes = []
        for i in range(num_nodes):
            # Node 0 is Voltage Source (12V), Node N-1 is Ground (0V), middle are floating potentials
            pot = 12.0 if i == 0 else (0.0 if i == num_nodes - 1 else 6.0 + np.random.uniform(-1, 1))
            nodes.append(IsomorphicNode(
                node_id=f"elec_node_{i}",
                domain=MediumDomain.PHYSICAL,
                potential=pot,
                capacity=1.0,  # Electrical capacitance C
                strain=0.1     # Thermal noise / strain
            ))

        edges = []
        for i in range(num_nodes - 1):
            edges.append(IsomorphicEdge(
                source_id=f"elec_node_{i}",
                target_id=f"elec_node_{i+1}",
                coupling_weight=1.0,
                impedance=0.5 * (i + 1),  # Resistor R_i
                tension=0.0
            ))

        return ConstraintTensorMatrix(nodes, edges)

    @staticmethod
    def create_fluid_medium(num_nodes: int = 5) -> ConstraintTensorMatrix:
        """
        Creates a fluid medium domain with pressure head, viscous pipe network, and sink.
        """
        nodes = []
        for i in range(num_nodes):
            # Node 0 is High Pressure Head (12.0 bar), Node N-1 is Sink (0.0 bar)
            pot = 12.0 if i == 0 else (0.0 if i == num_nodes - 1 else 6.0 + np.random.uniform(-1, 1))
            nodes.append(IsomorphicNode(
                node_id=f"fluid_node_{i}",
                domain=MediumDomain.FLUID,
                potential=pot,
                capacity=1.0,  # Fluid density / volume rho
                strain=0.1     # Phase viscosity eta
            ))

        edges = []
        for i in range(num_nodes - 1):
            edges.append(IsomorphicEdge(
                source_id=f"fluid_node_{i}",
                target_id=f"fluid_node_{i+1}",
                coupling_weight=1.0,
                impedance=0.5 * (i + 1),  # Viscous friction / hydraulic resistance
                tension=0.0
            ))

        return ConstraintTensorMatrix(nodes, edges)

    @staticmethod
    def create_logical_structure(num_nodes: int = 5) -> ConstraintTensorMatrix:
        """
        Creates a logical proposition domain with premise truth potential, contradiction strain, and deduction barriers.
        """
        nodes = []
        for i in range(num_nodes):
            # Node 0 is Axiom Premise (12.0 Truth Potential), Node N-1 is Target Conclusion (0.0)
            pot = 12.0 if i == 0 else (0.0 if i == num_nodes - 1 else 6.0 + np.random.uniform(-1, 1))
            nodes.append(IsomorphicNode(
                node_id=f"logic_node_{i}",
                domain=MediumDomain.LOGICAL,
                potential=pot,
                capacity=1.0,  # Logical capacity K
                strain=0.1     # Contradiction strain S
            ))

        edges = []
        for i in range(num_nodes - 1):
            edges.append(IsomorphicEdge(
                source_id=f"logic_node_{i}",
                target_id=f"logic_node_{i+1}",
                coupling_weight=1.0,
                impedance=0.5 * (i + 1),  # Contradiction / Inference barrier
                tension=0.0
            ))

        return ConstraintTensorMatrix(nodes, edges)
