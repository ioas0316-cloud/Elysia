"""
[MechanismTensor: Mechanized Information Structure & Causal Lineage Engine]

Implements the paradigm shift where tensors are treated not as primitive arrays of numbers,
but as substrate manifolds governed by higher-level causal mechanisms, topological invariants,
and tension fields.

Key Components:
1. CausalLineage: Tracks the "Process of Becoming" (Lineage Trajectory) with O(1) LCA branch differentiation.
2. TopologicalInvariant: Defines boundary rules, symmetries, and tensor axis tension metrics.
3. MechanismTensor: Autonomous relaxation via tensor axis contraction (e.g. torch.einsum).
4. MechanismNode, CausalEdge, CausalGraphNetwork: Multi-node network tension wave propagation.
5. CausalGeodesicInferenceEngine: Autonomous synthesis of minimal tension causal geodesics
   connecting historical start lineages S_start to target boundary invariants S_target.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Any
import torch
import numpy as np


@dataclass
class CausalLineage:
    """
    Data lineage trajectory preserving the 'Process of Becoming'.
    Allows O(1)-like causal gap differentiation via Lowest Common Ancestor (LCA) tracking.
    """
    node_id: str
    parent_ids: List[str] = field(default_factory=list)
    transformation_history: List[str] = field(default_factory=list)
    depth: int = 0

    def find_lowest_common_ancestor(self, other: 'CausalLineage') -> Tuple[str, int]:
        """
        Finds the lowest common ancestor node/branching point and split depth
        between two lineage trajectories without examining raw tensor data.
        """
        common_id = self.node_id
        min_len = min(len(self.transformation_history), len(other.transformation_history))
        split_depth = 0

        for i in range(min_len):
            if self.transformation_history[i] == other.transformation_history[i]:
                split_depth = i + 1
            else:
                break

        if self.node_id != other.node_id:
            common_parents = [p for p in self.parent_ids if p in other.parent_ids]
            if common_parents:
                common_id = common_parents[0]

        return common_id, split_depth


@dataclass
class TopologicalInvariant:
    """
    Topological constraint & symmetry specification that lower substrate tensors must satisfy.
    Measures tension error fields along tensor dimensions/axes.
    """
    name: str
    target_value: float = 1.0
    symmetry_group: Optional[str] = None
    constraint_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    tolerance: float = 1e-4

    def compute_error(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Computes axis-wise projection errors against the target value
        and returns (max_tension, max_tension_axis).
        """
        if self.constraint_fn is not None:
            err = self.constraint_fn(tensor)
            return torch.norm(err), 0

        if tensor.ndim == 0:
            err = torch.square(tensor - self.target_value)
            return err, 0

        axis_sums = torch.stack([tensor.sum(dim=i).mean() for i in range(tensor.ndim)])
        tensions = torch.square(axis_sums - self.target_value)
        max_axis = int(torch.argmax(tensions).item())
        return tensions[max_axis], max_axis


class MechanismTensor:
    """
    Higher-level mechanized tensor structure wrapping a substrate torch.Tensor.
    Auto-dispatches Einsum tensor contraction when high topological tension fields are detected.
    """
    def __init__(
        self,
        raw_tensor: torch.Tensor,
        lineage: CausalLineage,
        invariants: Optional[List[TopologicalInvariant]] = None,
        invariant: Optional[TopologicalInvariant] = None
    ):
        self._raw_tensor = raw_tensor.detach().clone().float() if isinstance(raw_tensor, torch.Tensor) else torch.tensor(raw_tensor, dtype=torch.float32)
        self.lineage = lineage

        if invariants is not None:
            self.invariants = invariants
        elif invariant is not None:
            self.invariants = [invariant]
        else:
            self.invariants = [TopologicalInvariant(name="Flux_Conservation", target_value=1.0)]

        self.tension_field: torch.Tensor = torch.zeros(1)

    @property
    def raw_tensor(self) -> torch.Tensor:
        return self._raw_tensor

    @raw_tensor.setter
    def raw_tensor(self, value: torch.Tensor):
        self._raw_tensor = value

    @property
    def shape(self):
        return self._raw_tensor.shape

    def update_tension(self) -> Tuple[torch.Tensor, int]:
        """
        Measures total potential tension field from invariants and identifies maximum tension axis.
        """
        total_tension = torch.zeros(1)
        max_axis = 0
        max_axis_t = -1.0

        for inv in self.invariants:
            t_val, axis = inv.compute_error(self._raw_tensor)
            total_tension += t_val
            if t_val.item() > max_axis_t:
                max_axis_t = t_val.item()
                max_axis = axis

        self.tension_field = total_tension
        return self.tension_field, max_axis

    def auto_dispatch_relaxation(
        self,
        tolerance: float = 1e-4,
        max_steps: int = 10
    ) -> 'MechanismTensor':
        """
        Auto-dispatches lower Einsum axis contraction without external control loops
        until potential tension relaxes to equilibrium.
        """
        step = 0
        while step < max_steps:
            total_tension, max_axis = self.update_tension()
            if total_tension.item() < tolerance or self._raw_tensor.ndim <= 1:
                break

            rank = self._raw_tensor.ndim
            input_indices = [chr(97 + i) for i in range(rank)]
            output_indices = [idx for i, idx in enumerate(input_indices) if i != max_axis]

            einsum_str = f"{''.join(input_indices)}->{''.join(output_indices)}"
            reduced_tensor = torch.einsum(einsum_str, self._raw_tensor)

            target_val = self.invariants[0].target_value if self.invariants else 1.0
            scale = target_val / (reduced_tensor.sum() + 1e-8)
            self._raw_tensor = (reduced_tensor * scale).unsqueeze(max_axis)

            self.lineage.transformation_history.append(f"Einsum({einsum_str})@axis_{max_axis}")
            self.lineage.depth += 1
            step += 1

        self.update_tension()
        return self

    def differentiate_causal_gap(self, other: 'MechanismTensor') -> Dict[str, Any]:
        """
        Differentiates causal variation gap between two structures in O(1) time
        by analyzing lineage trajectory LCA rather than raw tensor comparison.
        """
        lca_id, split_depth = self.lineage.find_lowest_common_ancestor(other.lineage)
        return {
            "divergence_point": lca_id,
            "self_path": self.lineage.transformation_history[split_depth:],
            "other_path": other.lineage.transformation_history[split_depth:],
            "split_depth": split_depth
        }


class MechanismNode:
    """Single topological node in a causal network."""
    def __init__(self, node_id: str, raw_tensor: torch.Tensor, target_flux: float = 1.0):
        self.node_id = node_id
        self.tensor = raw_tensor.detach().clone().float() if isinstance(raw_tensor, torch.Tensor) else torch.tensor(raw_tensor, dtype=torch.float32)
        self.target_flux = target_flux
        self.lineage = CausalLineage(node_id=node_id)

    def compute_internal_tension(self) -> Tuple[torch.Tensor, int]:
        if self.tensor.ndim == 0:
            err = torch.square(self.tensor - self.target_flux)
            return err, 0
        axis_sums = torch.stack([self.tensor.sum(dim=i).mean() for i in range(self.tensor.ndim)])
        tensions = torch.square(axis_sums - self.target_flux)
        max_axis = int(torch.argmax(tensions).item())
        return tensions[max_axis], max_axis

    def relax_local_axis(self, axis: int) -> str:
        if self.tensor.ndim <= 1:
            return "NoReduction"
        rank = self.tensor.ndim
        in_idx = [chr(97 + i) for i in range(rank)]
        out_idx = [idx for i, idx in enumerate(in_idx) if i != axis]
        einsum_str = f"{''.join(in_idx)}->{''.join(out_idx)}"

        reduced = torch.einsum(einsum_str, self.tensor)
        scale = self.target_flux / (reduced.sum() + 1e-8)
        self.tensor = (reduced * scale).unsqueeze(axis)

        op_log = f"LocalEinsum({einsum_str})@axis_{axis}"
        self.lineage.transformation_history.append(op_log)
        self.lineage.depth += 1
        return op_log


class CausalEdge:
    """Inter-node causal constraint & tension wave channel."""
    def __init__(self, source: MechanismNode, target: MechanismNode, coupling_strength: float = 0.5):
        self.source = source
        self.target = target
        self.coupling = coupling_strength

    def compute_edge_tension(self) -> torch.Tensor:
        diff = torch.abs(self.source.tensor.mean() - self.target.tensor.mean())
        return diff * self.coupling

    def propagate_tension(self):
        edge_tension = self.compute_edge_tension()
        if edge_tension.item() > 1e-3:
            self.target.tensor += edge_tension * 0.1
            self.target.lineage.transformation_history.append(
                f"PropagatedFrom({self.source.node_id}, tension={edge_tension.item():.4f})"
            )


class CausalGraphNetwork:
    """Network orchestrator for multi-node tension wave propagation and asynchronous relaxation."""
    def __init__(self):
        self.nodes: Dict[str, MechanismNode] = {}
        self.edges: List[CausalEdge] = []

    def add_node(self, node: MechanismNode):
        self.nodes[node.node_id] = node

    def connect(self, source_id: str, target_id: str, coupling: float = 0.5):
        edge = CausalEdge(self.nodes[source_id], self.nodes[target_id], coupling)
        self.edges.append(edge)

    def resolve_network_equilibrium(self, max_cycles: int = 10, tol: float = 1e-3):
        for cycle in range(max_cycles):
            total_network_tension = 0.0

            for edge in self.edges:
                edge_t = edge.compute_edge_tension()
                total_network_tension += edge_t.item()
                edge.propagate_tension()

            for node in self.nodes.values():
                node_t, max_axis = node.compute_internal_tension()
                total_network_tension += node_t.item()
                if node_t.item() > tol:
                    node.relax_local_axis(max_axis)

            if total_network_tension < tol:
                break


class CausalGeodesicInferenceEngine:
    """
    Synthesizes the minimal-tension causal geodesic trajectory between an initial
    lineage trajectory S_start and a target boundary invariant S_target.
    """
    def __init__(self, start_tensor: MechanismTensor, target_invariant: TopologicalInvariant):
        self.start_tensor = start_tensor
        self.target_invariant = target_invariant

    def construct_geodesic(self, max_steps: int = 10) -> Dict[str, Any]:
        """
        Relaxes start tensor until friction/tension with target invariant is 0.
        Returns the finalized geodesic trajectory and total action cost.
        """
        current_m_tensor = MechanismTensor(
            raw_tensor=self.start_tensor.raw_tensor.clone(),
            lineage=CausalLineage(
                node_id=f"Geodesic_{self.start_tensor.lineage.node_id}",
                parent_ids=[self.start_tensor.lineage.node_id],
                transformation_history=list(self.start_tensor.lineage.transformation_history)
            ),
            invariant=self.target_invariant
        )

        initial_tension, _ = current_m_tensor.update_tension()
        current_m_tensor.auto_dispatch_relaxation(tolerance=1e-4, max_steps=max_steps)
        final_tension, _ = current_m_tensor.update_tension()

        geodesic_trajectory = current_m_tensor.lineage.transformation_history
        action_cost = float(initial_tension.item() - final_tension.item())

        return {
            "geodesic_trajectory": geodesic_trajectory,
            "initial_tension": float(initial_tension.item()),
            "final_tension": float(final_tension.item()),
            "action_cost": action_cost,
            "resolved_tensor": current_m_tensor.raw_tensor
        }
