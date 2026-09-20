"""
Dynamic Graph Rewriting Engine (동적 그래프 재배열 엔진)

Implements categorical Double Pushout (DPO) rewriting on multi-scale causal hypergraph G = (V, E, H, S):
  L <-- K --> R
  G_fracture <-- G_context --> G_reconstructed

3 Topological Surgeries triggered when L_friction > theta_critical:
1. Node Unfolding (노드 분할): Unfolds single node V_i into causal chain (V_i,1 -> V_i,2)
2. Edge Rewiring (간선 단절 및 재연결): Severs broken edge E(A -> B) and rewires via foreign node V_ext -> (A -> V_ext -> C)
3. Scale Elevation (스케일 승격): Elevates micro loop subgraphs to macro scale node V_macro at s_k+1
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, List, Optional, Set, Tuple
import torch
import numpy as np


class SurgeryType(Enum):
    NODE_UNFOLDING = auto()
    EDGE_REWIRING = auto()
    SCALE_ELEVATION = auto()


@dataclass
class CausalNode:
    node_id: str
    concept_tensor: torch.Tensor
    scale_coordinate: float          # s in [0, 1]
    friction_strain: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalEdge:
    edge_id: str
    source_id: str
    target_id: str
    transition_probability: float
    causal_weight: float
    is_severed: bool = False


@dataclass
class HyperEdge:
    hyperedge_id: str
    member_node_ids: List[str]
    macro_scale_target: float
    binding_tensor: torch.Tensor


@dataclass
class SurgeryLog:
    surgery_type: SurgeryType
    affected_nodes: List[str]
    affected_edges: List[str]
    new_nodes_added: List[str]
    new_edges_added: List[str]
    description: str


class DynamicGraphRewriter:
    """
    Multi-Scale Causal Hypergraph G = (V, E, H, S) Engine executing Categorical Double Pushout (DPO)
    Topological Surgery upon exogenous shock friction overflow.
    """

    def __init__(self, dimension: int = 64, dtype=torch.float32):
        self.dimension = dimension
        self.dtype = dtype

        # Multi-scale Hypergraph structures
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: Dict[str, CausalEdge] = {}
        self.hyperedges: Dict[str, HyperEdge] = {}

        # Surgery history
        self.surgery_history: List[SurgeryLog] = []

    def add_node(self, node_id: str, concept_tensor: torch.Tensor, scale: float = 0.0) -> CausalNode:
        """Adds a CausalNode at scale s in [0, 1]."""
        tensor_norm = concept_tensor.to(self.dtype)
        if tensor_norm.shape[0] != self.dimension:
            tensor_norm = torch.resize_as_(tensor_norm, torch.zeros(self.dimension, dtype=self.dtype))

        node = CausalNode(
            node_id=node_id,
            concept_tensor=tensor_norm,
            scale_coordinate=scale
        )
        self.nodes[node_id] = node
        return node

    def add_edge(self, source_id: str, target_id: str, prob: float = 1.0, weight: float = 1.0) -> CausalEdge:
        """Adds a directed CausalEdge between two nodes."""
        edge_id = f"edge_{source_id}_to_{target_id}"
        edge = CausalEdge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            transition_probability=prob,
            causal_weight=weight
        )
        self.edges[edge_id] = edge
        return edge

    def add_hyperedge(self, hyperedge_id: str, member_node_ids: List[str], macro_scale: float) -> HyperEdge:
        """Binds multiple micro nodes to a higher scale target."""
        binding_vec = torch.zeros(self.dimension, dtype=self.dtype)
        for nid in member_node_ids:
            if nid in self.nodes:
                binding_vec += self.nodes[nid].concept_tensor
        binding_vec /= (len(member_node_ids) + 1e-8)

        hyperedge = HyperEdge(
            hyperedge_id=hyperedge_id,
            member_node_ids=member_node_ids,
            macro_scale_target=macro_scale,
            binding_tensor=binding_vec
        )
        self.hyperedges[hyperedge_id] = hyperedge
        return hyperedge

    def execute_double_pushout_rewriting(
        self,
        exogenous_shock_vector: torch.Tensor,
        friction_magnitude: float
    ) -> List[SurgeryLog]:
        """
        Executes Double Pushout (DPO) topological surgery L <-- K --> R on fractured subgraphs.
        Triggered when friction magnitude is high.
        """
        logs: List[SurgeryLog] = []
        shock_t = exogenous_shock_vector.to(self.dtype)

        # 1. Localize Fracture Subgraph G_fracture based on node strain + shock alignment
        fractured_node_ids = []
        for nid, node in self.nodes.items():
            dot_shock = torch.dot(node.concept_tensor, shock_t).item() if shock_t.shape[0] == self.dimension else 1.0
            node.friction_strain += friction_magnitude * abs(dot_shock) * 0.1
            if node.friction_strain > 1.0:
                fractured_node_ids.append(nid)

        if not fractured_node_ids:
            # Fallback: pick node with highest strain
            highest_nid = max(self.nodes.keys(), key=lambda k: self.nodes[k].friction_strain, default=None)
            if highest_nid:
                fractured_node_ids.append(highest_nid)

        # 2. Execute 3 Surgeries based on friction pattern
        for fnid in fractured_node_ids:
            fnode = self.nodes[fnid]

            # Condition A: Node Unfolding if strain is extreme (> 3.0)
            if fnode.friction_strain > 3.0:
                log = self._node_unfolding_surgery(fnode, shock_t)
                logs.append(log)

            # Condition B: Edge Rewiring if broken edges connected to fnode exist
            connected_edges = [e for e in self.edges.values() if (e.source_id == fnid or e.target_id == fnid) and not e.is_severed]
            if connected_edges:
                log = self._edge_rewiring_surgery(fnode, connected_edges[0], shock_t)
                logs.append(log)

            # Condition C: Scale Elevation if micro nodes form dense cluster
            if fnode.scale_coordinate < 0.5:
                log = self._scale_elevation_surgery(fnode)
                logs.append(log)

            # Reset strain post surgery
            fnode.friction_strain *= 0.1

        self.surgery_history.extend(logs)
        return logs

    def _node_unfolding_surgery(self, node: CausalNode, shock_t: torch.Tensor) -> SurgeryLog:
        """Node Unfolding (노드 분할): Unfolds node V_i into causal chain (V_i,1 -> V_i,2)."""
        sub_id1 = f"{node.node_id}_seq1"
        sub_id2 = f"{node.node_id}_seq2"

        tensor1 = node.concept_tensor * 0.5
        tensor2 = node.concept_tensor * 0.5 + shock_t * 0.2

        n1 = self.add_node(sub_id1, tensor1, scale=node.scale_coordinate)
        n2 = self.add_node(sub_id2, tensor2, scale=node.scale_coordinate)

        new_edge = self.add_edge(sub_id1, sub_id2, prob=0.9, weight=1.5)

        # Re-route edges coming to node to n1, and edges leaving node to n2
        for edge in list(self.edges.values()):
            if edge.target_id == node.node_id:
                edge.target_id = sub_id1
            if edge.source_id == node.node_id:
                edge.source_id = sub_id2

        log = SurgeryLog(
            surgery_type=SurgeryType.NODE_UNFOLDING,
            affected_nodes=[node.node_id],
            affected_edges=[],
            new_nodes_added=[sub_id1, sub_id2],
            new_edges_added=[new_edge.edge_id],
            description=f"Unfolded node {node.node_id} into causal sequence ({sub_id1} -> {sub_id2})."
        )
        return log

    def _edge_rewiring_surgery(self, node: CausalNode, edge: CausalEdge, shock_t: torch.Tensor) -> SurgeryLog:
        """Edge Rewiring (간선 단절 및 재연결): Severs edge E(A->B) and rewires via V_ext (A -> V_ext -> C)."""
        edge.is_severed = True

        v_ext_id = f"node_ext_{node.node_id}"
        ext_node = self.add_node(v_ext_id, shock_t, scale=node.scale_coordinate)

        edge1 = self.add_edge(edge.source_id, v_ext_id, prob=0.85, weight=1.2)
        edge2 = self.add_edge(v_ext_id, edge.target_id, prob=0.85, weight=1.2)

        log = SurgeryLog(
            surgery_type=SurgeryType.EDGE_REWIRING,
            affected_nodes=[edge.source_id, edge.target_id],
            affected_edges=[edge.edge_id],
            new_nodes_added=[v_ext_id],
            new_edges_added=[edge1.edge_id, edge2.edge_id],
            description=f"Severed edge {edge.edge_id} and rewired through foreign exogenous node {v_ext_id}."
        )
        return log

    def _scale_elevation_surgery(self, node: CausalNode) -> SurgeryLog:
        """Scale Elevation (스케일 승격): Elevates micro node to macro scale node V_macro."""
        macro_id = f"macro_{node.node_id}"
        new_scale = min(1.0, node.scale_coordinate + 0.5)

        macro_node = self.add_node(macro_id, node.concept_tensor.clone(), scale=new_scale)
        hyper = self.add_hyperedge(f"hyper_{macro_id}", [node.node_id, macro_id], macro_scale=new_scale)

        log = SurgeryLog(
            surgery_type=SurgeryType.SCALE_ELEVATION,
            affected_nodes=[node.node_id],
            affected_edges=[],
            new_nodes_added=[macro_id],
            new_edges_added=[hyper.hyperedge_id],
            description=f"Elevated node {node.node_id} from scale {node.scale_coordinate:.2f} to macro scale {new_scale:.2f} ({macro_id})."
        )
        return log
