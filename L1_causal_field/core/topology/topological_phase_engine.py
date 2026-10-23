"""
Topological Voluntary Coupling and Phase Transition Engine (Gas -> Liquid -> Ice)

Implements dynamic phase state transitions based on native topological alignment:
- Gas (Unstructured/Unbound): Nodes float independently without fixed boundary pointers.
- Liquid (Active Coupling/Alignment): Boundary pointers between topologies actively search, match, and align interface links.
- Ice (Crystallized Macro-Topology): Phase-locked pointer binding producing a single unified macro-topology.
  Once in Ice state, traversal and query require zero external decoding or parsing (FLOPs = 0).
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum
from core.topology.native_topological_manifold import (
    NativeTopologicalNode,
    NativeTopologicalManifold,
    TopologyType
)
from core.topology.invariance_divergence_tracker import (
    InvarianceDivergenceTracker,
    CausalDiscontinuityDetector
)


class TopologicalPhaseState(Enum):
    GAS = "Gas (Unbound/Floating)"
    LIQUID = "Liquid (Active Coupling & Boundary Alignment)"
    ICE = "Ice (Crystallized Phase-Locked Macro-Topology, FLOPs=0)"


class TopologicalPhaseEngine:
    """
    Engine executing voluntary coupling and Gas -> Liquid -> Ice phase transitions.
    """

    def __init__(self, engine_id: str):
        self.engine_id: str = engine_id
        self.phase_state: TopologicalPhaseState = TopologicalPhaseState.GAS
        self.discontinuity_detector = CausalDiscontinuityDetector()
        self.crystallized_macro_manifold: Optional[NativeTopologicalManifold] = None

    def observe_gas_state(
        self,
        manifolds: List[NativeTopologicalManifold]
    ) -> Dict[str, Any]:
        """
        Reports Gas state: Manifolds exist as independent, unbound native topologies.
        """
        self.phase_state = TopologicalPhaseState.GAS
        return {
            "phase": self.phase_state.value,
            "flops_cost": 0,
            "status": "Unbound native topologies floating independently.",
            "manifold_ids": [m.manifold_id for m in manifolds],
            "total_unbound_nodes": sum(len(m.nodes) for m in manifolds)
        }

    def initiate_liquid_coupling(
        self,
        manifold_a: NativeTopologicalManifold,
        manifold_b: NativeTopologicalManifold
    ) -> Dict[str, Any]:
        """
        Transitions to Liquid state: Boundary pointer alignment and structural interface matching.
        Discontinuities are internalised into a bridging macro-node.
        """
        self.phase_state = TopologicalPhaseState.LIQUID

        # 1. Observe Invariance & Divergence
        invariant = InvarianceDivergenceTracker.observe_invariant(manifold_a, manifold_b)
        divergence = InvarianceDivergenceTracker.observe_divergence(manifold_a, manifold_b)

        # 2. Reconstruct Causal Bridge across boundary discontinuity
        bridge_node = self.discontinuity_detector.detect_and_reconstruct_bridge(manifold_a, manifold_b)

        return {
            "phase": self.phase_state.value,
            "status": "Liquid coupling active. Boundary interfaces aligned via structural bridge.",
            "invariant_principle": invariant["principle"],
            "divergence_rule": divergence["divergence_rule"],
            "bridge_node_id": bridge_node.node_id
        }

    def crystallize_to_ice(
        self,
        manifold_a: NativeTopologicalManifold,
        manifold_b: NativeTopologicalManifold
    ) -> NativeTopologicalManifold:
        """
        Transitions to Ice state: Phase-locks all boundary pointers, binding manifold_a, manifold_b,
        and the bridging macro-node into a unified macro-manifold.

        Zero external decoding or parsing is required during future traversals (FLOPs = 0).
        """
        if self.phase_state != TopologicalPhaseState.LIQUID:
            # Execute liquid coupling first if needed
            self.initiate_liquid_coupling(manifold_a, manifold_b)

        macro_id = f"ice_macro_{manifold_a.manifold_id}_{manifold_b.manifold_id}"
        macro_manifold = NativeTopologicalManifold(
            manifold_id=macro_id,
            topology_type=TopologyType.HYBRID_MACRO
        )

        # Merge all nodes into macro_manifold
        for node in manifold_a.nodes.values():
            macro_manifold.add_node(node)

        for node in manifold_b.nodes.values():
            macro_manifold.add_node(node)

        # Retrieve bridge nodes
        for bridge_info in self.discontinuity_detector.reconstructed_bridges:
            bridge_node = bridge_info["bridge_node"]
            macro_manifold.add_node(bridge_node, is_root=True)

        self.crystallized_macro_manifold = macro_manifold
        self.phase_state = TopologicalPhaseState.ICE

        return macro_manifold

    def query_ice_macro_topology(
        self,
        start_node_id: str,
        traversal_path: List[str]
    ) -> Dict[str, Any]:
        """
        Traverses the crystallized Ice macro-topology natively without parsing or float computation.
        FLOPs = 0.
        """
        if self.phase_state != TopologicalPhaseState.ICE or not self.crystallized_macro_manifold:
            raise RuntimeError("Engine must be in Ice state before querying macro-topology.")

        current_node = self.crystallized_macro_manifold.nodes.get(start_node_id)
        if not current_node:
            return {"error": f"Node {start_node_id} not found in Ice topology."}

        visited_nodes = [current_node.node_id]

        for step in traversal_path:
            # First check direct adjacencies
            if step in current_node.adjacencies and current_node.adjacencies[step]:
                current_node = current_node.adjacencies[step][0]
                visited_nodes.append(current_node.node_id)
            # Check boundary coupling pointers
            elif step in current_node.coupled_pointers:
                current_node = current_node.coupled_pointers[step]
                visited_nodes.append(current_node.node_id)
            else:
                break

        return {
            "phase": self.phase_state.value,
            "flops_performed": 0,  # Zero FLOPs
            "final_node_id": current_node.node_id,
            "final_payload": current_node.payload,
            "traversal_path": visited_nodes
        }
