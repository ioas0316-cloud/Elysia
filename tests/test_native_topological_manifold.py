"""
Tests for Native Topological Data Structures, Invariance/Divergence Tracking,
Boundary Discontinuity Reconstruction, and Gas -> Liquid -> Ice Phase Transitions.
"""

import pytest
from core.topology.native_topological_manifold import (
    SequentialOffsetTopology,
    SpatialGridTopology,
    HierarchicalBranchTopology,
    TopologyType
)
from core.topology.invariance_divergence_tracker import (
    InvarianceDivergenceTracker,
    CausalDiscontinuityDetector
)
from core.topology.topological_phase_engine import (
    TopologicalPhaseEngine,
    TopologicalPhaseState
)


def test_native_topological_manifolds_creation():
    seq = SequentialOffsetTopology("s1", "HELLO")
    assert len(seq.nodes) == 5
    assert seq.nodes["s1_seq_0"].topology_type == TopologyType.SEQUENTIAL
    assert seq.nodes["s1_seq_0"].adjacencies["next"][0].payload == "E"

    grid = SpatialGridTopology("g1", [[1, 2], [3, 4]])
    assert len(grid.nodes) == 4
    assert grid.nodes["g1_grid_0_0"].topology_type == TopologyType.SPATIAL_GRID
    assert grid.nodes["g1_grid_0_0"].adjacencies["east"][0].payload == 2

    hier = HierarchicalBranchTopology("h1", {"key": "value", "list": [10, 20]})
    assert len(hier.nodes) > 0
    assert hier.nodes["h1_root"].topology_type == TopologyType.HIERARCHICAL


def test_invariance_and_divergence_tracker():
    grid = SpatialGridTopology("g1", [[1, 2], [3, 4]])
    hier = HierarchicalBranchTopology("h1", {"key": "value"})

    inv = InvarianceDivergenceTracker.observe_invariant(grid, hier)
    assert inv["is_continuous"] is True
    assert inv["principle"] == "Relational Topological Adjacency"

    div = InvarianceDivergenceTracker.observe_divergence(grid, hier)
    assert div["type_a"] == "spatial_grid"
    assert div["type_b"] == "hierarchical"
    assert "Spatial Grid expands across 2D orthogonal neighbor meshes" in div["explanation"]


def test_causal_discontinuity_reconstruction():
    grid = SpatialGridTopology("g1", [[1, 2], [3, 4]])
    hier = HierarchicalBranchTopology("h1", {"key": "value"})

    detector = CausalDiscontinuityDetector()
    bridge_node = detector.detect_and_reconstruct_bridge(grid, hier)

    assert bridge_node.topology_type == TopologyType.HYBRID_MACRO
    assert len(bridge_node.adjacencies) > 0
    assert "bridge_to_h1" in grid.nodes["g1_grid_0_0"].coupled_pointers


def test_topological_phase_engine_gas_liquid_ice():
    grid = SpatialGridTopology("g1", [[1, 2], [3, 4]])
    hier = HierarchicalBranchTopology("h1", {"key": "value"})

    engine = TopologicalPhaseEngine("test_engine")

    # Gas State
    gas_res = engine.observe_gas_state([grid, hier])
    assert gas_res["phase"] == TopologicalPhaseState.GAS.value

    # Liquid Coupling
    liquid_res = engine.initiate_liquid_coupling(grid, hier)
    assert liquid_res["phase"] == TopologicalPhaseState.LIQUID.value

    # Ice Crystallization
    ice_macro = engine.crystallize_to_ice(grid, hier)
    assert engine.phase_state == TopologicalPhaseState.ICE
    assert len(ice_macro.nodes) > len(grid.nodes) + len(hier.nodes)

    # Query with 0 FLOPs
    bridge_id = list(ice_macro.root_nodes)[0].node_id
    traversal_res = engine.query_ice_macro_topology(bridge_id, ["connected_boundary_a:g1_grid_0_0"])
    assert traversal_res["flops_performed"] == 0
    assert traversal_res["final_node_id"] == "g1_grid_0_0"
