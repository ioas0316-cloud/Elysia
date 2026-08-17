"""
Unit tests for synaptic_architecture/hierarchical_emergence.py.

Verifies:
1. Molecule Primitive & Lineage O(1) LCA gap differentiation.
2. Cell Node Type State Pattern (Unrelaxed -> Relaxed) & Type State Violation Gating.
3. Tissue Layer (TopologicalFieldNetwork) Delta Tension Wave Propagation & Chain Relaxation.
4. Homotopy Type Checker, Auto-Lowering, and Reframing Type Elevation on Singularity.
"""

import pytest
import torch
import numpy as np

from synaptic_architecture.hierarchical_emergence import (
    AtomicTensor,
    MoleculePrimitive,
    MechanismNode,
    Unrelaxed,
    Relaxed,
    create_causal_cell_node,
    TopologicalFieldNetwork,
    MetaInvariant,
    HomotopyTypeChecker,
)


def test_molecule_primitive_and_lca():
    """Verifies Level 1 MoleculePrimitive creation and O(1) LCA divergence tracking."""
    tensor_a = AtomicTensor([1.0, 2.0, 3.0])
    primitive_a = MoleculePrimitive(tensor_a, origin_tag="Branch_A", parent_ids=["Root"])
    primitive_a.lineage.history.extend(["Op1", "Op2_Split", "Op3_A"])

    tensor_b = AtomicTensor([1.0, 2.0, 4.0])
    primitive_b = MoleculePrimitive(tensor_b, origin_tag="Branch_B", parent_ids=["Root"])
    primitive_b.lineage.history.extend(["Op1", "Op2_Split", "Op3_B"])

    gap = primitive_a.compute_causal_gap(primitive_b)
    # History starts with origin_tag ("Branch_A" vs "Branch_B"), then ["Op1", "Op2_Split", "Op3_A"]
    # So index 0 ("Branch_A" != "Branch_B") differs, but common parents contains "Root"
    assert gap["lca_id"] == "Root"
    assert gap["split_depth"] == 0
    assert "Op1" in gap["self_divergence"]
    assert "Op1" in gap["other_divergence"]


def test_type_state_pattern_and_gating():
    """Verifies Type State semantics and method gating for Unrelaxed vs Relaxed states."""
    # Create unrelaxed cell node with initial flux sum = 4.0 (Target Flux = 1.0, Epsilon = 0.05)
    node = create_causal_cell_node([2.0, 2.0], origin_tag="Cell_1", epsilon=0.05)
    assert node.tension > 0.05

    # Accessing into_geodesic_path() on Unrelaxed node must raise TypeError
    with pytest.raises(TypeError) as exc_info:
        node.into_geodesic_path()
    assert "Type State Violation" in str(exc_info.value)

    # Relax the node
    relaxed_node = node.relax()
    assert isinstance(relaxed_node, MechanismNode)
    assert relaxed_node.tension <= 0.05

    # Accessing into_geodesic_path() on Relaxed node succeeds
    path = relaxed_node.into_geodesic_path()
    assert "Geodesic Path Established" in path


def test_topological_field_network_chain_relaxation():
    """Verifies Tissue Level asynchronous delta tension wave propagation and chain relaxation."""
    network = TopologicalFieldNetwork(node_ids=["Node_A", "Node_B", "Node_C"])

    node_a = create_causal_cell_node([5.0, 5.0], origin_tag="A", epsilon=0.05)
    node_b = create_causal_cell_node([1.0, 0.0], origin_tag="B", epsilon=0.05)
    node_c = create_causal_cell_node([0.5, 0.5], origin_tag="C", epsilon=0.05)

    network.add_node("Node_A", node_a)
    network.add_node("Node_B", node_b)
    network.add_node("Node_C", node_c)

    network.set_coupling("Node_A", "Node_B", 0.5)
    network.set_coupling("Node_B", "Node_C", 0.3)

    success, cycles = network.resolve_chain_relaxation(max_cycles=15, global_tol=0.1)
    assert success is True
    assert cycles <= 15
    # Verify nodes in network have relaxed tensions
    for nid, node in network.nodes.items():
        assert node.tension <= 0.1


def test_homotopy_type_checker_and_reframing_elevation():
    """Verifies Homotopy Type Checker and Reframing Type Elevation when singularity T -> inf is encountered."""
    checker = HomotopyTypeChecker(
        meta_invariant=MetaInvariant(
            name="OrganMeta",
            symmetry_group="SO(3)",
            singularity_threshold=50.0
        )
    )

    # Create node with extreme flux tension (Sum = 500.0 >> threshold 50.0)
    huge_tensor = torch.ones((10, 10)) * 5.0  # Sum = 500.0
    cell_node = create_causal_cell_node(huge_tensor, origin_tag="SingularityNode", epsilon=0.05)
    assert cell_node.tension >= 50.0

    # Auto lowering pass triggers Reframing Type Elevation
    relaxed_node, synthesized_meta = checker.auto_lower_and_relax(cell_node)

    assert synthesized_meta is not None
    assert "SO(3) x U(1)" in synthesized_meta.symmetry_group
    assert "ReframingTypeElevation" in relaxed_node.primitive.lineage.history[-1] or \
           any("ReframingTypeElevation" in h for h in relaxed_node.primitive.lineage.history)
