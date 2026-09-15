"""
Unit tests for ExclusivityCoexistenceTopologyEngine
"""

import pytest
import numpy as np

from core.topology.causal_structure import InformationTopology, CausalSymbol, TopologyLink
from core.topology.exclusivity_coexistence_topology import (
    ExclusivityCoexistenceTopologyEngine,
    ExclusivityBoundary,
    CoexistenceLayer
)


def create_sample_topology(name: str, symbol_name: str, tension: float) -> InformationTopology:
    topo = InformationTopology(name=name)
    sym = CausalSymbol(
        id=f"sym_{symbol_name}",
        name=symbol_name,
        material_vector=np.array([1.0, 0.5, 0.2, 0.0], dtype=np.float32),
        causal_trajectory=["origin", "evolution"],
        logical_category="physical_phenomenon",
        relational_links=[
            TopologyLink("origin", f"sym_{symbol_name}", "causal", 0.8, tension)
        ],
        intrinsic_tension=tension
    )
    topo.add_symbol(sym)
    return topo


def test_exclusivity_coexistence_topology_init():
    engine = ExclusivityCoexistenceTopologyEngine()
    assert engine.coexistence_map.primary_layer_id == "layer_root"
    assert "layer_root" in engine.coexistence_map.layers


def test_detect_exclusivity_boundary_exclusive():
    engine = ExclusivityCoexistenceTopologyEngine(exclusivity_threshold=0.5)

    vec_a = np.array([1.0, 0.0, 0.0, 0.0, 0.1], dtype=np.float32)
    vec_b = np.array([0.0, 2.0, 1.0, 0.5, 0.9], dtype=np.float32)
    context = np.array([1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    boundary = engine.detect_exclusivity_boundary(
        node_a_id="node_a",
        node_b_id="node_b",
        node_a_vector=vec_a,
        node_b_vector=vec_b,
        context_vector=context
    )

    assert boundary.is_mutually_exclusive is True
    assert boundary.friction_magnitude >= 0.5
    assert boundary.layer_b_id.startswith("layer_mitosis_")


def test_topological_mitosis_and_fusion():
    engine = ExclusivityCoexistenceTopologyEngine()

    # Mitosis
    boundary = ExclusivityBoundary(
        node_a_id="node_a",
        node_b_id="node_b",
        condition_vector=np.zeros(4, dtype=np.float32),
        friction_magnitude=0.9,
        is_mutually_exclusive=True,
        layer_a_id="layer_root",
        layer_b_id="layer_mitosis_1"
    )

    mitotic_layer = engine.perform_topological_mitosis(
        source_layer_id="layer_root",
        exclusive_node_id="node_b",
        boundary=boundary
    )

    assert mitotic_layer.layer_id == "layer_mitosis_1"
    assert "layer_mitosis_1" in engine.coexistence_map.layers

    # Fusion
    fused_layer = engine.perform_topological_fusion(
        layer_a_id="layer_root",
        layer_b_id="layer_mitosis_1",
        coexistent_nodes=[("node_a", "node_b")]
    )
    assert fused_layer.layer_id == "fused_layer_root_layer_mitosis_1"


def test_process_stimulus():
    self_topo = create_sample_topology("SelfTopo", "Light", tension=0.1)
    alien_topo = create_sample_topology("AlienTopo", "Light", tension=0.9)

    engine = ExclusivityCoexistenceTopologyEngine(primary_topology=self_topo)
    context = np.array([1.0, 0.5, 0.0, 0.2, 0.1], dtype=np.float32)

    res = engine.process_stimulus(alien_topo, context)
    assert res["boundaries_detected_count"] >= 1
    assert res["total_coexistence_layers"] >= 1
