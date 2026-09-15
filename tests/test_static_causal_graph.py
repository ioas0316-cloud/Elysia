"""
Unit tests for StaticCausalGraph and topological memory wave propagation.
"""

import pytest
from core.memory.static_causal_graph import (
    BoundaryType,
    LayerMetadata,
    CausalSignal,
    CausalNode,
    CausalEdge,
    StaticCausalGraph
)


def test_boundary_type_and_metadata():
    layer = LayerMetadata(
        boundary=BoundaryType.PERCEPTION_LAYER,
        label="Perception",
        description="Sensory boundary",
        controllability=0.2
    )
    assert layer.is_self is True

    ext_layer = LayerMetadata(
        boundary=BoundaryType.EXTERNAL_WORLD,
        label="External",
        description="Outside world",
        controllability=0.0
    )
    assert ext_layer.is_self is False


def test_causal_node_energy_reception():
    triggered_nodes = []

    def sample_action(node_id, potential):
        triggered_nodes.append((node_id, potential))

    node = CausalNode(
        node_id="N1",
        threshold=2.0,
        boundary=BoundaryType.PROCESSING_LAYER,
        action_vector=sample_action
    )

    # Receive sub-threshold energy
    overflow1 = node.receive_energy(1.0)
    assert overflow1 == 0.0
    assert node.potential == 1.0
    assert len(triggered_nodes) == 0

    # Receive threshold-exceeding energy
    overflow2 = node.receive_energy(1.5)
    # total potential 2.5 >= threshold 2.0 -> overflow 0.5 + 1.0 = 1.5
    assert overflow2 == 1.5
    assert node.potential == 0.0
    assert len(triggered_nodes) == 1
    assert triggered_nodes[0][0] == "N1"


def test_causal_edge_transmission_and_friction():
    edge = CausalEdge(
        source_id="A",
        target_id="B",
        tension=0.8,
        resistance=0.2
    )
    # Transmit energy = 10.0
    # Effective energy = 10.0 * 0.8 * (1.0 - 0.2) = 6.4
    # Friction loss = 10.0 - 6.4 = 3.6
    transmitted = edge.transmit(10.0)
    assert pytest.approx(transmitted, 0.01) == 6.4
    assert pytest.approx(edge.accumulated_friction, 0.01) == 3.6
    assert edge.traversal_count == 1


def test_graph_wave_propagation_and_plasticity():
    graph = StaticCausalGraph()

    graph.add_node("STIMULUS", threshold=1.0, boundary=BoundaryType.PERCEPTION_LAYER, depth=1.0)
    graph.add_node("FILTER", threshold=1.0, boundary=BoundaryType.PROCESSING_LAYER, depth=3.0)
    graph.add_node("CORE", threshold=1.0, boundary=BoundaryType.MEMORY_LAYER, depth=10.0)

    edge1 = graph.connect("STIMULUS", "FILTER", tension=0.8, resistance=0.1)
    edge2 = graph.connect("FILTER", "CORE", tension=0.9, resistance=0.1)

    initial_tension1 = edge1.tension

    trace = graph.propagate("STIMULUS", energy=5.0, plasticity_alpha=0.02)
    assert "STIMULUS" in trace
    assert "FILTER" in trace
    assert "CORE" in trace

    # Check plasticity reinforcement on edge1
    assert edge1.tension > initial_tension1
    assert edge1.accumulated_friction > 0.0
