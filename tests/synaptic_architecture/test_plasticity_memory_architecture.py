import pytest
import numpy as np
from synaptic_architecture.plasticity_memory_architecture import (
    CausalNode,
    CausalEdge,
    MetaCognitiveObservation,
    FrictionSensor,
    NodeAutopoiesis,
    GraphMutator,
    ConsolidationLoop,
    PlasticityMemoryArchitecture
)


def test_friction_sensor():
    sensor = FrictionSensor(impedance_threshold=0.3)
    nodes = {
        "node_1": CausalNode(
            id="node_1",
            feature_vector=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        )
    }
    edges = [
        CausalEdge(source_id="node_1", target_id="node_1", resistance_mask=0.2)
    ]

    # Similar vector -> lower impedance and friction
    similar_ctx = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
    f_low, meta_low = sensor.detect_friction(similar_ctx, nodes, edges)
    assert f_low >= 0.0
    assert isinstance(meta_low, MetaCognitiveObservation)

    # Distant vector -> higher impedance and friction
    distant_ctx = np.array([0.0, 10.0, 10.0, 10.0], dtype=np.float32)
    f_high, meta_high = sensor.detect_friction(distant_ctx, nodes, edges)
    assert f_high > f_low
    assert "FrictionSensor" in meta_high.narrative


def test_node_autopoiesis():
    autopoiesis = NodeAutopoiesis(friction_threshold=0.2, split_factor=0.2)
    nodes = {
        "node_base": CausalNode(
            id="node_base",
            feature_vector=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        )
    }

    # Low friction -> no autopoietic nodes spawned
    low_friction_nodes = autopoiesis.trigger_autopoiesis(
        friction=0.1,
        context_vector=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        nodes=nodes
    )
    assert len(low_friction_nodes) == 0

    # High friction with different vector -> spawns hypothesis nodes
    high_friction_nodes = autopoiesis.trigger_autopoiesis(
        friction=0.8,
        context_vector=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        nodes=nodes
    )
    assert len(high_friction_nodes) > 0
    assert high_friction_nodes[0].node_type == "autopoietic_hypothesis"


def test_graph_mutator():
    mutator = GraphMutator(attraction_rate=0.3, resistance_decay=0.1)
    nodes = {
        "n1": CausalNode(id="n1", feature_vector=np.array([1.0, 0.0], dtype=np.float32)),
        "n2": CausalNode(id="n2", feature_vector=np.array([0.9, 0.1], dtype=np.float32))
    }
    edges = [
        CausalEdge(source_id="n1", target_id="n2", weight=1.0, resistance_mask=0.5)
    ]
    auto_nodes = [
        CausalNode(id="n3_auto", feature_vector=np.array([0.85, 0.15], dtype=np.float32), node_type="autopoietic_hypothesis")
    ]

    mutated_edges = mutator.mutate_topology(nodes, edges, auto_nodes)
    assert len(mutated_edges) > len(edges)
    # The weight between n1 and n2 should increase due to high similarity (sameness)
    assert mutated_edges[0].weight >= 1.0


def test_consolidation_loop():
    consolidation = ConsolidationLoop(consolidation_threshold=0.5)
    nodes = {
        "n_auto": CausalNode(
            id="n_auto",
            feature_vector=np.array([0.5, 0.5], dtype=np.float32),
            node_type="autopoietic_hypothesis",
            energy=1.0
        )
    }
    edges = [
        CausalEdge(source_id="n_auto", target_id="n_auto", weight=1.0, resistance_mask=0.2)
    ]

    # Friction reduction -> consolidates hypothesis node into persistent memory
    updated_nodes, updated_edges, consolidated_ids = consolidation.evaluate_and_consolidate(
        nodes=nodes,
        edges=edges,
        initial_friction=0.8,
        final_friction=0.3
    )

    assert "n_auto" in consolidated_ids
    assert updated_nodes["n_auto"].node_type == "consolidated"


def test_plasticity_memory_architecture_end_to_end():
    architecture = PlasticityMemoryArchitecture(base_feature_dim=16)

    # 1. Process Event with vector context
    ctx_vec = np.random.randn(16).astype(np.float32)
    record_1 = architecture.process_event(ctx_vec)
    assert "initial_friction" in record_1
    assert "final_friction" in record_1
    assert record_1["total_nodes"] >= 1

    # 2. Process Event with text string context
    record_2 = architecture.process_event("High friction external event stimulus")
    assert record_2["total_nodes"] >= 1

    # 3. Test Reconstructive Memory
    reconstructed_vec, node_ids = architecture.reconstruct_memory("High friction external event stimulus")
    assert reconstructed_vec.shape == (16,)
    assert len(node_ids) > 0
