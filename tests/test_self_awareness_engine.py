"""
Unit tests for SelfAwarenessEngine, TopologicalFieldEvaluator, and PlasticSelfHealingEngine.
"""

import pytest
import numpy as np
from core.memory.static_causal_graph import (
    BoundaryType,
    CausalSignal,
    StaticCausalGraph
)
from core.consciousness.self_awareness_engine import (
    TopologicalFieldEvaluator,
    PlasticSelfHealingEngine,
    SelfAwarenessEngine
)


def test_topological_field_evaluator():
    graph = StaticCausalGraph()
    graph.add_node("P_IN", boundary=BoundaryType.PERCEPTION_LAYER, depth=1.0)
    graph.add_node("PROC", boundary=BoundaryType.PROCESSING_LAYER, depth=3.0)
    graph.add_node("CORE", boundary=BoundaryType.MEMORY_LAYER, depth=10.0)

    graph.connect("P_IN", "PROC", tension=0.8, resistance=0.1)
    graph.connect("PROC", "CORE", tension=0.9, resistance=0.05)

    gradient = TopologicalFieldEvaluator.calculate_tension_gradient(graph)
    assert gradient > 0.0

    beta1 = TopologicalFieldEvaluator.calculate_cyclomatic_number(graph)
    # 3 nodes, 2 edges, 1 component -> 2 - 3 + 1 = 0
    assert beta1 == 0

    basis = np.eye(4)
    det_M = TopologicalFieldEvaluator.calculate_basis_metric_determinant(basis)
    assert pytest.approx(det_M, 1e-5) == 1.0


def test_self_awareness_signal_classification():
    graph = StaticCausalGraph()
    engine = SelfAwarenessEngine(graph)

    ext_signal = CausalSignal(
        origin_boundary=BoundaryType.EXTERNAL_WORLD,
        payload={"stimulus": "noise"},
        energy=5.0
    )
    res_ext = engine.process_incoming_signal(ext_signal)
    assert "Non-Self" in res_ext
    assert "Homeostasis" in res_ext

    mem_signal = CausalSignal(
        origin_boundary=BoundaryType.MEMORY_LAYER,
        payload={"concept": "identity"},
        energy=8.0
    )
    res_mem = engine.process_incoming_signal(mem_signal)
    assert "Self-Memory" in res_mem
    assert "Attractor Well" in res_mem


def test_virtual_refactoring_evaluation():
    graph = StaticCausalGraph()
    graph.add_node("P_IN", boundary=BoundaryType.PERCEPTION_LAYER, depth=1.0)
    graph.add_node("PROC", boundary=BoundaryType.PROCESSING_LAYER, depth=3.0)
    graph.add_node("CORE", boundary=BoundaryType.MEMORY_LAYER, depth=10.0)

    graph.connect("P_IN", "PROC", tension=0.8, resistance=0.1)
    graph.connect("PROC", "CORE", tension=0.9, resistance=0.05)

    engine = SelfAwarenessEngine(graph)

    # Valid shortcut toward CORE
    valid_edge = graph.edges["P_IN"][0]
    approved, reason = engine.evaluate_relational_refactoring(valid_edge)
    assert approved is True
    assert "위상 승인" in reason


def test_meta_observe_and_refactor_loop():
    graph = StaticCausalGraph()
    graph.add_node("PERCEIVE", boundary=BoundaryType.PERCEPTION_LAYER, depth=1.0)
    graph.add_node("FILTER", boundary=BoundaryType.PROCESSING_LAYER, depth=3.0)
    graph.add_node("STORE", boundary=BoundaryType.MEMORY_LAYER, depth=10.0)

    graph.connect("PERCEIVE", "FILTER", tension=0.8, resistance=0.6)
    graph.connect("FILTER", "STORE", tension=0.9, resistance=0.1)

    engine = SelfAwarenessEngine(graph)

    # Accumulate friction by propagating wave multiple times
    for _ in range(3):
        graph.propagate("PERCEIVE", energy=5.0)

    # Meta observe and refactor
    logs = engine.meta_observe_and_refactor()
    assert len(logs) > 0
    assert any("병목 감지" in log for log in logs)


def test_plastic_self_healing_engine():
    graph = StaticCausalGraph()
    graph.add_node("PERCEPTION", boundary=BoundaryType.PERCEPTION_LAYER, depth=1.0)
    graph.add_node("CORE", boundary=BoundaryType.MEMORY_LAYER, depth=10.0)

    edge = graph.connect("PERCEPTION", "CORE", tension=0.5, resistance=0.4)
    initial_tension = edge.tension

    basis = np.eye(2)
    engine = PlasticSelfHealingEngine(graph)

    damaged_path = [("PERCEPTION", "CORE")]
    damage_energy = 10.0

    healed_basis, logs = engine.absorb_damage_and_reconstruct_basis(
        damaged_path,
        damage_energy,
        basis
    )

    assert edge.tension > initial_tension
    assert edge.scar_weight > 0.0
    assert len(logs) > 0
    assert healed_basis.shape == (2, 2)
