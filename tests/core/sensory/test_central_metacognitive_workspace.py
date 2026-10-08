import pytest
import numpy as np
from core.sensory.central_metacognitive_workspace import (
    GroundingManifold,
    QualitativeDiscrepancyEngine,
    MetacognitiveReflectionLoop,
    SchemaTopologyRewiringUnit,
    CentralMetacognitiveWorkspace,
)


def test_grounding_manifold():
    manifold = GroundingManifold(dim=16)
    sym_vec = np.ones(16, dtype=np.float32)
    wave = np.sin(np.linspace(0, 2 * np.pi, 16)).astype(np.float32)

    grounded = manifold.ground(sym_vec, wave)
    assert grounded.shape == (16,)
    assert np.isclose(np.linalg.norm(grounded), 1.0)


def test_qualitative_discrepancy_engine():
    engine = QualitativeDiscrepancyEngine(dim=16, num_symbols=4)

    int_m = np.random.randn(16).astype(np.float32)
    world_r = np.random.randn(16).astype(np.float32)

    report = engine.reason_discrepancy(int_m, world_r)
    assert report.what_symbol_id.startswith("SYMBOL_NODE_")
    assert report.how_much_magnitude >= 0.0
    assert report.how_causal_type in ["LENS_MISALIGNMENT", "CONTEXT_SHIFT", "STRUCTURAL_GAP"]


def test_metacognitive_reflection_loop():
    engine = QualitativeDiscrepancyEngine(dim=16, num_symbols=4)
    reflection = MetacognitiveReflectionLoop(dim=16)

    int_m = np.ones(16, dtype=np.float32)
    world_r = np.ones(16, dtype=np.float32) * 5.0  # Large discrepancy

    report = engine.reason_discrepancy(int_m, world_r)
    reflect_data = reflection.reflect(report)

    assert reflect_data["target_symbol"] == report.what_symbol_id
    assert reflect_data["updated_validity"] < 1.0
    assert reflect_data["active_seeking_value"] > 0.0


def test_schema_topology_rewiring_unit():
    engine = QualitativeDiscrepancyEngine(dim=16, num_symbols=4)
    reflection = MetacognitiveReflectionLoop(dim=16)
    rewiring = SchemaTopologyRewiringUnit(engine)

    int_m = np.ones(16, dtype=np.float32)
    world_r = np.ones(16, dtype=np.float32) * 10.0

    report = engine.reason_discrepancy(int_m, world_r)
    reflect_data = reflection.reflect(report)
    action = rewiring.evolve_schema(report, reflect_data)

    assert hasattr(action, "action_type")
    assert hasattr(action, "target_symbol")


def test_central_metacognitive_workspace_full():
    cmw = CentralMetacognitiveWorkspace(dim=16, num_symbols=4)

    symbol_vec = np.random.randn(16).astype(np.float32)
    sensory_wave = np.random.randn(16).astype(np.float32)
    world_response = np.random.randn(16).astype(np.float32)

    out = cmw.process_cognition(symbol_vec, sensory_wave, world_response)

    assert "grounded_manifold" in out
    assert "discrepancy_report" in out
    assert "reflection_data" in out
    assert "evolution_action" in out
