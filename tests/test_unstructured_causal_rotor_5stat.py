"""
Unit and Integration Tests for Unstructured Data Topological & Rotor Extraction
and 5-Stat Environmental Principle Closed-Loop Feedback Engine.
"""

import pytest
import numpy as np
from core.ingestion.unstructured_topological_extractor import UnstructuredTopologicalExtractor
from core.causal_world.five_stat_closed_loop import FiveStatClosedLoopEngine, FiveStatVector


def test_topological_extractor_text():
    extractor = UnstructuredTopologicalExtractor(dim=4)
    text = "Elysia engine transforms unstructured reality into continuous causal rotors."

    result = extractor.extract_from_unstructured_text(text)

    assert result["num_tokens"] == 9
    assert len(result["metric_tensor_g"]) == 4
    assert len(result["rotors"]) == 8  # N-1 rotors for 9 tokens
    assert "ricci_scalar" in result
    assert "attractor_potential" in result
    assert 0.0 <= result["attractor_potential"] <= 1.0


def test_rotor_bivector_computation():
    extractor = UnstructuredTopologicalExtractor(dim=4)
    v1 = np.array([0.0, 1.0, 0.0, 0.0])
    v2 = np.array([0.1, 0.0, 2.0, 0.0])

    B_matrix, r_data = extractor.compute_bivector_and_rotor(v1, v2)

    assert B_matrix.shape == (4, 4)
    assert np.allclose(B_matrix, -B_matrix.T)  # Anti-symmetric bivector
    assert "scalar" in r_data
    assert len(r_data["bivector"]) == 6
    assert r_data["bivector_norm"] > 0.0


def test_five_stat_mapping_and_closed_loop():
    extractor = UnstructuredTopologicalExtractor(dim=4)
    engine = FiveStatClosedLoopEngine()

    extraction = extractor.extract_from_unstructured_text("The social environment experiences high tension and friction.")
    mapped_stat = engine.map_extraction_to_5stat(extraction)

    assert isinstance(mapped_stat, FiveStatVector)
    assert 0.0 <= mapped_stat.energy_consumption <= 1.0
    assert 0.0 <= mapped_stat.info_bandwidth <= 1.0
    assert 0.0 <= mapped_stat.friction_resistance <= 1.0
    assert 0.0 <= mapped_stat.adaptation_speed <= 1.0
    assert 0.0 <= mapped_stat.equilibrium_stability <= 1.0

    # Test Closed Loop Simulation & Self-Regulating Action
    fatigued_npc = FiveStatVector(
        energy_consumption=0.9,
        info_bandwidth=0.5,
        friction_resistance=0.6,
        adaptation_speed=0.4,
        equilibrium_stability=0.2
    )

    updated_npc, summary = engine.step_closed_loop(
        npc_stat=fatigued_npc,
        extraction_data=extraction,
        action_intent="work_or_exert"
    )

    # Overly fatigued NPC should automatically shift action to rest and recover
    assert summary["executed_action"] == "rest_and_recover"
    assert updated_npc.energy_consumption < 0.9
    assert updated_npc.equilibrium_stability > 0.2
