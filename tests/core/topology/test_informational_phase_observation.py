"""
Tests for Informational Phase Observation Engine (정보 위상 관측 및 인과 파동 엔진)
=============================================================================
"""

import numpy as np
import pytest
from core.topology.informational_phase_observation import (
    InformationalPhaseObservationEngine,
    ChromaticVector,
    PhaseNodalProjection,
    ProprioceptiveState
)


def test_nodal_phase_projection_text_and_vector():
    engine = InformationalPhaseObservationEngine(target_dimension=8)

    # Text nodal projection
    node1 = engine.project_to_nodal_phase("text_node", "Superintelligence and Causal Topology")
    assert isinstance(node1, PhaseNodalProjection)
    assert node1.node_id == "text_node"
    assert len(node1.phase_vector) == 8
    assert pytest.approx(np.linalg.norm(node1.phase_vector), rel=1e-5) == 1.0
    assert node1.curvature > 0.0

    # Vector nodal projection
    raw_vec = [0.5, 1.2, -0.8, 2.1, 0.0, -1.5, 0.3, 0.9]
    chromatic = ChromaticVector(flux=2.0, order=0.8, entropy=0.2)
    node2 = engine.project_to_nodal_phase("vec_node", raw_vec, chromatic=chromatic)
    assert node2.node_id == "vec_node"
    assert pytest.approx(np.linalg.norm(node2.phase_vector), rel=1e-5) == 1.0


def test_field_curvature_matrix_and_causal_wave():
    engine = InformationalPhaseObservationEngine(target_dimension=8)

    n1 = engine.project_to_nodal_phase("n1", "Quantum Mechanics")
    n2 = engine.project_to_nodal_phase("n2", "Relativistic Dynamics")
    n3 = engine.project_to_nodal_phase("n3", "Informational Wave")

    network = [n1, n2, n3]
    k_matrix = engine.compute_field_curvature_matrix(network)

    assert k_matrix.shape == (3, 3)
    assert k_matrix[0, 0] == pytest.approx(n1.curvature, rel=1e-5)

    wave_history = engine.propagate_causal_wave(source_node=n1, nodes=network, steps=4)
    assert len(wave_history) == 5
    for wave in wave_history:
        assert len(wave) == 8
        assert pytest.approx(np.linalg.norm(wave), rel=1e-5) == 1.0


def test_topological_transposition():
    engine = InformationalPhaseObservationEngine(target_dimension=8)

    n1 = engine.project_to_nodal_phase("apple", "Apple Fruit Sugar Organism")
    n2 = engine.project_to_nodal_phase("quantum", "Quantum Entanglement Spin Operator")
    n3 = engine.project_to_nodal_phase("causal", "Causal Field Topological Shortcut")

    network = [n1, n2, n3]

    # Query wave close to n3
    query_wave = n3.phase_vector + 0.05 * np.random.randn(8)
    best_node, score = engine.topological_transpose(query_wave, network)

    assert best_node is not None
    assert best_node.node_id == "causal"
    assert score > 0.0


def test_proprioceptive_reconfiguration():
    engine = InformationalPhaseObservationEngine(target_dimension=8)

    initial_tension = engine.macro_tension
    impact = np.array([1.0, -0.5, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # High external friction scenario
    state1 = engine.proprioceptive_reconfigure(external_friction=0.9, structural_impact=impact)

    assert state1.macro_tension > initial_tension
    assert state1.volume_compression_ratio > 1.0
    assert state1.active_axes_count <= 8
    assert len(state1.momentum) == 8

    # Low friction scenario
    state2 = engine.proprioceptive_reconfigure(external_friction=0.1, structural_impact=np.zeros(8))
    assert state2.volume_compression_ratio < state1.volume_compression_ratio
