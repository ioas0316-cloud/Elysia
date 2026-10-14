"""
Unit Tests for Prismatic Axis Spectrum Engine & Multi-Layered Wave Memory Stack
================================================================================

Tests the 4 fundamental capabilities of the Prismatic Spectrum Paradigm:
1. Continuous wave field evaluation and threshold boundary projection (0/1 emergence).
2. Prismatic scale/frequency bifurcation and label-free relationship determination.
3. Multi-layered wave memory stack interlocked chromatic persistence.
4. Phase resonance search (Wave collapse) and Phase-locked focus with noise suppression.
"""

import pytest
import numpy as np
from core.physics.prismatic_axis_spectrum import (
    PrismaticAxisSpectrumEngine,
    MultiLayeredWaveMemoryStack,
    SpectralNode
)


def test_continuous_wave_and_threshold_projection():
    engine = PrismaticAxisSpectrumEngine(base_frequency=10.0, num_bifurcation_levels=2)
    spatial_x = np.linspace(-10.0, 10.0, 200)

    # Evaluate continuous wave
    wave = engine.evaluate_continuous_wave(t=0.0, spatial_x=spatial_x)
    assert len(wave) == 200
    assert np.std(wave) > 0.0  # Wave is active and non-trivial

    # Project to discrete binary 0/1
    proj = engine.project_discrete_states(wave, threshold=0.0)
    binary = proj["discrete_binary"]

    assert set(np.unique(binary)).issubset({0, 1})
    assert proj["zero_crossing_count"] > 0
    assert len(proj["continuous_margin"]) == 200


def test_prismatic_bifurcation_and_hierarchy():
    engine = PrismaticAxisSpectrumEngine(base_frequency=10.0, num_bifurcation_levels=3, branches_per_node=2)

    # Check total nodes created: 1 (root) + 2 + 4 + 8 = 15
    assert len(engine.nodes) == 15
    assert "root_white_light" in engine.nodes
    assert "root_white_light_b1" in engine.nodes
    assert "root_white_light_b1_b1" in engine.nodes

    # Derive relationship between child branches without statistical labeling
    rel = engine.determine_prismatic_relationship("root_white_light_b1", "root_white_light_b2")
    assert rel["common_root"] == "root_white_light"
    assert "phase_difference_rad" in rel
    assert "chromatic_similarity" in rel
    assert 0.0 <= rel["structural_affinity"] <= 1.0


def test_multi_layered_wave_memory_stack():
    stack = MultiLayeredWaveMemoryStack(spatial_dim=32)
    assert len(stack.layers) == 3

    node = SpectralNode(
        node_id="test_node",
        parent_id=None,
        frequency=15.0,
        phase=np.pi / 4.0,
        amplitude=1.0,
        threshold=0.0,
        chromatic_signature=np.array([0.6, 0.3, 0.1]),
        depth=1
    )

    stack.imprint_node(node)

    field_full = stack.compute_cross_relational_field()
    assert field_full.shape == (32, 32)
    assert np.max(np.abs(field_full)) > 0.0

    # Gate off the Red layer
    stack.layers["red_flux_layer"].set_active_state(False)
    field_gated = stack.compute_cross_relational_field()

    # Field should remain structurally intact via Blue and Yellow persistence
    assert field_gated.shape == (32, 32)
    assert np.max(np.abs(field_gated)) > 0.0


def test_phase_resonance_search():
    engine = PrismaticAxisSpectrumEngine(base_frequency=10.0, num_bifurcation_levels=2)

    # Search for a frequency close to root
    results = engine.execute_phase_resonance_search(query_frequency=10.0, query_phase=0.0)

    assert len(results) == len(engine.nodes)
    top_result = results[0]

    # Top result should be the root white light node due to perfect match
    assert top_result.target_node_id == "root_white_light"
    assert top_result.resonance_score > 0.0
    assert top_result.constructive_energy > top_result.destructive_cancellation


def test_phase_locked_focus_and_noise_suppression():
    engine = PrismaticAxisSpectrumEngine(base_frequency=10.0, num_bifurcation_levels=2)

    target_id = "root_white_light_b1"
    initial_target_amp = engine.nodes[target_id].amplitude

    focus_res = engine.apply_phase_locked_focus(
        target_node_id=target_id,
        boost_factor=2.0,
        noise_suppression_factor=0.1
    )

    # Target node amplitude boosted
    assert engine.nodes[target_id].amplitude == initial_target_amp * 2.0
    assert focus_res["focused_target"] == target_id
    assert "energy_efficiency_ratio" in focus_res
