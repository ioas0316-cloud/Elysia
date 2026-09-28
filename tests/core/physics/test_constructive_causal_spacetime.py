"""
Unit tests for core/physics/constructive_causal_spacetime.py
"""

import numpy as np
import pytest
from core.physics.constructive_causal_spacetime import (
    ConstructiveSpacetimeAxis,
    ConstructiveLogicDiscriminator,
    HierarchicalScaleCoupler,
)


def test_constructive_spacetime_axis():
    axis = ConstructiveSpacetimeAxis(dimension=4)
    state = axis.apply_causal_impulse(np.eye(4) * 0.5, dt=0.1)

    assert "algebraic" in state
    assert "geometric" in state
    assert "causal" in state

    assert state["algebraic"]["impedance"] == 0.0
    assert state["geometric"]["effective_radius"] > 0.0
    assert state["causal"]["impedance"] >= 0.0


def test_constructive_logic_discriminator():
    discriminator = ConstructiveLogicDiscriminator(feature_dim=8)
    wave_a = np.ones(8) * 2.0
    wave_b = np.ones(8) * 0.5
    wave_c = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float64)

    # Parallel vectors should have 0 residual impedance
    res_parallel = discriminator.measure_residual_impedance(wave_a, wave_b)
    assert res_parallel["residual_impedance_delta_z"] < 1e-4
    assert res_parallel["is_isomorphic"]

    # Orthogonal vectors should have high residual impedance
    res_orthogonal = discriminator.measure_residual_impedance(wave_a, wave_c)
    assert res_orthogonal["residual_impedance_delta_z"] > 0.5
    assert not res_orthogonal["is_isomorphic"]


def test_hierarchical_scale_coupler():
    coupler = HierarchicalScaleCoupler(num_micro_nodes=8, dim=4)
    sb_res = coupler.trigger_spontaneous_symmetry_breaking(perturbation_strength=1.0)
    assert "bit_states" in sb_res

    pl_res = coupler.execute_phase_lock_knotting(lock_threshold=0.5)
    assert pl_res["conservation"]["maintained"]
    assert pl_res["macro_mass"] >= 1.0
