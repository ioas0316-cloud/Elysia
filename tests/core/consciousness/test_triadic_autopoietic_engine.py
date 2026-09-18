"""
Unit tests for Triadic Autopoietic Engine (Triadic Integration Loop).
Verifies Homeostatic Valence, Active Inference Action Spillover,
Attractor Memory Topology, and Diagnostic Metrics.
"""

import numpy as np
import pytest
from core.consciousness.triadic_autopoietic_engine import (
    TriadicAutopoieticEngine,
    quaternion_multiply,
    quaternion_inverse,
    normalize_quaternion,
)


def test_quaternion_math_utilities():
    q1 = normalize_quaternion(np.array([1.0, 2.0, 3.0, 4.0]))
    q_inv = quaternion_inverse(q1)
    res = quaternion_multiply(q1, q_inv)
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(res, identity, atol=1e-6)


def test_homeostatic_strain_and_valence():
    target_h = np.array([1.0, 0.5, 0.0])
    engine = TriadicAutopoieticEngine(homeo_dim=3, target_homeo=target_h, seed=123)

    # Initial strain when h == target_h
    strain_at_target = engine.compute_homeostatic_strain(target_h)
    assert strain_at_target == pytest.approx(0.0, abs=1e-6)

    # Displaced state
    displaced_h = np.array([2.0, 1.5, 1.0])
    strain_displaced = engine.compute_homeostatic_strain(displaced_h)
    assert strain_displaced > 0.0

    # Test Valence calculation (moving towards target -> positive valence)
    engine.prev_D_H = strain_displaced
    current_strain = strain_displaced - 0.5
    valence = engine.compute_valence(current_strain, dt=0.1)
    assert valence > 0.0  # Strain decreased => Positive Valence

    # Moving away from target -> negative valence
    current_strain_worse = strain_displaced + 0.5
    valence_negative = engine.compute_valence(current_strain_worse, dt=0.1)
    assert valence_negative < 0.0  # Strain increased => Negative Valence


def test_dynamic_relaxation_and_thermal_agitation():
    engine = TriadicAutopoieticEngine(gamma_0=0.5, gamma_max=2.0, T_0=0.01, T_max=0.5)

    gamma_pos = engine.get_dynamic_relaxation(valence=2.0)
    gamma_neg = engine.get_dynamic_relaxation(valence=-2.0)
    assert gamma_pos > gamma_neg

    T_pos = engine.get_effective_thermal_agitation(valence=2.0)
    T_neg = engine.get_effective_thermal_agitation(valence=-2.0)
    assert T_neg > T_pos


def test_action_spillover_threshold():
    # Set theta_action low to force spillover vs high
    engine = TriadicAutopoieticEngine(theta_action=0.01, num_nodes=4, action_dim=3, seed=42)

    # Boundary signal with large mismatch
    q_bound_orthogonal = np.array([0.0, 1.0, 0.0, 0.0])
    telemetry = engine.step(q_bound_orthogonal, dt=0.05)

    # High tension should trigger non-zero action
    assert telemetry["T_phase"] > 0.01
    assert np.any(telemetry["action"] != 0.0)

    # Now with high threshold -> no action spillover
    engine_high_thresh = TriadicAutopoieticEngine(theta_action=10.0, num_nodes=4, action_dim=3, seed=42)
    telemetry_no_action = engine_high_thresh.step(q_bound_orthogonal, dt=0.05)
    np.testing.assert_allclose(telemetry_no_action["action"], np.zeros(3))


def test_attractor_memory_anchoring():
    engine = TriadicAutopoieticEngine(num_nodes=4, seed=42)

    target_attractor = normalize_quaternion(np.random.randn(4, 4))
    engine.register_attractor(target_attractor)

    assert len(engine.attractor_basins) == 1

    # Force step and verify attractor force calculation
    force = engine.compute_attractor_force()
    assert force.shape == (4, 4)


def test_diagnostics_and_winding_number():
    engine = TriadicAutopoieticEngine(num_nodes=8, seed=42)
    q_bound = np.array([1.0, 0.0, 0.0, 0.0])

    telemetry = engine.step(q_bound, dt=0.05)
    diag = telemetry["diagnostics"]

    assert "R_H" in diag
    assert "rho_D" in diag
    assert "dot_S_int" in diag
    assert 0.0 <= diag["R_H"] <= 1.0
    assert diag["rho_D"] >= 0.0
    assert diag["dot_S_int"] >= 0.0
    assert isinstance(telemetry["winding_number"], float)
