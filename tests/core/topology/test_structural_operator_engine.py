"""
Unit tests for StructuralOperatorToken and StructuralOperatorEngine
===================================================================
Tests non-commutativity, stereoisomer chirality (sign inversion),
100% exact reversibility, attractor relaxation, and subject-object topological inversion.
"""

import pytest
import numpy as np
from core.topology.structural_operator_engine import (
    StructuralOperatorToken,
    StructuralOperatorEngine,
    SubjectivityMode
)


def test_token_creation():
    token = StructuralOperatorToken("Health", [1.0, 0.0, 0.0])
    assert token.name == "Health"
    assert np.allclose(token.bivector, [1.0, 0.0, 0.0])
    assert pytest.approx(token.magnitude) == 1.0


def test_lie_bracket_and_non_commutativity():
    # Health (e23) and Fire (e12)
    t_health = StructuralOperatorToken("Health", [1.0, 0.0, 0.0])
    t_fire = StructuralOperatorToken("Fire", [0.0, 0.0, 1.0])

    chain_A = t_health.compose_bch(t_fire, order=2)  # Health -> Fire
    chain_B = t_fire.compose_bch(t_health, order=2)  # Fire -> Health

    # Non-commutativity
    assert not np.allclose(chain_A.bivector, chain_B.bivector)

    # e31 component (emergent 3rd dimension from Lie Bracket)
    # [e23, e12] = 2 * (e23 x e12) = 2 * (-e31) = -2 * e31
    # 0.5 * bracket = -1.0 * e31
    assert chain_A.bivector[1] == pytest.approx(-1.0)
    assert chain_B.bivector[1] == pytest.approx(+1.0)

    # Chirality / Stereoisomer Sign Inversion
    assert chain_A.bivector[1] == -chain_B.bivector[1]


def test_sandwich_product_and_100_percent_reversibility():
    t_rune = StructuralOperatorToken("Rune_R", [0.5, 0.8, -0.3])
    initial_state = np.array([1.0, 0.2, -0.4, 0.6], dtype=np.float64)

    # Forward sandwich: Psi' = R * Psi * R_dagger
    forward_state = t_rune.apply_sandwich(initial_state)
    assert not np.allclose(forward_state, initial_state)

    # Reverse sandwich: Psi = R_dagger * Psi' * R
    restored_state = t_rune.apply_inverse_sandwich(forward_state)

    # 100% Lossless state recovery
    assert np.allclose(restored_state, initial_state, atol=1e-12)


def test_engine_fold_chain_and_gauge_curvature():
    engine = StructuralOperatorEngine(boundary_capacity=5.0)
    t_health = StructuralOperatorToken("Health", [1.0, 0.0, 0.0])
    t_fire = StructuralOperatorToken("Fire", [0.0, 0.0, 1.0])

    engine.register_token(t_health)
    engine.register_token(t_fire)

    eff_operator = engine.fold_chain_bch(["Health", "Fire"], order=2)
    curvature = engine.compute_gauge_curvature(eff_operator)

    # B_eff = [1.0, -1.0, 1.0] -> ||B_eff||^2 = 3.0 -> V(F) = 0.5 * 3.0 = 1.5
    assert pytest.approx(curvature) == 1.5


def test_attractor_relaxation():
    engine = StructuralOperatorEngine()
    eff_op = StructuralOperatorToken("Eff", [1.0, 0.0, 0.0])
    initial_state = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)

    relaxed_state, energy_history = engine.relax_to_attractor(
        initial_state, eff_op, steps=20, learning_rate=0.2
    )

    # Energy should strictly decrease / relax towards attractor valley
    assert energy_history[-1] < energy_history[0]


def test_topological_inversion_drinking_vs_drowning():
    engine = StructuralOperatorEngine(boundary_capacity=5.0)
    subject_state = np.array([1.0, 0.1, 0.1, 0.1], dtype=np.float64)

    # 1. Controlled Assimilation (Drinking Water): F_ext <= boundary_capacity
    small_water_op = StructuralOperatorToken("WaterGlass", [0.5, 0.5, 0.0])
    res_absorb = engine.evaluate_topological_inversion(subject_state, small_water_op)

    assert res_absorb.mode == SubjectivityMode.ABSORB
    assert res_absorb.boundary_integrity > 0.8
    assert not res_absorb.is_dissolved()

    # 2. Overwhelming Envelopment (Drowning / System Breakdown): F_ext > boundary_capacity
    tsunami_op = StructuralOperatorToken("Tsunami", [4.0, 4.0, 4.0])  # ||B||^2 = 48 -> V(F) = 24.0 > 5.0
    res_enveloped = engine.evaluate_topological_inversion(subject_state, tsunami_op)

    assert res_enveloped.mode == SubjectivityMode.ENVELOPED
    assert res_enveloped.boundary_integrity == 0.0
    assert res_enveloped.is_dissolved()
