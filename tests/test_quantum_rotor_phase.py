"""
tests/test_quantum_rotor_phase.py

Test suite for 5D Clifford Rotor & Quantum Phase Transition Dynamics
"""

import pytest
import numpy as np

from elysia_core import (
    Clifford5DRotorEngine,
    LandauGinzburgPotentialEngine,
    QuantumRotorPhaseIntegrator
)


def test_clifford_5d_rotor_generator_skew_symmetry():
    rotor_engine = Clifford5DRotorEngine(dim=5)
    for idx in range(10):
        J = rotor_engine.get_bivector_generator(idx)
        # Check skew-symmetry: J^T = -J
        assert np.allclose(J.T, -J)
        assert J.shape == (5, 5)


def test_clifford_5d_rotor_norm_preservation():
    rotor_engine = Clifford5DRotorEngine(dim=5)
    v_5d = np.array([1.0, 2.0, 0.5, -1.2, 0.8])
    initial_norm = np.linalg.norm(v_5d)

    # Random 10D bivector angle vector
    np.random.seed(42)
    theta_vector = np.random.randn(10) * 0.5

    Omega = rotor_engine.build_bivector_omega(theta_vector)
    R = rotor_engine.exponential_map(Omega)
    v_rotated = rotor_engine.sandwich_transform(v_5d, R)

    transformed_norm = np.linalg.norm(v_rotated)
    assert np.isclose(initial_norm, transformed_norm, atol=1e-6)


def test_landau_ginzburg_bifurcation_and_equilibrium():
    lg_engine = LandauGinzburgPotentialEngine(lambda_c=1.0, beta=0.5)

    # Sub-critical control parameter (lambda < lambda_c)
    left_sub, right_sub = lg_engine.get_equilibrium_angles(control_lambda=0.8)
    assert left_sub == 0.0 and right_sub == 0.0

    # Super-critical control parameter (lambda > lambda_c) -> Symmetry breaking
    left_super, right_super = lg_engine.get_equilibrium_angles(control_lambda=1.5)
    expected_theta = np.sqrt((1.5 - 1.0) / 0.5)  # sqrt(0.5 / 0.5) = 1.0
    assert np.isclose(right_super, expected_theta)
    assert np.isclose(left_super, -expected_theta)


def test_wkb_quantum_tunneling_probability():
    lg_engine = LandauGinzburgPotentialEngine(lambda_c=1.0, beta=0.5, hbar=1.0)

    # Sub-critical -> 0.0 probability
    prob_sub = lg_engine.compute_wkb_tunneling_probability(control_lambda=0.8)
    assert prob_sub == 0.0

    # Super-critical -> Non-zero tunneling probability in [0, 1]
    prob_super = lg_engine.compute_wkb_tunneling_probability(control_lambda=1.5)
    assert 0.0 < prob_super <= 1.0


def test_quantum_rotor_phase_integrator_step():
    integrator = QuantumRotorPhaseIntegrator(lambda_c=1.0, tunneling_threshold=0.01)

    # Step in super-critical regime
    state_res = integrator.step(control_lambda=1.5, external_torque_10d=np.ones(10) * 0.01, dt=0.05)

    assert state_res.is_bifurcated is True
    assert state_res.state_vector.shape == (5,)
    assert np.isclose(np.linalg.norm(state_res.state_vector), 1.0)
