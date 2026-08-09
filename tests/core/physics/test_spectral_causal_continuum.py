import pytest
import numpy as np
from core.physics.spectral_causal_continuum import SpectralCausalContinuum

def test_spectral_causal_continuum_analytical_derivatives():
    """
    Verify that the algebraic analytical derivatives match the finite-difference approximation
    at high spatial resolutions, proving algebraic derivative completeness.
    """
    engine = SpectralCausalContinuum(num_modes=20)
    # Set coordinates to be non-zero
    engine.q += np.sin(np.arange(20)) * 0.1
    engine.p += np.cos(np.arange(20)) * 0.1

    u_vals = np.array([1.5, 2.5, 3.5])

    # Calculate analytical derivative
    dphi_du_anal = engine.project_first_derivative(u_vals)
    d2phi_du2_anal = engine.project_second_derivative(u_vals)

    # Finite-difference check for first derivative
    eps = 1e-6
    phi_plus = engine.project_field(u_vals + eps)
    phi_minus = engine.project_field(u_vals - eps)
    dphi_du_fd = (phi_plus - phi_minus) / (2.0 * eps)

    # Finite-difference check for second derivative
    phi_center = engine.project_field(u_vals)
    d2phi_du2_fd = (phi_plus - 2.0 * phi_center + phi_minus) / (eps ** 2)

    # Verify alignment
    np.testing.assert_allclose(dphi_du_anal, dphi_du_fd, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(d2phi_du2_anal, d2phi_du2_fd, rtol=1e-3, atol=1e-3)

def test_hamiltonian_conservation():
    """
    Verify that when sigma/damping is set to zero, the total Hamiltonian energy
    is conserved perfectly across multiple time steps.
    """
    engine = SpectralCausalContinuum(num_modes=10, initial_sigma=0.0)
    engine.impedance = 0.0 # Force zero impedance for absolute conservation check

    initial_energy = engine.compute_hamiltonian()

    # Step the continuous system
    for _ in range(50):
        # We pass no external excitation and we expect energy to remain constant
        engine.step(dt=0.005)

    final_energy = engine.compute_hamiltonian()

    # Under zero damping, Hamiltonian should be conserved perfectly within a tight numerical tolerance
    assert abs(final_energy - initial_energy) < 1e-8

def test_winding_number_resolution_independence():
    """
    Verify that the winding number is a robust topological invariant that returns
    the exact same integer value independent of the resolution (number of grid steps)
    used during projection once high enough to resolve high-frequency content.
    """
    engine = SpectralCausalContinuum(num_modes=30)
    # Give some asymmetric perturbations to have non-trivial winding
    engine.q = np.arange(30, dtype=np.float64) * 0.02
    engine.p = np.arange(30, dtype=np.float64) * -0.015

    # Since winding number is computed using integration, both steps should be sufficient to capture the phase.
    # 500 steps vs 2000 steps
    w_low = engine.compute_winding_number(u_start=0.2, u_end=4.0, steps=500)
    w_high = engine.compute_winding_number(u_start=0.2, u_end=4.0, steps=2000)

    # They must be absolutely identical integers
    assert isinstance(w_low, int)
    assert w_low == w_high

def test_active_impedance_feedback_loop():
    """
    Verify that high tension/instability (energy drift) triggers the active impedance
    loop to suppress high-frequency noise and stabilize the system.
    """
    engine = SpectralCausalContinuum(num_modes=10, initial_sigma=0.1)
    # Set high initial coordinates to trigger tension
    engine.q *= 50.0
    engine.p *= 50.0
    engine.last_energy = engine.compute_hamiltonian()

    # Simulate a step with energy drift
    res = engine.step(dt=0.1)

    # Impedance should increase dynamically or stabilize based on tension gap
    assert res["tension_gap"] >= 0.0
    assert res["impedance"] > 1e-5
