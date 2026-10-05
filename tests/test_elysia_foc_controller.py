import numpy as np
import pytest
from elysia_engine.flux_weakening_controller import (
    ElysiaFluxWeakeningController,
    ElysiaPIClosedLoopController
)

def test_flux_weakening_factor_normal():
    controller = ElysiaFluxWeakeningController(vram_limit_mb=3072.0, safety_margin=0.15)
    # Under safety margin (2611.2MB threshold), scaling should be 1.0
    gamma_d = controller.compute_flux_weakening_factor(current_vram_mb=2000.0)
    assert gamma_d == 1.0

def test_flux_weakening_factor_high_vram():
    controller = ElysiaFluxWeakeningController(vram_limit_mb=3072.0, safety_margin=0.15, lambda_gain=0.005)
    # Over threshold (2800MB)
    gamma_d = controller.compute_flux_weakening_factor(current_vram_mb=2800.0)
    assert gamma_d < 1.0
    assert gamma_d >= controller.min_flux_floor

def test_apply_foc_scaling():
    controller = ElysiaFluxWeakeningController(vram_limit_mb=3072.0, safety_margin=0.15)
    d_context = np.array([1.0, 2.0, 3.0])
    q_momentum = np.array([4.0, 5.0, 6.0])

    scaled_d, preserved_q = controller.apply_foc_scaling(d_context, q_momentum, current_vram_mb=2800.0)

    # Q-axis momentum must be 100% preserved
    np.testing.assert_array_equal(preserved_q, q_momentum)
    # D-axis context must be scaled down
    assert np.all(scaled_d < d_context)

def test_clarke_park_transformation_reversibility():
    controller = ElysiaFluxWeakeningController()

    # Generate 3-phase sine waves
    t = np.linspace(0, 2*np.pi, 100)
    angles = t

    a = np.sin(t)
    b = np.sin(t - 2*np.pi/3)
    c = np.sin(t + 2*np.pi/3)

    input_abc = np.stack([a, b, c], axis=-1)

    dq0 = controller.clarke_park_transform(input_abc, angles)
    assert dq0.shape == (100, 3)

    # Check zero component is near 0 for balanced 3-phase system
    np.testing.assert_allclose(dq0[:, 2], 0.0, atol=1e-5)

    # Check inverse transform reconstructs original abc
    reconstructed_abc = controller.inverse_park_clarke_transform(dq0, angles)
    np.testing.assert_allclose(reconstructed_abc, input_abc, atol=1e-4)

def test_pi_closed_loop_controller():
    pi_ctrl = ElysiaPIClosedLoopController(kp=10.0, ki=0.1, dt=0.01)
    target_q = 10.0
    current_q = 0.0

    for _ in range(600):
        current_q, error = pi_ctrl.step(target_q, current_q)

    # Current Q should converge to target Q
    assert abs(current_q - target_q) < 0.01
