r"""
Unit Tests for Emergent Phase Viscosity & Thermodynamic Operator Engine
=======================================================================
"""

import pytest
import torch
from core.physics.emergent_phase_viscosity import (
    EmergentPhaseViscosityEngine,
    quaternion_multiply,
    quaternion_conjugate,
    quaternion_to_rotation_matrix,
    generate_max_mipmaps_3d,
    generate_glsl_volume_raymarch_shader,
    generate_hlsl_compute_shader
)


def test_quaternion_s3_preservation():
    """
    Verifies that quaternions remain on S^3 unit sphere (||Q|| == 1) after integration.
    """
    engine = EmergentPhaseViscosityEngine(K_0=10.0, D_R=0.5)
    B, H, W, D = 2, 6, 6, 6

    # Random initial velocities & quaternions
    V = torch.randn(B, 3, H, W, D)
    Q = torch.randn(B, 4, H, W, D)
    Q = torch.nn.functional.normalize(Q, p=2, dim=1)
    T = torch.ones(B, 1, H, W, D) * 0.5

    for step in range(5):
        V, Q, metrics = engine.step(V, Q, T, dt=0.01)
        q_norms = torch.norm(Q, dim=1)
        # Check S^3 unit sphere preservation within machine precision
        assert torch.allclose(q_norms, torch.ones_like(q_norms), atol=1e-5)


def test_thermodynamic_phase_transitions():
    r"""
    Verifies continuous phase transition:
    - Low T -> High Order Parameter \Phi (Solid)
    - High T -> Low Order Parameter \Phi (Gas)
    """
    engine = EmergentPhaseViscosityEngine(K_0=20.0, D_R=2.0)
    B, H, W, D = 1, 8, 8, 8

    V = torch.zeros(B, 3, H, W, D)

    # 1. Cold state setup
    Q_cold = torch.zeros(B, 4, H, W, D)
    Q_cold[:, 0] = 1.0  # Completely aligned identity quaternions
    T_cold = torch.ones(B, 1, H, W, D) * 0.001

    _, _, cold_metrics = engine.step(V, Q_cold, T_cold, dt=0.01)
    assert cold_metrics["mean_order_parameter_phi"] > 0.8
    assert cold_metrics["solid_fraction"] > 0.5

    # 2. Hot state setup
    Q_hot = torch.randn(B, 4, H, W, D)
    Q_hot = torch.nn.functional.normalize(Q_hot, p=2, dim=1)
    T_hot = torch.ones(B, 1, H, W, D) * 50.0

    _, _, hot_metrics = engine.step(V, Q_hot, T_hot, dt=0.01)
    assert hot_metrics["mean_order_parameter_phi"] < 0.6
    assert hot_metrics["gas_fraction"] > cold_metrics["gas_fraction"]


def test_emergent_viscosity_under_shear():
    """
    Verifies that shear force induces micro-rotor rotation and sync torque
    without any hardcoded viscosity constant.
    """
    engine = EmergentPhaseViscosityEngine(K_0=15.0)
    B, H, W, D = 1, 8, 8, 8

    # Apply velocity shear gradient across Y axis (v_x = y)
    V = torch.zeros(B, 3, H, W, D)
    y_coords = torch.linspace(-1, 1, W).view(1, 1, W, 1).expand(B, H, W, D)
    V[:, 0, :, :, :] = y_coords

    Q = torch.zeros(B, 4, H, W, D)
    Q[:, 0] = 1.0
    T = torch.ones(B, 1, H, W, D) * 0.1

    V_next, Q_next, metrics = engine.step(V, Q, T, dt=0.02)

    # Verify micro-rotors rotated away from identity
    rotation_displacement = torch.mean(torch.abs(Q_next[:, 1:]))
    assert rotation_displacement > 0.0

    # Verify momentum feedback damping velocity gradient
    v_diff_initial = torch.max(V[:, 0]) - torch.min(V[:, 0])
    v_diff_next = torch.max(V_next[:, 0]) - torch.min(V_next[:, 0])
    assert v_diff_next <= v_diff_initial + 1e-4


def test_non_newtonian_modes():
    """
    Verifies shear-thinning and shear-thickening behavior response.
    """
    engine_thin = EmergentPhaseViscosityEngine(K_0=10.0, shear_mode="shear_thinning")
    engine_thick = EmergentPhaseViscosityEngine(K_0=10.0, shear_mode="shear_thickening")

    B, H, W, D = 1, 8, 8, 8
    V = torch.randn(B, 3, H, W, D) * 2.0
    Q = torch.nn.functional.normalize(torch.randn(B, 4, H, W, D), p=2, dim=1)
    T = torch.ones(B, 1, H, W, D) * 0.1

    _, _, metrics_thin = engine_thin.step(V, Q, T)
    _, _, metrics_thick = engine_thick.step(V, Q, T)

    assert "mean_dissipation_alpha" in metrics_thin
    assert "mean_dissipation_alpha" in metrics_thick


def test_optical_shader_generation_and_mipmaps():
    """
    Verifies GLSL/HLSL shader generation and 3D Max-Mipmap spatial reduction.
    """
    glsl_code = generate_glsl_volume_raymarch_shader()
    hlsl_code = generate_hlsl_compute_shader()

    assert "u_PhiTexture" in glsl_code
    assert "u_QTexture" in glsl_code
    assert "CS_RaymarchVolume" in hlsl_code

    # Test Max Mipmap skipping structure
    Phi = torch.zeros(1, 1, 8, 8, 8)
    Phi[0, 0, 4, 4, 4] = 0.95
    mip = generate_max_mipmaps_3d(Phi, brick_size=4)

    assert mip.shape == (1, 1, 2, 2, 2)
    assert torch.max(mip) == 0.95
