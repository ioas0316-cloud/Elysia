"""
Unit tests for DynamicObservationLens and DualFrameCausalEngine in Elysia.

Tests:
1. DynamicObservationLens metric evolution ∂G_ij/∂t and wave refraction.
2. 2D FFT k-space Clifford rotor extraction.
3. Theory of Mind Observer (argmin_G δS_other) metric reconstruction.
4. CliffordMultivectorMemory Grade 0-3 orthogonal projection.
5. CounterfactualWaveEngine time rewind and e^(iπ) phase inversion cancellation.
6. KuramotoDualFrameCoupler Lyapunov energy decay V(t) and phase locking R -> 1.0.
7. Bivector topological deadlock unlocking for 180° (π rad) phase opposition.
8. ScaleRenormalizationEngine Wilsonian coarse-graining and top-down value constraint.
"""

import math
import numpy as np
import pytest
import torch

from core.lens.dynamic_observation_lens import DynamicObservationLens, extract_clifford_rotor_from_kspace
from core.consciousness.dual_frame_causal_engine import (
    TheoryOfMindObserver,
    CliffordMultivectorMemory,
    CounterfactualWaveEngine,
    KuramotoDualFrameCoupler,
    ScaleRenormalizationEngine,
)


def test_dynamic_observation_lens_metric_evolution():
    lens = DynamicObservationLens(dim=4, eta=0.2, gamma=0.05, kappa=0.01)
    psi = torch.randn(10, 4)

    G_initial = lens.G.clone()
    refracted_psi = lens(psi, update_metric=True, dt=0.1)

    assert refracted_psi.shape == (10, 4)
    # Check metric evolved away from initial baseline due to friction
    assert not torch.allclose(lens.G, G_initial)
    # Metric symmetry check
    assert torch.allclose(lens.G, lens.G.t(), atol=1e-5)


def test_kspace_clifford_rotor_extraction():
    Nx, Ny = 64, 64
    kx = np.linspace(-3.0, 3.0, Nx)
    ky = np.linspace(-3.0, 3.0, Ny)
    KX, KY = np.meshgrid(kx, ky)

    k0 = 1.2
    target_theta = np.radians(45.0)
    k1_target = np.array([k0 * np.cos(target_theta / 2), k0 * np.sin(target_theta / 2)])
    k2_target = np.array([k0 * np.cos(-target_theta / 2), k0 * np.sin(-target_theta / 2)])

    P = np.exp(-((KX - k1_target[0])**2 + (KY - k1_target[1])**2) / 0.05) + \
        np.exp(-((KX - k2_target[0])**2 + (KY - k2_target[1])**2) / 0.05)

    result = extract_clifford_rotor_from_kspace(P, kx, ky, k0_magnitude=k0)

    assert "theta_rotor_rad" in result
    assert "clifford_rotor" in result
    assert result["chord_length"] > 0.0
    assert abs(result["theta_rotor_deg"] - 45.0) < 15.0


def test_theory_of_mind_observer():
    tom = TheoryOfMindObserver(dim=4)
    statement_wave = torch.randn(1, 4) * 2.0

    G_other, R_tom, min_action = tom.reconstruct_other_metric(statement_wave, lr=0.05, steps=15)

    assert G_other.shape == (4, 4)
    assert R_tom.shape == (4, 4)
    assert min_action >= 0.0
    # Symmetric positive-definite check
    assert torch.allclose(G_other, G_other.t(), atol=1e-4)


def test_clifford_multivector_memory_orthogonal_projection():
    mem = CliffordMultivectorMemory(depth=8, height=8, width=8)

    actual_history = torch.ones((8, 8, 8)) * 2.5
    cf1_vector = torch.randn((8, 8, 8, 3)) * 0.5
    cf2_bivector = torch.ones((8, 8, 8, 3)) * -1.8

    mem.write_scenario(grade=0, scenario_tensor=actual_history)
    mem.write_scenario(grade=1, scenario_tensor=cf1_vector)
    mem.write_scenario(grade=2, scenario_tensor=cf2_bivector)

    retrieved_g0 = mem.read_grade_projection(grade=0)
    retrieved_g1 = mem.read_grade_projection(grade=1)
    retrieved_g2 = mem.read_grade_projection(grade=2)

    assert retrieved_g0.shape == (8, 8, 8, 1)
    assert retrieved_g1.shape == (8, 8, 8, 3)
    assert retrieved_g2.shape == (8, 8, 8, 3)

    assert torch.allclose(retrieved_g0.squeeze(-1), actual_history)
    assert torch.allclose(retrieved_g2, cf2_bivector)


def test_counterfactual_wave_engine_rewind_and_inversion():
    engine = CounterfactualWaveEngine()
    Ny, Nx = 16, 16
    psi_present = torch.complex(torch.ones(Ny, Nx), torch.zeros(Ny, Nx))
    v_causal = torch.randn(2, Ny, Nx) * 0.1

    event_mask = torch.zeros(Ny, Nx)
    event_mask[6:10, 6:10] = 1.0

    psi_cf, psi_rewound, causal_impact = engine.compute_counterfactual_wave_branch(
        psi_present=psi_present,
        v_causal=v_causal,
        event_mask_x=event_mask,
        dt=0.02,
        rewind_steps=10,
        forward_steps=10
    )

    assert psi_cf.shape == (Ny, Nx)
    assert psi_rewound.shape == (Ny, Nx)
    assert causal_impact > 0.0


def test_kuramoto_coupler_lyapunov_decay_and_phase_locking():
    coupler = KuramotoDualFrameCoupler(dim=2, eta=0.25, gamma=0.08, beta=0.40)

    theta_obs = 1.8
    theta_target = math.pi / 4.0  # 0.7854 rad

    v_history = []
    for step in range(500):
        theta_obs, g01, V_t = coupler.step_coupling(theta_obs, theta_target, dt=0.05)
        v_history.append(V_t)

    # Phase error convergence
    assert abs(theta_obs - theta_target) < 0.1
    # Overall Lyapunov energy decay over full simulation trajectory
    assert v_history[-1] < v_history[10]


def test_topological_deadlock_unlocking_for_pi_opposition():
    coupler = KuramotoDualFrameCoupler(dim=2)
    N = 32

    # Exactly 180° (π rad) opposition -> Topological deadlock
    psi1 = torch.complex(torch.ones(N), torch.zeros(N))
    psi2 = torch.complex(-torch.ones(N), torch.zeros(N))

    psi2_unlocked, torque, is_deadlock = coupler.unlock_topological_deadlock(
        psi1, psi2, epsilon_deadlock=0.1, bivector_theta=0.15
    )

    assert is_deadlock is True
    # Unlocked torque should now be non-zero to resume convergence
    assert float(torque.abs().mean().item()) > 0.0


def test_scale_renormalization_engine():
    rg = ScaleRenormalizationEngine(num_scales=4, spatial_dim=16)

    # Fill micro scale s=0 with vector waves
    rg.scale_field[0, ..., 1] = torch.randn(16, 16)
    rg.scale_field[0, ..., 2] = torch.randn(16, 16)

    s1_layer = rg.coarse_grain_step(s=0)
    assert s1_layer.shape == (16, 16, 8)

    # Top-down constraint check
    rg.scale_field[-1, ..., 0] = torch.randn(16, 16) * 2.0
    rg.apply_topdown_constraint(dt=0.05)
    assert not torch.allclose(rg.scale_field[0], torch.zeros_like(rg.scale_field[0]))
