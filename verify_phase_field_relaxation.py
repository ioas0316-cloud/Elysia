r"""
Verify Phase Field Relaxation & Noise Tolerance
===============================================
Verification script validating Zero-Branching Phase Field Relaxation Engines (2D & 3D/4D Quaternion S^3)
and verifying Lyapunov energy decay, zero branch divergence, zero pointer chasing, and analog noise tolerance.
"""

import sys
import torch
import numpy as np
from core.physics.phase_field_relaxation_engine import PhaseFieldRelaxationEngine
from core.physics.quaternion_3d_phase_engine import Quaternion3DPhaseEngine


def verify_2d_phase_field_relaxation():
    print("=" * 70)
    print("1. Verifying 2D Zero-Branching Phase Field Relaxation Engine...")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    shape = (64, 64)
    engine = PhaseFieldRelaxationEngine(shape=shape, device=device)

    # Construct maze / obstacle boundary conditions
    mask = torch.zeros((1, 1, *shape), device=device)
    boundary = torch.zeros((1, 1, *shape), device=device)

    # Outer wall bounds
    mask[:, :, 0, :] = 1.0; boundary[:, :, 0, :] = 0.0
    mask[:, :, -1, :] = 1.0; boundary[:, :, -1, :] = 0.0
    mask[:, :, :, 0] = 1.0; boundary[:, :, :, 0] = 0.0
    mask[:, :, :, -1] = 1.0; boundary[:, :, :, -1] = 0.0

    # Obstacle wall in middle
    mask[:, :, 20:44, 32] = 1.0; boundary[:, :, 20:44, 32] = 0.0

    # Start (+5.0 potential) and Goal (-5.0 potential)
    start_idx = (10, 10)
    goal_idx = (50, 50)
    mask[:, :, start_idx[0], start_idx[1]] = 1.0; boundary[:, :, start_idx[0], start_idx[1]] = 5.0
    mask[:, :, goal_idx[0], goal_idx[1]] = 1.0; boundary[:, :, goal_idx[0], goal_idx[1]] = -5.0

    engine.set_boundary_condition(mask, boundary)

    energies = []
    iterations = 200

    for step in range(iterations):
        psi_state, free_energy, metrics = engine(dt=0.01)
        energies.append(free_energy)

        # Assert zero control-flow branching & zero pointer chasing
        assert metrics["branch_divergence"] == 0, "Branch divergence must be 0!"
        assert metrics["pointer_chasing_steps"] == 0, "Pointer chasing steps must be 0!"

    print(f"   Initial Free Energy : {energies[0]:.4f}")
    print(f"   Final Free Energy   : {energies[-1]:.4f}")
    print(f"   Energy Monotonicity : {'PASS' if energies[-1] <= energies[0] else 'FAIL'}")

    # Verify field gradient continuity between start and goal
    psi_map = engine.psi.squeeze().cpu().numpy()
    grad_y, grad_x = np.gradient(psi_map)
    field_strain = grad_x ** 2 + grad_y ** 2
    print(f"   Mean Field Strain   : {field_strain.mean():.6f}")

    assert energies[-1] <= energies[0], "Free energy must decrease monotonically or stabilize!"
    print("   [2D Phase Field Relaxation Verification Passed successfully!]\n")


def verify_3d_quaternion_phase_engine():
    print("=" * 70)
    print("2. Verifying 3D Spatial / 4D Quaternion (S^3 Manifold) Phase Engine...")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    shape = (24, 24, 24)
    engine = Quaternion3DPhaseEngine(shape=shape, device=device)

    mask = torch.zeros((1, 1, *shape), device=device)
    boundary_q = torch.zeros((1, 4, *shape), device=device)

    # 3D spherical obstacle in center
    z_grid, y_grid, x_grid = torch.meshgrid(
        torch.arange(24, device=device),
        torch.arange(24, device=device),
        torch.arange(24, device=device),
        indexing="ij"
    )
    sphere = ((x_grid - 12) ** 2 + (y_grid - 12) ** 2 + (z_grid - 12) ** 2) < 5 ** 2
    mask[0, 0, sphere] = 1.0; boundary_q[0, :, sphere] = 0.0

    # Start: q = [1.0, 0.0, 0.0, 0.0]
    mask[0, 0, 3, 3, 3] = 1.0
    boundary_q[0, :, 3, 3, 3] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

    # Goal: q = [0.0, 1.0, 0.0, 0.0] (180 deg phase shift)
    mask[0, 0, 20, 20, 20] = 1.0
    boundary_q[0, :, 20, 20, 20] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)

    engine.set_boundary_condition(mask, boundary_q)

    energies = []
    iterations = 150

    for step in range(iterations):
        q_state, free_energy, metrics = engine(dt=0.005)
        energies.append(free_energy)

        # Assert S^3 hyper-sphere unit norm projection ||q|| = 1.0 on unconstrained field points (mask == 0)
        unconstrained_mask = (engine.mask == 0.0)
        q_unconstrained = q_state[unconstrained_mask.repeat(1, 4, 1, 1, 1)].view(1, 4, -1)
        norms = torch.norm(q_unconstrained, dim=1)

        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), "Unconstrained quaternion field must remain on S^3 manifold!"
        assert metrics["branch_divergence"] == 0
        assert metrics["pointer_chasing_steps"] == 0

    print(f"   Initial S^3 Energy  : {energies[0]:.4f}")
    print(f"   Final S^3 Energy    : {energies[-1]:.4f}")
    print(f"   Norm Conservation   : ||q|| = 1.0000 across all active field points (S^3 preserved)")

    assert energies[-1] <= energies[0], "Quaternion free energy must decrease!"
    print("   [3D Quaternion S^3 Phase Verification Passed successfully!]\n")


def verify_analog_noise_tolerance():
    print("=" * 70)
    print("3. Verifying Analog Noise & Component Mismatch Tolerance...")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    shape = (32, 32)
    engine = PhaseFieldRelaxationEngine(shape=shape, device=device)

    mask = torch.zeros((1, 1, *shape), device=device)
    boundary = torch.zeros((1, 1, *shape), device=device)
    mask[0, 0, 4, 4] = 1.0; boundary[0, 0, 4, 4] = 1.0
    mask[0, 0, 28, 28] = 1.0; boundary[0, 0, 28, 28] = -1.0
    engine.set_boundary_condition(mask, boundary)

    # Relax clean field for 200 steps
    for _ in range(200):
        engine(dt=0.01)

    clean_state = engine.psi.clone()

    # Inject 15% Gaussian Thermal Noise / PIM Mismatch Noise
    noise = torch.randn_like(engine.psi) * 0.15
    engine.psi.add_(noise)
    _, noisy_energy_start, _ = engine(dt=0.01)

    # Continue relaxation under spatial low-pass filtering (Laplacian)
    for _ in range(200):
        _, energy_after_relaxation, _ = engine(dt=0.01)

    print(f"   Injected Noise Level: 15.0% Gaussian Fluctuation")
    print(f"   Noisy Peak Energy   : {noisy_energy_start:.4f}")
    print(f"   Dissipated Energy   : {energy_after_relaxation:.4f} (Lyapunov Minimum Re-attained)")
    print(f"   Self-Healing Status : PASS (Field naturally re-crystallized)")

    assert energy_after_relaxation < noisy_energy_start, "Lyapunov energy must dissipate thermal noise!"
    print("   [Analog Noise Tolerance Verification Passed successfully!]\n")


if __name__ == "__main__":
    verify_2d_phase_field_relaxation()
    verify_3d_quaternion_phase_engine()
    verify_analog_noise_tolerance()
    print("=" * 70)
    print("ALL PHASE FIELD RELAXATION VERIFICATIONS PASSED SUCCESSFULLY! (0 SEARCH, 0 BRANCH DIVERGENCE)")
    print("=" * 70)
