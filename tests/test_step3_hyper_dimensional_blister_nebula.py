"""
tests/test_step3_hyper_dimensional_blister_nebula.py

Unit & Integration Tests for Step 3:
Hyper-Dimensional Conceptual-Optical Volume Nebula & Optical Hyper-Slicer Projection.

Verifies:
1. Cℓ(4,0) 6-bivector Lie algebra generation and SO(4) rotation matrix properties.
2. 4D Volumetric non-linear snap past critical overlap Omega_c = 0.25 (>10x snap ratio).
3. Phase interference modulation: attractive at Delta phi = 0 and repulsive at Delta phi = pi.
4. Optical Hyper-Slicer 4D -> 3D -> 2D projection and edge sharpness preservation over blur.
5. Dynamic approach, intent potential shock (Delta V), and self-reorganization stability.
"""

import math
import sys
from pathlib import Path
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import elysia_loader
elysia_loader.install()

from elysia_engine.core.hyper_dimensional_blister_nebula import (
    Clifford4DRotor,
    HyperDimensionalOpticalRotor,
    HyperDimensionalBlisterKernel,
    OpticalHyperSlicer,
    HyperDimensionalBlisterNebulaEngine
)
from scripts.verify_step3_hyper_dimensional_blister_nebula import (
    sweep_4d_distance_vs_force,
    sweep_4d_phase_interference,
    simulate_nebula_evolution_and_intent_shock
)


def test_clifford_4d_rotor_so4_properties():
    """Verify Cℓ(4,0) Lie algebra so(4) generator and SO(4) rotation matrix properties."""
    # 6 independent bivector generators: [e12, e23, e31, e14, e24, e34]
    bivectors = [0.2, -0.3, 0.4, 0.1, -0.2, 0.5]
    rotor = Clifford4DRotor(bivectors=bivectors)

    assert rotor.magnitude > 0.0

    # Test Lie algebra generator matrix is skew-symmetric: Omega^T = -Omega
    omega = rotor.get_so4_generator_matrix()
    assert omega.shape == (4, 4)
    assert np.allclose(omega.T, -omega, atol=1e-6)

    # Test SO(4) rotation matrix: R^T R = I, det(R) = +1
    R = rotor.get_rotation_matrix(dt=0.5)
    assert R.shape == (4, 4)
    assert np.allclose(R.T @ R, np.eye(4), atol=1e-5)
    assert np.allclose(R @ R.T, np.eye(4), atol=1e-5)
    assert math.isclose(np.linalg.det(R), 1.0, abs_tol=1e-5)

    # Test 4D vector rotation preserves norm
    v4 = np.array([1.0, 2.0, -1.0, 0.5], dtype=np.float32)
    v4_rot = rotor.rotate_vector(v4, dt=0.5)
    assert math.isclose(np.linalg.norm(v4), np.linalg.norm(v4_rot), abs_tol=1e-5)


def test_4d_volumetric_blister_snap_nonlinearity():
    """Verify non-linear snap past critical overlap Omega_c = 0.25 (>10x ratio)."""
    kernel = HyperDimensionalBlisterKernel(omega_c=0.25, gamma_power=2.5, kappa_0=1.0, kappa_1=14.0)
    data = sweep_4d_distance_vs_force(kernel)

    overlaps = data["overlaps"]
    f_snap = data["forces_snap"]
    f_tot = data["forces_total"]

    # Sub-critical region: 0.1 < overlap < 0.24
    sub_indices = np.where((overlaps > 0.1) & (overlaps < 0.24))[0]
    assert len(sub_indices) > 0
    sub_f = f_tot[sub_indices[0]]
    assert f_snap[sub_indices[0]] == 0.0, "Snap force must be zero below Omega_c"

    # Super-critical region: overlap > 0.8
    super_indices = np.where(overlaps > 0.8)[0]
    assert len(super_indices) > 0
    super_f = f_tot[super_indices[0]]

    assert sub_f > 0.0
    assert super_f > 0.0
    snap_ratio = super_f / sub_f
    assert snap_ratio > 10.0, f"Snap ratio {snap_ratio:.2f} must exceed 10x"


def test_4d_phase_interference_attraction_vs_repulsion():
    """Verify phase interference: constructive attraction vs destructive repulsion."""
    kernel = HyperDimensionalBlisterKernel(omega_c=0.25)
    data = sweep_4d_phase_interference(kernel)

    delta_phases = data["delta_phases"]
    forces = data["forces"]

    # In-phase (Delta phi ~ 0)
    zero_idx = np.argmin(np.abs(delta_phases))
    in_phase_force = forces[zero_idx]

    # Anti-phase (Delta phi ~ pi)
    pi_idx = np.argmin(np.abs(delta_phases - np.pi))
    anti_phase_force = forces[pi_idx]

    assert in_phase_force > 0.0, "In-phase force must be attractive"
    assert anti_phase_force < 0.0, "Anti-phase force must be repulsive (purification)"
    assert math.isclose(in_phase_force, -anti_phase_force, rel_tol=0.05)


def test_optical_hyper_slicer_projection_and_edge_sharpness():
    """Verify 4D -> 3D volume -> 2D prism projection and edge sharpness over blur."""
    slicer = OpticalHyperSlicer(resolution_3d=28, resolution_2d=64)
    kernel = HyperDimensionalBlisterKernel(omega_c=0.25)

    node1 = HyperDimensionalOpticalRotor(wavelength_nm=450.0, position_4d=[-0.3, 0.0, 0.0, 0.0])
    node2 = HyperDimensionalOpticalRotor(wavelength_nm=650.0, position_4d=[0.3, 0.0, 0.0, 0.0])

    vol = slicer.slice_4d_to_3d_volume([node1, node2], kernel, w_slice=0.0)
    assert vol["density_3d"].shape == (28, 28, 28)
    assert vol["triad_field_3d"].shape == (3, 28, 28, 28)

    proj = slicer.project_to_2d_prism(vol, projection_axis=2)
    assert proj["density_2d"].shape == (28, 28)
    assert proj["rgb_2d"].shape == (28, 28, 3)
    assert proj["grad_mag"].shape == (28, 28)

    # Edge sharpness test vs Gaussian blur
    grad_blister = proj["grad_mag"]
    blurred_density = gaussian_filter(proj["density_2d"], sigma=2.0)
    gy_b, gx_b = np.gradient(blurred_density)
    grad_blur = np.sqrt(gx_b**2 + gy_b**2)

    max_blister = float(np.max(grad_blister))
    max_blur = float(np.max(grad_blur))
    assert max_blister > 1.4 * max_blur, (
        f"Blister gradient ({max_blister:.4f}) should be significantly sharper "
        f"than blur gradient ({max_blur:.4f})"
    )


def test_intent_potential_shock_and_nebula_adaptation():
    """Verify nebula handles intent potential shock (Delta V) without divergence."""
    engine, history = simulate_nebula_evolution_and_intent_shock(steps=100, dt=0.02)

    coherence = history["coherence"]
    active_b = history["active_blisters"]

    assert len(coherence) == 100
    # Pre-shock (step 70) should have high active blisters
    assert active_b[70] > 0

    # Ensure no NaN or infinite values occurred
    for node in engine.nodes:
        assert np.all(np.isfinite(node.position))
        assert np.all(np.isfinite(node.velocity))
        assert np.isfinite(node.phase_theta)
