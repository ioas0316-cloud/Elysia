"""
Unit tests for Riemannian Causal Field Engine (`synaptic_architecture/riemannian_causal_field_engine.py`).
Tests mathematical principles:
1. Potential field decomposition (V_ext, V_self, V_will).
2. Spontaneous symmetry breaking under thermal fluctuations xi(tau).
3. Volitional metric deformation g_ij^will and orthogonal noise suppression.
4. Damped covariant geodesic motion and Phase-Locking convergence.
5. Causal erosion PDE and memory terrain update.
6. Phenomenal Present / Conscious Residual Energy E_present.
7. Cognitive fluid dynamics and phase transitions (evaporation/rainfall).
8. O(N) -> O(1) Spatial Tensor Grid rasterization and fast sampling.
"""

import math
import pytest
import torch
from synaptic_architecture.riemannian_causal_field_engine import (
    RiemannianCausalFieldEngine,
    CognitiveFieldState,
    PotentialWell
)


def test_potential_field_decomposition():
    engine = RiemannianCausalFieldEngine(dimension=16)

    # Add a memory well at center (0, ..., 0) with depth 2.0
    center = torch.zeros(16)
    engine.add_potential_well("well_1", center=center, depth=2.0, sigma=1.0)

    # Set volitional target at (2, ..., 2)
    # V_will at origin = 0.5 * ||0 - [2]*16||^2 = 0.5 * 16 * 4 = 32.0
    target = torch.ones(16) * 2.0
    engine.set_volitional_target(target)

    psi = torch.zeros(16)
    v_total, grad_v_total = engine.compute_total_potential(psi)

    # V_self at origin = -2.0
    # V_will at origin = 32.0
    # V_total = 1.0 * (-2.0) + 2.0 * 32.0 = 62.0
    assert abs(v_total.item() - 62.0) < 1e-4
    assert grad_v_total.shape == (16,)


def test_spontaneous_symmetry_breaking():
    engine = RiemannianCausalFieldEngine(dimension=16)

    # Saddle point setup at origin
    center_1 = torch.ones(16) * 2.0
    center_2 = torch.ones(16) * -2.0
    engine.add_potential_well("well_1", center=center_1, depth=2.0, sigma=1.0)
    engine.add_potential_well("well_2", center=center_2, depth=2.0, sigma=1.0)

    # State starting at origin (unstable equilibrium) with temperature > 0
    state = CognitiveFieldState(
        psi=torch.zeros(16),
        velocity=torch.zeros(16),
        temperature=2.0
    )

    # Step forward with thermal fluctuation
    new_state = engine.step_geodesic_motion(state, dt=0.1)

    # Position should have moved away from exact origin due to xi(tau) thermal kick
    assert not torch.equal(state.psi, new_state.psi)
    assert torch.norm(new_state.psi).item() > 0.0


def test_volitional_metric_deformation():
    engine = RiemannianCausalFieldEngine(dimension=4)
    target = torch.tensor([1.0, 0.0, 0.0, 0.0])
    engine.set_volitional_target(target)

    psi = torch.zeros(4)
    g = engine.compute_metric_tensor(psi)

    # Orthogonal movement [0, 1, 0, 0] should encounter higher metric resistance than parallel [1, 0, 0, 0]
    parallel_vec = torch.tensor([1.0, 0.0, 0.0, 0.0])
    orthogonal_vec = torch.tensor([0.0, 1.0, 0.0, 0.0])

    res_parallel = torch.dot(parallel_vec, torch.matmul(g, parallel_vec)).item()
    res_ortho = torch.dot(orthogonal_vec, torch.matmul(g, orthogonal_vec)).item()

    assert res_ortho > res_parallel


def test_geodesic_flow_and_phase_lock():
    engine = RiemannianCausalFieldEngine(dimension=4, damping_gamma=5.0)
    center = torch.tensor([1.0, 1.0, 0.0, 0.0])
    engine.add_potential_well("target_well", center=center, depth=5.0, sigma=1.5)

    state = CognitiveFieldState(
        psi=torch.tensor([0.95, 0.95, 0.0, 0.0]),
        velocity=torch.zeros(4),
        temperature=0.001  # Very low noise for smooth convergence
    )

    locked = False
    for _ in range(50):
        state = engine.step_geodesic_motion(state, dt=0.05)
        if state.phase_locked:
            locked = True
            break

    assert locked
    assert torch.norm(state.psi - center).item() < 0.5


def test_causal_erosion_update():
    engine = RiemannianCausalFieldEngine(dimension=4)
    center = torch.zeros(4)
    engine.add_potential_well("erosion_well", center=center, depth=1.0, sigma=1.0)

    initial_depth = engine.wells["erosion_well"].depth
    engine.apply_causal_erosion("erosion_well", erosion_depth=0.5)

    assert engine.wells["erosion_well"].depth == initial_depth + 0.5


def test_phenomenal_present_residual_energy():
    engine = RiemannianCausalFieldEngine(dimension=4)
    psi = torch.zeros(4)
    sensory = torch.ones(4) * 2.0

    energy = engine.compute_phenomenal_present_energy(psi, sensory)
    assert energy > 0.0


def test_cognitive_fluid_dynamics():
    engine = RiemannianCausalFieldEngine(dimension=4, grid_resolution=16)

    initial_fluid = engine.step_cognitive_fluid_dynamics(dt=0.05)
    assert "mean_density" in initial_fluid
    assert "max_vorticity" in initial_fluid
    assert initial_fluid["mean_density"] > 0.0


def test_O1_spatial_grid_rasterization_phase_transition():
    # N_critical = 5
    engine = RiemannianCausalFieldEngine(dimension=4, n_critical=5, grid_resolution=16)

    assert not engine.is_rasterized

    for i in range(6):
        center = torch.ones(4) * float(i)
        engine.add_potential_well(f"well_{i}", center=center, depth=1.0)

    assert engine.is_rasterized
    assert engine.spatial_tensor_grid is not None

    psi = torch.zeros(4)
    v_val, v_grad = engine.compute_V_self(psi)
    assert isinstance(v_val.item(), float)
    assert v_grad.shape == (4,)
