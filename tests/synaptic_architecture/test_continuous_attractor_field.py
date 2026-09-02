import torch
import pytest
from synaptic_architecture.continuous_attractor_field import ContinuousAttractorField


def test_continuous_attractor_bump_drift():
    cann = ContinuousAttractorField(spatial_dim=32, tau_r=0.05, gamma_error=0.8)

    r = cann.initialize_bump(center=0.0, width=0.8)
    sensory = cann.initialize_bump(center=0.5, width=0.8)  # Shifted sensory target
    pred = torch.zeros(32)

    initial_center = cann.compute_bump_center(r)

    # Field relaxation under shifted error field
    for _ in range(30):
        r, error_field = cann.compute_field_step(r, sensory, pred, dt=0.01)

    final_center = cann.compute_bump_center(r)

    # Distance to target (0.5) should decrease
    assert abs(final_center - 0.5) < abs(initial_center - 0.5)


def test_stokes_boundary_reduction():
    cann = ContinuousAttractorField(spatial_dim=8)
    bulk_field_3d = torch.randn(8, 8, 8)

    boundary_flux = cann.stokes_boundary_flux(bulk_field_3d)

    # 6 boundary surface faces of shape (8, 8)
    assert boundary_flux.shape == (6, 8, 8)


def test_hodge_laplacian_resonance():
    cann = ContinuousAttractorField(spatial_dim=32)

    # Smooth harmonic 1-form error field
    x = cann.x_grid
    error_1form = torch.sin(x)

    free_energy, harmonic_norm = cann.compute_free_energy_and_resonance(error_1form)

    assert free_energy > 0.0
    assert harmonic_norm >= 0.0
