import torch
import pytest
from elysia_engine.core.causal_erosion import CausalErosionLandscape

def test_causal_erosion_basic():
    landscape = CausalErosionLandscape(state_dim=3, nc_threshold=5)
    point = torch.tensor([1.0, 1.0, 1.0])

    # Erode landscape
    landscape.erode_trajectory(point)
    assert landscape.well_centers.size(0) == 1
    assert not landscape.is_phase_transformed

    # Potential evaluation
    v = landscape(point)
    assert isinstance(v, torch.Tensor)

    # Force calculation
    force = landscape.compute_gradient_force(point)
    assert force.shape == (3,)

def test_causal_erosion_phase_transition():
    landscape = CausalErosionLandscape(state_dim=3, nc_threshold=5)
    point = torch.tensor([1.0, 0.0, 0.0])

    for _ in range(5):
        landscape.erode_trajectory(point)

    assert landscape.is_phase_transformed
    v = landscape(point)
    force = landscape.compute_gradient_force(point)
    assert force.shape == (3,)

def test_memory_replay():
    landscape = CausalErosionLandscape(state_dim=3, nc_threshold=5)
    for _ in range(3):
        landscape.erode_trajectory(torch.tensor([0.0, 0.0, 0.0]))

    start_pos = torch.tensor([1.0, 1.0, 1.0])
    next_pos = landscape.memory_replay_step(start_pos)
    assert not torch.allclose(start_pos, next_pos)
