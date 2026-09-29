"""
Unit test suite for DynamicLightning3D & Hebbian Field Plasticity Engine.
"""

import torch
import pytest
from synaptic_architecture.dynamic_lightning_3d import DynamicLightning3D, HebbianFieldPlasticityEngine


def test_dynamic_lightning_3d_channel_collapse():
    solver = DynamicLightning3D(shape=(16, 16, 16), device='cpu')
    grid = torch.ones((16, 16, 16), dtype=torch.float32)

    start_pt = (2, 2, 2)
    goal_pt = (13, 13, 13)

    # Calculate lightning channel
    channel_3d = solver.step(start_pt, goal_pt, grid, gamma=16.0, relax_steps=20)
    assert channel_3d.shape == (16, 16, 16)
    assert (channel_3d >= 0.0).all()
    assert (channel_3d <= 1.0).all()

    # Active voxels should exist forming a channel between start and goal
    active_voxels = (channel_3d > 0.1).sum().item()
    assert active_voxels > 0


def test_dynamic_lightning_3d_obstacle_avoidance():
    solver = DynamicLightning3D(shape=(16, 16, 16), device='cpu')
    grid = torch.ones((16, 16, 16), dtype=torch.float32)

    start_pt = (2, 2, 2)
    goal_pt = (13, 13, 13)

    # Place an obstacle blocking direct path
    grid[7:10, 7:10, 7:10] = 0.0

    channel_3d = solver.step(start_pt, goal_pt, grid, gamma=16.0, relax_steps=5)
    assert channel_3d.shape == (16, 16, 16)
    # Obstacle region should have zero voltage channel
    assert (channel_3d[7:10, 7:10, 7:10] == 0.0).all()


def test_hebbian_field_plasticity():
    hebb = HebbianFieldPlasticityEngine(shape=(16, 16, 16), device='cpu')
    assert hebb.sigma.shape == (1, 1, 16, 16, 16)

    # Create potential field with strong gradient in center
    potential = torch.zeros((16, 16, 16), dtype=torch.float32)
    potential[8, :, :] = 1.0

    sigma_updated = hebb.update_plasticity(potential, dt=0.1)
    assert sigma_updated.shape == (16, 16, 16)
    # Conductivity near gradient boundary should change
    assert not torch.all(sigma_updated == 0.5)
