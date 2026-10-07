"""
tests/test_metric_rotor_pipeline.py

Test suite for Integrated Metric Wave to Rotor Pipeline & Closed-Loop Autograd Pipeline
"""

import pytest
import numpy as np
import torch

from elysia_core import (
    IntegratedMetricRotorPipeline,
    TrainableMetricRotorPipeline,
    SwarmLiftField
)


def test_integrated_metric_rotor_pipeline_execution():
    grid_res = (8, 8, 8)
    pipeline = IntegratedMetricRotorPipeline(grid_shape=grid_res, coupling_gain=2.0)

    raw_h = np.random.randn(3, 3, *grid_res) * 0.1
    h_symmetric = 0.5 * (raw_h + np.swapaxes(raw_h, 0, 1))

    v_5d = np.array([1.0, 0.0, 0.0, 0.0, 0.0])

    res = pipeline.process(h_symmetric, current_state_5d=v_5d, wave_number_k0=1.0)

    assert "theta_10d" in res
    assert len(res["theta_10d"]) == 10
    assert np.isclose(res["norm_preserved"], 1.0)


def test_trainable_metric_rotor_pipeline_convergence():
    device = "cpu"
    pipeline = TrainableMetricRotorPipeline(num_scales=3, base_dim=8, device=device)
    optimizer = torch.optim.Adam(pipeline.parameters(), lr=0.08)

    mock_theta_10d = np.array([0.5, -0.2, 0.8, 0.1, -0.4, 0.3, 0.9, -0.1, 0.6, 0.2])
    mock_initial_v5d = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    target_memory = torch.ones((3, 8, 3), device=device) * 0.7071

    initial_loss = torch.mean((pipeline(mock_theta_10d, mock_initial_v5d) - target_memory) ** 2).item()

    for step in range(10):
        optimizer.zero_grad()
        out_memory = pipeline(mock_theta_10d, mock_initial_v5d)
        loss = torch.mean((out_memory - target_memory) ** 2)
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    assert final_loss < initial_loss


def test_swarm_lift_field_dislocation_and_re_locking():
    swarm = SwarmLiftField(num_drones=6)
    assert len(swarm.drones) == 6

    # Apply severe wind gust shock to trigger phase dislocations
    wind_gust = np.array([3.5, -2.8, 2.0])
    swarm.apply_wind_gust_shock(wind_gust)

    # Detect and resolve dislocations via 5D Clifford Rotor jumps
    res = swarm.detect_and_resolve_dislocations()

    assert res["all_phase_locked"] is True
    assert res["phase_coherence"] > 0.0
