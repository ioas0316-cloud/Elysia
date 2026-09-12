import numpy as np
import pytest
import torch

from core.engine.causal_nexus_pipeline import (
    CausalToControlNetPipeline,
    CausalSubspaceProjectionLayer,
    calculate_betti_2d,
    BitwiseCausalSimulator,
    CausalNexusRenderEngine,
)


def test_causal_to_controlnet_pipeline_injection():
    pipeline = CausalToControlNetPipeline(height=32, width=32, device="cpu")
    traj = np.zeros((32, 32), dtype=np.uint8)
    hitbox = np.zeros((32, 32), dtype=np.uint8)

    traj[4:12, 16] = 1
    hitbox[8:16, 12:20] = 1

    tensor = pipeline.inject_nexus_bits(traj, hitbox)
    assert tensor.shape == (1, 3, 32, 32)

    # Convert to numpy for value verification
    if isinstance(tensor, torch.Tensor):
        arr = tensor.numpy()
    else:
        arr = tensor

    # Red Channel: Trajectory
    assert np.array_equal(arr[0, 0], (traj > 0).astype(np.float32))
    # Green Channel: Hitbox
    assert np.array_equal(arr[0, 1], (hitbox > 0).astype(np.float32))
    # Blue Channel: Logical OR
    expected_blue = np.logical_or(traj > 0, hitbox > 0).astype(np.float32)
    assert np.array_equal(arr[0, 2], expected_blue)


def test_causal_subspace_projection_layer():
    layer = CausalSubspaceProjectionLayer()

    z_latent = torch.randn(1, 4, 16, 16, requires_grad=True)
    z_prior = torch.ones(1, 4, 16, 16)
    v_causal_mask = torch.zeros(1, 1, 16, 16)
    v_causal_mask[0, 0, 4:8, 4:8] = 1.0  # Active causal domain

    z_bounded = layer.forward(z_latent, v_causal_mask, z_prior)
    assert z_bounded.shape == (1, 4, 16, 16)

    # Inside causal mask: should match z_latent
    assert torch.allclose(z_bounded[0, :, 4:8, 4:8], z_latent[0, :, 4:8, 4:8])
    # Outside causal mask: should match z_prior
    assert torch.allclose(z_bounded[0, :, :4, :4], z_prior[0, :, :4, :4])

    # Test Gradient Masking
    loss = z_bounded.sum()
    loss.backward()

    # Gradients inside causal mask should be 1.0
    assert torch.allclose(z_latent.grad[0, :, 4:8, 4:8], torch.ones(4, 4, 4))
    # Gradients outside causal mask must be strictly 0.0 (Leakage zero)
    assert torch.allclose(z_latent.grad[0, :, :4, :4], torch.zeros(4, 4, 4))


def test_betti_number_calculation():
    # 1. Solid square (b0 = 1, b1 = 0)
    grid_solid = np.zeros((8, 8), dtype=np.uint8)
    grid_solid[2:6, 2:6] = 1
    b0, b1 = calculate_betti_2d(grid_solid, 8, 8)
    assert b0 == 1
    assert b1 == 0

    # 2. Ring with a 2x2 hole in center (b0 = 1, b1 = 1)
    grid_ring = np.zeros((8, 8), dtype=np.uint8)
    grid_ring[1:7, 1:7] = 1
    grid_ring[3:5, 3:5] = 0
    b0_ring, b1_ring = calculate_betti_2d(grid_ring, 8, 8)
    assert b0_ring == 1
    assert b1_ring == 1

    # 3. Two disconnected components (b0 = 2, b1 = 0)
    grid_two = np.zeros((8, 8), dtype=np.uint8)
    grid_two[0:2, 0:2] = 1
    grid_two[5:7, 5:7] = 1
    b0_two, b1_two = calculate_betti_2d(grid_two, 8, 8)
    assert b0_two == 2
    assert b1_two == 0


def test_bitwise_causal_simulator_audio_visual():
    inputs = np.array([0xFFFFFFFFFFFFFFFF], dtype=np.uint64)
    weights = np.array([0xFFFFFFFFFFFFFFFF], dtype=np.uint64)

    act = BitwiseCausalSimulator.execute_bitwise_xnor_popcnt(inputs, weights)
    assert len(act) == 64
    assert np.all(act == 1)

    traj = np.zeros((16, 16), dtype=np.uint8)
    hitbox = np.zeros((16, 16), dtype=np.uint8)
    traj[4, 2] = 1
    hitbox[8, 12] = 1

    res = BitwiseCausalSimulator.split_audio_visual_voltage(traj, hitbox, 16, 16)
    assert res["visual_voltage_rgb"].shape == (3, 16, 16)
    assert len(res["audio_dsp_registers"]) == 6
    assert np.all(res["audio_dsp_registers"] >= 0.0)


def test_causal_nexus_render_engine_step_and_rollback():
    engine = CausalNexusRenderEngine(height=16, width=16, device="cpu")

    traj = np.zeros((16, 16), dtype=np.uint8)
    hitbox = np.zeros((16, 16), dtype=np.uint8)
    traj[2:6, 2:6] = 1  # b0 = 1, b1 = 0

    # Step 1: Normal execution with expected betti (1, 0)
    step1 = engine.render_step(traj, hitbox, expected_betti=(1, 0))
    assert step1["status"] == "SIGNALED"
    assert step1["topological_loss"] == 0
    assert step1["betti_numbers"] == (1, 0)

    # Step 2: Trigger Topological Rollback by specifying unexpected betti (5, 5)
    step2 = engine.render_step(traj, hitbox, expected_betti=(5, 5))
    assert step2["status"] == "TOPOLOGICAL_ROLLBACK"
    assert step2["topological_loss"] > 0


def test_shared_memory_view_binding():
    pipeline = CausalToControlNetPipeline(height=16, width=16, device="cpu")
    size = 16 * 16
    buf = bytearray(size * 2)
    buf[0:size] = b"\x01" * size
    buf[size:size * 2] = b"\x02" * size

    traj_view, hitbox_view = pipeline.bind_shared_memory_view(buf)
    assert traj_view.shape == (16, 16)
    assert hitbox_view.shape == (16, 16)
    assert np.all(traj_view == 1)
    assert np.all(hitbox_view == 2)
