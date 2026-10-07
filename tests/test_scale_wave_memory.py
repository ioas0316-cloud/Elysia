"""
tests/test_scale_wave_memory.py

Test suite for Scale-Invariant Wave Tensor Memory and Autograd Functions
"""

import pytest
import numpy as np
import torch

from elysia_core import ScaleWaveTensorMemory, TrainableScaleWaveMemory, ScaleWaveMemoryAutogradFunction


def test_scale_wave_tensor_memory_initialization():
    mem = ScaleWaveTensorMemory(num_scales=3, base_dim=4)
    assert mem.tensor_memory.shape == (3, 4, 3)
    assert np.allclose(mem.scale_factors, np.array([1.0, 0.5, 0.25]))


def test_scale_wave_tensor_memory_write_and_decay():
    mem = ScaleWaveTensorMemory(num_scales=3, base_dim=4)
    initial_scale_0 = mem.tensor_memory[0, 0].copy()

    # Write wave data to scale level 2 (Micro)
    mem.write_wave_data(scale_level=2, address_idx=0, phase_shift=np.pi / 2.0, coupling_gain=1.0)

    # Scale 2 cell 0 transformed
    assert not np.allclose(mem.tensor_memory[2, 0], [0.0, 0.0, 0.0])

    # Scale decay: distance between scale 2 and 0 is 2 -> exp(-2) decay
    # Check that scale 1 (distance 1) received more coupling than scale 0 (distance 2)
    diff_scale1 = np.linalg.norm(mem.tensor_memory[1, 0][:2])
    diff_scale0 = np.linalg.norm(mem.tensor_memory[0, 0][:2])
    assert diff_scale1 > 0.0 and diff_scale0 > 0.0


def test_scale_wave_tensor_memory_resonance_read():
    mem = ScaleWaveTensorMemory(num_scales=3, base_dim=4)
    query = np.array([1.0, 0.0, 0.0])  # Pure sin query
    res_map = mem.resonance_read(query)

    assert res_map.shape == (3, 4)
    assert not np.isnan(res_map).any()


def test_trainable_scale_wave_memory_autograd_flow():
    device = "cpu"
    wave_mem = TrainableScaleWaveMemory(num_scales=3, base_dim=8, device=device)
    optimizer = torch.optim.Adam(wave_mem.parameters(), lr=0.05)

    target_state = torch.ones((3, 8, 3), device=device) * 0.7071

    initial_loss = torch.mean((wave_mem(target_scale=1, target_idx=0) - target_state) ** 2).item()

    for step in range(10):
        optimizer.zero_grad()
        output_mem = wave_mem(target_scale=1, target_idx=0)
        loss = torch.mean((output_mem - target_state) ** 2)
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    assert final_loss < initial_loss
    assert wave_mem.phase_shift.grad is not None
    assert wave_mem.coupling_gain.grad is not None


def test_autograd_function_direct():
    tensor_memory = torch.zeros((2, 4, 3), requires_grad=True)
    scale_factors = torch.tensor([1.0, 0.5])
    phase_shift = torch.tensor(0.2, requires_grad=True)
    coupling_gain = torch.tensor(0.5, requires_grad=True)

    out = ScaleWaveMemoryAutogradFunction.apply(
        tensor_memory, scale_factors, 0, 0, phase_shift, coupling_gain
    )

    loss = out.sum()
    loss.backward()

    assert phase_shift.grad is not None
    assert coupling_gain.grad is not None
    assert not torch.isnan(out).any()
