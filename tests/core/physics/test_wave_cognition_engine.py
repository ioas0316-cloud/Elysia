"""
Unit Tests for Wave Cognition & Continuous Latent Memory Engine
================================================================
Tests for:
1. MultimodalFrictionLayer (entropy & surprise to R_causal mapping)
2. SDFCognitionEngine (Level Set PDE step, diffusion decay, phase-locking, smooth minimum)
3. SDFLatentMemoryLayer (Sphere Tracing read, autograd gradient flow for queries and value grid, smin write)
4. SDFTransformerDecoder (forward pass, loss computation, autograd flow across parameters, KV-cache-free token generation)
"""

import pytest
import torch
from core.physics.wave_cognition_engine import (
    MultimodalFrictionLayer,
    SDFCognitionEngine,
    SDFLatentMemoryLayer,
    SDFTransformerConfig,
    SDFTransformerDecoder
)


def test_multimodal_friction_layer():
    device = torch.device("cpu")
    batch_size = 4
    feature_dim = 64
    grid_res = 16

    friction_layer = MultimodalFrictionLayer(feature_dim=feature_dim, grid_res=grid_res)
    x_multimodal = torch.randn(batch_size, feature_dim, device=device)

    R_causal = friction_layer(x_multimodal)

    assert R_causal.shape == (grid_res, grid_res, grid_res)
    assert not torch.isnan(R_causal).any()
    assert (R_causal >= 0).all()


def test_sdf_cognition_engine():
    grid_res = 16
    engine = SDFCognitionEngine(grid_res=grid_res, nu_decay=0.01)

    initial_vol = (engine.sdf < 0.0).sum().item()
    R_causal = torch.rand(grid_res, grid_res, grid_res) * 0.1

    # Advance 3 simulation steps
    for _ in range(3):
        engine.step(R_causal, F0=0.3)

    assert engine.sdf.shape == (grid_res, grid_res, grid_res)
    assert engine.phase_lock.shape == (grid_res, grid_res, grid_res)
    assert (engine.phase_lock >= 0.0).all() and (engine.phase_lock <= 1.0).all()

    # Topological fusion test (Smooth Minimum)
    sdf_b = torch.norm(engine.sdf, dim=-1, keepdim=True).repeat(1, 1, grid_res) - 0.2
    merged = engine.smooth_minimum(engine.sdf, sdf_b, k=10.0)
    assert merged.shape == (grid_res, grid_res, grid_res)


def test_sdf_latent_memory_layer():
    hidden_size = 32
    grid_res = 16
    batch_size = 4

    memory_layer = SDFLatentMemoryLayer(hidden_size=hidden_size, grid_res=grid_res)
    query = torch.randn(batch_size, hidden_size, requires_grad=True)

    # 1. Memory Read & Autograd Gradient Flow (Queries & Value Grid)
    out = memory_layer(query)
    assert out.shape == (batch_size, hidden_size)

    loss = out.sum()
    loss.backward()

    assert query.grad is not None and query.grad.norm().item() > 0.0
    assert memory_layer.value_grid.grad is not None and memory_layer.value_grid.grad.norm().item() > 0.0

    # 2. Non-destructive Smooth Minimum Write Test
    old_sdf = memory_layer.sdf_grid.detach().clone()
    old_value = memory_layer.value_grid.detach().clone()

    new_sdf = torch.ones_like(old_sdf) * 0.5
    new_value = torch.randn_like(old_value) * 0.1

    memory_layer.write_memory(new_sdf, new_value)

    # Confirm SDF grid updated smoothly without NaN
    assert not torch.isnan(memory_layer.sdf_grid).any()
    # Smooth minimum ensures updated_sdf <= min(old_sdf, new_sdf)
    assert (memory_layer.sdf_grid <= old_sdf + 1e-4).all()


def test_sdf_transformer_decoder():
    config = SDFTransformerConfig(
        hidden_size=32,
        intermediate_size=64,
        num_decoder_layers=2,
        grid_res=16
    )
    vocab_size = 100
    batch_size = 2
    seq_len = 8

    model = SDFTransformerDecoder(config, vocab_size=vocab_size)

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Forward pass with target IDs (computes loss & dynamic friction write)
    logits, loss = model(input_ids, target_ids=target_ids)

    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert loss is not None
    assert loss.item() > 0.0

    loss.backward()

    # Check model gradients
    for layer in model.layers:
        assert layer.sdf_memory.value_grid.grad is not None

    # Generation pass (O(1) continuous memory, no KV cache)
    prompt = torch.randint(0, vocab_size, (1, 3))
    generated = model.generate(prompt, max_new_tokens=4)

    assert generated.shape == (1, 7)
