"""
Unit tests for the 4-Layer Sandwich Hybrid Architecture with Closed Feedback Loop & Active Inference.
"""

import pytest
import torch
import torch.nn as nn

from synaptic_architecture.sandwich_hybrid_architecture import (
    STEBinarize,
    TopologyLSHProjection,
    SinkhornSoftOTLoss,
    NeuralPerceptionLayer,
    TopologicalTransducer,
    TopologicalCausalCore,
    ConstrainedSynthesisLayer,
    SandwichHybridArchitecture,
)
from core.physics.mgris_engine import MGRISCausalBridge, Polarity, StickyEnd


def test_ste_binarization_forward_backward():
    x = torch.tensor([-2.0, -0.5, 0.1, 3.0], requires_grad=True)
    out = STEBinarize.apply(x, 5.0)

    # Check forward hard sign
    expected = torch.tensor([-1.0, -1.0, 1.0, 1.0])
    assert torch.allclose(out, expected)

    # Check backward gradient flow
    loss = torch.sum(out * torch.tensor([1.0, 2.0, 3.0, 4.0]))
    loss.backward()

    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))
    assert x.grad.shape == x.shape


def test_topology_lsh_projection_and_fields():
    batch_size = 4
    in_features = 32
    layer = TopologyLSHProjection(in_features=in_features, out_bits=64)

    x = torch.randn(batch_size, in_features)
    binary_code, logits = layer(x)

    assert binary_code.shape == (batch_size, 64)
    assert logits.shape == (batch_size, 64)
    assert torch.all((binary_code == 1.0) | (binary_code == -1.0))

    # Test bit fields extraction
    bit_fields = TopologyLSHProjection.extract_bit_fields(binary_code)
    assert bit_fields["macro"].shape == (batch_size, 16)
    assert bit_fields["meso"].shape == (batch_size, 32)
    assert bit_fields["micro"].shape == (batch_size, 16)


def test_sinkhorn_soft_ot_loss():
    loss_fn = SinkhornSoftOTLoss(eps=0.05, max_iter=20)

    d1 = torch.randn(2, 16, 4, requires_grad=True)
    d2 = torch.randn(2, 16, 4, requires_grad=True)

    # Loss between d1 and d1 should be close to 0 (Unbiased divergence property)
    self_loss = loss_fn(d1, d1)
    assert self_loss.item() < 1e-4

    # Loss between d1 and d2 should be non-negative
    diff_loss = loss_fn(d1, d2)
    assert diff_loss.item() >= 0.0

    # Test backward pass
    diff_loss.backward()
    assert d1.grad is not None
    assert d2.grad is not None


def test_neural_perception_threshold_phase_transition():
    layer = NeuralPerceptionLayer(in_dim=10, latent_dim=16, threshold=5.0)

    # Low energy input
    low_input = torch.randn(2, 10) * 0.1
    latents, energy, is_critical = layer(low_input)
    assert is_critical.tolist() == [False, False]

    # High energy input triggering critical threshold phase shift
    high_input = torch.ones(2, 10) * 10.0
    latents_h, energy_h, is_critical_h = layer(high_input)
    assert is_critical_h.tolist() == [True, True]


def test_topological_transducer_sticky_end_conversion():
    transducer = TopologicalTransducer(latent_dim=16, out_bits=64)
    latents = torch.randn(1, 16)
    binary_code, logits, bit_fields = transducer(latents)

    sticky_end = transducer.convert_to_sticky_end(binary_code[0])
    assert isinstance(sticky_end, StickyEnd)
    assert 0 <= sticky_end.pattern <= 0xFFFFFFFFFFFFFFFF


def test_constrained_synthesis_layer():
    synthesis = ConstrainedSynthesisLayer(latent_dim=16, vocab_dim=100)
    latents = torch.randn(2, 16)
    constraint_mask = torch.ones(2, 64)

    logits, allowed_mask = synthesis(latents, constraint_mask)
    assert logits.shape == (2, 100)
    assert allowed_mask.shape == (2, 100)


def test_sandwich_hybrid_architecture_end_to_end():
    model = SandwichHybridArchitecture(in_dim=16, latent_dim=32, vocab_dim=50)
    raw_input = torch.randn(2, 16)
    target_barcodes = torch.randn(2, 16, 4)

    out = model(raw_input, target_barcodes=target_barcodes)

    assert "latents" in out
    assert "accumulated_energy" in out
    assert "binary_code" in out
    assert "constrained_logits" in out
    assert "sinkhorn_loss" in out
    assert out["sinkhorn_loss"] is not None
    assert out["sinkhorn_loss"].item() >= 0.0


def test_active_inference_step():
    model = SandwichHybridArchitecture(in_dim=16, latent_dim=32, vocab_dim=50)

    obs = torch.randn(1, 32)
    internal_state = torch.randn(1, 32)
    candidate_actions = torch.randn(3, 16)

    def mock_env(state, action):
        # Linear dummy environment response
        return state + action.sum() * 0.1

    new_state, best_action, new_obs = model.active_inference_step(
        observation=obs,
        internal_state=internal_state,
        candidate_actions=candidate_actions,
        env_transition_fn=mock_env,
        lr=0.01
    )

    assert new_state.shape == internal_state.shape
    assert best_action is not None
    assert new_obs.shape == obs.shape
