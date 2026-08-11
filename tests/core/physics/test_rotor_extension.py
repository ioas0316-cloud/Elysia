"""
test_rotor_extension.py
========================
High-precision testing suite for Clifford Geometric Algebra Rotor Sandwich Layer ($R v R^\\dagger$).
Validates 4 core geometric invariants and PyTorch Autograd compliance.
"""

import pytest
import torch
import numpy as np
from core.physics.rotor_extension.rotor_layer import (
    rotor_sandwich_native_pytorch,
    RotorSandwichLayer,
    CognitiveRotorNetwork,
    apply_rotor_sandwich
)
from core.physics.ga_rotor_field import CognitiveThoughtTrajectory

def test_gram_schmidt_orthonormalize_invariant():
    """Validates that u and w are perfectly orthonormalized (u . w = 0, ||u|| = ||w|| = 1)."""
    layer = RotorSandwichLayer(features=32)
    p_u = torch.randn(10, 32)
    p_w = torch.randn(10, 32)

    u, w = layer._gram_schmidt_orthonormalize(p_u, p_w)

    # 1. Norm checks
    u_norms = torch.norm(u, dim=-1)
    w_norms = torch.norm(w, dim=-1)
    assert torch.allclose(u_norms, torch.ones_like(u_norms), atol=1e-6)
    assert torch.allclose(w_norms, torch.ones_like(w_norms), atol=1e-6)

    # 2. Orthogonality check <u, w> = 0
    dot_products = torch.sum(u * w, dim=-1)
    assert torch.allclose(dot_products, torch.zeros_like(dot_products), atol=1e-6)

def test_orthogonal_invariance():
    """
    Validates that any vector v_perp fully orthogonal to the rotation plane B = u ^ w
    remains completely invariant under Clifford Rotor Sandwich rotation.
    """
    # 3D Space example
    u = torch.tensor([[1.0, 0.0, 0.0]])
    w = torch.tensor([[0.0, 1.0, 0.0]])
    theta = torch.tensor([[1.2345]]) # random rotation angle

    # v_perp is fully orthogonal to the u-w plane
    v_perp = torch.tensor([[0.0, 0.0, 5.0]])

    rotated = rotor_sandwich_native_pytorch(v_perp, u, w, theta)
    assert torch.allclose(rotated, v_perp, atol=1e-6)

def test_analytical_gradient_and_autograd_compliance():
    """
    Validates the autograd backpropagation and gradcheck compliance of
    the Clifford Rotor Sandwich operation using double precision.
    """
    from torch.autograd import gradcheck

    B, D = 4, 8
    v = torch.randn(B, D, dtype=torch.float64, requires_grad=True)
    u = torch.randn(B, D, dtype=torch.float64, requires_grad=True)
    w = torch.randn(B, D, dtype=torch.float64, requires_grad=True)
    theta = torch.rand(B, 1, dtype=torch.float64, requires_grad=True)

    # Wrap vectors to ensure orthonormalized input during gradcheck to satisfy invariants
    # using Gram-Schmidt
    def grad_test_fn(v_, u_cand, w_cand, theta_):
        u_norm = torch.nn.functional.normalize(u_cand, dim=-1, eps=1e-8)
        proj = torch.sum(w_cand * u_norm, dim=-1, keepdim=True) * u_norm
        w_norm = torch.nn.functional.normalize(w_cand - proj, dim=-1, eps=1e-8)
        return apply_rotor_sandwich(v_, u_norm, w_norm, theta_)

    # Perform gradcheck comparing PyTorch Autograd with numerical gradients
    test_passed = gradcheck(grad_test_fn, (v, u, w, theta), eps=1e-6, atol=1e-4)
    assert test_passed, "Gradcheck failed for Clifford Rotor Sandwich Operation!"

def test_cognitive_thought_trajectory_differentiable():
    """
    Validates the end-to-end differentiability of the CognitiveThoughtTrajectory
    using the new differentiable navigation method.
    """
    thought_engine = CognitiveThoughtTrajectory(embedding_dim=8, contradiction_threshold=1.5)

    # Ensure starting point is well within the contradiction threshold zone
    # to guarantee the Clifford Rotor logic is always executed and validated.
    # Note: Use an operation that keeps the tensor as a leaf tensor or retain grads,
    # or simply initialize with requires_grad=True.
    start = (torch.zeros(8) + 0.1).detach().requires_grad_(True)
    goal = torch.randn(8, requires_grad=True)

    contradiction_center = np.zeros(8, dtype=np.float32)
    thought_engine.add_contradiction_zone(center_emb=contradiction_center, radius=0.5)

    # Run differentiable navigation
    trajectory = thought_engine.navigate_thought_differentiable(start, goal, steps=10, dt=0.05)

    final_pos = trajectory[-1]
    loss = torch.sum(final_pos ** 2)

    # Verify we can backpropagate gradients all the way to start and goal embeddings
    loss.backward()

    assert start.grad is not None
    assert goal.grad is not None
    assert torch.sum(torch.abs(start.grad)) > 1e-8
    assert torch.sum(torch.abs(goal.grad)) > 1e-8

def test_cognitive_rotor_network_training():
    """Verifies that CognitiveRotorNetwork with layers runs forward and backward on synthetic training batch."""
    torch.manual_seed(42)
    model = CognitiveRotorNetwork(in_features=16, hidden_dim=32, num_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    X = torch.randn(8, 16)
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1], dtype=torch.long)

    # Training step
    optimizer.zero_grad()
    logits = model(X)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()

    assert loss.item() > 0.0
