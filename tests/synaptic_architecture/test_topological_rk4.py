import torch
import pytest
from synaptic_architecture.topological_rk4_autograd import TopologicalRK4Layer


def test_topological_rk4_autograd():
    N = 16
    s = torch.randn(N, dtype=torch.float32, requires_grad=True)
    W = torch.randn(N, N, dtype=torch.float32, requires_grad=True)
    W0 = W.clone().detach().requires_grad_(True)
    stress_grad = torch.ones(N, dtype=torch.float32, requires_grad=True)

    layer = TopologicalRK4Layer(dt=0.001, tau_s=0.1, tau_w=10.0, k_elastic=0.1, lambda_w=0.01)

    # 1. Forward Pass
    s_next, W_next = layer(s, W, stress_grad, W0)

    assert s_next.shape == (N,)
    assert W_next.shape == (N, N)

    # 2. Loss & Backward Pass
    loss = (s_next ** 2).sum() + (W_next ** 2).sum()
    loss.backward()

    # 3. Gradient checks
    assert s.grad is not None
    assert W.grad is not None
    assert stress_grad.grad is not None
    assert W0.grad is not None
    assert not torch.isnan(s.grad).any()
    assert not torch.isnan(W.grad).any()


def test_topological_rk4_relaxation():
    N = 16
    s = torch.randn(N, dtype=torch.float32) * 0.1
    W = torch.eye(N, dtype=torch.float32) * 0.05
    W0 = W.clone()
    stress_grad = torch.zeros(N, dtype=torch.float32)
    stress_grad[0:5] = 0.5  # Bounded stress injection

    layer = TopologicalRK4Layer(dt=0.001, tau_s=0.1, tau_w=10.0, k_elastic=0.1, lambda_w=0.01)

    s_curr, W_curr = s.clone(), W.clone()

    for _ in range(10):
        s_curr, W_curr = layer(s_curr, W_curr, stress_grad, W0)

    assert not torch.isnan(s_curr).any()
    assert not torch.isnan(W_curr).any()
