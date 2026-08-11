import torch
import torch.nn as nn
import torch.nn.functional as F


def test_harmonic_fourier_projection():
    from core.physics.continuous_field_encoder import HarmonicFourierProjection
    proj = HarmonicFourierProjection(n_harmonics=16, max_freq=50.0)
    x = torch.rand(4, 32)
    out = proj(x)
    assert out.shape == (4, 32, 32)


def test_continuous_field_encoder_invariants():
    from core.physics.continuous_field_encoder import ContinuousFieldEncoder
    d_model = 256
    encoder = ContinuousFieldEncoder(d_model=d_model, n_harmonics=32)

    # Batch size 2, sequence length 64
    x = torch.rand(2, 64)
    v0, u0, w0, theta0 = encoder(x)

    assert v0.shape == (2, d_model)
    assert u0.shape == (2, d_model)
    assert w0.shape == (2, d_model)
    assert theta0.shape == (2, 1)

    # 1. ||v0|| == 1.0 (초구면 위상 상태 벡터)
    v0_norm = torch.norm(v0, p=2, dim=-1)
    assert torch.allclose(v0_norm, torch.ones_like(v0_norm), atol=1e-5)

    # 2. ||u0|| == 1.0, ||w0|| == 1.0
    u0_norm = torch.norm(u0, p=2, dim=-1)
    w0_norm = torch.norm(w0, p=2, dim=-1)
    assert torch.allclose(u0_norm, torch.ones_like(u0_norm), atol=1e-5)
    assert torch.allclose(w0_norm, torch.ones_like(w0_norm), atol=1e-5)

    # 3. <u0, w0> == 0.0 (그람-슈미트 직교성 만족)
    inner_uw = (u0 * w0).sum(dim=-1)
    assert torch.allclose(inner_uw, torch.zeros_like(inner_uw), atol=1e-5)

    # 4. θ0 > 0.0 (Softplus 사용으로 양수 보장)
    assert (theta0 > 0.0).all()
