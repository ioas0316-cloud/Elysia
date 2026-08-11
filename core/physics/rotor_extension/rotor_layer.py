"""
rotor_layer.py
==============
Clifford Geometric Algebra Rotor Sandwich Layer ($R v R^\\dagger$) with hybrid execution support.
Features dynamic binding of PyTorch CUDA Extension and ultra-optimized Native PyTorch Autograd Fallback.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

HAS_CUDA_EXT = False

try:
    if torch.cuda.is_available():
        import rotor_cuda_ext
        HAS_CUDA_EXT = True
except ImportError:
    HAS_CUDA_EXT = False


# =====================================================================
# 1. Native PyTorch Autograd Fallback (Analytical & Numerically Stable)
# =====================================================================
def rotor_sandwich_native_pytorch(
    v: torch.Tensor,
    u: torch.Tensor,
    w: torch.Tensor,
    theta: torch.Tensor
) -> torch.Tensor:
    """
    Pure PyTorch equivalent implementation of Clifford Rotor Sandwich:
        R v R^\\dagger = v + sin(\\theta) A(v) + (1 - cos(\\theta)) A^2(v)
        where A(v) = w * (u . v) - u * (w . v)
    """
    # 1. Projection components (Dot Products)
    # v_dot_u = <v, u>, v_dot_w = <v, w>
    v_dot_u = torch.sum(v * u, dim=-1, keepdim=True)  # [Batch, 1]
    v_dot_w = torch.sum(v * w, dim=-1, keepdim=True)  # [Batch, 1]

    # 2. Skew-Symmetric Operator A(v) and A^2(v)
    Av = w * v_dot_u - u * v_dot_w                   # [Batch, D]
    A2v = -(u * v_dot_u + w * v_dot_w)               # [Batch, D]

    # 3. Trigonometric coefficients
    sin_t = torch.sin(theta)                         # [Batch, 1] or Broadcastable
    one_minus_cos_t = 1.0 - torch.cos(theta)         # [Batch, 1] or Broadcastable

    # 4. Combine terms
    v_rotated = v + sin_t * Av + one_minus_cos_t * A2v
    return v_rotated


# =====================================================================
# 2. Custom CUDA Extension Autograd Wrapper
# =====================================================================
if HAS_CUDA_EXT:
    class RotorSandwichFunctionCUDA(torch.autograd.Function):
        @staticmethod
        def forward(ctx, v, u, w, theta):
            # Input validation & continuous memory layout ensuring
            v = v.contiguous()
            u = u.contiguous()
            w = w.contiguous()
            theta = theta.contiguous()

            ctx.save_for_backward(v, u, w, theta)
            out = rotor_cuda_ext.forward(v, u, w, theta)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            v, u, w, theta = ctx.saved_tensors
            # Trigger custom C++/CUDA fast backward kernel
            grad_v, grad_u, grad_w, grad_theta = rotor_cuda_ext.backward(
                grad_output.contiguous(), v, u, w, theta
            )
            return grad_v, grad_u, grad_w, grad_theta
else:
    RotorSandwichFunctionCUDA = None


# =====================================================================
# 3. Dynamic Dispatcher
# =====================================================================
def apply_rotor_sandwich(
    v: torch.Tensor,
    u: torch.Tensor,
    w: torch.Tensor,
    theta: torch.Tensor
) -> torch.Tensor:
    """
    Seamless dispatcher that runs CUDA Extension when available and on GPU,
    otherwise falls back to native PyTorch Autograd execution.
    """
    if HAS_CUDA_EXT and v.is_cuda:
        return RotorSandwichFunctionCUDA.apply(v, u, w, theta)
    else:
        return rotor_sandwich_native_pytorch(v, u, w, theta)


# =====================================================================
# 4. Clifford Rotor Sandwich PyTorch Module (Neural Layer)
# =====================================================================
class RotorSandwichLayer(nn.Module):
    """
    N-dimensional Clifford Geometric Algebra Rotor Layer.
    Orthonormalizes u and w via Gram-Schmidt and rotates x by theta.
    """
    def __init__(self, features: int):
        super().__init__()
        self.features = features

        # Projections to dynamically generate rotation plane candidates & angles from state x
        self.proj_u = nn.Linear(features, features, bias=False)
        self.proj_w = nn.Linear(features, features, bias=False)
        self.proj_theta = nn.Linear(features, 1)

        self.reset_parameters()

    def reset_parameters(self):
        # Orthogonal initialization for u & w projections to stabilize training
        nn.init.orthogonal_(self.proj_u.weight)
        nn.init.orthogonal_(self.proj_w.weight)
        nn.init.zeros_(self.proj_theta.weight)

    def _gram_schmidt_orthonormalize(self, p_u: torch.Tensor, p_w: torch.Tensor):
        """
        Orthonormalizes projection candidates p_u, p_w to maintain geometric invariants:
            <u, w> = 0, ||u|| = ||w|| = 1
        """
        u = F.normalize(p_u, dim=-1, eps=1e-8)

        # Remove projection of p_w onto u
        proj_u_w = torch.sum(p_w * u, dim=-1, keepdim=True) * u
        w = F.normalize(p_w - proj_u_w, dim=-1, eps=1e-8)
        return u, w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Project to candidate space
        p_u = self.proj_u(x)
        p_w = self.proj_w(x)
        theta = self.proj_theta(x)  # [Batch, 1]

        # 2. Orthonormalize via Gram-Schmidt
        u, w = self._gram_schmidt_orthonormalize(p_u, p_w)

        # 3. Apply the dynamic Clifford Rotor rotation sandwich
        x_rotated = apply_rotor_sandwich(x, u, w, theta)
        return x_rotated


# =====================================================================
# 5. Cognitive Rotor Neural Network
# =====================================================================
class CognitiveRotorNetwork(nn.Module):
    """
    Multi-layer neural network leveraging Clifford Geometric Algebra rotor layers
    for high-dimensional continuous latent space optimization.
    """
    def __init__(self, in_features: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.input_proj = nn.Linear(in_features, hidden_dim)

        self.rotor1 = RotorSandwichLayer(hidden_dim)
        self.act1 = nn.SiLU()
        self.rotor2 = RotorSandwichLayer(hidden_dim)
        self.act2 = nn.SiLU()

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.rotor1(h)
        h = self.act1(h)
        h = self.rotor2(h)
        h = self.act2(h)
        logits = self.classifier(h)
        return logits
