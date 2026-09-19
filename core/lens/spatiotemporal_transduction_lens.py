"""
Spatiotemporal Transduction Lens & High-Dimensional Spinor-Riemannian Autograd Layer.

This module implements:
1. StructuralReceptor: Transduces external energy/stimulus into internal state signals (S_bound).
2. TopologicalMapper: Maps transduced signals into tangent shift vectors (v_shift).
3. SpinorRiemannianStepFunction: Custom PyTorch autograd function performing Riemannian
   natural gradient flow and Spin(N)/SO(N) frame rotation via matrix exponentials.
4. HighDimSpinorRiemannianLayer: High-dimensional trainable PyTorch module.
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class StructuralReceptor(nn.Module):
    """
    Transduction Module:
    Destroys external raw physical force/stimulus and transduces it into internal
    meaning signal (S_bound) through structural coupling parameter W_rec.
    """

    def __init__(self, ext_dim: int, bound_dim: int, threshold: float = 0.05):
        super().__init__()
        self.W_rec = nn.Parameter(torch.randn(ext_dim) * 0.1)
        self.transduce_layer = nn.Linear(ext_dim, bound_dim)
        self.threshold = nn.Parameter(torch.tensor(threshold))

    def forward(self, Xi_ext: torch.Tensor) -> torch.Tensor:
        # 1. Structural coupling resonance
        coupled = Xi_ext * torch.sigmoid(self.W_rec)

        # 2. Threshold gating for random noise rejection
        thresh_abs = torch.abs(self.threshold)
        masked = torch.where(
            torch.abs(coupled) > thresh_abs,
            coupled,
            torch.zeros_like(coupled),
        )

        # 3. Transduction into internal boundary language (S_bound)
        S_bound = torch.tanh(self.transduce_layer(masked))
        return S_bound


class TopologicalMapper(nn.Module):
    """
    Topological Mapping Module:
    Projects transduced internal boundary signals (S_bound) into state shift
    vectors (v_shift) on the system tangent space.
    """

    def __init__(self, bound_dim: int, state_dim: int):
        super().__init__()
        self.Pi_map = nn.Linear(bound_dim, state_dim, bias=False)

    def forward(self, S_bound: torch.Tensor) -> torch.Tensor:
        return self.Pi_map(S_bound)


class SpinorRiemannianStepFunction(Function):
    r"""
    N-Dimensional Spinor-Riemannian Dynamics Step (Custom Autograd Function).
    Supports exact Lie Algebra \mathfrak{so}(N) skew-symmetric bivector torque integration
    and Riemannian natural gradient flows on spatially varying metric fields.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        Q: torch.Tensor,
        target: torch.Tensor,
        anchors: torch.Tensor,
        metric_deforms: torch.Tensor,
        bivector_weights: torch.Tensor,
        dt: float = 0.04,
        sigma: float = 0.8,
    ):
        """
        Forward Pass:
        x: [B, N] - Current state coordinates
        Q: [B, N, N] - Current spinor orientation frame in SO(N)
        target: [B, N] - Target memory attractor
        anchors: [M, N] - Anchor positions in state space
        metric_deforms: [M, N, N] - Local metric tensor deformation matrices
        bivector_weights: [M, N, N] - Raw bivector torque matrices
        """
        B, N = x.shape
        M = anchors.shape[0]

        # 1. RBF Kernel weights
        diff = x.unsqueeze(1) - anchors.unsqueeze(0)  # [B, M, N]
        sq_dist = torch.sum(diff**2, dim=-1)  # [B, M]
        weights = torch.exp(-sq_dist / (2 * (sigma**2)))  # [B, M]

        # 2. Spatially varying Riemannian metric g_mem(x) & inverse
        g_base = torch.eye(N, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(B, 1, 1)
        g_deform = torch.sum(weights.view(B, M, 1, 1) * metric_deforms.unsqueeze(0), dim=1)
        g_x = g_base + g_deform  # [B, N, N]
        g_inv = torch.linalg.inv(g_x)  # [B, N, N]

        # 3. Riemannian Natural Gradient step
        grad_V = x - target  # [B, N]
        riemannian_grad = torch.bmm(g_inv, grad_V.unsqueeze(-1)).squeeze(-1)  # [B, N]
        x_next = x - dt * riemannian_grad

        # 4. Skew-symmetric Lie Algebra \mathfrak{so}(N) torque \Omega(x)
        skew_bivectors = 0.5 * (bivector_weights - bivector_weights.transpose(-1, -2))
        Omega = torch.sum(weights.view(B, M, 1, 1) * skew_bivectors.unsqueeze(0), dim=1)  # [B, N, N]

        # 5. Spinor frame rotation update Q_{t+1} = Q_t * exp(-\Delta t * \Omega(x))
        rot_step = torch.matrix_exp(-dt * Omega)  # [B, N, N]
        Q_next = torch.bmm(Q, rot_step)

        ctx.save_for_backward(
            x, Q, target, anchors, metric_deforms, bivector_weights, weights, diff, g_inv, Omega, rot_step
        )
        ctx.dt = dt
        ctx.sigma = sigma

        return x_next, Q_next

    @staticmethod
    def backward(ctx, grad_x_next, grad_Q_next):
        (
            x,
            Q,
            target,
            anchors,
            metric_deforms,
            bivector_weights,
            weights,
            diff,
            g_inv,
            Omega,
            rot_step,
        ) = ctx.saved_tensors
        dt = ctx.dt
        sigma = ctx.sigma

        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            Q_in = Q.detach().requires_grad_(True)
            anchors_in = anchors.detach().requires_grad_(True)
            metric_deforms_in = metric_deforms.detach().requires_grad_(True)
            bivector_weights_in = bivector_weights.detach().requires_grad_(True)

            B, N = x_in.shape
            M = anchors_in.shape[0]

            diff_in = x_in.unsqueeze(1) - anchors_in.unsqueeze(0)
            sq_dist_in = torch.sum(diff_in**2, dim=-1)
            weights_in = torch.exp(-sq_dist_in / (2 * (sigma**2)))

            g_base = torch.eye(N, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(B, 1, 1)
            g_deform_in = torch.sum(weights_in.view(B, M, 1, 1) * metric_deforms_in.unsqueeze(0), dim=1)
            g_inv_in = torch.linalg.inv(g_base + g_deform_in)

            grad_V_in = x_in - target
            riemannian_grad_in = torch.bmm(g_inv_in, grad_V_in.unsqueeze(-1)).squeeze(-1)
            x_next_func = x_in - dt * riemannian_grad_in

            skew_biv = 0.5 * (bivector_weights_in - bivector_weights_in.transpose(-1, -2))
            Omega_in = torch.sum(weights_in.view(B, M, 1, 1) * skew_biv.unsqueeze(0), dim=1)
            rot_step_in = torch.matrix_exp(-dt * Omega_in)
            Q_next_func = torch.bmm(Q_in, rot_step_in)

            grads = torch.autograd.grad(
                outputs=[x_next_func, Q_next_func],
                inputs=[x_in, Q_in, anchors_in, metric_deforms_in, bivector_weights_in],
                grad_outputs=[grad_x_next, grad_Q_next],
                retain_graph=False,
                allow_unused=True,
            )

        grad_x, grad_Q, grad_anchors, grad_deforms, grad_bivectors = grads
        return grad_x, grad_Q, None, grad_anchors, grad_deforms, grad_bivectors, None, None


class HighDimSpinorRiemannianLayer(nn.Module):
    """
    High-Dimensional Spinor-Riemannian Neural Layer.
    Wraps SpinorRiemannianStepFunction as a trainable PyTorch module.
    """

    def __init__(self, state_dim: int, num_anchors: int, dt: float = 0.04, sigma: float = 0.8):
        super().__init__()
        self.state_dim = state_dim
        self.num_anchors = num_anchors
        self.dt = dt
        self.sigma = sigma

        # Learnable parameters
        self.anchors = nn.Parameter(torch.randn(num_anchors, state_dim) * 0.5)

        # Symmetric positive-definite matrix parameterization
        raw_deforms = torch.randn(num_anchors, state_dim, state_dim) * 0.1
        self.metric_deforms = nn.Parameter(torch.bmm(raw_deforms, raw_deforms.transpose(-1, -2)))

        # Skew-symmetric Lie algebra torque parameters
        self.bivector_weights = nn.Parameter(torch.randn(num_anchors, state_dim, state_dim) * 0.2)

    def forward(self, x: torch.Tensor, Q: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return SpinorRiemannianStepFunction.apply(
            x,
            Q,
            target,
            self.anchors,
            self.metric_deforms,
            self.bivector_weights,
            self.dt,
            self.sigma,
        )
