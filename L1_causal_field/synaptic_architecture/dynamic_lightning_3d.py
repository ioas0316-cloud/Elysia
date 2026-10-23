"""
DynamicLightning3D & Hebbian Field Plasticity Engine
Elysia Engine Core Architectural Module

Implements:
1. DynamicLightning3D: Dual-potential injection (+1 leader, -1 or +1 streamer), 3D conv3d wave diffusion,
   time-varying conductivity grid C(x,y,z,t), and non-linear power-law channel collapse (gamma exponent).
2. HebbianFieldPlasticityEngine: Continuous 3D field plasticity where flux density J = -Sigma * grad(Psi)
   drives self-organizing axonal highway formation via flux reinforcement (alpha * |J|^eta),
   spontaneous dissipation (-beta * Sigma), and spatial smoothing (D * laplacian(Sigma)).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicLightning3D(nn.Module):
    """
    3D Dynamic Lightning Wave Relaxation & Non-linear Channel Collapse System.
    Propagates dual potential waves (start and goal) across a 3D conductivity manifold C(x,y,z,t).
    """

    def __init__(self, shape=(32, 32, 32), device='cpu'):
        super().__init__()
        self.d, self.h, self.w = shape
        self.device = torch.device(device if isinstance(device, torch.device) else (device if torch.cuda.is_available() and device != 'cpu' else 'cpu'))

        # 3D Laplacian Kernel (6-neighbor stencil)
        kernel_data = torch.zeros((1, 1, 3, 3, 3), dtype=torch.float32)
        kernel_data[0, 0, 1, 1, 0] = 1 / 6.0  # Front
        kernel_data[0, 0, 1, 1, 2] = 1 / 6.0  # Back
        kernel_data[0, 0, 1, 0, 1] = 1 / 6.0  # Top
        kernel_data[0, 0, 1, 2, 1] = 1 / 6.0  # Bottom
        kernel_data[0, 0, 0, 1, 1] = 1 / 6.0  # Left
        kernel_data[0, 0, 2, 1, 1] = 1 / 6.0  # Right
        self.register_buffer("kernel", kernel_data)

        # Potential fields (1, 1, D, H, W)
        self.register_buffer("v_start", torch.zeros((1, 1, self.d, self.h, self.w), dtype=torch.float32))
        self.register_buffer("v_goal", torch.zeros((1, 1, self.d, self.h, self.w), dtype=torch.float32))

    def reset_fields(self):
        """Resets v_start and v_goal to zeros."""
        self.v_start.zero_()
        self.v_goal.zero_()

    def step(self, start_pos: tuple, goal_pos: tuple, conductivity_grid: torch.Tensor, gamma: float = 16.0, relax_steps: int = 3) -> torch.Tensor:
        """
        Performs continuous potential diffusion and collapses non-linear lightning channel.

        start_pos, goal_pos: (z, y, x) coordinates
        conductivity_grid: (D, H, W) tensor with values in [0, 1]
        returns: lightning_3d channel (D, H, W)
        """
        c_mask = conductivity_grid.unsqueeze(0).unsqueeze(0).to(self.device)

        # Clamping Boundary Conditions (Leader & Streamer Injectors)
        sz, sy, sx = start_pos
        gz, gy, gx = goal_pos

        self.v_start[0, 0, sz, sy, sx] = 1.0
        self.v_goal[0, 0, gz, gy, gx] = 1.0

        # Continuous Field Diffusion
        for _ in range(relax_steps):
            self.v_start = F.conv3d(self.v_start, self.kernel, padding=1) * c_mask
            self.v_goal = F.conv3d(self.v_goal, self.kernel, padding=1) * c_mask

            # Re-enforce boundary clamping
            self.v_start[0, 0, sz, sy, sx] = 1.0
            self.v_goal[0, 0, gz, gy, gx] = 1.0

        # Minimal action coupling energy and non-linear channel collapse
        product_field = self.v_start * self.v_goal
        max_val = torch.max(product_field)

        if max_val > 0:
            norm_field = product_field / max_val
            lightning_3d = torch.pow(norm_field, gamma).squeeze()
        else:
            lightning_3d = torch.zeros((self.d, self.h, self.w), device=self.device)

        return lightning_3d


class HebbianFieldPlasticityEngine(nn.Module):
    """
    3D Hebbian Plasticity Engine.
    Extends Hebbian learning ("cells that fire together wire together") to a continuous 3D field tensor.
    Updates conductivity Sigma(x,y,z,t) based on wave current density flux J = -Sigma * grad(Psi).

    dSigma/dt = alpha * |J|^eta - beta * Sigma + D * laplacian(Sigma)
    """

    def __init__(self, shape=(32, 32, 32), alpha: float = 0.5, beta: float = 0.02, eta: float = 1.5, D: float = 0.01, device='cpu'):
        super().__init__()
        self.d, self.h, self.w = shape
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.D = D
        self.device = torch.device(device if isinstance(device, torch.device) else (device if torch.cuda.is_available() and device != 'cpu' else 'cpu'))

        # Conductivity field Sigma (1, 1, D, H, W)
        self.register_buffer("sigma", torch.ones((1, 1, self.d, self.h, self.w), dtype=torch.float32, device=self.device) * 0.5)

        # 3D Laplacian kernel for diffusion
        kernel_data = torch.zeros((1, 1, 3, 3, 3), dtype=torch.float32)
        kernel_data[0, 0, 1, 1, 0] = 1 / 6.0
        kernel_data[0, 0, 1, 1, 2] = 1 / 6.0
        kernel_data[0, 0, 1, 0, 1] = 1 / 6.0
        kernel_data[0, 0, 1, 2, 1] = 1 / 6.0
        kernel_data[0, 0, 0, 1, 1] = 1 / 6.0
        kernel_data[0, 0, 2, 1, 1] = 1 / 6.0
        kernel_data[0, 0, 1, 1, 1] = -1.0
        self.register_buffer("laplacian_kernel", kernel_data)

    def compute_gradient_3d(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Computes 3D gradient grad(Psi) using central differences.
        psi: (1, 1, D, H, W) tensor
        returns: magnitude of gradient |grad(Psi)| of shape (1, 1, D, H, W)
        """
        # Padding for boundary differences
        p = F.pad(psi, (1, 1, 1, 1, 1, 1), mode='replicate')

        grad_z = (p[:, :, 2:, 1:-1, 1:-1] - p[:, :, :-2, 1:-1, 1:-1]) / 2.0
        grad_y = (p[:, :, 1:-1, 2:, 1:-1] - p[:, :, 1:-1, :-2, 1:-1]) / 2.0
        grad_x = (p[:, :, 1:-1, 1:-1, 2:] - p[:, :, 1:-1, 1:-1, :-2]) / 2.0

        grad_mag = torch.sqrt(grad_z**2 + grad_y**2 + grad_x**2 + 1e-8)
        return grad_mag

    def update_plasticity(self, potential_field: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        """
        Updates conductivity field Sigma according to Hebbian dynamics:
        J = -Sigma * grad(Psi)
        dSigma = (alpha * |J|^eta - beta * Sigma + D * laplacian(Sigma)) * dt
        """
        if potential_field.dim() == 3:
            potential_field = potential_field.unsqueeze(0).unsqueeze(0)

        # 1. Flux density J
        grad_mag = self.compute_gradient_3d(potential_field)
        flux_J = self.sigma * grad_mag

        # 2. Reinforcement term alpha * |J|^eta
        reinforcement = self.alpha * torch.pow(flux_J, self.eta)

        # 3. Dissipation term -beta * Sigma
        dissipation = -self.beta * self.sigma

        # 4. Spatial diffusion D * laplacian(Sigma)
        diffusion = self.D * F.conv3d(self.sigma, self.laplacian_kernel, padding=1)

        # 5. Time step update
        d_sigma = (reinforcement + dissipation + diffusion) * dt
        self.sigma = torch.clamp(self.sigma + d_sigma, min=0.001, max=1.0)

        return self.sigma.squeeze()
