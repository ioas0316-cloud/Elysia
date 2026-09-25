"""
elysia_engine/core/causal_erosion.py

Causal Erosion Landscape & Phase Transition Engine
--------------------------------------------------
This module implements dynamic trajectory crystallization into static potential paths (Geodesics)
via Gaussian Causal Erosion, memory decay hysteresis, potential gradient relaxation (-∇V),
and phase transition rasterization into 3D/ND Spatial Potential Fields for O(1) sampling.
"""

import math
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any


class CausalErosionLandscape(nn.Module):
    """
    Causal Erosion Landscape Module.

    Dynamic trajectory states ψ(t) continuously erode potential landscape V(ψ) using a
    Gaussian erosion kernel K(r). As well density reaches critical density N >= N_c,
    it transitions from discrete multi-body well iteration O(M x N) into a continuous
    spatial tensor field grid for O(M) / O(1) sampling and gradient evaluation.
    """
    def __init__(
        self,
        state_dim: int = 3,
        base_k: float = 0.2,
        erosion_rate: float = 0.15,
        kernel_sigma: float = 0.4,
        decay_alpha: float = 0.01,
        nc_threshold: int = 256,
        grid_res: Tuple[int, int, int] = (64, 64, 64),
        bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)),
        device: str = "cpu"
    ):
        super().__init__()
        self.state_dim = state_dim
        self.base_k = base_k
        self.erosion_rate = erosion_rate
        self.kernel_sigma = kernel_sigma
        self.decay_alpha = decay_alpha
        self.nc_threshold = nc_threshold
        self.grid_res = grid_res
        self.bounds = bounds
        self.device = torch.device(device)

        # Eroded wells storage: centers [N, D], depths [N]
        self.register_buffer("well_centers", torch.empty((0, state_dim), device=self.device, dtype=torch.float32))
        self.register_buffer("well_depths", torch.empty((0,), device=self.device, dtype=torch.float32))

        # Phase transition continuous 3D Spatial Grid
        self.is_phase_transformed = False
        self.register_buffer(
            "grid_v",
            torch.zeros((grid_res[2], grid_res[1], grid_res[0]), device=self.device, dtype=torch.float32)
        )

        # Precompute spatial bounds and cell sizes
        self.min_bound = torch.tensor([bounds[0][0], bounds[1][0], bounds[2][0]], device=self.device, dtype=torch.float32)
        self.max_bound = torch.tensor([bounds[0][1], bounds[1][1], bounds[2][1]], device=self.device, dtype=torch.float32)
        self.cell_size = (self.max_bound - self.min_bound) / torch.tensor(grid_res, device=self.device, dtype=torch.float32)
        self.inv_cell_size = 1.0 / self.cell_size

    def base_potential(self, states: torch.Tensor) -> torch.Tensor:
        """
        Calculates harmonic base potential V_0(ψ) = 0.5 * k * ||ψ||^2
        """
        return 0.5 * self.base_k * torch.sum(states ** 2, dim=-1)

    def erode_trajectory(self, trajectory_point: torch.Tensor, depth_boost: float = 1.0):
        """
        Record a dynamic state point, excavating a new Gaussian well in the landscape.
        """
        with torch.no_grad():
            point = trajectory_point.to(self.device).detach().float()
            if point.ndim == 1:
                point = point.unsqueeze(0)

            num_new = point.size(0)
            depths = torch.full((num_new,), self.erosion_rate * depth_boost, device=self.device, dtype=torch.float32)

            self.well_centers = torch.cat([self.well_centers, point], dim=0)
            self.well_depths = torch.cat([self.well_depths, depths], dim=0)

            # Check for phase transition threshold
            if not self.is_phase_transformed and self.well_centers.size(0) >= self.nc_threshold:
                self.rasterize_to_continuous_grid()

            if self.is_phase_transformed:
                self._update_grid_with_new_wells(point, depths)

    def apply_memory_decay(self, dt: float = 0.05):
        """
        Applies memory decay hysteresis: exp(-alpha * dt) to eroded wells.
        """
        with torch.no_grad():
            if self.well_depths.size(0) > 0:
                decay_factor = math.exp(-self.decay_alpha * dt)
                self.well_depths *= decay_factor

                if self.is_phase_transformed:
                    self.grid_v *= decay_factor

    def rasterize_to_continuous_grid(self):
        """
        Rasterizes N discrete Gaussian wells into a continuous 3D Spatial Grid Field for O(1) sampling.
        """
        with torch.no_grad():
            self.is_phase_transformed = True
            gz, gy, gx = self.grid_res

            # Create 3D Grid coordinates
            z_coords = torch.linspace(self.min_bound[2] + 0.5 * self.cell_size[2], self.max_bound[2] - 0.5 * self.cell_size[2], gz, device=self.device)
            y_coords = torch.linspace(self.min_bound[1] + 0.5 * self.cell_size[1], self.max_bound[1] - 0.5 * self.cell_size[1], gy, device=self.device)
            x_coords = torch.linspace(self.min_bound[0] + 0.5 * self.cell_size[0], self.max_bound[0] - 0.5 * self.cell_size[0], gx, device=self.device)

            grid_z, grid_y, grid_x = torch.meshgrid(z_coords, y_coords, x_coords, indexing="ij")
            grid_pts = torch.stack([grid_x, grid_y, grid_z], dim=-1) # [GZ, GY, GX, 3]

            # Clear grid
            self.grid_v.zero_()

            N = self.well_centers.size(0)
            inv_two_sigma_sq = 1.0 / (2.0 * self.kernel_sigma ** 2)

            for j in range(N):
                c = self.well_centers[j]
                d = self.well_depths[j]
                dist_sq = torch.sum((grid_pts - c) ** 2, dim=-1)
                self.grid_v -= d * torch.exp(-dist_sq * inv_two_sigma_sq)

    def _update_grid_with_new_wells(self, new_centers: torch.Tensor, new_depths: torch.Tensor):
        """
        Splat newly added wells onto the existing continuous spatial grid.
        """
        gz, gy, gx = self.grid_res
        z_coords = torch.linspace(self.min_bound[2] + 0.5 * self.cell_size[2], self.max_bound[2] - 0.5 * self.cell_size[2], gz, device=self.device)
        y_coords = torch.linspace(self.min_bound[1] + 0.5 * self.cell_size[1], self.max_bound[1] - 0.5 * self.cell_size[1], gy, device=self.device)
        x_coords = torch.linspace(self.min_bound[0] + 0.5 * self.cell_size[0], self.max_bound[0] - 0.5 * self.cell_size[0], gx, device=self.device)

        grid_z, grid_y, grid_x = torch.meshgrid(z_coords, y_coords, x_coords, indexing="ij")
        grid_pts = torch.stack([grid_x, grid_y, grid_z], dim=-1)

        inv_two_sigma_sq = 1.0 / (2.0 * self.kernel_sigma ** 2)

        for j in range(new_centers.size(0)):
            c = new_centers[j]
            d = new_depths[j]
            dist_sq = torch.sum((grid_pts - c) ** 2, dim=-1)
            self.grid_v -= d * torch.exp(-dist_sq * inv_two_sigma_sq)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Evaluates potential energy V(ψ) = V_0(ψ) + ΔV(ψ).
        Supports states tensor of shape [M, 3] or [3].
        """
        squeeze_out = False
        if states.ndim == 1:
            states = states.unsqueeze(0)
            squeeze_out = True

        states = states.to(self.device)
        v = self.base_potential(states)

        if self.well_centers.size(0) == 0:
            return v.squeeze(0) if squeeze_out else v

        if not self.is_phase_transformed:
            # Discrete Gaussian well summation O(M x N)
            diff = states.unsqueeze(1) - self.well_centers.unsqueeze(0) # [M, N, 3]
            dist_sq = torch.sum(diff ** 2, dim=-1) # [M, N]
            inv_two_sigma_sq = 1.0 / (2.0 * self.kernel_sigma ** 2)
            wells_v = torch.sum(self.well_depths.unsqueeze(0) * torch.exp(-dist_sq * inv_two_sigma_sq), dim=-1)
            v = v - wells_v
        else:
            # Continuous Field O(M) sampling via trilinear interpolation
            v_field = self._sample_grid_v(states)
            v = v + v_field

        return v.squeeze(0) if squeeze_out else v

    def _sample_grid_v(self, states: torch.Tensor) -> torch.Tensor:
        """
        Samples continuous potential grid V_field via normalized trilinear grid_sample.
        """
        # Normalize states to [-1, 1] range for grid_sample
        norm_states = 2.0 * (states - self.min_bound) / (self.max_bound - self.min_bound) - 1.0
        # reshape for grid_sample: [1, 1, 1, M, 3]
        grid_input = norm_states.view(1, 1, 1, -1, 3)
        grid_v_5d = self.grid_v.view(1, 1, self.grid_res[2], self.grid_res[1], self.grid_res[0])

        sampled = torch.nn.functional.grid_sample(
            grid_v_5d, grid_input, mode="bilinear", padding_mode="border", align_corners=True
        )
        return sampled.view(-1)

    def compute_gradient_force(self, states: torch.Tensor) -> torch.Tensor:
        """
        Computes relaxation force vector F(ψ) = -∇V(ψ).
        Calculated analytically or via autograd gradient flow.
        """
        squeeze_out = False
        if states.ndim == 1:
            states = states.unsqueeze(0)
            squeeze_out = True

        states = states.to(self.device)
        M = states.size(0)

        # Base force: -k * ψ
        forces = -self.base_k * states

        if self.well_centers.size(0) == 0:
            return forces.squeeze(0) if squeeze_out else forces

        if not self.is_phase_transformed:
            # Discrete analytical force sum
            diff = states.unsqueeze(1) - self.well_centers.unsqueeze(0) # [M, N, 3]
            dist_sq = torch.sum(diff ** 2, dim=-1) # [M, N]
            inv_two_sigma_sq = 1.0 / (2.0 * self.kernel_sigma ** 2)
            inv_sigma_sq = 1.0 / (self.kernel_sigma ** 2)

            well_v = self.well_depths.unsqueeze(0) * torch.exp(-dist_sq * inv_two_sigma_sq) # [M, N]
            coeff = well_v * inv_sigma_sq # [M, N]
            forces_wells = torch.sum(coeff.unsqueeze(-1) * diff, dim=1) # [M, 3]
            forces = forces + forces_wells
        else:
            # Numerical central difference gradient on grid
            eps = 1e-3
            for dim in range(self.state_dim):
                delta = torch.zeros_like(states)
                delta[:, dim] = eps
                v_plus = self._sample_grid_v(states + delta)
                v_minus = self._sample_grid_v(states - delta)
                grad_dim = (v_plus - v_minus) / (2.0 * eps)
                forces[:, dim] -= grad_dim

        return forces.squeeze(0) if squeeze_out else forces

    def memory_replay_step(self, current_states: torch.Tensor, dt: float = 0.05, speed: float = 0.4) -> torch.Tensor:
        """
        Spontaneous memory replay step: states slide along eroded geodesics under -∇V without external forces.
        """
        with torch.no_grad():
            forces = self.compute_gradient_force(current_states)
            next_states = current_states + speed * forces * dt
            return next_states
