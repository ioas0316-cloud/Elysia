"""
Clifford Spinor Recall Engine.

This module implements:
1. CliffordSpinorRecallEngine:
   - Computes Riemannian natural gradient recall trajectories towards target memory attractors.
   - Integrates Clifford spinor precession dynamics along curved Riemannian metric fields.
   - Computes variational geodesic path optimization minimizing action energy.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim


class CliffordSpinorRecallEngine(nn.Module):
    """
    Clifford Spinor Recall Engine.
    Executes intuitive geodesic recall flows and spinor frame precession over deformed Riemannian metric fields.
    """

    def __init__(self, state_dim: int):
        super().__init__()
        self.state_dim = state_dim

    def riemannian_distance(self, x_a: torch.Tensor, x_b: torch.Tensor, g_mem: torch.Tensor) -> torch.Tensor:
        """
        Calculates local Riemannian metric distance under metric g_mem:
        d_R = sqrt( (x_a - x_b)^T g_mem (x_a - x_b) )
        """
        diff = (x_a - x_b).unsqueeze(-1)  # [B, N, 1]
        dist_sq = torch.bmm(torch.bmm(diff.transpose(-1, -2), g_mem.unsqueeze(0)), diff)
        return torch.sqrt(torch.clamp(dist_sq.squeeze(), min=1e-8))

    def recall_dynamics(
        self,
        x_init: torch.Tensor,
        Q_init: torch.Tensor,
        target_attractor: torch.Tensor,
        g_mem: torch.Tensor,
        Omega_skew: torch.Tensor,
        steps: int = 100,
        dt: float = 0.04,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
        """
        Simulates coupled position Riemannian natural gradient flow and spinor frame precession:
        dx/dt = - g_mem^-1 * (x - target)
        dQ/dt = -0.5 * Q * exp(-dt * Omega_skew)
        """
        x_traj = [x_init.clone()]
        Q_traj = [Q_init.clone()]
        energy_history = []

        x = x_init.clone()
        Q = Q_init.clone()

        g_inv = torch.linalg.inv(g_mem)

        for step_idx in range(steps):
            # 1. Potential gradient
            grad_V = x - target_attractor  # [B, N]

            # 2. Riemannian natural gradient step
            riemannian_grad = torch.matmul(g_inv, grad_V.T).T
            x = x - dt * riemannian_grad

            # 3. Spinor precession step
            rot_step = torch.matrix_exp(-dt * Omega_skew)
            Q = torch.bmm(Q, rot_step)

            # Energy calculation
            dist_r = self.riemannian_distance(x, target_attractor, g_mem)
            energy_history.append(dist_r.mean().item())

            x_traj.append(x.clone())
            Q_traj.append(Q.clone())

        return torch.stack(x_traj, dim=0), torch.stack(Q_traj, dim=0), energy_history

    def optimize_geodesic_path(
        self,
        x_start: torch.Tensor,
        x_target: torch.Tensor,
        g_mem: torch.Tensor,
        num_waypoints: int = 30,
        opt_steps: int = 150,
    ) -> torch.Tensor:
        """
        Variational geodesic path optimization:
        Minimizes Action Energy E = sum_i (delta x_i)^T g_mem (delta x_i)
        """
        # Linear initialization
        alpha = torch.linspace(0, 1, num_waypoints, device=x_start.device).unsqueeze(1)
        waypoints = (1 - alpha) * x_start + alpha * x_target

        middle_waypoints = nn.Parameter(waypoints[1:-1].clone())
        optimizer = optim.Adam([middle_waypoints], lr=0.01)

        for _ in range(opt_steps):
            optimizer.zero_grad()

            full_path = torch.cat([x_start, middle_waypoints, x_target], dim=0)
            diffs = full_path[1:] - full_path[:-1]  # [N-1, state_dim]

            # Action energy E
            energy = torch.sum(torch.matmul(diffs, g_mem) * diffs)
            energy.backward()
            optimizer.step()

        full_geodesic = torch.cat([x_start, middle_waypoints.detach(), x_target], dim=0)
        return full_geodesic
