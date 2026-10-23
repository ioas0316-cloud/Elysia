import torch
import torch.nn as nn
import numpy as np


class ContinuousAttractorField(nn.Module):
    r"""
    Continuous Attractor Neural Network (CANN) Field with Boundary Error Coupling
    and Differential Geometric Resonance.

    Mathematical Principles:
    1. Continuous Field Dynamics:
       \tau_r \frac{\partial r(\mathbf{x}, t)}{\partial t} = -r(\mathbf{x}, t) + f\left( \int J(\mathbf{x}, \mathbf{x}') r(\mathbf{x}', t) d\mathbf{x}' + \gamma \varepsilon(\mathbf{x}, t) \right)
       where J is Mexican-hat recurrent connectivity kernel.

    2. Bump Drift Dynamics:
       \tau_{\text{drift}} \frac{d\mathbf{x}_0}{dt} = \int \nabla_{\mathbf{x}} \varepsilon(\mathbf{x}, t) \cdot r(\mathbf{x} - \mathbf{x}_0(t)) d\mathbf{x}

    3. 3D Bulk to 2D Boundary Reduction & Stokes' Theorem:
       \int_V (d\boldsymbol{\alpha}) = \int_{\partial V} \boldsymbol{\alpha}

    4. Hodge Laplacian Harmonic Resonance:
       \Delta_H \boldsymbol{\alpha} = (d d^* + d^* d) \boldsymbol{\alpha} \to 0 \implies \boldsymbol{\alpha} \in \mathcal{H}^1(M)
    """
    def __init__(self, spatial_dim=32, tau_r=0.1, gamma_error=0.5, sigma_ex=0.5, sigma_in=1.0):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.tau_r = tau_r
        self.gamma_error = gamma_error

        # Construct 1D circular position grid \mathbf{x} \in [-\pi, \pi)
        x_grid = torch.linspace(-np.pi, np.pi, spatial_dim)
        self.register_buffer('x_grid', x_grid)

        # Normalized Mexican-hat kernel J(x, x')
        dx = x_grid.unsqueeze(0) - x_grid.unsqueeze(1)
        dx = torch.remainder(dx + np.pi, 2 * np.pi) - np.pi
        kernel = torch.exp(-dx**2 / (2 * sigma_ex**2)) - 0.5 * torch.exp(-dx**2 / (2 * sigma_in**2))
        kernel = kernel / (spatial_dim * 0.1)  # Scale normalized kernel
        self.register_buffer('kernel', kernel)

        self.act = torch.relu

    def initialize_bump(self, center=0.0, width=0.5, amplitude=1.0):
        """Initializes a Gaussian attractor bump r(x) centered at center."""
        dx = torch.remainder(self.x_grid - center + np.pi, 2 * np.pi) - np.pi
        r = amplitude * torch.exp(-dx**2 / (2 * width**2))
        return r

    def compute_field_step(self, r, sensory_input, top_down_pred, dt=0.01):
        r"""
        Calculates one time-step update for continuous field dynamics driven by local error tensor.
        \varepsilon(x) = s(x) - p(x)
        """
        # Dimension-preserving local error tensor field \varepsilon(x)
        error_field = sensory_input - top_down_pred  # \in R^{spatial_dim}

        # Recurrent field excitation: \int J(x, x') r(x') dx'
        recurrent_input = torch.matmul(self.kernel, r)

        # Total activation driving force
        total_input = recurrent_input + self.gamma_error * error_field
        target_r = self.act(total_input)

        # Field dynamics update
        dr = (-r + target_r) / self.tau_r
        r_next = r + dt * dr
        return r_next, error_field

    def compute_bump_center(self, r):
        r"""Calculates Center of Mass / Peak position \mathbf{x}_0 of the bump."""
        sin_avg = torch.sum(r * torch.sin(self.x_grid))
        cos_avg = torch.sum(r * torch.cos(self.x_grid))
        center = torch.atan2(sin_avg, cos_avg)
        return center.item()

    def stokes_boundary_flux(self, bulk_field_3d):
        r"""
        3D Bulk to 2D Boundary Reduction via Stokes' Theorem:
        \int_V (\nabla \cdot F) dV = \int_{\partial V} (F \cdot n) dA
        Converts 3D volume tensor (D, D, D) to 2D surface boundary flux (6, D, D).
        """
        front = bulk_field_3d[-1, :, :]
        back = bulk_field_3d[0, :, :]
        top = bulk_field_3d[:, -1, :]
        bottom = bulk_field_3d[:, 0, :]
        right = bulk_field_3d[:, :, -1]
        left = bulk_field_3d[:, :, 0]

        boundary_flux = torch.stack([front, back, top, bottom, right, left], dim=0)
        return boundary_flux

    def hodge_laplacian_error(self, error_1form):
        r"""
        Computes Discrete Hodge Laplacian \Delta_H \alpha = (d d^* + d^* d) \alpha
        for 1-form error field \alpha to verify harmonic convergence (\Delta_H \alpha \to 0).
        """
        pad_error = torch.cat([error_1form[-1:], error_1form, error_1form[:1]])
        laplacian = -(pad_error[2:] - 2 * pad_error[1:-1] + pad_error[:-2])
        return laplacian

    def compute_free_energy_and_resonance(self, error_field):
        r"""
        Calculates Free Energy E = 0.5 * \int \alpha \wedge \star \alpha
        and Hodge Laplacian norm to assess topological resonance (\Delta \phi = 0).
        """
        free_energy = 0.5 * torch.sum(error_field ** 2).item()
        hodge_lap = self.hodge_laplacian_error(error_field)
        harmonic_norm = torch.norm(hodge_lap).item()
        return free_energy, harmonic_norm
