"""
Dynamic Observation Lens Engine for Elysia.

This module implements the continuous physical-informational observation architecture:
1. DynamicObservationLens:
   - Evaluates observation friction stress tensor F_ij from state wave gradients / outer products.
   - Dynamically evolves local metric tensor G_ij via metric differential equation:
     ∂G_ij/∂t = -κ R_ij + η F_ij - γ (G_ij - G_0)
   - Performs spectral wave refraction Ψ' = Ψ @ G^(-1) without discrete conditional branching.
2. 2D FFT k-Space Spectrum Analysis & Clifford Rotor Extraction:
   - Tracks spectral splitting in wavenumber domain P(k_x, k_y).
   - Inverts Clifford 2-bivector rotation angle θ_rotor, chord length, and rotor components R(t).
"""

import math
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn


class DynamicObservationLens(nn.Module):
    """
    Dynamic Observation Lens operating as a variable physical metric substrate G_ij.
    Converts observation friction F_ij into real-time metric curvature shifts
    and refracts wave packets via inverse metric tensor operators.
    """

    def __init__(
        self,
        dim: int = 4,
        eta: float = 0.15,
        gamma: float = 0.05,
        kappa: float = 0.01,
        device: Optional[str] = None
    ):
        super().__init__()
        self.dim = dim
        self.eta = eta      # Friction sensitivity
        self.gamma = gamma    # Baseline relaxation rate
        self.kappa = kappa    # Ricci / curvature damping rate

        if device is None:
            self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device_str = device

        dev = torch.device(self.device_str)

        # Baseline isotropic metric G_0 = I
        self.register_buffer("G_base", torch.eye(dim, dtype=torch.float32, device=dev))
        # Current dynamic metric G_ij
        self.register_buffer("G", torch.eye(dim, dtype=torch.float32, device=dev))

    def _compute_friction_tensor(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Calculates observation friction shear stress tensor F_ij from input wave tensor psi.
        psi shape: [Batch, Dim] or [Batch, ..., Dim]
        """
        if psi.dim() > 2:
            psi_flat = psi.reshape(-1, self.dim)
        else:
            psi_flat = psi

        grad_psi = psi_flat - psi_flat.mean(dim=0, keepdim=True)  # [Batch, Dim]
        outer_product = torch.matmul(grad_psi.unsqueeze(2), grad_psi.unsqueeze(1))  # [Batch, Dim, Dim]
        F_raw = outer_product.mean(dim=0)  # [Dim, Dim]

        # Extract shear stress component: F_shear = F_raw - (1/n) * Tr(G^(-1) F_raw) * G
        G_inv = torch.linalg.inv(self.G)
        trace_F = torch.trace(torch.matmul(G_inv, F_raw))
        F_shear = F_raw - (1.0 / self.dim) * trace_F * self.G
        return F_shear

    def _compute_pseudo_ricci(self) -> torch.Tensor:
        """
        Approximates spatial tension curvature R_ij based on metric deviation from baseline.
        """
        diff = self.G - self.G_base
        return torch.matmul(diff, diff.t())

    def update_lens_metric(self, psi: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        """
        Updates dynamic metric tensor G_ij via differential evolution ∂G_ij/∂t.
        """
        with torch.no_grad():
            F_ij = self._compute_friction_tensor(psi)
            R_ij = self._compute_pseudo_ricci()

            # Differential rate: dG/dt = -kappa * R_ij + eta * F_ij - gamma * (G - G_0)
            dG_dt = -self.kappa * R_ij + self.eta * F_ij - self.gamma * (self.G - self.G_base)

            # Continuous Euler update
            self.G += dG_dt * dt

            # Enforce symmetry and positive-definiteness
            self.G = 0.5 * (self.G + self.G.t())

        return self.G

    def forward(self, psi: torch.Tensor, update_metric: bool = True, dt: float = 0.1) -> torch.Tensor:
        """
        Refracts wave packet Ψ through variable metric G_ij.
        Output: Refracted wave tensor Ψ' = Ψ @ G^(-1)
        """
        G_inv = torch.linalg.inv(self.G)
        if psi.dim() == 1:
            refracted_psi = torch.matmul(psi.unsqueeze(0), G_inv).squeeze(0)
        else:
            refracted_psi = torch.matmul(psi, G_inv)

        if update_metric:
            self.update_lens_metric(psi, dt=dt)

        return refracted_psi


def extract_clifford_rotor_from_kspace(
    power_spectrum: np.ndarray,
    kx_axis: np.ndarray,
    ky_axis: np.ndarray,
    k0_magnitude: float = 1.2
) -> Dict[str, Any]:
    """
    Extracts wavenumber peak separation (Δk_x, Δk_y) from 2D FFT power spectrum P(k_x, k_y)
    and reconstructs Clifford 2-bivector rotation angle θ_rotor and rotor R components.
    """
    flat_indices = np.argsort(power_spectrum.ravel())[::-1]

    # Primary peak 1
    idx1 = flat_indices[0]
    iy1, ix1 = np.unravel_index(idx1, power_spectrum.shape)
    k1 = np.array([kx_axis[ix1], ky_axis[iy1]], dtype=np.float64)

    # Primary peak 2 (with minimum spatial separation)
    k2 = None
    min_separation_idx = 3
    for idx in flat_indices[1:]:
        iy2, ix2 = np.unravel_index(idx, power_spectrum.shape)
        if np.hypot(ix2 - ix1, iy2 - iy1) >= min_separation_idx:
            k2 = np.array([kx_axis[ix2], ky_axis[iy2]], dtype=np.float64)
            break

    if k2 is None:
        # Fallback if single peak
        k2 = k1.copy()

    # Peak displacement & chord length
    delta_k = k1 - k2
    delta_kx, delta_ky = float(delta_k[0]), float(delta_k[1])
    chord_length = float(np.linalg.norm(delta_k))

    # Invert Clifford rotor angle θ_rotor = 2 * arcsin(chord / (2 * k0))
    sin_half_theta = np.clip(chord_length / (2.0 * max(k0_magnitude, 1e-6)), -1.0, 1.0)
    theta_rotor_rad = float(2.0 * np.arcsin(sin_half_theta))

    phi_bivector = float(np.arctan2(delta_ky, delta_kx)) if chord_length > 1e-6 else 0.0
    bit_crossover_ratio = float(theta_rotor_rad / (2.0 * math.pi))

    rotor_scalar = float(np.cos(theta_rotor_rad / 2.0))
    rotor_bivector_e1e2 = float(-np.sin(theta_rotor_rad / 2.0))

    return {
        "k1_vector": k1,
        "k2_vector": k2,
        "delta_kx": delta_kx,
        "delta_ky": delta_ky,
        "chord_length": chord_length,
        "theta_rotor_rad": theta_rotor_rad,
        "theta_rotor_deg": float(np.degrees(theta_rotor_rad)),
        "phi_bivector_deg": float(np.degrees(phi_bivector)),
        "bit_crossover_ratio": bit_crossover_ratio,
        "clifford_rotor": {
            "scalar_part": rotor_scalar,
            "bivector_e1e2": rotor_bivector_e1e2
        }
    }
