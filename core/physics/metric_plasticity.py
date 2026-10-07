"""
metric_plasticity.py: Nonlinear Metric Tensor Plasticity & Stress-Energy Flow Engine
======================================================================================

Adheres to "Do not calculate, let it flow." - Continuous Causal Intelligence Principles.

This module governs the Riemannian dynamics of the observation metric tensor g_ij.
The metric tensor g_ij is not static; it dynamically deforms under boundary stress-energy T_ij,
Ricci curvature flow R_ij, and elastic memory recovery.

Mathematical Formulation:
1. Metric Distance: ds^2 = ∑_{i,j} g_ij(x, t) dx^i dx^j
2. Boundary Stress-Energy Tensor: T_ij = ∇_i y ⊗ ∇_j y - 1/2 g_ij ||∇y||_g^2
3. Metric Plasticity Flow Equation:
   ∂g_ij/∂t = -2γ R_ij + α T_ij - λ (g_ij - g_ij^(0))
4. Positive-Definiteness Protection: Ensures metric tensor remains symmetric positive definite.
"""

import numpy as np
from typing import Tuple, Optional


class MetricPlasticityEngine:
    """
    Manages non-linear metric tensor deformation under stress-energy tension and Ricci-like curvature flow.
    """

    def __init__(
        self,
        dim: int = 3,
        gamma: float = 0.05,
        alpha: float = 0.1,
        lambda_recovery: float = 0.02,
        min_eigenvalue: float = 1e-4
    ):
        self.dim = dim
        self.gamma = gamma  # Curvature damping factor
        self.alpha = alpha  # Plasticity stress response factor
        self.lambda_rec = lambda_recovery  # Elastic memory recovery factor
        self.min_eigenvalue = min_eigenvalue

        # Initial background metric tensor g_ij^(0) (Euclidean default)
        self.g0 = np.eye(dim, dtype=float)
        # Active metric tensor g_ij
        self.g = self.g0.copy()

    def compute_stress_energy_tensor(self, gradient_y: np.ndarray) -> np.ndarray:
        """
        Computes the boundary Stress-Energy Tensor T_ij:
        T_ij = ∇_i y ⊗ ∇_j y - 1/2 g_ij ||∇y||_g^2
        """
        grad = np.asarray(gradient_y, dtype=float)
        if grad.shape[0] != self.dim:
            raise ValueError(f"Gradient dimension {grad.shape[0]} does not match metric dimension {self.dim}")

        # Norm under metric g: ||∇y||_g^2 = grad^T · g^(-1) · grad
        g_inv = np.linalg.inv(self.g)
        norm_sq = float(grad.T @ g_inv @ grad)

        # Outer product ∇_i y ⊗ ∇_j y
        outer_grad = np.outer(grad, grad)

        # T_ij = outer_grad - 0.5 * g * norm_sq
        T_ij = outer_grad - 0.5 * self.g * norm_sq
        return T_ij

    def compute_ricci_curvature_approx(self) -> np.ndarray:
        """
        Approximates Ricci curvature tensor R_ij as structural distortion relative to base metric.
        R_ij ≈ g - g^(0)
        """
        return self.g - self.g0

    def enforce_positive_definiteness(self, metric_matrix: np.ndarray) -> np.ndarray:
        """
        Guarantees that the metric tensor remains symmetric positive definite (SPD).
        """
        # Ensure symmetry
        sym_m = (metric_matrix + metric_matrix.T) / 2.0

        # Eigenvalue decomposition
        evals, evecs = np.linalg.eigh(sym_m)

        # Clip eigenvalues below minimum threshold
        evals_clipped = np.maximum(evals, self.min_eigenvalue)

        # Reconstruct SPD matrix
        spd_m = evecs @ np.diag(evals_clipped) @ evecs.T
        return spd_m

    def step_plasticity_flow(self, gradient_y: np.ndarray, dt: float = 0.1) -> Tuple[np.ndarray, float]:
        """
        Integrates one step of the metric flow equation:
        ∂g_ij/∂t = -2γ R_ij + α T_ij - λ (g_ij - g_ij^(0))
        Returns updated g_ij and total metric strain distortion.
        """
        T_ij = self.compute_stress_energy_tensor(gradient_y)
        R_ij = self.compute_ricci_curvature_approx()

        # Flow differential
        dg_dt = -2.0 * self.gamma * R_ij + self.alpha * T_ij - self.lambda_rec * (self.g - self.g0)

        # Euler integration step
        unclamped_g = self.g + dt * dg_dt

        # Apply SPD enforcement
        self.g = self.enforce_positive_definiteness(unclamped_g)

        # Strain distortion metric relative to Euclidean metric
        strain_distortion = float(np.linalg.norm(self.g - self.g0, ord='fro'))
        return self.g.copy(), strain_distortion
