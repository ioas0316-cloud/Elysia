"""
Self-Supervised Structural Adaptation (SSA) Engine
===================================================

This module implements the Self-Supervised Structural Adaptation (SSA) architecture.
Unlike conventional BPE/WordPiece tokenization and static gradient-descent neural networks,
the SSA engine operates on continuous sensory wave fields and an internal Riemannian manifold.

Key Principles:
1. Triadic Contrast & Dynamic Field Coupling:
   - Evaluates interactions across Observer (Internal Manifold), Environment (External Causal Field),
     and Relational Edge (Causal Boundary Coupling).
2. Continuous Causal Phase Friction Energy (F):
   - Friction is calculated between projected sensory fields and internal manifold phase dynamics,
     regularized by scalar curvature R(g_ij).
3. Dual-Process Physical Adaptation:
   - Phase Synchronization (Kuramoto Phase Dynamics): Internal phase theta_i aligns with external dynamics.
   - Geometry Adaptation (Modified Ricci Flow): Metric tensor g_ij deforms based on internal Ricci curvature
     and Hessian of phase friction (-2 R_ij - eta_g * grad_i grad_j F).
4. Boundary Emergence & Volumetric Manifold Carving:
   - When phase friction falls below a resonance threshold and metric tensor stabilizes,
     connected phase-locked clusters emerge as "Carved Volumetric Manifold" tokens.
5. Multi-Lens Co-Registration:
   - Projects inputs across Mathematical, Physical, and Semantic/Relational lenses,
     dynamically selecting the projection with minimal topological friction.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ContinuousSensoryField:
    """
    Represents continuous spatiotemporal wave inputs S(x, t) from sensory organs / external field.
    """
    spatial_dim: int = 16
    temporal_freq: float = 1.0
    field_matrix: np.ndarray = field(default_factory=lambda: np.zeros((16, 16)))

    @classmethod
    def generate_wave(cls, spatial_dim: int = 16, t: float = 0.0, freq: float = 1.0, phase_shift: float = 0.0) -> "ContinuousSensoryField":
        x = np.linspace(-np.pi, np.pi, spatial_dim)
        y = np.linspace(-np.pi, np.pi, spatial_dim)
        xx, yy = np.meshgrid(x, y)
        field = np.sin(freq * xx + t + phase_shift) * np.cos(freq * yy - t)
        return cls(spatial_dim=spatial_dim, temporal_freq=freq, field_matrix=field)


@dataclass
class InternalManifoldState:
    """
    Internal Topological Manifold State defined by Metric Tensor g_ij,
    Phase Vector Field theta(z, t), Intrinsic Frequencies omega_i, and Adjacency A_ij.
    """
    grid_size: int
    metric_g: np.ndarray        # [grid_size, grid_size, 2, 2] Riemannian metric tensor
    phase_theta: np.ndarray     # [grid_size, grid_size] phase angles theta(z, t)
    frequency_omega: np.ndarray # [grid_size, grid_size] intrinsic natural frequencies
    adjacency: np.ndarray       # [grid_size, grid_size, 4] 4-neighbor coupling matrix

    @classmethod
    def initialize(cls, grid_size: int = 16) -> "InternalManifoldState":
        # Initial euclidean metric g_ij = delta_ij
        metric_g = np.zeros((grid_size, grid_size, 2, 2), dtype=np.float64)
        for i in range(grid_size):
            for j in range(grid_size):
                metric_g[i, j] = np.eye(2, dtype=np.float64)

        phase_theta = np.random.uniform(0, 2 * np.pi, size=(grid_size, grid_size))
        frequency_omega = np.random.normal(loc=1.0, scale=0.1, size=(grid_size, grid_size))

        # Standard 2D grid adjacency weights
        adjacency = np.ones((grid_size, grid_size, 4), dtype=np.float64)
        return cls(grid_size=grid_size, metric_g=metric_g, phase_theta=phase_theta, frequency_omega=frequency_omega, adjacency=adjacency)


class MultiLensProjectionOperator:
    """
    Co-registers external wave fields through Mathematical, Physical, and Semantic Lenses.
    """
    def __init__(self, spatial_dim: int = 16):
        self.spatial_dim = spatial_dim

    def project_mathematical(self, sensory_field: np.ndarray) -> np.ndarray:
        """Fourier phase and spatial gradient projection."""
        fft_field = np.fft.fft2(sensory_field)
        phase_projection = np.angle(fft_field)
        return phase_projection

    def project_physical(self, sensory_field: np.ndarray) -> np.ndarray:
        """Energy gradient and wave curvature projection."""
        grad_y, grad_x = np.gradient(sensory_field)
        curvature = np.arctan2(grad_y, grad_x)
        return curvature

    def project_semantic(self, sensory_field: np.ndarray) -> np.ndarray:
        """Attractor / relational contrast projection."""
        semantic_map = np.sin(sensory_field * np.pi)
        return semantic_map

    def project_best_lens(self, sensory_field: np.ndarray, internal_phase: np.ndarray) -> Tuple[np.ndarray, str]:
        p_math = self.project_mathematical(sensory_field)
        p_phys = self.project_physical(sensory_field)
        p_sem = self.project_semantic(sensory_field)

        f_math = np.mean((internal_phase - p_math) ** 2)
        f_phys = np.mean((internal_phase - p_phys) ** 2)
        f_sem = np.mean((internal_phase - p_sem) ** 2)

        frictions = {"math": f_math, "physical": f_phys, "semantic": f_sem}
        best_lens = min(frictions, key=frictions.get)

        if best_lens == "math":
            return p_math, "mathematical"
        elif best_lens == "physical":
            return p_phys, "physical"
        else:
            return p_sem, "semantic"


class SelfSupervisedStructuralAdaptationEngine:
    """
    Core SSA Engine implementing Phase Dynamics, Ricci Flow Metric Adaptation,
    Causal Phase Friction Minimization, and Volumetric Boundary Carving.
    """
    def __init__(
        self,
        grid_size: int = 16,
        kuramoto_coupling_K: float = 1.5,
        learning_rate_phase: float = 0.2,
        learning_rate_metric: float = 0.05,
        curvature_reg_lambda: float = 0.01,
        resonance_threshold: float = 0.05
    ):
        self.grid_size = grid_size
        self.K = kuramoto_coupling_K
        self.eta_theta = learning_rate_phase
        self.eta_g = learning_rate_metric
        self.lambda_R = curvature_reg_lambda
        self.resonance_threshold = resonance_threshold

        self.manifold = InternalManifoldState.initialize(grid_size=grid_size)
        self.lens_operator = MultiLensProjectionOperator(spatial_dim=grid_size)

    def compute_ricci_tensor(self, metric_g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes 2D discretized Ricci Curvature Tensor R_ij and Scalar Curvature R
        for metric_g tensor field [grid_size, grid_size, 2, 2].
        """
        grid = self.grid_size
        ricci_tensor = np.zeros_like(metric_g)
        scalar_curvature = np.zeros((grid, grid), dtype=np.float64)

        # Numerical finite difference for metric spatial derivatives
        dg_dx = np.gradient(metric_g, axis=1) # x derivative
        dg_dy = np.gradient(metric_g, axis=0) # y derivative

        for i in range(grid):
            for j in range(grid):
                g = metric_g[i, j]
                g_inv = np.linalg.pinv(g)

                # Laplace-Beltrami approx for 2D Ricci curvature R_ij ~ -0.5 * Laplace(g_ij)
                laplace_g_00 = (
                    metric_g[(i+1)%grid, j, 0, 0] + metric_g[(i-1)%grid, j, 0, 0] +
                    metric_g[i, (j+1)%grid, 0, 0] + metric_g[i, (j-1)%grid, 0, 0] - 4 * g[0, 0]
                )
                laplace_g_11 = (
                    metric_g[(i+1)%grid, j, 1, 1] + metric_g[(i-1)%grid, j, 1, 1] +
                    metric_g[i, (j+1)%grid, 1, 1] + metric_g[i, (j-1)%grid, 1, 1] - 4 * g[1, 1]
                )
                laplace_g_01 = (
                    metric_g[(i+1)%grid, j, 0, 1] + metric_g[(i-1)%grid, j, 0, 1] +
                    metric_g[i, (j+1)%grid, 0, 1] + metric_g[i, (j-1)%grid, 0, 1] - 4 * g[0, 1]
                )

                R_ij = -0.5 * np.array([[laplace_g_00, laplace_g_01], [laplace_g_01, laplace_g_11]])
                ricci_tensor[i, j] = R_ij

                # Scalar curvature R = g^ij R_ij
                scalar_curvature[i, j] = np.trace(g_inv @ R_ij)

        return ricci_tensor, scalar_curvature

    def compute_causal_phase_friction(
        self,
        projected_field: np.ndarray,
        scalar_curvature: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Computes Causal Phase Friction F(t) = int || d_theta/dt - T_ext(S) ||^2 dz + lambda * int R(g_ij) dz
        Returns:
            friction_energy: Total friction scalar F
            phase_gradient: dF / d_theta
            friction_hessian_g: Spatial Hessian tensor for metric adaptation
        """
        grid = self.grid_size
        phase_diff = np.sin(self.manifold.phase_theta - projected_field)
        local_friction = phase_diff ** 2

        friction_energy = np.mean(local_friction) + self.lambda_R * np.mean(np.abs(scalar_curvature))
        phase_gradient = 2.0 * phase_diff * np.cos(self.manifold.phase_theta - projected_field)

        # Compute Spatial Hessian of friction w.r.t coordinates mapped to metric deformation
        grad_y, grad_x = np.gradient(local_friction)
        hess_xx = np.gradient(grad_x, axis=1)
        hess_yy = np.gradient(grad_y, axis=0)
        hess_xy = np.gradient(grad_x, axis=0)

        friction_hessian_g = np.zeros_like(self.manifold.metric_g)
        for i in range(grid):
            for j in range(grid):
                friction_hessian_g[i, j] = np.array([[hess_xx[i, j], hess_xy[i, j]], [hess_xy[i, j], hess_yy[i, j]]])

        return friction_energy, phase_gradient, friction_hessian_g

    def compute_kuramoto_coupling(self) -> np.ndarray:
        """Computes internal oscillator Kuramoto phase coupling sum_j A_ij sin(theta_j - theta_i)."""
        grid = self.grid_size
        kuramoto_term = np.zeros((grid, grid), dtype=np.float64)
        for i in range(grid):
            for j in range(grid):
                neighbors = [
                    self.manifold.phase_theta[(i+1)%grid, j],
                    self.manifold.phase_theta[(i-1)%grid, j],
                    self.manifold.phase_theta[i, (j+1)%grid],
                    self.manifold.phase_theta[i, (j-1)%grid]
                ]
                coupling_sum = sum(np.sin(nb - self.manifold.phase_theta[i, j]) for nb in neighbors)
                kuramoto_term[i, j] = (self.K / 4.0) * coupling_sum
        return kuramoto_term

    def step(self, sensory_field: ContinuousSensoryField, time_delta: float = 0.1) -> Dict[str, Any]:
        """
        Executes single SSA iteration loop:
        1. Multi-lens projection of external continuous field
        2. Compute Ricci curvature & scalar curvature R(g_ij)
        3. Compute Causal Phase Friction energy F and gradients
        4. Kuramoto & Friction Phase dynamics update
        5. Modified Ricci Flow Metric adaptation update: dg_ij/dt = -2 R_ij - eta_g * grad_i grad_j F
        6. Volumetric boundary carving check
        """
        grid = self.grid_size
        # 1. Multi-Lens Projection
        projected_field, active_lens = self.lens_operator.project_best_lens(
            sensory_field.field_matrix, self.manifold.phase_theta
        )

        # 2. Ricci Curvature
        ricci_tensor, scalar_curvature = self.compute_ricci_tensor(self.manifold.metric_g)

        # 3. Causal Phase Friction
        friction_energy, phase_gradient, friction_hessian = self.compute_causal_phase_friction(
            projected_field, scalar_curvature
        )

        # 4. Phase Dynamics Update (Kuramoto + Friction Gradient Descent)
        kuramoto_term = self.compute_kuramoto_coupling()

        dtheta_dt = self.manifold.frequency_omega + kuramoto_term - self.eta_theta * phase_gradient
        self.manifold.phase_theta = (self.manifold.phase_theta + dtheta_dt * time_delta) % (2 * np.pi)

        # 5. Modified Ricci Flow Metric Adaptation
        # dg_ij / dt = -2 * R_ij - eta_g * Friction_Hessian
        metric_update = -2.0 * ricci_tensor - self.eta_g * friction_hessian
        self.manifold.metric_g += metric_update * time_delta

        # Ensure metric tensor remains positive-definite
        for i in range(grid):
            for j in range(grid):
                g = self.manifold.metric_g[i, j]
                g = (g + g.T) / 2.0 # Symmetry
                eigvals, eigvecs = np.linalg.eigh(g)
                eigvals = np.maximum(eigvals, 1e-4) # Positivity
                self.manifold.metric_g[i, j] = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # 6. Boundary Emergence Check (Carved Volumetric Manifold)
        carved_volumes = self.extract_carved_volumetric_boundaries(friction_energy)

        return {
            "friction_energy": friction_energy,
            "active_lens": active_lens,
            "mean_scalar_curvature": float(np.mean(scalar_curvature)),
            "carved_volumes_count": len(carved_volumes),
            "carved_volumes": carved_volumes
        }

    def extract_carved_volumetric_boundaries(self, current_friction: float) -> List[Dict[str, Any]]:
        """
        Extracts phase-locked connected components as self-carved volumetric manifold tokens.
        """
        grid = self.grid_size
        visited = np.zeros((grid, grid), dtype=bool)
        carved_components = []

        phase = self.manifold.phase_theta

        for i in range(grid):
            for j in range(grid):
                if not visited[i, j]:
                    # Breadth-first search for phase synchrony (|theta_i - theta_j| < threshold)
                    component_nodes = []
                    queue = [(i, j)]
                    visited[i, j] = True

                    while queue:
                        curr_i, curr_j = queue.pop(0)
                        component_nodes.append((curr_i, curr_j))

                        neighbors = [
                            ((curr_i + 1) % grid, curr_j),
                            ((curr_i - 1) % grid, curr_j),
                            (curr_i, (curr_j + 1) % grid),
                            (curr_i, (curr_j - 1) % grid)
                        ]

                        for ni, nj in neighbors:
                            if not visited[ni, nj]:
                                phase_diff = abs(np.sin(phase[curr_i, curr_j] - phase[ni, nj]))
                                if phase_diff < self.resonance_threshold:
                                    visited[ni, nj] = True
                                    queue.append((ni, nj))

                    if len(component_nodes) >= 3:
                        # Calculate volumetric manifold properties
                        volumes = [np.linalg.det(self.manifold.metric_g[ni, nj]) for ni, nj in component_nodes]
                        total_volume = float(np.sum(np.sqrt(np.maximum(0, volumes))))
                        carved_components.append({
                            "token_id": f"volumetric_token_{len(carved_components)+1}",
                            "node_count": len(component_nodes),
                            "topological_volume": total_volume,
                            "center_coordinate": (
                                float(np.mean([n[0] for n in component_nodes])),
                                float(np.mean([n[1] for n in component_nodes]))
                            )
                        })

        return carved_components
