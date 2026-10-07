"""
boundary_level_set.py: Level-Set Topological Boundary Extraction & Causal Tension Engine
========================================================================================

Adheres to "Do not calculate, let it flow." - Continuous Causal Intelligence Principles.

This module defines entities not as fixed classes (`class Fish`), but as level-set
iso-surfaces (∂Ω) extracted from an unbounded continuous scalar potential field Φ(x).

Mathematical Formulation:
1. Unbounded Field: Φ : R^N -> R^K
2. Observation Projection: y(x) = P_θ(Φ(x))
3. Gradient Field: g(x) = ∇y(x)
4. Topological Boundary Extraction (Iso-surface): ∂Ω = { x ∈ R^N | y(x) = C }
5. Boundary Coupling Tension & Emergent Causal Force: F_action = ∫_{∂Ω} T(x) · n̂(x) dA
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any


class ContinuousPotentialField:
    """
    Unbounded continuous scalar/vector potential field Φ(x) in N-dimensional space.
    """

    def __init__(self, bounds: Tuple[float, float] = (-50.0, 50.0), spatial_dim: int = 3):
        self.bounds = bounds
        self.spatial_dim = spatial_dim
        self.sources: List[Dict[str, Any]] = []

    def add_potential_source(self, center: np.ndarray, intensity: float, sigma: float) -> None:
        """
        Adds a Gaussian potential source/well into the field.
        """
        center_arr = np.asarray(center, dtype=float)
        if center_arr.shape[0] != self.spatial_dim:
            raise ValueError(f"Center dimension {center_arr.shape[0]} does not match spatial_dim {self.spatial_dim}")
        self.sources.append({
            "center": center_arr,
            "intensity": float(intensity),
            "sigma": float(sigma)
        })

    def sample_at(self, pos: np.ndarray) -> float:
        """
        Samples the scalar potential value Φ(pos) at position `pos`.
        """
        pos_arr = np.asarray(pos, dtype=float)
        val = 0.0
        for src in self.sources:
            diff = pos_arr - src["center"]
            dist_sq = np.sum(diff ** 2)
            val += src["intensity"] * np.exp(-dist_sq / (2.0 * src["sigma"] ** 2))
        return float(val)

    def compute_gradient(self, pos: np.ndarray, delta: float = 1e-4) -> np.ndarray:
        """
        Computes the numerical gradient ∇Φ(pos) using central finite differences.
        """
        pos_arr = np.asarray(pos, dtype=float)
        grad = np.zeros(self.spatial_dim, dtype=float)

        for i in range(self.spatial_dim):
            pos_plus = pos_arr.copy()
            pos_minus = pos_arr.copy()
            pos_plus[i] += delta
            pos_minus[i] -= delta
            grad[i] = (self.sample_at(pos_plus) - self.sample_at(pos_minus)) / (2.0 * delta)

        return grad


class LevelSetBoundaryExtractor:
    """
    Extracts topological iso-surface boundaries (∂Ω) and unit normal vectors n̂ from field Φ(x).
    """

    def __init__(self, cutoff_threshold: float = 0.5, spatial_dim: int = 3):
        self.threshold = cutoff_threshold
        self.spatial_dim = spatial_dim

    def extract_boundary_samples(
        self,
        field: ContinuousPotentialField,
        center: np.ndarray,
        radius: float = 5.0,
        num_samples: int = 32
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Extracts sample points near the level-set boundary ∂Ω = { x | y(x) = C } and unit normal vectors.
        """
        center_arr = np.asarray(center, dtype=float)
        boundary_pts: List[np.ndarray] = []
        surface_normals: List[np.ndarray] = []

        if self.spatial_dim == 2:
            angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
            directions = np.column_stack([np.cos(angles), np.sin(angles)])
        elif self.spatial_dim == 3:
            # Fibonacci sphere sampling for uniform 3D directional vectors
            indices = np.arange(0, num_samples, dtype=float) + 0.5
            phi = np.arccos(1.0 - 2.0 * indices / num_samples)
            theta = np.pi * (1.0 + 5.0 ** 0.5) * indices
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            directions = np.column_stack([x, y, z])
        else:
            # Random unit sphere sampling for higher dimensions
            raw = np.random.randn(num_samples, self.spatial_dim)
            norms = np.linalg.norm(raw, axis=1, keepdims=True) + 1e-12
            directions = raw / norms

        for d in directions:
            sample_pt = center_arr + radius * d
            val = field.sample_at(sample_pt)

            # Check if point is within iso-surface tolerance delta
            if abs(val - self.threshold) < 0.25:
                grad = field.compute_gradient(sample_pt)
                norm_grad = np.linalg.norm(grad) + 1e-8
                normal_vec = grad / norm_grad
                boundary_pts.append(sample_pt)
                surface_normals.append(normal_vec)

        return boundary_pts, surface_normals


class BoundaryTensionCalculator:
    """
    Computes boundary difference tension T = ΔΦ across iso-surfaces and calculates
    emergent causal forces F_action = ∫_{∂Ω} T(x) · n̂(x) dA.
    """

    def __init__(self, field: ContinuousPotentialField, extractor: LevelSetBoundaryExtractor):
        self.field = field
        self.extractor = extractor

    def compute_causal_force_and_tension(
        self,
        center: np.ndarray,
        radius: float = 5.0,
        sample_delta: float = 0.5
    ) -> Tuple[np.ndarray, float, int]:
        """
        Computes emergent causal force vector and total integrated boundary tension.
        """
        boundary_pts, normals = self.extractor.extract_boundary_samples(
            self.field, center, radius=radius
        )

        dim = center.shape[0]
        net_force = np.zeros(dim, dtype=float)
        total_tension = 0.0

        if not boundary_pts:
            return net_force, total_tension, 0

        for pt, normal in zip(boundary_pts, normals):
            inner_val = self.field.sample_at(pt - normal * sample_delta)
            outer_val = self.field.sample_at(pt + normal * sample_delta)

            # Boundary Tension T = ΔΦ
            tension = inner_val - outer_val
            total_tension += abs(tension)

            # Force emergence along normal direction
            net_force += tension * normal

        return net_force, total_tension, len(boundary_pts)
