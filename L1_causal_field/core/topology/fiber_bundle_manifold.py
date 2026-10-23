"""
Fiber Bundle Manifold & Presentation Projection Engine for Elysia.

This module implements a 4D Fiber Bundle Differential Manifold (E = B x F):
- Base Space B (1D): Temporal trajectory axis t ∈ ℝ^1 (Guarantees temporal monotonicity dt/dτ > 0)
- Fiber Space F_t (3D): Structural volume x = (x1, x2, x3) ∈ ℝ^3
- 5 Primal Human Sensory Input Ports: Vision, Audition, Somatosensory, Olfaction, Gustation
  (Raw external wave impact conduits that dynamically reshape the 4D causal topography)
- Non-Backprop Geodesic Flow along Christoffel symbols Γ^μ_αβ
- CausalSectionCache: Slices 4D holographic trajectories into 2D/3D presentation cache sections.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any


# 5 Primal Human Sensory Input Ports
SENSORY_PORTS = ["VISION", "AUDITION", "SOMATOSENSORY", "OLFACTION", "GUSTATION"]
NUM_SENSORY_PORTS = len(SENSORY_PORTS)


class FiberBundleManifold:
    """
    4D Smooth Manifold E with Ehresmann Connection 1-form ω and Metric Tensor g_μν.
    State representation:
        p = (t, x1, x2, x3) - 4D spacetime coordinates
        v = (dt/dτ, dx1/dτ, dx2/dτ, dx3/dτ) - Geodesic flow velocity
    """

    def __init__(self, num_points: int = 1000, device: str = "cpu"):
        self.num_points = num_points
        self.device = torch.device(device)

        # 4D Spacetime Coordinates: [num_points, 4] -> (t, x1, x2, x3)
        self.coords = torch.zeros((num_points, 4), dtype=torch.float32, device=self.device)
        # Initialize time t with slight spread along temporal axis
        self.coords[:, 0] = torch.linspace(0.0, 1.0, num_points, device=self.device)
        # Initialize 3D fiber coordinates with small initial distribution
        self.coords[:, 1:] = torch.randn((num_points, 3), dtype=torch.float32, device=self.device) * 0.1

        # Geodesic Velocity: [num_points, 4] -> (v_t, v_x1, v_x2, v_x3)
        self.velocity = torch.zeros((num_points, 4), dtype=torch.float32, device=self.device)
        self.velocity[:, 0] = 1.0  # Monotonic initial forward flow on temporal axis
        self.velocity[:, 1:] = torch.randn((num_points, 3), dtype=torch.float32, device=self.device) * 0.05

        # 3D Fiber Spatial Metric h_ij: [num_points, 3, 3]
        self.h_metric = torch.eye(3, dtype=torch.float32, device=self.device).repeat(num_points, 1, 1)

        # Gauge Potential A_t^i: [num_points, 3]
        self.gauge_A_t = torch.zeros((num_points, 3), dtype=torch.float32, device=self.device)

        # Sensory Port Boundary Occupancy Weights: [num_points, 5] (sum = 1.0)
        self.sensory_weights = torch.full((num_points, NUM_SENSORY_PORTS), 1.0 / NUM_SENSORY_PORTS, dtype=torch.float32, device=self.device)

        # Sensory port metric bases
        self.sensory_metrics = self._init_sensory_metric_bases()

    def _init_sensory_metric_bases(self) -> torch.Tensor:
        """Initialize 5 orthogonal sensory metric bases in 3D fiber space."""
        bases = torch.zeros((NUM_SENSORY_PORTS, 3, 3), dtype=torch.float32, device=self.device)
        for s in range(NUM_SENSORY_PORTS):
            # Each sensory port imparts a unique wave curvature signature
            scale = 0.8 + 0.1 * s
            diag = torch.diag(torch.tensor([scale, 1.0 / scale, 1.0], device=self.device))
            bases[s] = diag
        return bases

    def update_sensory_occupancy(self, weights: torch.Tensor):
        """Update sensory port wave occupancy weights and recompute fiber metric h_ij."""
        assert weights.shape == (self.num_points, NUM_SENSORY_PORTS)
        self.sensory_weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        # h_ij = sum_s (w_s * h_{ij}^(s))
        self.h_metric = torch.einsum('ns,sij->nij', self.sensory_weights, self.sensory_metrics)

    def set_gauge_potential(self, gauge_A: torch.Tensor):
        """Set temporal gauge connection A_t^i."""
        assert gauge_A.shape == (self.num_points, 3)
        self.gauge_A_t = gauge_A

    def compute_christoffel_symbols(self) -> torch.Tensor:
        """
        Compute 4D Christoffel symbols Γ^μ_αβ from gauge potential A_t and metric h_ij.
        Returns tensor of shape [num_points, 4, 4, 4].
        """
        gamma = torch.zeros((self.num_points, 4, 4, 4), dtype=torch.float32, device=self.device)

        # Gauge connection induces Christoffel components Γ^i_00, Γ^i_0i, Γ^i_i0
        for i in range(1, 4):
            A_i = self.gauge_A_t[:, i - 1]
            gamma[:, i, 0, 0] = -0.5 * A_i
            gamma[:, i, 0, i] = 0.5 * A_i
            gamma[:, i, i, 0] = 0.5 * A_i

        return gamma

    def step_geodesic_flow(self, d_tau: float = 0.01):
        """
        Evolve 4D state and velocity using Symplectic Euler integration along Geodesic Flow:
        d^2 z^μ / dτ^2 = - Γ^μ_αβ v^α v^β
        """
        gamma = self.compute_christoffel_symbols()  # [N, 4, 4, 4]

        # accel[μ] = - sum_{α, β} Γ^μ_αβ v^α v^β
        accel = -torch.einsum('nmab,na,nb->nm', gamma, self.velocity, self.velocity)

        # Ensure temporal velocity is strictly monotonic (dt/dτ > 0)
        v_t_next = self.velocity[:, 0] + accel[:, 0] * d_tau
        accel_t = torch.where(v_t_next <= 0.05, torch.zeros_like(accel[:, 0]), accel[:, 0])
        accel[:, 0] = accel_t

        # Symplectic update
        self.velocity += accel * d_tau
        self.coords += self.velocity * d_tau

    def inject_sensory_wave_impact(self, port_idx: int, impact_magnitude: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Inject a raw external sensory wave impact (e.g. SOMATOSENSORY thermal/pressure wave)
        and observe multi-sensory phase shift resonance across the 4D scale layer.
        """
        # Inject wave perturbation into gauge potential along sensory port's preferred axis
        delta_A = torch.zeros_like(self.gauge_A_t)
        axis = port_idx % 3
        delta_A[:, axis] += impact_magnitude

        self.gauge_A_t += delta_A

        # Re-compute Christoffel symbols and step geodesic
        self.step_geodesic_flow(d_tau=0.02)

        # Measure phase shift resonance for each sensory port
        resonances = {}
        for s, name in enumerate(SENSORY_PORTS):
            # Sensory resonance = projection of metric response onto sensory port basis
            proj = torch.einsum('nij,ij->n', self.h_metric, self.sensory_metrics[s])
            resonances[name] = proj.mean()

        return resonances


class CausalSectionCache:
    """
    Presentation Cache Slicing Layer:
    Slices 4D holographic trajectories along temporal section t_slice or spatial section planes
    and projects them into lightweight 2D/3D GPU/RAM caches for fast rendering without causal distortion.
    """

    def __init__(self, manifold: FiberBundleManifold):
        self.manifold = manifold
        self.section_cache: Dict[str, Any] = {}

    def slice_temporal_section(self, t_slice: float, tol: float = 0.1) -> Dict[str, torch.Tensor]:
        """
        Slices the 4D Fiber Bundle at a specific temporal leaf F_{t_slice} = π^(-1)(t_slice).
        Returns the 3D fiber coordinates and sensory port weights at that instant.
        """
        t_coords = self.manifold.coords[:, 0]
        mask = torch.abs(t_coords - t_slice) <= tol

        if not mask.any():
            # If exact slice is empty, pick closest points
            _, idxs = torch.topk(torch.abs(t_coords - t_slice), k=min(10, self.manifold.num_points), largest=False)
            mask = torch.zeros_like(mask, dtype=torch.bool)
            mask[idxs] = True

        sliced_fiber_coords = self.manifold.coords[mask, 1:]  # [K, 3]
        sliced_velocities = self.manifold.velocity[mask, 1:]   # [K, 3]
        sliced_sensory_weights = self.manifold.sensory_weights[mask] # [K, 5]

        section = {
            "t_slice": t_slice,
            "fiber_3d": sliced_fiber_coords,
            "velocity_3d": sliced_velocities,
            "sensory_weights": sliced_sensory_weights,
            "num_points": int(mask.sum().item())
        }

        self.section_cache[f"t_{t_slice:.2f}"] = section
        return section

    def rasterize_section_to_2d_projection(self, section: Dict[str, torch.Tensor], proj_axis: Tuple[int, int] = (0, 1)) -> np.ndarray:
        """
        Rasterizes a 3D fiber section into a 2D cache projection matrix (e.g. 64x64)
        for front-end rendering or display.
        """
        pts = section["fiber_3d"][:, list(proj_axis)].cpu().numpy() # [K, 2]
        weights = section["sensory_weights"].cpu().numpy()         # [K, 5]

        grid_size = 64
        grid = np.zeros((grid_size, grid_size, NUM_SENSORY_PORTS), dtype=np.float32)

        if len(pts) == 0:
            return grid

        # Normalize points to grid range [0, grid_size - 1]
        pts_min = pts.min(axis=0, keepdims=True)
        pts_max = pts.max(axis=0, keepdims=True)
        denom = np.where((pts_max - pts_min) < 1e-5, 1.0, pts_max - pts_min)
        norm_pts = ((pts - pts_min) / denom * (grid_size - 1)).astype(int)

        for i, (gx, gy) in enumerate(norm_pts):
            gx = np.clip(gx, 0, grid_size - 1)
            gy = np.clip(gy, 0, grid_size - 1)
            grid[gx, gy] += weights[i]

        return grid
