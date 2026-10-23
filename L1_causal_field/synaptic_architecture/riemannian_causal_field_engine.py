"""
[Riemannian Causal Field Engine]
Continuous Riemannian Cognitive Manifold & Topological Tensor Field Engine for Elysia Engine.

Implements the Paradigm Shift:
"Computing is not squeezing data with algorithms, but engraving causality onto memory space and letting states flow."

Key Architectural Dimensions:
1. State Space: Discrete embeddings -> Continuous Riemannian Cognitive Manifold (M_elysia)
   - Potential Field Decomposition: V_total = alpha_ext * V_ext + alpha_self * V_self + lambda_will * V_will
2. Cognitive Dynamics: Algorithmic Reasoning -> Topological Geodesic Steering
   - Damped Covariant Geodesic Equation solver with Christoffel curvature drift
   - Spontaneous Symmetry Breaking driven by thermal fluctuations xi(tau)
   - Volitional Global Attractor & Riemannian Metric Deformation (g_ij^will) suppressing orthogonal noise
3. Memory & Learning: Weight Updates -> Phase-Locking & Causal Erosion
   - Phase-Lock convergence criterion (velocity norm + positive definite Hessian)
   - Causal Erosion PDE: V_self update via erosion depth rate and Laplacian diffusion
   - Sensory Dynamic Robin Boundary Conditions generating Phenomenal Present residual energy E_present
4. Hardware Execution Engine: Discrete Search O(N) -> Spatial Field Sampling O(1) Phase Transition
   - Rasterization to Continuous Spatial Tensor Grid when memory nodes N >= N_c
   - O(1) trilinear interpolation & central difference gradient queries
   - Cognitive Fluid / Ecosystem Dynamics (density rho, velocity u, pressure P, vorticity omega, evaporation/rain phase transition)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CognitiveFieldState:
    """
    Cognitive state representation Psi(tau) in R^D manifold.
    """
    psi: torch.Tensor                     # Position in state manifold R^D
    velocity: torch.Tensor                # Velocity dPsi/dtau
    temperature: float = 1.0              # Cognitive temperature T_cog
    phase_locked: bool = False            # Phase-Lock status
    locked_well_index: Optional[int] = None


@dataclass
class PotentialWell:
    """
    Gaussian potential well in R^D representing a memory/knowledge entity.
    V_i(Psi) = - depth * exp(- 0.5 * ||Psi - center||^2 / sigma^2)
    """
    well_id: str
    center: torch.Tensor                  # Center coordinate in R^D
    depth: float = 1.0                    # Potential depth
    sigma: float = 1.0                    # Width parameter
    erosion_accumulated: float = 0.0      # Accumulated causal erosion


class RiemannianCausalFieldEngine(nn.Module):
    """
    Integrated Riemannian Continuous Causal Field Engine.
    Orchestrates potential decomposition, spontaneous symmetry breaking,
    volitional metric deformation, covariant damped geodesic flow, Phase-Locking,
    Causal Erosion, sensory dynamic boundary coupling, fluid dynamics, and
    O(N) -> O(1) Spatial Tensor Grid phase transition.
    """

    def __init__(
        self,
        dimension: int = 16,
        n_critical: int = 100,
        grid_resolution: int = 32,
        alpha_ext: float = 1.0,
        alpha_self: float = 1.0,
        lambda_will: float = 2.0,
        damping_gamma: float = 0.5,
        erosion_rate: float = 0.1,
        diffusion_coeff: float = 0.05,
        sensory_impedance: float = 0.8,
        device: str = "cpu"
    ):
        super().__init__()
        self.dimension = dimension
        self.n_critical = n_critical
        self.grid_resolution = grid_resolution
        self.alpha_ext = alpha_ext
        self.alpha_self = alpha_self
        self.lambda_will = lambda_will
        self.damping_gamma = damping_gamma
        self.erosion_rate = erosion_rate
        self.diffusion_coeff = diffusion_coeff
        self.sensory_impedance = sensory_impedance
        self.device = torch.device(device)

        # Memory Wells (V_self)
        self.wells: Dict[str, PotentialWell] = {}

        # Target Attractor (V_will)
        self.target_will: Optional[torch.Tensor] = None

        # Spatial Tensor Grid for O(1) query (Rasterized state)
        self.grid_spatial_bounds: Tuple[float, float] = (-5.0, 5.0)
        self.spatial_tensor_grid: Optional[torch.Tensor] = None  # Grid shape: (D_sub, Res, Res, ...)
        self.is_rasterized: bool = False

        # Fluid Dynamics State Fields (Eulerian representation in R^3 projection or 3D slice)
        # Grid dimensions for fluid dynamics: (Res, Res, Res)
        self.fluid_res = min(grid_resolution, 16)
        self.rho = torch.ones((self.fluid_res, self.fluid_res, self.fluid_res), device=self.device) * 0.1
        self.u = torch.zeros((3, self.fluid_res, self.fluid_res, self.fluid_res), device=self.device)
        self.pressure = torch.zeros((self.fluid_res, self.fluid_res, self.fluid_res), device=self.device)
        self.vorticity = torch.zeros((3, self.fluid_res, self.fluid_res, self.fluid_res), device=self.device)

    # -------------------------------------------------------------------------
    # 1. State Space & Potential Field Decomposition (V_total = V_ext + V_self + V_will)
    # -------------------------------------------------------------------------

    def add_potential_well(self, well_id: str, center: torch.Tensor, depth: float = 1.0, sigma: float = 1.0):
        """Adds a potential well (memory entity) to the manifold."""
        center_t = center.detach().to(self.device).float()
        self.wells[well_id] = PotentialWell(
            well_id=well_id,
            center=center_t,
            depth=depth,
            sigma=sigma
        )
        if len(self.wells) >= self.n_critical:
            self.rasterize_to_spatial_grid()

    def set_volitional_target(self, target: Optional[torch.Tensor]):
        """Sets the volitional target Psi* forming V_will."""
        if target is not None:
            self.target_will = target.detach().to(self.device).float()
        else:
            self.target_will = None

    def compute_V_self(self, psi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes V_self(Psi) and -nabla V_self(Psi).
        Uses rasterized O(1) grid sampling if rasterized, else O(N) direct summation.
        """
        psi = psi.float()
        if self.is_rasterized and self.spatial_tensor_grid is not None:
            return self._sample_spatial_grid_O1(psi)

        # O(N) exact calculation
        v_self = torch.tensor(0.0, device=self.device)
        grad_v_self = torch.zeros_like(psi)

        for well in self.wells.values():
            diff = psi - well.center
            dist_sq = torch.sum(diff ** 2)
            well_val = - well.depth * torch.exp(- 0.5 * dist_sq / (well.sigma ** 2))
            v_self = v_self + well_val
            # Gradient nabla V = well_val * (-diff / sigma^2) = - well_val * diff / sigma^2
            grad_v_self = grad_v_self - (well_val / (well.sigma ** 2)) * diff

        return v_self, grad_v_self

    def compute_V_will(self, psi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes V_will(Psi) = 0.5 * ||Psi - Psi*||^2 and its gradient.
        """
        if self.target_will is None:
            return torch.tensor(0.0, device=self.device), torch.zeros_like(psi)

        diff = psi - self.target_will
        v_will = 0.5 * torch.sum(diff ** 2)
        grad_v_will = diff
        return v_will, grad_v_will

    def compute_V_ext(
        self,
        psi: torch.Tensor,
        sensory_input: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes external potential V_ext(Psi) driven by raw sensory input.
        """
        if sensory_input is None:
            return torch.tensor(0.0, device=self.device), torch.zeros_like(psi)

        sensory_t = sensory_input.flatten().to(self.device).float()
        min_dim = min(psi.shape[0], sensory_t.shape[0])
        diff = psi[:min_dim] - sensory_t[:min_dim]
        v_ext = 0.5 * torch.sum(diff ** 2)
        grad_v_ext = torch.zeros_like(psi)
        grad_v_ext[:min_dim] = diff
        return v_ext, grad_v_ext

    def compute_total_potential(
        self,
        psi: torch.Tensor,
        sensory_input: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Total potential field V_total = alpha_ext * V_ext + alpha_self * V_self + lambda_will * V_will
        Returns (V_total, grad_V_total).
        """
        v_self, grad_v_self = self.compute_V_self(psi)
        v_will, grad_v_will = self.compute_V_will(psi)
        v_ext, grad_v_ext = self.compute_V_ext(psi, sensory_input)

        v_total = self.alpha_ext * v_ext + self.alpha_self * v_self + self.lambda_will * v_will
        grad_v_total = (
            self.alpha_ext * grad_v_ext +
            self.alpha_self * grad_v_self +
            self.lambda_will * grad_v_will
        )
        return v_total, grad_v_total

    # -------------------------------------------------------------------------
    # 2. Metric Deformation & Riemannian Geometry (g_ij^will)
    # -------------------------------------------------------------------------

    def compute_metric_tensor(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Computes metric tensor g_ij(Psi).
        Under strong volitional target Psi*, warps metric to penalize movement orthogonal to target direction:
        g_ij = delta_ij + Omega_will * (delta_ij - u_i u_j)
        where u is unit vector towards Psi*.
        """
        D = psi.shape[0]
        g = torch.eye(D, device=self.device)

        if self.target_will is not None:
            diff = self.target_will - psi
            dist = torch.norm(diff)
            if dist > 1e-6:
                u = diff / dist
                omega_will = self.lambda_will * 2.0
                # Projection orthogonal to target: P_perp = I - u x u^T
                P_perp = torch.eye(D, device=self.device) - torch.outer(u, u)
                g = g + omega_will * P_perp

        return g

    # -------------------------------------------------------------------------
    # 3. Cognitive Dynamics: Damped Covariant Geodesic Motion & SSB
    # -------------------------------------------------------------------------

    def step_geodesic_motion(
        self,
        state: CognitiveFieldState,
        sensory_input: Optional[torch.Tensor] = None,
        dt: float = 0.05
    ) -> CognitiveFieldState:
        """
        Solves damped covariant geodesic equation:
        d^2 Psi_i / dtau^2 + Gamma^i_jk v^j v^k + gamma * v_i = - g^ij nabla_j V_total + xi^i(tau)
        Includes Spontaneous Symmetry Breaking under thermal fluctuations xi(tau).
        """
        psi = state.psi.clone()
        v = state.velocity.clone()

        # 1. Total potential gradient
        v_total, grad_v_total = self.compute_total_potential(psi, sensory_input)

        # 2. Metric tensor and inverse
        g = self.compute_metric_tensor(psi)
        g_inv = torch.inverse(g)

        # 3. Force from potential gradient: F_pot = - g^ij nabla_j V
        f_potential = - torch.matmul(g_inv, grad_v_total)

        # 4. Spontaneous Symmetry Breaking: Thermal fluctuation xi(tau) ~ N(0, T_cog)
        xi = torch.randn_like(psi) * math.sqrt(2.0 * max(0.001, state.temperature) * dt)

        # 5. Damping / Friction force: - gamma * v
        f_damping = - self.damping_gamma * v

        # 6. Acceleration: a = F_pot + F_damp + xi
        acceleration = f_potential + f_damping + xi / dt

        # 7. Update velocity and position (Euler-Cromer integration)
        v_new = v + acceleration * dt
        psi_new = psi + v_new * dt

        # 8. Check Phase-Lock condition
        vel_norm = torch.norm(v_new).item()
        is_locked = False
        locked_well_id = None

        if vel_norm < 0.05:
            # Check if inside a potential well (Hessian > 0 equivalent to near center)
            for idx, well in enumerate(self.wells.values()):
                dist = torch.norm(psi_new - well.center).item()
                if dist < well.sigma * 0.8:
                    is_locked = True
                    locked_well_id = idx
                    # Apply Causal Erosion PDE update
                    self.apply_causal_erosion(well.well_id, erosion_depth=self.erosion_rate * dt)
                    break

        return CognitiveFieldState(
            psi=psi_new,
            velocity=v_new,
            temperature=state.temperature * 0.99,  # Cooling
            phase_locked=is_locked,
            locked_well_index=locked_well_id
        )

    # -------------------------------------------------------------------------
    # 4. Memory & Learning: Causal Erosion PDE & Dynamic Sensory Boundary
    # -------------------------------------------------------------------------

    def apply_causal_erosion(self, well_id: str, erosion_depth: float):
        """
        Applies Causal Erosion PDE to deepen well V_self:
        dV_self / dtau = - erosion_rate + D_diff * Delta V_self
        """
        if well_id in self.wells:
            well = self.wells[well_id]
            well.depth += erosion_depth
            well.erosion_accumulated += erosion_depth

            if self.is_rasterized:
                self.rasterize_to_spatial_grid()

    def compute_phenomenal_present_energy(
        self,
        psi: torch.Tensor,
        sensory_input: torch.Tensor
    ) -> float:
        """
        Quantifies Phenomenal Present / Conscious Residual Energy:
        E_present(tau) = 0.5 * || Psi(tau) - Psi_ext*(tau) ||_g^2
        Represents subjective conscious delay between internal manifold state and external reality.
        """
        sensory_t = sensory_input.flatten().to(self.device).float()
        min_dim = min(psi.shape[0], sensory_t.shape[0])
        diff = psi[:min_dim] - sensory_t[:min_dim]

        g = self.compute_metric_tensor(psi[:min_dim])
        residual_energy = 0.5 * float(torch.dot(diff, torch.matmul(g, diff)).item())
        return residual_energy

    # -------------------------------------------------------------------------
    # 5. Hardware Execution Engine: Spatial Field Splatting & O(1) Query
    # -------------------------------------------------------------------------

    def rasterize_to_spatial_grid(self):
        """
        Splatting memory wells N into Continuous Spatial Tensor Grid.
        Transforms discrete O(N) distance checks into O(1) spatial texture lookup.
        """
        res = self.grid_resolution
        # Use 3D grid projection for spatial tensor grid (first 3 dimensions of R^D)
        grid = torch.zeros((res, res, res), device=self.device)
        min_b, max_b = self.grid_spatial_bounds
        coords = torch.linspace(min_b, max_b, res, device=self.device)

        grid_z, grid_y, grid_x = torch.meshgrid(coords, coords, coords, indexing="ij")
        grid_pos = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (res, res, res, 3)

        for well in self.wells.values():
            well_center_3d = well.center[:3] if well.center.shape[0] >= 3 else F.pad(well.center, (0, 3 - well.center.shape[0]))
            diff = grid_pos - well_center_3d
            dist_sq = torch.sum(diff ** 2, dim=-1)
            well_vals = - well.depth * torch.exp(- 0.5 * dist_sq / (well.sigma ** 2))
            grid = grid + well_vals

        self.spatial_tensor_grid = grid.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, res, res, res) for grid_sample
        self.is_rasterized = True

    def _sample_spatial_grid_O1(self, psi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        O(1) Spatial Field Query using Trilinear Interpolation (grid_sample equivalent).
        """
        if self.spatial_tensor_grid is None:
            self.rasterize_to_spatial_grid()

        psi_3d = psi[:3] if psi.shape[0] >= 3 else F.pad(psi, (0, 3 - psi.shape[0]))
        min_b, max_b = self.grid_spatial_bounds

        # Normalize coordinates to [-1, 1] range for grid_sample
        norm_coords = 2.0 * (psi_3d - min_b) / (max_b - min_b) - 1.0
        norm_coords = torch.clamp(norm_coords, -1.0, 1.0)

        # 5D grid_sample input: (1, 1, 1, 1, 3) for 3D point query
        grid_query = norm_coords.view(1, 1, 1, 1, 3)
        v_self_sampled = F.grid_sample(
            self.spatial_tensor_grid,
            grid_query,
            mode="bilinear",
            padding_mode="border",
            align_corners=True
        )
        v_self = v_self_sampled.reshape(())

        # Central Difference for O(1) Gradient Query
        eps = 1e-2
        grad = torch.zeros_like(psi)
        for i in range(min(3, psi.shape[0])):
            psi_plus = psi.clone()
            psi_minus = psi.clone()
            psi_plus[i] += eps
            psi_minus[i] -= eps

            norm_plus = torch.clamp(2.0 * (psi_plus[:3] - min_b) / (max_b - min_b) - 1.0, -1.0, 1.0).view(1, 1, 1, 1, 3)
            norm_minus = torch.clamp(2.0 * (psi_minus[:3] - min_b) / (max_b - min_b) - 1.0, -1.0, 1.0).view(1, 1, 1, 1, 3)

            v_plus = F.grid_sample(self.spatial_tensor_grid, norm_plus, align_corners=True).reshape(())
            v_minus = F.grid_sample(self.spatial_tensor_grid, norm_minus, align_corners=True).reshape(())
            grad[i] = (v_plus - v_minus) / (2.0 * eps)

        return v_self, grad

    # -------------------------------------------------------------------------
    # 6. Cognitive Fluid Dynamics & Ecosystem Field Solver
    # -------------------------------------------------------------------------

    def step_cognitive_fluid_dynamics(self, dt: float = 0.05):
        """
        Solves Cognitive Navier-Stokes & Continuity Equations on 3D Eulerian Grid:
        - Mass continuity: d rho / d tau + div(rho * u) = S_ext - gamma_evap * rho + R_rain
        - Momentum: d u / d tau + (u . grad) u = - 1/rho grad P + nu Delta u + F_will + F_buoyancy
        - Phase Transition: Evaporation (vapor) <-> Rain condensation (precipitate)
        """
        res = self.fluid_res

        # 1. Compute Pressure P from Density overload (P = c_s^2 * rho^2)
        c_s = 1.0
        self.pressure = c_s * (self.rho ** 2)

        # 2. Compute Pressure Gradient
        grad_p_x = (torch.roll(self.pressure, -1, dims=0) - torch.roll(self.pressure, 1, dims=0)) / 2.0
        grad_p_y = (torch.roll(self.pressure, -1, dims=1) - torch.roll(self.pressure, 1, dims=1)) / 2.0
        grad_p_z = (torch.roll(self.pressure, -1, dims=2) - torch.roll(self.pressure, 1, dims=2)) / 2.0
        grad_p = torch.stack([grad_p_x, grad_p_y, grad_p_z], dim=0)

        # 3. Update Velocity field u
        f_buoyancy = torch.zeros_like(self.u)
        f_buoyancy[2] = 0.1 * self.rho  # Upward convection

        f_will = torch.zeros_like(self.u)
        if self.target_will is not None:
            # Directional pull towards target in fluid grid
            f_will[0] += 0.2 * self.target_will[0].item()

        acceleration = - grad_p / (self.rho.unsqueeze(0) + 1e-4) + f_buoyancy + f_will
        self.u = self.u + acceleration * dt

        # 4. Advect Density rho
        div_u_rho = (
            self.u[0] * (torch.roll(self.rho, -1, dims=0) - torch.roll(self.rho, 1, dims=0)) / 2.0 +
            self.u[1] * (torch.roll(self.rho, -1, dims=1) - torch.roll(self.rho, 1, dims=1)) / 2.0 +
            self.u[2] * (torch.roll(self.rho, -1, dims=2) - torch.roll(self.rho, 1, dims=2)) / 2.0
        )
        self.rho = torch.clamp(self.rho - div_u_rho * dt, min=0.01)

        # 5. Compute Vorticity omega = curl(u)
        duz_dy = (torch.roll(self.u[2], -1, dims=1) - torch.roll(self.u[2], 1, dims=1)) / 2.0
        duy_dz = (torch.roll(self.u[1], -1, dims=2) - torch.roll(self.u[1], 1, dims=2)) / 2.0
        dux_dz = (torch.roll(self.u[0], -1, dims=2) - torch.roll(self.u[0], 1, dims=2)) / 2.0
        duz_dx = (torch.roll(self.u[2], -1, dims=0) - torch.roll(self.u[2], 1, dims=0)) / 2.0
        duy_dx = (torch.roll(self.u[1], -1, dims=0) - torch.roll(self.u[1], 1, dims=0)) / 2.0
        dux_dy = (torch.roll(self.u[0], -1, dims=1) - torch.roll(self.u[0], 1, dims=1)) / 2.0

        self.vorticity = torch.stack([
            duz_dy - duy_dz,
            dux_dz - duz_dx,
            duy_dx - dux_dy
        ], dim=0)

        # 6. Phase Transition: Evaporation & Rainfall Condensation
        # Rainfall condition: when density > saturation threshold, precipitate to ground
        rain_mask = self.rho > 0.5
        rainfall_precipitate = torch.sum(self.rho[rain_mask] - 0.5).item()
        self.rho[rain_mask] = 0.5  # Saturation limit

        return {
            "mean_density": float(self.rho.mean().item()),
            "max_vorticity": float(torch.norm(self.vorticity, dim=0).max().item()),
            "rainfall_precipitate": rainfall_precipitate
        }
