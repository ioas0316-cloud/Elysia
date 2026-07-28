import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from core.physics.causal_field import InformationVoxel

class PhaseTransitionEngine:
    """
    [Phase Transition Engine: Ginzburg-Landau & Cahn-Hilliard Dynamics]
    Simulates a continuous 2D information density field.
    Allows a uniform or perturbed field of information to separate under the
    Ginzburg-Landau double-well potential into dense concepts (high density) and
    vacuum backgrounds/firmaments (low density).

    Energy Functional: F(rho) = \int [ f(rho) + 0.5 * gamma * |grad rho|^2 ] dV
    Double Well: f(rho) = alpha * (rho - rho_vacuum)^2 * (rho - rho_entity)^2
    """
    def __init__(self, size: int = 32, alpha: float = 1.0, gamma: float = 0.5, rho_vacuum: float = 0.0, rho_entity: float = 1.0):
        self.size = size
        self.alpha = alpha     # Double-well potential barrier height
        self.gamma = gamma     # Surface tension/interface thickness parameter
        self.rho_vacuum = rho_vacuum
        self.rho_entity = rho_entity

        # Information density field (2D grid)
        # Start near a uniform phase with small random perturbations (high entropy initial state)
        self.density = np.full((size, size), 0.5, dtype=np.float32) + np.random.normal(0, 0.05, (size, size)).astype(np.float32)
        self.density = np.clip(self.density, 0.0, 1.0)

        # Chromatic grid: [Red (Flux), Blue (Order), Yellow (Entropy)]
        self.chromatic_grid = np.zeros((3, size, size), dtype=np.float32)
        self.chromatic_grid[0, :, :] = 0.33  # Red
        self.chromatic_grid[1, :, :] = 0.33  # Blue
        self.chromatic_grid[2, :, :] = 0.34  # Yellow

    def inject_disturbance(self, x_norm: float, y_norm: float, intensity: float, chromatic_impact: Optional[np.ndarray] = None):
        """
        Inject an external information disturbance (energy injection) into the density field.
        """
        cx = int(np.clip(x_norm * (self.size - 1), 0, self.size - 1))
        cy = int(np.clip(y_norm * (self.size - 1), 0, self.size - 1))

        radius = 3
        for i in range(max(0, cx - radius), min(self.size, cx + radius + 1)):
            for j in range(max(0, cy - radius), min(self.size, cy + radius + 1)):
                dist = np.sqrt((i - cx)**2 + (j - cy)**2)
                if dist <= radius:
                    factor = 1.0 - (dist / radius)
                    self.density[i, j] += intensity * factor
                    if chromatic_impact is not None:
                        self.chromatic_grid[:, i, j] = (1.0 - factor * 0.5) * self.chromatic_grid[:, i, j] + factor * 0.5 * chromatic_impact

        self.density = np.clip(self.density, 0.0, 1.0)

    def calculate_free_energy(self) -> Tuple[float, float]:
        """
        Calculates the bulk Ginzburg-Landau energy and interface gradient energy.
        Returns: (bulk_energy, gradient_energy)
        """
        # Bulk potential energy: f(rho) = alpha * (rho - rho_vac)^2 * (rho - rho_ent)^2
        bulk = self.alpha * (self.density - self.rho_vacuum)**2 * (self.density - self.rho_entity)**2
        bulk_energy = float(np.sum(bulk))

        # Gradient square energy: 0.5 * gamma * |grad rho|^2
        grad_x = np.diff(self.density, axis=0, append=self.density[-1:, :])
        grad_y = np.diff(self.density, axis=1, append=self.density[:, -1:])
        grad_sq = grad_x**2 + grad_y**2
        gradient_energy = float(0.5 * self.gamma * np.sum(grad_sq))

        return bulk_energy, gradient_energy

    def step(self, dt: float = 0.1):
        """
        Advances Cahn-Hilliard-like phase separation.
        Chemical Potential: \mu = df/d_rho - \gamma * \nabla^2 \rho
        Conservation law: d_rho/dt = M * \nabla^2 \mu
        We modulate Mobility (M) with the Chromatic Red (Flux) field.
        We modulate interface thickness with the Chromatic Blue (Order) field.
        Yellow (Entropy) acts as thermal fluctuation/noise.
        """
        # 1. Modulation parameters from chromatic grids
        # Red (Flux) scales Mobility (speed of phase separation)
        mobility = 0.1 + 0.9 * self.chromatic_grid[0]
        # Blue (Order) increases boundary stiffness/interface energy
        order_stiffness = self.gamma * (0.5 + 1.5 * self.chromatic_grid[1])
        # Yellow (Entropy) scales Brownian random fluctuations
        entropy_fluctuation = 0.05 * self.chromatic_grid[2]

        # 2. Bulk derivative: df/d_rho = 2 * alpha * (rho - vacuum) * (rho - entity) * (2 * rho - vacuum - entity)
        rho = self.density
        vac = self.rho_vacuum
        ent = self.rho_entity
        df_drho = 2.0 * self.alpha * (rho - vac) * (rho - ent) * (2.0 * rho - vac - ent)

        # 3. Discrete Laplacian of density: \nabla^2 \rho
        laplacian_rho = (
            np.roll(rho, 1, axis=0) + np.roll(rho, -1, axis=0) +
            np.roll(rho, 1, axis=1) + np.roll(rho, -1, axis=1) - 4 * rho
        )

        # Chemical Potential: \mu = df/d_rho - gamma_effective * \nabla^2 \rho
        mu = df_drho - order_stiffness * laplacian_rho

        # 4. Laplacian of Chemical Potential: \nabla^2 \mu
        laplacian_mu = (
            np.roll(mu, 1, axis=0) + np.roll(mu, -1, axis=0) +
            np.roll(mu, 1, axis=1) + np.roll(mu, -1, axis=1) - 4 * mu
        )

        # 5. Cahn-Hilliard equation step: d_rho/dt = \nabla * (M * \nabla \mu)
        # Approximated as: M * \nabla^2 \mu
        d_density_dt = mobility * laplacian_mu

        # 6. Apply thermal noise from Yellow (Entropy)
        noise = (np.random.normal(0, 1.0, rho.shape).astype(np.float32)) * entropy_fluctuation

        # Update density and enforce conservation and boundary limits
        self.density += (d_density_dt + noise) * dt
        self.density = np.clip(self.density, 0.0, 1.0)

        # 7. Chromatic grid diffusion & conservation
        for c in range(3):
            lap_c = (
                np.roll(self.chromatic_grid[c], 1, axis=0) + np.roll(self.chromatic_grid[c], -1, axis=0) +
                np.roll(self.chromatic_grid[c], 1, axis=1) + np.roll(self.chromatic_grid[c], -1, axis=1) - 4 * self.chromatic_grid[c]
            )
            self.chromatic_grid[c] += 0.05 * lap_c * dt

        # Re-normalize chromatic grid locally
        total_chromatic = np.sum(self.chromatic_grid, axis=0, keepdims=True)
        total_chromatic = np.where(total_chromatic > 0, total_chromatic, 1.0)
        self.chromatic_grid /= total_chromatic


class DensityFluidGravity:
    """
    [Density Fluid Gravity Engine]
    O(N) Fluid-Based Gravitational Alignment.
    Instead of calculating O(N^2) mutual distances, voxels read the pressure P and
    surface tension gradient vectors of the continuous phase transition density field.
    Voxels flow down the pressure gradient (-\nabla P) into the energy sinks (high density zones).

    Fluid force: F = -grad P + viscosity * laplacian(v)
    """
    def __init__(self, size: int = 32, pressure_scaling: float = 2.0, viscosity: float = 0.1):
        self.size = size
        self.pressure_scaling = pressure_scaling
        self.viscosity = viscosity  # Fluid shear resistance/viscosity damping

    def apply_gravity(self, voxels: List[InformationVoxel], phase_field: PhaseTransitionEngine, dt: float = 0.1):
        """
        Calculates and applies the pressure-gradient gravity and fluid viscosity
        to the voxels flowing on the continuous 2D coordinate space.
        Assumes voxels have 2D positions/velocities mapped from their 3D vectors
        or mapped onto the 2D density grid boundary [0, 1] x [0, 1].
        """
        if not voxels:
            return

        # 1. Compute pressure gradient field from Phase density field
        # High density = High structural concentration = Low local pressure (energy sink)
        # So, pressure P = 1.0 - density
        density = phase_field.density
        pressure = 1.0 - density

        # Calculate spatial gradients of pressure using finite differences
        grad_P_x, grad_P_y = np.gradient(pressure)

        # 2. Extract voxel coordinates and map to phase grid index Space
        for voxel in voxels:
            # Map 3D coordinates (assuming bounded, e.g. -10 to 10) to 2D range [0, 1]
            # If position is not initialized or invalid, give random
            pos = voxel.position
            if pos is None:
                continue

            # Project 3D to 2D normalized space
            x_norm = float(np.clip((pos[0] + 10.0) / 20.0, 0.0, 1.0))
            y_norm = float(np.clip((pos[1] + 10.0) / 20.0, 0.0, 1.0))

            # Find cell index on the grid
            idx_x = int(x_norm * (self.size - 1))
            idx_y = int(y_norm * (self.size - 1))

            # Retrieve gradient force at voxel position: F_g = - \nabla P
            # Since gradient is calculated along indices, scale appropriately
            force_x = -grad_P_x[idx_x, idx_y] * self.pressure_scaling
            force_y = -grad_P_y[idx_x, idx_y] * self.pressure_scaling

            # 3. Apply viscous fluid drag (viscosity * \nabla^2 v)
            # Lap velocity is approximated from neighbor voxel velocities
            # O(N) neighbor interaction using grid binning to maintain performance
            # Simple approximation: localize and damp based on voxel density
            local_density = density[idx_x, idx_y]
            damping_factor = 1.0 - (self.viscosity * local_density * dt)
            damping_factor = float(np.clip(damping_factor, 0.5, 1.0))

            # Apply gravity acceleration
            voxel.velocity[0] += force_x * dt
            voxel.velocity[1] += force_y * dt

            # Damp velocities (viscous resistance)
            voxel.velocity[0] *= damping_factor
            voxel.velocity[1] *= damping_factor

            # Modulate potential based on local density concentration (resonance depth)
            voxel.potential = float(local_density)

            # Chromatic coupling: Voxel absorbs chromatic characteristics from the field
            field_chromatic = phase_field.chromatic_grid[:, idx_x, idx_y]

            # Robustness check to avoid AttributeError if voxel.chromatic_vector does not exist
            voxel_chrom_vec = getattr(voxel, 'chromatic_vector', None)
            if voxel_chrom_vec is None:
                # Default fallback
                voxel.chromatic_vector = np.array([0.33, 0.33, 0.34], dtype=np.float32)
                voxel_chrom_vec = voxel.chromatic_vector

            voxel.chromatic_vector = 0.8 * voxel_chrom_vec + 0.2 * field_chromatic
            # Re-normalize chromatic vector
            total_chromatic = np.sum(voxel.chromatic_vector)
            if total_chromatic > 0:
                voxel.chromatic_vector /= total_chromatic
