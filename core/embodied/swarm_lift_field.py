"""
swarm_lift_field.py: N-Drone Swarm Distributed Potential Field & Autonomous Phase Re-Locking Engine
===================================================================================================

Adheres to "Do not calculate, let it flow." - Continuous Causal Intelligence Principles.

This module models N drone agents in a 3D/5D phase lattice space. Instead of brute-force propeller energy
consumption, drones form a distributed potential field that superimposes pressure gradients to create
a synthetic virtual gliding lift field (Lift Field).

When subjected to external wind gust shock, the field temporarily dissipates/scatters, and the system
autonomously undergoes phase re-locking relaxation back to a stable order parameter R ≈ 1.0.

Mathematical Formulation:
1. Kuramoto Phase Locking Dynamics: dθ_i/dt = ω_i + K/N ∑_{j=1}^N sin(θ_j - θ_i) + S_i(wind)
2. Order Parameter: R e^(i Ψ) = 1/N ∑_{j=1}^N e^(i θ_j)
3. Synthetic Lift Field Energy Saved Ratio: E_saved = R · (1.0 - Dissipation_Loss)
4. Phase Re-locking Relaxation: ΔΦ = 1.0 - R -> 0 as t -> ∞
"""

import numpy as np
from typing import Tuple, Dict, Any, List


class DroneSwarmLiftFieldSimulator:
    """
    Simulates N drones creating a distributed aerodynamic potential field,
    reacting to wind gust shocks, and executing phase re-locking relaxation.
    """

    def __init__(
        self,
        num_drones: int = 32,
        coupling_K: float = 2.5,
        natural_freq_std: float = 0.1,
        spatial_dim: int = 3
    ):
        self.num_drones = num_drones
        self.coupling_K = coupling_K
        self.spatial_dim = spatial_dim

        # Natural frequencies ω_i
        np.random.seed(100)
        self.omega = np.random.normal(1.0, natural_freq_std, num_drones)

        # Drone phase angles θ_i ∈ [-π, π]
        self.phases = np.random.uniform(-np.pi, np.pi, num_drones)

        # Drone 3D lattice positions
        self.positions = np.random.uniform(-10.0, 10.0, (num_drones, spatial_dim))

        # Wind gust shock vector
        self.wind_gust = np.zeros(spatial_dim, dtype=float)

    def compute_order_parameter(self) -> Tuple[float, float]:
        """
        Computes Kuramoto macro order parameter R ∈ [0, 1] and average phase Ψ.
        R = |1/N ∑ e^(i θ_j)|
        """
        complex_sum = np.mean(np.exp(1j * self.phases))
        R = float(np.abs(complex_sum))
        Psi = float(np.angle(complex_sum))
        return R, Psi

    def apply_wind_gust_shock(self, gust_vector: np.ndarray, intensity: float = 5.0) -> None:
        """
        Applies external wind gust shock vector, disrupting drone phase synchronization.
        """
        self.wind_gust = np.asarray(gust_vector, dtype=float) * intensity

        # Disruption perturbation directly added to phase angles
        phase_pert = np.random.uniform(-np.pi * 0.8, np.pi * 0.8, self.num_drones)
        self.phases += phase_pert
        self.phases = (self.phases + np.pi) % (2.0 * np.pi) - np.pi

    def step_phase_locking_dynamics(self, dt: float = 0.05) -> Tuple[float, float, float]:
        """
        Integrates Kuramoto non-linear coupled differential equations:
        dθ_i/dt = ω_i + K/N ∑ sin(θ_j - θ_i) - Wind_Influence
        """
        N = self.num_drones
        phases_tile = np.tile(self.phases, (N, 1))
        phase_diffs = phases_tile - phases_tile.T

        # Coupling force: K/N * sum_j sin(θ_j - θ_i)
        coupling = (self.coupling_K / N) * np.sum(np.sin(phase_diffs), axis=1)

        # Wind shock influence on phase derivative
        wind_magnitude = float(np.linalg.norm(self.wind_gust))
        wind_influence = wind_magnitude * 0.1 * np.sin(self.phases)

        # Differential dθ_i/dt
        dtheta_dt = self.omega + coupling - wind_influence

        # Decay wind gust over time (dissipation)
        self.wind_gust *= 0.85

        # Integration step
        self.phases += dt * dtheta_dt
        self.phases = (self.phases + np.pi) % (2.0 * np.pi) - np.pi

        # Order parameter R and Psi
        R, Psi = self.compute_order_parameter()

        # Phase Divergence Error ΔΦ
        delta_phi = 1.0 - R

        # Synthetic Lift Field Energy Saved Ratio (relative to brute-force hovering)
        energy_saved_ratio = R * 0.75  # Up to 75% energy saved under perfect lock R = 1.0

        return R, delta_phi, energy_saved_ratio

    def run_relaxation_until_phase_lock(
        self,
        target_R: float = 0.90,
        max_steps: int = 100
    ) -> Dict[str, Any]:
        """
        Runs phase re-locking relaxation loop until macro order parameter R exceeds target threshold.
        """
        history = []
        for step in range(1, max_steps + 1):
            R, delta_phi, energy_saved = self.step_phase_locking_dynamics(dt=0.05)
            history.append({"step": step, "R": R, "delta_phi": delta_phi, "energy_saved": energy_saved})

            if R >= target_R:
                break

        return {
            "converged": history[-1]["R"] >= target_R,
            "total_steps": len(history),
            "final_R": history[-1]["R"],
            "final_delta_phi": history[-1]["delta_phi"],
            "final_energy_saved": history[-1]["energy_saved"],
            "history": history
        }
