"""
1D-4D Commutative Manifold & Counterfactual Simulator Engine for Elysia.

This module implements:
1. MemoryTrajectory1D: 1D topological causal trajectory z(t) containing minimal state parameters.
2. SpacetimeExpansionBasis4D: Observation basis tensor O_basis expanding 1D trajectory z(t)
   into 4D spacetime manifold (x, y, z, t) via metric tensor inner product.
3. CounterfactualSimulator: Injecting virtual perturbation Δz at t_intervention, propagating
   causal waves, and computing baseline vs counterfactual spacetime divergence metrics.
4. IdentityCoreGuardrail: Scanning 4D spacetime manifold fields against existential identity core
   radius r_core and maximum topological tension threshold ΔT_max to approve/reject counterfactual futures.
"""

from dataclasses import dataclass
import numpy as np
from typing import Dict, Optional, Tuple


@dataclass
class MemoryTrajectory1D:
    """1D topological causal trajectory (minimal parameter sequence over time t)."""
    timestamps: np.ndarray          # 1D time array [T]
    causal_states: np.ndarray       # 1D causal parameter matrix [T, StateDim]


@dataclass
class VerificationResult:
    """Result of Identity Core Guardrail scanning."""
    is_approved: bool
    reason: str
    min_core_distance: float
    max_tension_delta: float


class SpacetimeExpansionBasis4D:
    """
    Observation basis tensor matrix O_basis mapping 1D latent states z(t)
    to 4D spacetime manifold (Time, SpatialNodes, Coordinates[x, y, z, t]).
    """

    def __init__(self, state_dim: int = 4, spatial_nodes: int = 5, seed: int = 42, spatial_offset: float = 2.0):
        self.state_dim = state_dim
        self.spatial_nodes = spatial_nodes
        self.spatial_offset = spatial_offset
        np.random.seed(seed)

        # Spatial projection basis matrix mapping 1D state dim -> N nodes x 3D (x, y, z)
        self.spatial_projection = np.random.randn(state_dim, spatial_nodes, 3) * 0.3

        # Metric tensor g_ij
        self.metric_tensor = np.eye(3)

    def expand_1d_to_4d(self, trajectory: MemoryTrajectory1D) -> np.ndarray:
        """
        Expands 1D trajectory z(t) into 4D spacetime manifold array [T, SpatialNodes, 4].
        Coordinates at last dim are [x, y, z, t].
        """
        T_steps = len(trajectory.timestamps)
        spacetime_manifold_4d = np.zeros((T_steps, self.spatial_nodes, 4))

        for t_idx, t_val in enumerate(trajectory.timestamps):
            z_t = trajectory.causal_states[t_idx] # 1D causal state at timestamp t

            # Linear/Metric projection: P_spatial = z_t . O_basis . M_tensor
            flat_proj = np.dot(z_t, self.spatial_projection.reshape(self.state_dim, -1))
            spatial_coords = flat_proj.reshape(self.spatial_nodes, 3) + self.spatial_offset
            spatial_coords = np.dot(spatial_coords, self.metric_tensor)

            spacetime_manifold_4d[t_idx, :, 0:3] = spatial_coords
            spacetime_manifold_4d[t_idx, :, 3] = t_val

        return spacetime_manifold_4d


class CounterfactualSimulator:
    """
    Counterfactual Reasoning & Predictive Simulation Engine.
    Injects virtual shock Δz at step t_intervention and evaluates causal divergence.
    """

    def __init__(self, basis_4d: SpacetimeExpansionBasis4D):
        self.basis = basis_4d

    def simulate_counterfactual_future(
        self,
        base_trajectory: MemoryTrajectory1D,
        intervention_step: int,
        delta_perturbation: np.ndarray,
        decay_rate: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        1. Expands baseline 1D trajectory to baseline 4D spacetime manifold.
        2. Injects counterfactual perturbation Δz at intervention step with causal wave decay.
        3. Expands counterfactual 1D trajectory to counterfactual 4D manifold.
        4. Calculates temporal spacetime divergence metrics.
        """
        # 1. Baseline 4D manifold expansion
        baseline_4d = self.basis.expand_1d_to_4d(base_trajectory)

        # 2. Counterfactual 1D state generation
        cf_states = base_trajectory.causal_states.copy()

        for t in range(intervention_step, len(base_trajectory.timestamps)):
            decay = np.exp(-decay_rate * (t - intervention_step))
            cf_states[t] += delta_perturbation * decay

        cf_trajectory = MemoryTrajectory1D(
            timestamps=base_trajectory.timestamps.copy(),
            causal_states=cf_states
        )

        # 3. Counterfactual 4D manifold expansion
        cf_4d = self.basis.expand_1d_to_4d(cf_trajectory)

        # 4. Spacetime divergence calculation
        spatial_diff = cf_4d[:, :, 0:3] - baseline_4d[:, :, 0:3]
        divergence_over_time = np.linalg.norm(spatial_diff, axis=2).mean(axis=1)

        metrics = {
            "intervention_time": float(base_trajectory.timestamps[intervention_step]),
            "max_divergence": float(np.max(divergence_over_time)),
            "final_divergence": float(divergence_over_time[-1])
        }

        return baseline_4d, cf_4d, metrics


class IdentityCoreGuardrail:
    """
    Absolute Identity Core Guardrail Engine.
    Scans counterfactual 4D spacetime manifold fields against core origin (0,0,0),
    core radius r_core, and maximum tension deformation limits ΔT_max.
    """

    def __init__(
        self,
        core_origin: np.ndarray = np.array([0.0, 0.0, 0.0]),
        core_radius: float = 0.8,
        max_tension_delta: float = 1.8
    ):
        self.core_origin = core_origin
        self.core_radius = core_radius
        self.max_tension_delta = max_tension_delta

    def verify_counterfactual_future(
        self,
        base_4d_manifold: np.ndarray,
        cf_4d_manifold: np.ndarray
    ) -> VerificationResult:
        """Scans counterfactual 4D field for existential breach or tension rupture."""
        # 1. Core distance scanning
        cf_xyz = cf_4d_manifold[:, :, 0:3]
        distances_to_core = np.linalg.norm(cf_xyz - self.core_origin, axis=2)
        min_distance = float(np.min(distances_to_core))

        # Check A: Identity core boundary breach (r < r_core)
        if min_distance < self.core_radius:
            t_step, node_idx = np.unravel_index(np.argmin(distances_to_core), distances_to_core.shape)
            t_val = float(cf_4d_manifold[t_step, node_idx, 3])
            return VerificationResult(
                is_approved=False,
                reason=f"REJECTED (Existential Breach): t={t_val:.2f}s, Node {node_idx} breached core boundary "
                       f"(min dist: {min_distance:.4f} < r_core: {self.core_radius:.4f})",
                min_core_distance=min_distance,
                max_tension_delta=0.0
            )

        # 2. Tension delta scanning
        base_xyz = base_4d_manifold[:, :, 0:3]
        tension_deltas = np.linalg.norm(cf_xyz - base_xyz, axis=2)
        max_delta = float(np.max(tension_deltas))

        # Check B: Topological tension rupture (ΔT > ΔT_max)
        if max_delta > self.max_tension_delta:
            return VerificationResult(
                is_approved=False,
                reason=f"REJECTED (Tension Rupture): Max tension delta ({max_delta:.4f}) exceeded threshold ({self.max_tension_delta:.4f})",
                min_core_distance=min_distance,
                max_tension_delta=max_delta
            )

        # Approval: Existential stability verified
        return VerificationResult(
            is_approved=True,
            reason=f"APPROVED: Counterfactual future preserves identity core "
                   f"(min dist: {min_distance:.4f}, max tension delta: {max_delta:.4f})",
            min_core_distance=min_distance,
            max_tension_delta=max_delta
        )
