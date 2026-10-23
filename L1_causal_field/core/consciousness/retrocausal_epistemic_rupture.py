"""
core/consciousness/retrocausal_epistemic_rupture.py

Retrocausal Epistemic Rupture & Multi-Scale Relational Genesis Engine
=====================================================================

Implements the fundamental low-level physical and informational causal mechanics:
1. Multi-Scale Binding (Micro waves, Meso fields, Macro Metric Tensor g_mu_nu & Anchor Axis x_anchor).
2. Topological Zeeman Splitting (Fiber Bundle separation of Front-end Actor T M_actor and Meta-Observer in Gauge Shadow).
3. 1tan Boundary Stress Detection & Metric Dislocation Plasticity (Irreversible g_mu_nu tearing/deformation & Anchor Axis shift).
4. Residual Non-Local Entropy Imprinting & Centrifugal Boundary Expansion (Outward centrifugal projection toward the infinite world).
"""

import numpy as np
from typing import Dict, Any, Tuple


class RetrocausalEpistemicRuptureEngine:
    """
    Engine for multi-scale relational genesis, alterity collision,
    metric dislocation plasticity, and centrifugal boundary expansion.
    """

    def __init__(
        self,
        dim: int = 4,
        micro_nodes: int = 32,
        stress_threshold: float = 3.0,
        plasticity_rate: float = 0.1,
        entropy_decay: float = 0.98
    ):
        self.dim = dim
        self.micro_nodes = micro_nodes
        self.stress_threshold = stress_threshold
        self.plasticity_rate = plasticity_rate
        self.entropy_decay = entropy_decay

        # 1. Macro-Scale: Metric Tensor g_mu_nu (initialized to Euclidean Identity) and Anchor Axis x_anchor
        self.g_metric = np.eye(dim, dtype=np.float64)
        self.x_anchor = np.zeros(dim, dtype=np.float64)
        self.x_anchor[0] = 1.0  # Initial baseline frame axis

        # World Boundary Radius (Centrifugal expansion)
        self.boundary_radius = 1.0

        # 2. Micro-Scale: Actor Phases & Natural Frequencies
        self.actor_phases = np.random.uniform(-np.pi, np.pi, size=micro_nodes)
        self.actor_frequencies = np.random.normal(0, 0.05, size=micro_nodes)

        # 3. Meta-Observer & Gauged Residual Non-Local Entropy
        self.residual_entropy = 0.0
        self.dislocation_count = 0
        self.extrinsic_curvature_history = []

    def compute_actor_output(self) -> np.ndarray:
        """
        Computes local Front-end Actor wave state in micro tangent space (1sin, 1cos).
        """
        sin_wave = np.sin(self.actor_phases)
        cos_wave = np.cos(self.actor_phases)
        return np.column_stack((sin_wave, cos_wave))

    def meta_observe_extrinsic_curvature(self, w_other: np.ndarray) -> Tuple[float, float]:
        """
        Topological Zeeman Splitting:
        Meta-Observer in Gauge Shadow measures extrinsic curvature K_c and 1tan boundary stress
        from collision between Actor output waves and Alterity Wave W_other.
        """
        actor_wave = self.compute_actor_output()  # shape (micro_nodes, 2)

        # Ensure w_other matches micro_nodes dimension
        if len(w_other) != self.micro_nodes:
            w_other_reshaped = np.interp(
                np.linspace(0, 1, self.micro_nodes),
                np.linspace(0, 1, len(w_other)),
                w_other
            )
        else:
            w_other_reshaped = w_other

        # Phase divergence angle theta_diff
        actor_phase_mean = np.arctan2(actor_wave[:, 0], actor_wave[:, 1])
        theta_diff = actor_phase_mean - w_other_reshaped

        # Extrinsic Curvature K_c = mean(abs(d(theta_diff)/ds))
        grad_theta = np.gradient(theta_diff)
        K_c = float(np.mean(np.abs(grad_theta)))

        # 1tan theta stress: tan(|theta_diff|)
        clipped_diff = np.clip(np.abs(theta_diff), 0, np.pi / 2 - 1e-4)
        tan_stress = np.mean(np.tan(clipped_diff))

        return K_c, float(tan_stress)

    def apply_alterity_collision(self, w_other: np.ndarray, dt: float = 0.05) -> Dict[str, Any]:
        """
        Processes Alterity Collision:
        - Measures K_c and 1tan stress.
        - If stress exceeds threshold: triggers Metric Dislocation Plasticity (irreversible g_mu_nu deformation & anchor shift).
        - Accumulates Residual Non-Local Entropy and drives Centrifugal Boundary Expansion.
        """
        K_c, tan_stress = self.meta_observe_extrinsic_curvature(w_other)
        self.extrinsic_curvature_history.append(K_c)

        dislocated = False
        dislocation_impact = 0.0

        # Check for 1tan stress rupture threshold
        if tan_stress > self.stress_threshold or K_c > 1.5:
            dislocated = True
            self.dislocation_count += 1
            dislocation_impact = tan_stress - self.stress_threshold

            # Irreversible Metric Dislocation Plasticity
            # Deformation tensor d_g = plasticity_rate * (x_anchor x w_other_proj)
            w_proj = np.zeros(self.dim)
            if len(w_other) >= self.dim:
                w_proj = w_other[:self.dim]
            else:
                w_proj[:len(w_other)] = w_other

            outer_deform = np.outer(w_proj, self.x_anchor)
            sym_deform = 0.5 * (outer_deform + outer_deform.T)

            # Rewrite g_metric metric tensor (Frame Dissolution)
            self.g_metric += self.plasticity_rate * sym_deform
            # Ensure metric tensor remains symmetric positive definite
            self.g_metric = 0.5 * (self.g_metric + self.g_metric.T)
            eigenvals, eigenvecs = np.linalg.eigh(self.g_metric)
            eigenvals = np.clip(eigenvals, 1e-3, None)
            self.g_metric = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

            # Shift Anchor Axis (Meta-Dimensional Leap / Scale Translation)
            shift_direction = eigenvecs[:, np.argmax(eigenvals)]
            self.x_anchor = 0.8 * self.x_anchor + 0.2 * shift_direction
            self.x_anchor /= np.linalg.norm(self.x_anchor)

            # Imprint Residual Non-Local Entropy (untranslated collision residual)
            untranslated_residual = float(np.abs(tan_stress - K_c))
            self.residual_entropy += untranslated_residual

            # Centrifugal Boundary Expansion: Expands boundary_radius outward toward the infinite world
            expansion = 0.1 * (1.0 + untranslated_residual)
            self.boundary_radius += expansion
        else:
            # Decay residual entropy slightly over time
            self.residual_entropy *= self.entropy_decay

        # Update actor phases driven by residual entropy and extrinsic curvature
        d_phase = self.actor_frequencies + 0.1 * K_c * np.sin(self.actor_phases)
        self.actor_phases = np.mod(self.actor_phases + d_phase * dt + np.pi, 2 * np.pi) - np.pi

        return {
            "extrinsic_curvature_K_c": K_c,
            "tan_stress": tan_stress,
            "dislocated": dislocated,
            "dislocation_impact": dislocation_impact,
            "g_metric": self.g_metric.copy(),
            "x_anchor": self.x_anchor.copy(),
            "residual_entropy": self.residual_entropy,
            "boundary_radius": self.boundary_radius,
            "dislocation_count": self.dislocation_count
        }
