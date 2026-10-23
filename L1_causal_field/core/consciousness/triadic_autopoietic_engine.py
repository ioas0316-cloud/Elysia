r"""
Triadic Autopoietic Engine: Triadic Integration Loop of Homeostatic Valence,
Active Inference, and Topological Attractor Memory.

Implements the unified state space system:
    \Psi(t) = (q(t), a(t), h(t))
where:
    - q(t) \in \mathbb{H}^N: Quaternion phase tensor field nodes.
    - a(t) \in \mathbb{R}^M: Extrinsic action vector emitted upon phase tension spillover.
    - h(t) \in \mathbb{R}^K: Internal thermodynamic homeostatic state vector.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiplies two quaternions or arrays of quaternions q = [w, x, y, z].
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.stack([w, x, y, z], axis=-1)


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    """
    Computes conjugate / norm_sq for unit quaternions or array of quaternions.
    For unit quaternions, inverse is equal to conjugate [w, -x, -y, -z].
    """
    norm_sq = np.sum(q ** 2, axis=-1, keepdims=True)
    norm_sq = np.where(norm_sq == 0, 1e-12, norm_sq)
    conj = np.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], axis=-1)
    return conj / norm_sq


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """Normalizes quaternions to unit norm."""
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1e-12, norm)
    return q / norm


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


class TriadicAutopoieticEngine:
    r"""
    Triadic Autopoietic Integration Engine.
    Combines:
      1. Homeostatic Strain & Thermodynamic Valence \mathcal{V}
      2. Active Inference Action Spillover a(t) & Sensory Feedback
      3. Topological Attractor Memory q_i \in \mathbb{H}^N
    """

    def __init__(
        self,
        num_nodes: int = 16,
        action_dim: int = 4,
        homeo_dim: int = 3,
        theta_action: float = 0.5,
        target_homeo: Optional[np.ndarray] = None,
        homeo_weight_matrix: Optional[np.ndarray] = None,
        gamma_0: float = 0.5,
        gamma_max: float = 2.0,
        alpha_v: float = 2.0,
        T_0: float = 0.01,
        T_max: float = 0.5,
        beta_v: float = 2.0,
        seed: Optional[int] = 42,
    ):
        if seed is not None:
            np.random.seed(seed)

        self.num_nodes = num_nodes
        self.action_dim = action_dim
        self.homeo_dim = homeo_dim
        self.theta_action = theta_action

        # Homeostatic target h* and weight matrix M
        self.h_star = target_homeo if target_homeo is not None else np.zeros(homeo_dim)
        if homeo_weight_matrix is not None:
            self.M = homeo_weight_matrix
        else:
            self.M = np.eye(homeo_dim)

        # Dynamic relaxation & thermal parameters
        self.gamma_0 = gamma_0
        self.gamma_max = gamma_max
        self.alpha_v = alpha_v
        self.T_0 = T_0
        self.T_max = T_max
        self.beta_v = beta_v

        # State Variables \Psi(t) = (q(t), a(t), h(t))
        raw_q = np.random.randn(num_nodes, 4)
        self.q = normalize_quaternion(raw_q)
        self.a = np.zeros(action_dim)
        self.h = self.h_star.copy() + np.random.randn(homeo_dim) * 0.1

        # Action mapping matrix (maps 3D vector of error to action_dim)
        self.W_action = np.random.randn(3, action_dim) * 0.5

        # Attractor Memory Topology (list of stored unit quaternion configurations)
        self.attractor_basins: List[np.ndarray] = []

        # Previous state tracking for differential calculations
        self.prev_D_H: float = self.compute_homeostatic_strain(self.h)
        self.prev_q: np.ndarray = self.q.copy()
        self.current_valence: float = 0.0
        self.accumulated_phase_winding: float = 0.0

        # Environment State x_env
        self.x_env = np.zeros(action_dim)

    def compute_homeostatic_strain(self, h: np.ndarray) -> float:
        """
        Calculates Mahalanobis Homeostatic Strain Energy:
        D_H(h) = 0.5 * (h - h*)^T M (h - h*)
        """
        diff = h - self.h_star
        return float(0.5 * diff.T @ self.M @ diff)

    def compute_valence(self, current_D_H: float, dt: float) -> float:
        r"""
        Thermodynamic Valence:
        \mathcal{V} = - d(D_H) / dt
        Positive Valence (\mathcal{V} > 0): System moving towards homeostasis (entropy reduction).
        Negative Valence (\mathcal{V} < 0): System disrupted away from homeostasis.
        """
        if dt <= 0:
            return 0.0
        return float(-(current_D_H - self.prev_D_H) / dt)

    def get_dynamic_relaxation(self, valence: float) -> float:
        r"""
        \gamma(\mathcal{V}) = \gamma_0 + \gamma_{\max} \cdot \sigma_s(\alpha_v \cdot \mathcal{V})
        Accelerates phase-locking when valence is positive.
        """
        return self.gamma_0 + self.gamma_max * float(sigmoid(np.array(self.alpha_v * valence)))

    def get_effective_thermal_agitation(self, valence: float) -> float:
        r"""
        T_{\text{eff}}(\mathcal{V}) = T_0 + T_{\max} \cdot \sigma_s(-\beta_v \cdot \mathcal{V})
        Spikes noise when valence is negative to force escape from toxic basins.
        """
        return self.T_0 + self.T_max * float(sigmoid(np.array(-self.beta_v * valence)))

    def register_attractor(self, target_q: np.ndarray) -> None:
        """Registers a unit quaternion field configuration into Attractor Memory Topology."""
        normalized_target = normalize_quaternion(target_q.copy())
        self.attractor_basins.append(normalized_target)

    def compute_attractor_force(self) -> np.ndarray:
        """
        Computes gradient attraction force toward nearest registered attractor topology basin.
        """
        if not self.attractor_basins:
            return np.zeros_like(self.q)

        # Find nearest attractor basin based on average quaternion dot product
        best_basin = max(
            self.attractor_basins,
            key=lambda basin: np.mean(np.abs(np.sum(self.q * basin, axis=-1))),
        )

        # Vector force pulling node q_i toward target basin q_target_i
        # force_i = q_target_i - q_i
        return best_basin - self.q

    def compute_diagnostics(self, dot_q: np.ndarray) -> Dict[str, float]:
        r"""
        Computes the 3 non-loss Observability Metrics:
          1. Quaternion Order Parameter R_\mathbb{H} (Kuramoto coherence)
          2. Gauge Defect Density \rho_D
          3. Entropy Dissipation Rate \dot{S}_{\text{int}}
        """
        # 1. Quaternion Order Parameter R_\mathbb{H}
        mean_q = np.mean(self.q, axis=0)
        R_H = float(np.linalg.norm(mean_q))

        # 2. Gauge Defect Density \rho_D (mismatch between neighbor nodes)
        q_inv = quaternion_inverse(self.q)
        # Shift nodes to simulate neighbor coupling in lattice
        q_neighbor = np.roll(self.q, shift=1, axis=0)
        mismatch = quaternion_multiply(q_neighbor, q_inv)
        # Difference from identity quaternion [1, 0, 0, 0]
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        defect = np.linalg.norm(mismatch - identity, axis=-1)
        rho_D = float(np.mean(defect ** 2))

        # 3. Entropy Dissipation Rate \dot{S}_{\text{int}}
        dot_S_int = float(np.mean(np.sum(dot_q ** 2, axis=-1)))

        return {
            "R_H": R_H,
            "rho_D": rho_D,
            "dot_S_int": dot_S_int,
        }

    def compute_winding_number(self) -> float:
        """
        Computes topological winding number of node phase trajectories around identity axis.
        """
        # Dot product with previous quaternion state
        dot_prod = np.clip(np.sum(self.q * self.prev_q, axis=-1), -1.0, 1.0)
        delta_angles = np.arccos(dot_prod)
        self.accumulated_phase_winding += float(np.sum(delta_angles))
        return self.accumulated_phase_winding / (2.0 * np.pi)

    def step(
        self,
        q_bound: np.ndarray,
        dt: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Executes one time step of the Triadic Integration Loop.

        Parameters:
            q_bound: Boundary sensory input unit quaternion [4] or [num_nodes, 4].
            dt: Continuous integration time step.

        Returns:
            Telemetry dict containing state variables, valence, action, and diagnostic metrics.
        """
        if q_bound.ndim == 1:
            q_bound = np.tile(q_bound, (self.num_nodes, 1))
        q_bound = normalize_quaternion(q_bound)

        # Step 1: Calculate Quaternion Phase Error and Phase Tension
        q_inv = quaternion_inverse(self.q)
        e = quaternion_multiply(q_bound, q_inv)  # Quaternion rotation error e_i
        e_vec = e[:, 1:4]  # Extract vector component [e_x, e_y, e_z]
        T_phase = float(0.5 * np.mean(np.sum(e_vec ** 2, axis=-1)))

        # Step 2: Active Inference Action Spillover
        # Action is triggered if T_phase > theta_action
        if T_phase > self.theta_action:
            mean_e_vec = np.mean(e_vec, axis=0)  # [3]
            raw_action = mean_e_vec @ self.W_action  # [action_dim]
            self.a = np.tanh(raw_action)  # Action spillover signal
        else:
            self.a = np.zeros(self.action_dim)

        # Step 3: Extrinsic Environment Perturbation & Sensory Feedback Impact on Homeostasis
        # Action a perturbs environment x_env
        self.x_env += self.a * dt
        # Environment and action affect internal homeostatic state h
        # Action dissipation impacts entropy rate and energy consumption in h
        env_impact = np.sum(self.a ** 2) * 0.1
        self.h[0] += (-0.2 * (self.h[0] - self.h_star[0]) + env_impact) * dt
        for k in range(1, self.homeo_dim):
            self.h[k] += (-0.1 * (self.h[k] - self.h_star[k])) * dt

        # Step 4: Compute Homeostatic Strain and Thermodynamic Valence
        current_D_H = self.compute_homeostatic_strain(self.h)
        self.current_valence = self.compute_valence(current_D_H, dt)
        self.prev_D_H = current_D_H

        # Step 5: Thermodynamic Control Parameters
        gamma_val = self.get_dynamic_relaxation(self.current_valence)
        T_eff = self.get_effective_thermal_agitation(self.current_valence)

        # Step 6: Valence-Coupled Quaternion Langevin Dynamics
        attractor_force = self.compute_attractor_force()
        # Torque vector towards boundary signal + attractor force
        restoring_torque = e_vec + attractor_force[:, 1:4]

        # Thermal noise \eta_i(t)
        thermal_noise = np.random.randn(self.num_nodes, 3) * np.sqrt(2.0 * T_eff * dt)

        # Compute rate of change \dot{q}
        dot_q_vec = gamma_val * restoring_torque + thermal_noise
        dot_q = np.column_stack([np.zeros(self.num_nodes), dot_q_vec])

        # Integrate and normalize quaternions
        self.prev_q = self.q.copy()
        self.q = normalize_quaternion(self.q + dot_q * dt)

        # Step 7: Observability Metrics and Diagnostics
        diagnostics = self.compute_diagnostics(dot_q)
        winding_number = self.compute_winding_number()

        return {
            "T_phase": T_phase,
            "action": self.a.copy(),
            "homeostatic_strain": current_D_H,
            "valence": self.current_valence,
            "relaxation_gamma": gamma_val,
            "thermal_T_eff": T_eff,
            "diagnostics": diagnostics,
            "winding_number": winding_number,
            "homeo_state": self.h.copy(),
            "env_state": self.x_env.copy(),
        }
