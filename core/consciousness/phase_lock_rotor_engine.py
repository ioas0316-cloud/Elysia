"""
Elysia Phase Lock Engine, 4-Layer Volitional Architecture & Predictive Resonance
==============================================================================
Implements:
1. ElysiaRotorEngine: 3D Vector Field Phase Friction & Quaternion Rotor Recalibration.
2. TensorPhaseLockEngine: N-Dimensional Riemannian Tensor Field with Skew-Symmetric Torque so(N).
3. VolitionalGatedArchitecture: 4-Layer Phase-Gated Architecture (L0 Immutable Substrate, L1 Passive Resonance Gate, L2 Volitional Routing Gate, L3 Local Phase Engine).
4. PredictiveResonanceGatedEngine: 4-Stage Event-Driven Architecture (Passive Resonance, Dissonance Gate, Flow Loss L_flow, Phase Transition Internalization).
"""

import math
from typing import Dict, Any, Tuple
import numpy as np


class ElysiaRotorEngine:
    """
    3D Vector Field Phase Lock Rotor Engine.
    Computes local phase friction torque (V_in x V_out) and updates internal
    Spin(3) quaternion rotor R via Lie exponential map.
    """
    def __init__(self, spatial_grid_shape: Tuple[int, int, int] = (16, 16, 3), gamma: float = 2.5, beta: float = 0.15):
        self.grid_shape = spatial_grid_shape
        self.gamma = gamma
        self.beta = beta
        # Rotor R (Quaternion: [w, x, y, z])
        self.R = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.omega_int = np.zeros(3, dtype=np.float64)  # Internal angular velocity

    def _rotor_to_matrix(self, q: np.ndarray) -> np.ndarray:
        """Converts unit quaternion/rotor q into 3x3 rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1.0 - 2.0*(y**2 + z**2), 2.0*(x*y - z*w),       2.0*(x*z + y*w)],
            [2.0*(x*y + z*w),       1.0 - 2.0*(x**2 + z**2), 2.0*(y*z - x*w)],
            [2.0*(x*z - y*w),       2.0*(y*z + x*w),       1.0 - 2.0*(x**2 + y**2)]
        ], dtype=np.float64)

    def step(self, V_base: np.ndarray, V_out: np.ndarray, dt: float = 0.01) -> Tuple[float, np.ndarray]:
        """
        V_base: (N, 3) Base vector field.
        V_out:  (N, 3) External reality vector field.
        """
        R_mat = self._rotor_to_matrix(self.R)
        V_in = np.dot(V_base, R_mat.T)

        # Local phase friction moment density: V_in x V_out
        torque_density = np.cross(V_in, V_out)
        tau_raw = np.mean(torque_density, axis=0)

        # Recalibration torque with damping
        tau_recal = self.gamma * tau_raw - self.beta * self.omega_int
        self.omega_int += tau_recal * dt

        # Lie Algebra (Exponential Map) Rotor update
        omega_mag = np.linalg.norm(self.omega_int)
        if omega_mag > 1e-8:
            axis = self.omega_int / omega_mag
            angle = omega_mag * dt * 0.5

            dR = np.array([np.cos(angle), *(np.sin(angle) * axis)], dtype=np.float64)

            # Quaternion Multiplication: R_new = dR * R_old
            w1, x1, y1, z1 = dR
            w2, x2, y2, z2 = self.R
            self.R = np.array([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2
            ], dtype=np.float64)
            self.R /= (np.linalg.norm(self.R) + 1e-12)

        # Phase lock error (geodesic distance / angle discrepancy)
        norm_in = np.linalg.norm(V_in, axis=-1) + 1e-8
        norm_out = np.linalg.norm(V_out, axis=-1) + 1e-8
        cos_sim = np.sum(V_in * V_out, axis=-1) / (norm_in * norm_out)
        phase_error = float(np.mean(np.arccos(np.clip(cos_sim, -1.0, 1.0))))

        return phase_error, tau_recal


class TensorPhaseLockEngine:
    """
    N-Dimensional Riemannian Tensor Field Phase Lock Engine.
    Uses skew-symmetric Lie algebra so(N) torque tensors and matrix exponential
    rotors SO(N) to align internal hypothesis tensor fields with external reality.
    """
    def __init__(self, dim_feature: int = 16, gamma: float = 2.5, beta: float = 0.15):
        self.N = dim_feature
        self.gamma = gamma
        self.beta = beta
        # so(N) Lie algebra state: skew-symmetric matrix (N, N)
        self.omega_raw = np.zeros((self.N, self.N), dtype=np.float64)
        # Interaction curvature weight tensor
        np.random.seed(42)
        self.W = np.random.randn(self.N, self.N) * 0.01

    def get_skew_symmetric_omega(self) -> np.ndarray:
        """Guarantees skew-symmetry: Omega = -Omega.T"""
        return 0.5 * (self.omega_raw - self.omega_raw.T)

    def _matrix_exp(self, M: np.ndarray) -> np.ndarray:
        """Computes matrix exponential using Padé approximation via scaling and squaring."""
        norm = np.linalg.norm(M, ord=1)
        if norm == 0:
            return np.eye(self.N, dtype=np.float64)

        s = max(0, int(np.ceil(np.log2(norm))))
        A = M / (2**s)

        # Padé approximant of order [6/6]
        c = [1.0, 0.5, 0.1, 1.0/84.0, 1.0/1680.0, 1.0/30240.0, 1.0/665280.0]
        A2 = np.dot(A, A)
        A4 = np.dot(A2, A2)
        A6 = np.dot(A4, A2)

        U = np.dot(A, (c[1]*np.eye(self.N) + c[3]*A2 + c[5]*A4 + c[6]*A6))
        V = c[0]*np.eye(self.N) + c[2]*A2 + c[4]*A4

        N_mat = V + U
        D_mat = V - U

        res = np.linalg.solve(D_mat, N_mat)
        for _ in range(s):
            res = np.dot(res, res)
        return res

    def step(self, T_base: np.ndarray, T_out: np.ndarray, dt: float = 0.01) -> Tuple[float, np.ndarray]:
        """
        T_base: (Spatial_Grid, N) Internal base tensor field.
        T_out:  (Spatial_Grid, N) External reality tensor field.
        """
        Omega = self.get_skew_symmetric_omega()
        R_mat = self._matrix_exp(Omega)

        # Internal state transformation
        T_in = np.matmul(T_base, R_mat.T)

        # Cross-modal projection and non-linear outer product interference field
        T_transformed = np.matmul(T_in, self.W)

        # Outer product: (Spatial, N, 1) x (Spatial, 1, N) -> (Spatial, N, N)
        outer_prod = np.einsum('si,sj->sij', T_transformed, T_out)
        interference_field = np.tanh(outer_prod)

        # Skew-symmetric torque density Q = F - F.T
        Q_density = interference_field - np.transpose(interference_field, (0, 2, 1))
        Q_recal = np.mean(Q_density, axis=0)  # Spatial contraction -> (N, N)

        # Update Lie algebra state dOmega = gamma * Q_recal - beta * Omega
        dOmega = self.gamma * Q_recal - self.beta * Omega
        self.omega_raw += dOmega * dt

        # Phase error (Frobenius distance)
        phase_error = float(np.linalg.norm(T_in - T_out, ord='fro') / T_in.size)

        return phase_error, R_mat


class VolitionalGatedArchitecture:
    """
    4-Layer Phase-Gated Architecture (Coexistence of Environment and Volitional Agency).
    Layer 0: Immutable Substrate (Absolute environmental metrics, C_max capacity, non-modifiable).
    Layer 1: Passive Resonance Gate (Automated dissonance detection threshold tau_min).
    Layer 2: Volitional Routing Gate (Subject's intentional attention vector & orientation tensor).
    Layer 3: Local Phase Engine (On-demand execution bounded by environmental constraint C_max).
    """
    def __init__(self, topology_dim: int = 16, c_max: float = 0.35, tau_min: float = 0.05):
        # Layer 0: Immutable Substrate (Hardcoded Environmental Constraints)
        self.topology_dim = topology_dim
        self.c_max = float(c_max)
        self.tau_min = float(tau_min)

        # Layer 2: Volitional Agency Parameters (Free internal trajectory selection)
        np.random.seed(42)
        self.volitional_attention = np.random.randn(topology_dim)
        self.orientation_tensor = np.random.randn(topology_dim, topology_dim) * 0.1

    def forward(self, x_input: np.ndarray, internal_state: np.ndarray) -> Dict[str, Any]:
        # 1. Layer 0: Absolute environmental metric prediction error
        prediction_error = float(np.linalg.norm(x_input - internal_state))

        # 2. Layer 1: Passive Gate Check (Automated trigger)
        if prediction_error < self.tau_min:
            # Passive resonance state: compute cost = 0, state passes through
            return {
                "new_internal_state": internal_state.copy(),
                "compute_cost": 0.0,
                "prediction_error": prediction_error,
                "passive_gated": True,
                "volitional_active": False
            }

        # 3. Layer 2: Volitional Gate Active
        # Softmax over attention vector weighted by prediction error
        exp_attn = np.exp(self.volitional_attention * prediction_error - np.max(self.volitional_attention))
        selected_focus = exp_attn / (np.sum(exp_attn) + 1e-12)

        directional_intent = np.matmul(self.orientation_tensor, selected_focus)

        # 4. Layer 3: Local Phase Engine Execution (Clamped by environmental C_max)
        bounded_compute = np.clip(directional_intent, -self.c_max, self.c_max)
        new_internal_state = internal_state + bounded_compute

        return {
            "new_internal_state": new_internal_state,
            "compute_cost": float(np.linalg.norm(bounded_compute)),
            "prediction_error": prediction_error,
            "passive_gated": False,
            "volitional_active": True,
            "selected_focus": selected_focus,
            "directional_intent": directional_intent
        }


class PredictiveResonanceGatedEngine:
    """
    Predictive Resonance Architecture Engine with Dissonance Gate & Flow Loss.
    4-Stage Mechanics:
    1. Passive Structural Resonance (O(1) execution when error <= tau_min)
    2. Dissonance & Residual Gate (Triggers active computation when error >= tau_min)
    3. Event-Driven Active Computation (Gated by alpha(t) and Flow Loss L_flow)
    4. Phase Transition & Internalization (Structural adaptation/assimilation)
    """
    def __init__(
        self,
        dim_feature: int = 16,
        tau_min: float = 0.05,
        c_max: float = 0.35,
        gamma_barrier: float = 10.0,
        lambda_cost: float = 1.0,
        lambda_smooth: float = 0.5
    ):
        self.tensor_engine = TensorPhaseLockEngine(dim_feature=dim_feature)
        self.volitional_arch = VolitionalGatedArchitecture(topology_dim=dim_feature, c_max=c_max, tau_min=tau_min)
        self.tau_min = tau_min
        self.c_max = c_max
        self.gamma_barrier = gamma_barrier
        self.lambda_cost = lambda_cost
        self.lambda_smooth = lambda_smooth

        self.last_error = 0.0
        self.plasticity_rate = 0.01

    def compute_flow_loss(self, E: float, dE_dt: float, alpha: float) -> Dict[str, Any]:
        """
        Computes L_flow loss components:
        - Boredom penalty: max(0, tau_min - E)^2
        - Panic barrier: gamma_barrier * max(0, E - c_max)^2
        - Dynamic compute cost: lambda_cost * alpha * (E - tau_min)
        - Phase volatility penalty: lambda_smooth * |dE/dt|
        """
        boredom_penalty = float(max(0.0, self.tau_min - E) ** 2)
        panic_barrier = float(self.gamma_barrier * (max(0.0, E - self.c_max) ** 2))
        compute_cost = float(self.lambda_cost * alpha * max(0.0, E - self.tau_min))
        volatility_penalty = float(self.lambda_smooth * abs(dE_dt))

        l_flow = boredom_penalty + panic_barrier + compute_cost + volatility_penalty

        zone = "FLOW"
        if E <= self.tau_min:
            zone = "BOREDOM"
        elif E > self.c_max:
            zone = "PANIC"

        return {
            "l_flow": float(l_flow),
            "boredom_penalty": boredom_penalty,
            "panic_barrier": panic_barrier,
            "compute_cost": compute_cost,
            "volatility_penalty": volatility_penalty,
            "zone": zone
        }

    def process(
        self,
        T_base: np.ndarray,
        T_out: np.ndarray,
        dt: float = 0.01
    ) -> Dict[str, Any]:
        """
        Processes input tensor field through 4-stage Predictive Resonance loop.
        """
        # Step 1: Passive Phase Lock Evaluation
        phase_error, R_mat = self.tensor_engine.step(T_base, T_out, dt=dt)
        dE_dt = (phase_error - self.last_error) / dt if dt > 0 else 0.0
        self.last_error = phase_error

        # Step 2: Dissonance & Residual Gate
        dissonance_triggered = phase_error > self.tau_min

        # Compute gating signal alpha(t)
        if phase_error <= self.tau_min:
            alpha = 0.0  # Passive resonance O(1)
        else:
            # Sigmoid activation above tau_min
            k = 10.0
            alpha = float(1.0 / (1.0 + math.exp(-k * (phase_error - self.tau_min))))

        # Step 3: Compute Flow Loss metrics
        flow_metrics = self.compute_flow_loss(phase_error, dE_dt, alpha)

        # Step 4: Event-Driven Active Recalibration & Volitional Gating
        internalized = False
        updated_T_base = T_base.copy()

        if dissonance_triggered and alpha > 0.0:
            # Pass through 4-layer volitional architecture
            volitional_res = self.volitional_arch.forward(T_out, T_base)

            # On-demand Active Recalibration: Adjust base tensor field toward T_out using plasticity
            self.plasticity_rate = min(0.5, self.plasticity_rate + 0.01 * alpha)
            T_in = np.matmul(T_base, R_mat.T)
            delta_T = T_out - T_in
            updated_T_base = volitional_res["new_internal_state"] + alpha * self.plasticity_rate * delta_T
            internalized = True
        else:
            # Decay plasticity rate in passive flow/boredom state
            self.plasticity_rate = max(0.005, self.plasticity_rate * 0.95)

        return {
            "phase_error": phase_error,
            "dE_dt": dE_dt,
            "dissonance_triggered": dissonance_triggered,
            "compute_gating_alpha": alpha,
            "flow_metrics": flow_metrics,
            "internalized": internalized,
            "R_mat": R_mat,
            "updated_T_base": updated_T_base,
            "plasticity_rate": self.plasticity_rate
        }
