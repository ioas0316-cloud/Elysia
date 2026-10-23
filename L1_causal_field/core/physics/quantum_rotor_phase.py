r"""
core/physics/quantum_rotor_phase.py

5D Clifford Rotor & Quantum Phase Transition Module
===================================================
Integrates C\l(5,0) Geometric Algebra (Lie Algebra so(5)) 10-bivector rotations
with Landau-Ginzburg potential bifurcation and WKB quantum phase tunneling dynamics.

Math Foundations:
-----------------
1. Rotor Kinematics: dR/dt = -0.5 * \Omega(t) * R
2. Bivector Generators: \Omega \in \mathfrak{so}(5), 10 independent planes
3. LG Potential: V(\theta, \lambda) = -0.5*(\lambda - \lambda_c)*\theta^2 + 0.25*\beta*\theta^4
4. Quantum Tunneling: \Gamma_{\text{tunnel}} = \hbar * exp( -\int \sqrt{2m(V(\theta)-E)} d\theta )
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from scipy.linalg import expm


class Clifford5DRotorEngine:
    """Clifford Algebra C\\l(5,0) / Lie Algebra \\mathfrak{so}(5) 10-Bivector Rotor Engine"""

    def __init__(self, dim: int = 5):
        self.dim = dim
        self.bivector_indices: List[Tuple[int, int]] = [
            (i, j) for i in range(self.dim) for j in range(i + 1, self.dim)
        ]
        self.num_bivectors = len(self.bivector_indices)  # \binom{5}{2} = 10

    def get_bivector_generator(self, idx: int) -> np.ndarray:
        """10개의 바이벡터 기저 중 idx번째 \\mathfrak{so}(5) 반대칭 생성자 행렬 (J_ij) 반환"""
        if idx < 0 or idx >= self.num_bivectors:
            raise ValueError(f"Bivector index out of bounds: {idx}")
        i, j = self.bivector_indices[idx]
        J = np.zeros((self.dim, self.dim), dtype=float)
        J[i, j] = -1.0
        J[j, i] = 1.0
        return J

    def build_bivector_omega(self, theta_vector: np.ndarray) -> np.ndarray:
        """10차원 회전각/각속도 벡터 theta_vector [10]로부터 \\mathfrak{so}(5) 바이벡터 행렬 \\Omega 생성"""
        theta_vec = np.asarray(theta_vector, dtype=float)
        if len(theta_vec) != self.num_bivectors:
            raise ValueError(f"Expected {self.num_bivectors} bivector angles, got {len(theta_vec)}")

        Omega = np.zeros((self.dim, self.dim), dtype=float)
        for k, angle in enumerate(theta_vec):
            if abs(angle) > 1e-12:
                Omega += angle * self.get_bivector_generator(k)
        return Omega

    def exponential_map(self, Omega: np.ndarray) -> np.ndarray:
        """지수 사상(Exponential Map)을 통한 Spin(5) / SO(5) 로터 변환 행렬 R = exp(\\Omega) 계산"""
        return expm(Omega)

    def sandwich_transform(self, v: np.ndarray, R: np.ndarray) -> np.ndarray:
        """샌드위치 연산 v' = R * v * R^\\dagger (SO(5) 회전 변환)"""
        v_vec = np.asarray(v, dtype=float)
        return np.dot(R, v_vec)


@dataclass
class PhaseTransitionState:
    """양자 위상 도약 및 상태 이완 결과 데이터 구조"""
    step: int
    control_lambda: float
    is_bifurcated: bool
    phase_tunneling_event: bool
    tunneling_probability: float
    current_bivector_angles: np.ndarray
    state_vector: np.ndarray
    rotor_matrix: np.ndarray


class LandauGinzburgPotentialEngine:
    """비선형 란다우-긴즈부르크 포텐셜 및 WKB 양자 위상 터널링 연산자"""

    def __init__(
        self,
        lambda_c: float = 1.0,
        beta: float = 0.5,
        coupling_kappa: float = 0.1,
        hbar: float = 1.0,
        mass: float = 1.0
    ):
        self.lambda_c = lambda_c  # 임계 제어 파라미터
        self.beta = beta          # 4차 포텐셜 계수
        self.kappa = coupling_kappa
        self.hbar = hbar
        self.mass = mass

    def evaluate_potential(self, theta: float, control_lambda: float) -> float:
        """V(\\theta, \\lambda) = -0.5 * (\\lambda - \\lambda_c) * \\theta^2 + 0.25 * \\beta * \\theta^4"""
        delta_lambda = control_lambda - self.lambda_c
        return -0.5 * delta_lambda * (theta**2) + 0.25 * self.beta * (theta**4)

    def potential_gradient(self, theta: float, control_lambda: float) -> float:
        """dV/d\\theta = -(\\lambda - \\lambda_c) * \\theta + \\beta * \\theta^3"""
        delta_lambda = control_lambda - self.lambda_c
        return -delta_lambda * theta + self.beta * (theta**3)

    def get_equilibrium_angles(self, control_lambda: float) -> Tuple[float, float]:
        """분기 상태에서의 안정 포텐셜 우물 위치 \\theta^* 계산"""
        if control_lambda <= self.lambda_c:
            return 0.0, 0.0

        theta_star = np.sqrt((control_lambda - self.lambda_c) / self.beta)
        return -theta_star, theta_star

    def compute_wkb_tunneling_probability(self, control_lambda: float, energy: float = 0.0) -> float:
        """WKB 수치 적분을 통한 직교 평면 간 양자 위상 터널링 확률 \\Gamma_{\\text{tunnel}} 산출"""
        if control_lambda <= self.lambda_c:
            return 0.0

        theta_left, theta_right = self.get_equilibrium_angles(control_lambda)
        if abs(theta_left - theta_right) < 1e-8:
            return 0.0

        num_samples = 100
        theta_grid = np.linspace(theta_left, theta_right, num_samples)
        d_theta = theta_grid[1] - theta_grid[0]

        integrand_sum = 0.0
        for th in theta_grid:
            v_val = self.evaluate_potential(th, control_lambda)
            v_bottom = self.evaluate_potential(theta_left, control_lambda)
            barrier_height = max(0.0, v_val - v_bottom - energy)
            integrand_sum += np.sqrt(2.0 * self.mass * barrier_height) * d_theta

        action_S = integrand_sum
        gamma = np.exp(-action_S / self.hbar)
        return float(np.clip(gamma, 0.0, 1.0))


class QuantumRotorPhaseIntegrator:
    """5D 로터 운동학 및 양자 위상 도약 통합 적분기"""

    def __init__(
        self,
        initial_state_5d: Optional[np.ndarray] = None,
        lambda_c: float = 1.0,
        tunneling_threshold: float = 0.05
    ):
        self.rotor_engine = Clifford5DRotorEngine(dim=5)
        self.lg_engine = LandauGinzburgPotentialEngine(lambda_c=lambda_c)
        self.tunneling_threshold = tunneling_threshold

        if initial_state_5d is None:
            raw_state = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        else:
            raw_state = np.asarray(initial_state_5d, dtype=float)

        self.state_5d = raw_state / (np.linalg.norm(raw_state) + 1e-12)

        self.bivector_angles = np.zeros(10, dtype=float)
        self.current_rotor = np.eye(5, dtype=float)
        self.step_counter = 0

    def step(
        self,
        control_lambda: float,
        external_torque_10d: Optional[np.ndarray] = None,
        dt: float = 0.05
    ) -> PhaseTransitionState:
        """단일 타임스텝 연산: 연속 이완 및 자율 양자 위상 도약 수행"""
        self.step_counter += 1

        if external_torque_10d is None:
            external_torque = np.zeros(10, dtype=float)
        else:
            external_torque = np.asarray(external_torque_10d, dtype=float)

        is_bifurcated = control_lambda > self.lg_engine.lambda_c
        tunneling_event = False
        gamma_prob = 0.0

        grad_e12 = self.lg_engine.potential_gradient(self.bivector_angles[0], control_lambda)
        omega_vec = np.zeros(10, dtype=float)
        omega_vec[0] = -grad_e12 + external_torque[0]
        omega_vec[1:] = external_torque[1:]

        if is_bifurcated:
            gamma_prob = self.lg_engine.compute_wkb_tunneling_probability(control_lambda)

            if gamma_prob > self.tunneling_threshold and abs(self.bivector_angles[8]) < 0.1:
                tunneling_event = True
                _, theta_star = self.lg_engine.get_equilibrium_angles(control_lambda)

                self.bivector_angles[8] += theta_star
                omega_vec[8] += gamma_prob * 2.0

        self.bivector_angles += omega_vec * dt
        Omega_matrix = self.rotor_engine.build_bivector_omega(omega_vec * dt)

        dR = self.rotor_engine.exponential_map(-0.5 * Omega_matrix)
        self.current_rotor = np.dot(dR, self.current_rotor)

        self.state_5d = self.rotor_engine.sandwich_transform(self.state_5d, self.current_rotor)
        self.state_5d /= (np.linalg.norm(self.state_5d) + 1e-12)

        return PhaseTransitionState(
            step=self.step_counter,
            control_lambda=control_lambda,
            is_bifurcated=is_bifurcated,
            phase_tunneling_event=tunneling_event,
            tunneling_probability=gamma_prob,
            current_bivector_angles=self.bivector_angles.copy(),
            state_vector=self.state_5d.copy(),
            rotor_matrix=self.current_rotor.copy()
        )
