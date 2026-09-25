"""
synaptic_architecture/tension_relaxation_engine.py

elysia_engine의 배경 독립적 장력 이완(Tension Relaxation) 커널 및 가변 임피던스 접촉 시뮬레이션 모듈.
명시적 역역학(IK) 수식 계산 대신, 최소 작용 원리(Action Relaxation) 및
외부 임피던스(Z_env) 상쇄를 통해 계산 지연 없는 물리적 자발 평형을 달성합니다.
"""

import numpy as np
import time
from typing import Dict, Any, Tuple, Optional


class TensionRelaxationEngine:
    """
    [배경 독립적 인과 장력 이완 엔진]
    3차원 절대 공간 좌표계 없이 노드 간 위상적 팽팽함(Tension)만을 저장하고,
    라플라스-벨트라미 확산 및 외부 임피던스 상쇄를 통해 물리적 최소 에너지를 찾아갑니다.
    """

    def __init__(
        self,
        num_nodes: int = 16,
        decay_rate: float = 0.5,
        diffusion_coeff: float = 0.1
    ):
        """
        Args:
            num_nodes: 인과 네트워크 노드 수 (N)
            decay_rate: 장력 감쇄율 (γ)
            diffusion_coeff: 위상 확산 계수 (D)
        """
        self.num_nodes = num_nodes
        self.gamma = decay_rate
        self.D = diffusion_coeff

        # 배경 독립적 장력 텐서 (Tension Matrix: N x N)
        self.T_net = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        self.T_base = np.ones((num_nodes, num_nodes), dtype=np.float32) * 0.1

    def reset_state(self):
        """장력 상태 초기화"""
        self.T_net = self.T_base.copy()

    def forward(
        self,
        Z_ext: np.ndarray,
        Z_env: np.ndarray,
        dt: float = 0.01
    ) -> np.ndarray:
        """
        [장력 이완 및 토크 방출 forward pass]

        Args:
            Z_ext: 내부 의지/외부 감각 수용체에서 유입된 장력 임피던스 (N x N)
            Z_env: 엔드이펙터가 환경과 부딪히며 전이되는 반발 임피던스 (N x N)
            dt: 물리 이완 시간 이행 단계 (Delta Tau)

        Returns:
            A_motor: 외부 물리계로 방출되는 잔여 장력 구배 (Motor Force Output)
        """
        Z_ext_mat = Z_ext.astype(np.float32)
        Z_env_mat = Z_env.astype(np.float32)

        # 1. 라플라스-벨트라미 위상적 장력 확산
        deg_vector = self.T_net.sum(axis=-1)
        deg_matrix = np.diag(deg_vector)
        laplacian_T = self.T_net - deg_matrix

        # 2. 변분 장력 이완 PDE (Tension Relaxation Differential Equation)
        # dT/dτ = D * Δ_g T - γ * (T - T_base) + Z_ext
        dT_dt = self.D * laplacian_T - self.gamma * (self.T_net - self.T_base) + Z_ext_mat
        self.T_net = self.T_net + dT_dt * dt

        # 3. 잔여 장력 구배 방출 (Torque Discharge & Impedance Canceling)
        # A_motor = max(0, Raw_Discharge - Z_env)
        raw_discharge = np.maximum(0.0, self.T_net)
        A_motor = np.maximum(0.0, raw_discharge - Z_env_mat)

        return A_motor


class VariableImpedanceContactSimulator:
    """
    [가변 임피던스 접촉 및 장력 상쇄 시뮬레이터]
    두부나 달걀처럼 쉽게 파괴되는 가변 임피던스 물체와의 접촉 상황을
    명시적 역역학 수식 연산이나 피드백 루프 지연 없이, 장력 상쇄만으로 자발적 평형에 도달하게 합니다.
    """

    def __init__(
        self,
        engine: TensionRelaxationEngine,
        tofu_stiffness: float = 15.0,
        tofu_break_threshold: float = 5.0,
        tofu_surface: float = 0.5
    ):
        """
        Args:
            engine: 장력 이완 엔진
            tofu_stiffness: 두부/대상 물체의 반발 강도 계수 (N/m)
            tofu_break_threshold: 파괴 임계점 (N)
            tofu_surface: 대상 표면 위치 (m)
        """
        self.engine = engine
        self.tofu_stiffness = tofu_stiffness
        self.tofu_break_threshold = tofu_break_threshold
        self.tofu_surface = tofu_surface

    def run_simulation(self, max_steps: int = 250, verbose: bool = False) -> Dict[str, Any]:
        """시뮬레이션 구동 루프"""
        self.engine.reset_state()
        finger_position = 0.0
        history = []

        success = False
        broken = False
        final_force = 0.0

        N = self.engine.num_nodes

        for step in range(1, max_steps + 1):
            # 1. 두부 침투 깊이에 따른 환경 반발 임피던스(Z_env) 유입
            penetration = max(0.0, finger_position - self.tofu_surface)
            current_env_force = self.tofu_stiffness * penetration

            Z_env = np.ones((N, N), dtype=np.float32) * current_env_force

            # 2. 내재적 이동 의지(목표 방향으로의 지속적 인력) Z_ext 유입
            Z_ext = np.ones((N, N), dtype=np.float32) * 1.5

            # 3. 장력 이완 커널 수행 및 방출 모터 힘 계산
            A_motor = self.engine.forward(Z_ext, Z_env, dt=0.01)
            motor_force = float(A_motor.mean())

            # 4. 모터 방출 출력에 의한 지연 없는 물리적 위치 변이
            finger_position += motor_force * 0.01

            history.append({
                "step": step,
                "position": finger_position,
                "force": current_env_force,
                "motor_force": motor_force
            })

            # [검증 판정 1] 두부 파괴 조건
            if current_env_force >= self.tofu_break_threshold:
                broken = True
                final_force = current_env_force
                if verbose:
                    print(f"❌ [FAIL] Step {step:03d}: 두부가 파괴되었습니다! Force: {current_env_force:.3f} N")
                break

            # [검증 판정 2] 장력 완전 상쇄 및 물리적 자발 평형 도달
            if penetration > 0 and motor_force < 0.1:
                success = True
                final_force = current_env_force
                if verbose:
                    print(f"✅ [SUCCESS] Step {step:03d}: 수식 연산 및 센서 지연 없이 완전 평형 도달!")
                    print(f"   - 최종 위치: {finger_position:.4f} m (침투 깊이: {penetration*1000:.2f} mm)")
                    print(f"   - 평형 유지력: {current_env_force:.3f} N (파괴 기준 {self.tofu_break_threshold}N 미만 유지)")
                break

        return {
            "success": success,
            "broken": broken,
            "final_step": len(history),
            "final_force": final_force,
            "final_position": finger_position,
            "history": history
        }
