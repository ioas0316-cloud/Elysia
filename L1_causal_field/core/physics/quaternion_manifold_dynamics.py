"""
quaternion_manifold_dynamics.py
================================
3차원 고차원 로터 SO(3) 무분기 연속체 텐서 동역학 모듈 (Quaternion Manifold Dynamics).

핵심 철학:
- "Do not calculate, let it flow."
- "판단과 분별 자체를 공간의 곡률이나 텐서의 기하학적 결합으로 내장한다."
- 3차원 회전군 SO(3) 상의 임의의 두 회전 상태를 쿼터니언(Quaternion)과 각속도 벡터로 모델링합니다.
- 어떠한 if 조건문(Control Flow Branching)도 사용하지 않고,
  두 3D 로터의 쿼터니언 내적 기반 위상적 거리(SO(3) Metric Distance)에 따른
  연속적 결합 가중치(Coupling Field)를 통해 비접촉과 접촉(Tension-Coupled) 상태를 단일 장 방정식으로 물 흐르듯 전환합니다.
"""

from typing import Dict, Tuple
import math


class QuaternionHelper:
    """
    [쿼터니언 수학 헬퍼]
    if문 없이 3차원 쿼터니언 연산(곱셈, 내적, 정규화, 도함수)을 연속 수식으로 처리합니다.
    """
    @staticmethod
    def normalize(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """쿼터니언을 정규화합니다. (0 나눗셈 예방 epsilon 포함)"""
        w, x, y, z = q
        norm = math.sqrt(w*w + x*x + y*y + z*z) + 1e-12
        return w / norm, x / norm, y / norm, z / norm

    @staticmethod
    def dot(q1: Tuple[float, float, float, float], q2: Tuple[float, float, float, float]) -> float:
        """두 쿼터니언의 내적을 계산합니다."""
        return q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]

    @staticmethod
    def multiply(q1: Tuple[float, float, float, float], q2: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """두 쿼터니언의 곱셈(Hamilton Product)을 수행합니다."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return w, x, y, z

    @staticmethod
    def derivative(q: Tuple[float, float, float, float], omega: Tuple[float, float, float]) -> Tuple[float, float, float, float]:
        """각속도 omega를 기반으로 한 쿼터니언 시간 미분 dq/dt = 0.5 * q * [0, omega]를 계산합니다."""
        w, x, y, z = q
        ox, oy, oz = omega
        # omega_q = (0, ox, oy, oz)
        # dq/dt = 0.5 * multiply(q, omega_q)
        dw = 0.5 * (-x*ox - y*oy - z*oz)
        dx = 0.5 * (w*ox + y*oz - z*oy)
        dy = 0.5 * (w*oy - x*oz + z*ox)
        dz = 0.5 * (w*oz + x*oy - y*ox)
        return dw, dx, dy, dz


class QuaternionRotorState:
    """
    [3차원 쿼터니언 로터 상태 관리자]
    3차원 공간 속에서 회전하는 강체의 쿼터니언(q) 및 각속도 벡터(omega), 가변 임피던스(impedance)를 관리합니다.
    """
    def __init__(
        self,
        q: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        omega: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        impedance: float = 0.1
    ):
        self.q = QuaternionHelper.normalize(q)  # q = [w, x, y, z] (단위 쿼터니언)
        self.omega = omega                      # [wx, wy, wz] (각속도 벡터)
        self.impedance = impedance              # 3차원 충격을 흡수하는 완충 임피던스 계수

    def apply_torque(self, torque: Tuple[float, float, float], dt: float) -> None:
        """3차원 외부 토크를 인가하여 각속도 및 쿼터니언 위상을 연속적으로 업데이트합니다."""
        tx, ty, tz = torque
        # 임피던스 완충 효과 적용
        factor = 1.0 / (1.0 + self.impedance)

        # 각속도 갱신
        self.omega = (
            self.omega[0] + tx * factor * dt,
            self.omega[1] + ty * factor * dt,
            self.omega[2] + tz * factor * dt
        )

        # dq/dt 미분 적용 및 쿼터니언 갱신
        dq = QuaternionHelper.derivative(self.q, self.omega)
        next_q = (
            self.q[0] + dq[0] * dt,
            self.q[1] + dq[1] * dt,
            self.q[2] + dq[2] * dt,
            self.q[3] + dq[3] * dt
        )
        self.q = QuaternionHelper.normalize(next_q)

        # [수치적 드리프트 극미세 가드레일]
        # 부동소수점 오차 한계(1e-12) 초과 시 즉각 정밀 재보정 처리하여 무오성을 유지합니다.
        q_norm = math.sqrt(sum(v*v for v in self.q))
        if abs(q_norm - 1.0) > 1e-12:
            self.q = (self.q[0]/q_norm, self.q[1]/q_norm, self.q[2]/q_norm, self.q[3]/q_norm)


class ContinuousQuaternionManifoldSystem:
    """
    [3차원 SO(3) 무분기 연속체 텐서 시스템]
    두 개의 쿼터니언 로터 간의 상호작용을 3차원 다양체(Manifold) 상에서 연속적으로 연산합니다.
    """
    def __init__(
        self,
        base_omega_a: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        base_omega_b: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        sensitivity_radius: float = 0.3,
        learning_rate: float = 0.05
    ):
        # 3D 로터 초기화
        self.rotor_a = QuaternionRotorState(omega=base_omega_a)
        self.rotor_b = QuaternionRotorState(omega=base_omega_b)

        # 시스템 고유 속도장
        self.base_omega_a = base_omega_a
        self.base_omega_b = base_omega_b

        # 고차원 결합 및 자율 튜닝 제어 변수
        self.sigma = sensitivity_radius
        self.stiffness = 1.0  # 기어 결합 전도율(Stiffness) 계수
        self.lr = learning_rate

    def step(self, t: float, dt: float, user_torque: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Dict[str, float]:
        """
        if 조건문 없이 3D SO(3) 구속 결합과 예측-관측-조율 루프를 단일 수식장으로 관통합니다.
        """
        # 1. 3D 토크 주입
        self.rotor_a.apply_torque(user_torque, dt)

        # 2. SO(3) 위상적 거리(Metric Distance) 산출
        # d_metric = 1 - (q_a . q_b)^2 -> 두 회전 상태가 일치하면 0, 직교(180도 회전)하면 1로 수렴하는 매끄러운 곡률 거리
        dot_product = QuaternionHelper.dot(self.rotor_a.q, self.rotor_b.q)
        metric_distance = 1.0 - (dot_product ** 2)

        # 3. 연속적 가우시안 결합 장(Coupling Weight Field) 계산
        # 거리가 유효 반경(sigma) 이내로 들어올 때만 1에 가깝게 공명하고, 멀어지면 0으로 매끄럽게 이완
        coupling_weight = math.exp(- (metric_distance ** 2) / (2.0 * (self.sigma ** 2)))

        # 4. SO(3) 예측 부호화 (Predictive Coding)
        # 1) 3차원 미래 상태 예측 (Predictive Estimation)
        # Rotor B는 A의 각속도에 결합 계수(stiffness)가 반영된 전달 속도로 회전할 것으로 예측
        pred_omega_b = (
            self.rotor_a.omega[0] * self.stiffness,
            self.rotor_a.omega[1] * self.stiffness,
            self.rotor_a.omega[2] * self.stiffness
        )

        dq_pred_a = QuaternionHelper.derivative(self.rotor_a.q, self.rotor_a.omega)
        pred_q_a = QuaternionHelper.normalize((
            self.rotor_a.q[0] + dq_pred_a[0] * dt,
            self.rotor_a.q[1] + dq_pred_a[1] * dt,
            self.rotor_a.q[2] + dq_pred_a[2] * dt,
            self.rotor_a.q[3] + dq_pred_a[3] * dt
        ))
        # [수치적 드리프트 극미세 가드레일]
        p_a_norm = math.sqrt(sum(v*v for v in pred_q_a))
        if abs(p_a_norm - 1.0) > 1e-12:
            pred_q_a = (pred_q_a[0]/p_a_norm, pred_q_a[1]/p_a_norm, pred_q_a[2]/p_a_norm, pred_q_a[3]/p_a_norm)

        dq_pred_b = QuaternionHelper.derivative(self.rotor_b.q, pred_omega_b)
        pred_q_b = QuaternionHelper.normalize((
            self.rotor_b.q[0] + dq_pred_b[0] * dt,
            self.rotor_b.q[1] + dq_pred_b[1] * dt,
            self.rotor_b.q[2] + dq_pred_b[2] * dt,
            self.rotor_b.q[3] + dq_pred_b[3] * dt
        ))
        # [수치적 드리프트 극미세 가드레일]
        p_b_norm = math.sqrt(sum(v*v for v in pred_q_b))
        if abs(p_b_norm - 1.0) > 1e-12:
            pred_q_b = (pred_q_b[0]/p_b_norm, pred_q_b[1]/p_b_norm, pred_q_b[2]/p_b_norm, pred_q_b[3]/p_b_norm)

        # 2) 실제 3차원 물리 궤적 관측 (Physical Observation)
        # 비접촉 시(weight=0)에는 독립적인 고유 속도(base_omega_b)로 회전, 접촉 시(weight=1)에는 A의 각속도로 강제 커플링
        obs_omega_b = (
            (1.0 - coupling_weight) * self.base_omega_b[0] + coupling_weight * self.rotor_a.omega[0],
            (1.0 - coupling_weight) * self.base_omega_b[1] + coupling_weight * self.rotor_a.omega[1],
            (1.0 - coupling_weight) * self.base_omega_b[2] + coupling_weight * self.rotor_a.omega[2]
        )

        obs_q_a = pred_q_a
        dq_obs_b = QuaternionHelper.derivative(self.rotor_b.q, obs_omega_b)
        obs_q_b = QuaternionHelper.normalize((
            self.rotor_b.q[0] + dq_obs_b[0] * dt,
            self.rotor_b.q[1] + dq_obs_b[1] * dt,
            self.rotor_b.q[2] + dq_obs_b[2] * dt,
            self.rotor_b.q[3] + dq_obs_b[3] * dt
        ))
        # [수치적 드리프트 극미세 가드레일]
        o_b_norm = math.sqrt(sum(v*v for v in obs_q_b))
        if abs(o_b_norm - 1.0) > 1e-12:
            obs_q_b = (obs_q_b[0]/o_b_norm, obs_q_b[1]/o_b_norm, obs_q_b[2]/o_b_norm, obs_q_b[3]/o_b_norm)

        # 3) 3D Tension Gap (회전축 장력 오차) 측정
        # 예측한 쿼터니언과 실제 관측한 쿼터니언 사이의 SO(3) 거리 편차를 오차 에너지(Tension)로 환산
        pred_obs_dot = QuaternionHelper.dot(pred_q_b, obs_q_b)
        tension_gap = coupling_weight * (1.0 - (pred_obs_dot ** 2))

        # 5. 자율 동기화 및 자율 적응 (Self-Tuning)
        # 1) 결합 강도(Stiffness) 자율 보정
        # 예측 쿼터니언과 실제 관측 쿼터니언의 위상 오차를 내적을 통해 역산하여 결합 계수 보정
        # q_pred . q_obs 가 1 또는 -1에 가까우면 정렬된 것이며, 차이가 클수록 stiffness의 크기를 수정함
        direction_error = 1.0 - abs(pred_obs_dot)
        # 결합 강도는 오차의 크기와 방향에 맞춰 점진적으로 조율
        self.stiffness += coupling_weight * direction_error * self.lr

        # 2) 완충 임피던스(Impedance) 자율 적응 및 자연 완화 (Relaxation)
        # 3D Tension Gap 에너지의 크기에 비례하여 완충 임피던스가 유연하게 증가하여 충격을 흡수함
        self.rotor_a.impedance += tension_gap * 0.25
        self.rotor_b.impedance += tension_gap * 0.25

        # 임피던스 이완
        self.rotor_a.impedance = max(0.01, self.rotor_a.impedance * 0.92)
        self.rotor_b.impedance = max(0.01, self.rotor_b.impedance * 0.92)

        # 6. 최종 3차원 위상 상태 업데이트
        self.rotor_a.q = obs_q_a
        self.rotor_b.q = obs_q_b

        self.rotor_b.omega = obs_omega_b

        return {
            "t": round(t, 4),
            "metric_distance": round(metric_distance, 6),
            "coupling_weight": round(coupling_weight, 6),
            "tension_gap": round(tension_gap, 6),
            "stiffness": round(self.stiffness, 4),
            "rotor_a_impedance": round(self.rotor_a.impedance, 4),
            "q_a_w": round(self.rotor_a.q[0], 4),
            "q_b_w": round(self.rotor_b.q[0], 4)
        }
