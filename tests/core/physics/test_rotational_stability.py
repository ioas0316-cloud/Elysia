import math
import numpy as np
import pytest
from core.physics.quaternion_manifold_dynamics import QuaternionRotorState, QuaternionHelper
from core.evolution.moulting_plasticity import MoultingPlasticityEngine
from core.physics.continuous_manifold_gear import ContinuousManifoldGearSystem


def test_quaternion_norm_conservation_10k_steps():
    """
    [우주적 회전 구속 - 제1칙 검증: 반대칭 연산과 에너지 보존]
    단위 쿼터니언 SO(3) 다양체 위에서의 10,000 스텝 장기 회전 시뮬레이션 동안,
    수치적 표류(Numerical Drift) 없이 노름(Norm)이 부동소수점 한계(1e-12) 내로 완벽히 보존되는지 검증합니다.
    """
    # 임의의 3D 회전 각속도 벡터 (회전 속도를 격렬하게 주기 위해 비교적 큰 값 설정)
    omega = (1.5, -2.0, 3.1)
    rotor = QuaternionRotorState(q=(1.0, 0.0, 0.0, 0.0), omega=omega, impedance=0.0)

    dt = 0.001  # 미소 시간 간격
    steps = 10000

    max_norm_error = 0.0

    # 10,000 스텝 동안 격렬한 토크/회전 상태 연속 주사
    for i in range(steps):
        # 일정 주기마다 임의의 요동 토크 추가
        torque = (
            math.sin(i * 0.1) * 0.5,
            math.cos(i * 0.1) * 0.5,
            math.sin(i * 0.2) * 0.3
        )
        rotor.apply_torque(torque, dt)

        # 쿼터니언 상태의 노름 계산
        w, x, y, z = rotor.q
        q_norm = math.sqrt(w*w + x*x + y*y + z*z)
        norm_error = abs(q_norm - 1.0)
        max_norm_error = max(max_norm_error, norm_error)

        # 매 프레임 수치 오차 한계선(2e-12) 초과가 절대 없음을 보증
        assert norm_error <= 2e-12, f"Step {i}: Norm error exceeds 2e-12: {norm_error:.2e}"

    print(f"\n[Test 1 Complete] Maximum SO(3) Norm Error over 10k steps: {max_norm_error:.2e}")
    assert max_norm_error <= 2e-12


def test_mode_collapse_prevention_orthogonal_guard():
    """
    [우주적 회전 구속 - 제2칙 검증: 불변 축과 궤도 원반의 위상 분리]
    극단적인 전단 응력(Shear Stress)과 장력 벡터(Tension Vector)가 지속적으로 주입되어도,
    시스템의 불변 질서 수호 축인 'Order (Index 1 / Blue)'는 수직 투사 보호막에 의해
    100% 원형을 유지(Invariant)하며, [Flux-Entropy] 평면에서만 가소적 변형이 일어나는지 검증합니다.
    """
    # 3차원 투사 가소성 엔진 초기화
    engine = MoultingPlasticityEngine(dimensions=3)

    # 초기 상태에서는 정확한 정규직교 Identity 행렬
    initial_matrix = np.array(engine.projection_matrix, dtype=np.float32)
    np.testing.assert_array_equal(initial_matrix[1, :], np.array([0.0, 1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(initial_matrix[:, 1], np.array([0.0, 1.0, 0.0], dtype=np.float32))

    # 극단적으로 편향되고 강력한 노이즈 입력 (Shear Stress 유도)
    extreme_inputs = [
        b"\xff\x00\xff\x00\xff\x00",
        b"\x00\xff\x00\xff\x00\xff",
        b"\xff\xff\x00\x00\xff\xff",
        b"\x12\x34\x56\x78\x90\xab"
    ]

    # 150 스텝 동안 다양한 충격을 연속적으로 가조
    for step in range(150):
        raw_input = extreme_inputs[step % len(extreme_inputs)]
        engine.receive_and_shape(raw_input, modality_hint="extreme_test")

        # 가소성 사영 행렬의 Snapshots
        P = np.array(engine.projection_matrix, dtype=np.float32)

        # 1) 불변 수호 축(Y축 / Order / Index 1)은 어떤 마찰 하에서도 완전 일치 보존 검증 (1e-7 floating error)
        np.testing.assert_allclose(P[1, :], np.array([0.0, 1.0, 0.0], dtype=np.float32), atol=1e-7)
        np.testing.assert_allclose(P[:, 1], np.array([0.0, 1.0, 0.0], dtype=np.float32), atol=1e-7)

        # 2) 궤도 원반 [Flux (Index 0), Entropy (Index 2)] 공간 상에서는 실제로 변화가 활발히 전개됨을 검증
        # 즉, X축과 Z축 관련 요소들은 초기 직교행렬에서 벗어나 유동적으로 가소적 전이를 일으킴
        x_z_changed = (abs(P[0, 0] - 1.0) > 1e-5) or (abs(P[2, 2] - 1.0) > 1e-5) or (abs(P[0, 2]) > 1e-5) or (abs(P[2, 0]) > 1e-5)
        # 어느 정도 마찰이 쌓인 이후(5 스텝 이상)에는 X-Z 평면의 궤도가 확실히 가소적으로 뒤틀림을 확인
        if step >= 5:
            assert x_z_changed, f"Step {step}: Rotational plane [X, Z] showed no plasticity warp!"

        # 나이테(Annual Rings) 매트릭스 역시 불변 질서(Order) 축을 완벽히 수호하는지 검증
        R = np.array(engine.annual_rings, dtype=np.float32)
        np.testing.assert_allclose(R[1, :], np.array([0.0, 0.0, 0.0], dtype=np.float32), atol=1e-7)
        np.testing.assert_allclose(R[:, 1], np.array([0.0, 0.0, 0.0], dtype=np.float32), atol=1e-7)

    print("\n[Test 2 Complete] Orthogonal Guard successfully preserved system Order axis invariant under 150 extreme shear steps.")


def test_phase_locking_orbital_resonance():
    """
    [우주적 회전 구속 - 제3칙 검증: 위상 고정 기반의 무분기 임피던스 적응]
    임의의 미스매치 위상을 지닌 기어/파동 상태가 외부 손실 함수 제어 없이,
    자발적 위상차(Phase Gap) 복원력과 Tension Gap에 의해 스스로 위상 고정(Phase-Locked)되어
    정재파 동기화(Tension Gap -> 0)되는 현상을 검증합니다.
    """
    # 기어 결합 전도율 자율 조율 계수를 높이고, 위상차가 확실한 결합 반경 설정
    system = ContinuousManifoldGearSystem(
        base_omega_a=3.0,
        base_omega_b=3.0,
        sensitivity_radius=0.4,
        learning_rate=0.1
    )

    # 초기 위상 오차가 큰 상태 (어긋난 궤도)
    system.rotor_a.phase = 0.0
    system.rotor_b.phase = 0.5

    # 초기 stiffness에 미스매치 주입하여 자율 조정 텐션 유도
    system.stiffness = 0.5

    # 초기 상태의 Tension Gap은 유의미하게 큼
    res_initial = system.step(t=0.0, dt=0.01)
    assert res_initial["tension_gap"] > 1e-4

    # 1,000 스텝 동안 외부 개입 없이 텐서장 스스로 공명을 주도록 주사
    last_tension_gap = res_initial["tension_gap"]
    tension_history = []

    for i in range(1000):
        res = system.step(t=0.01 * (i + 1), dt=0.01)
        tension_history.append(res["tension_gap"])

    # 후반 100 스텝 동안의 평균 Tension Gap 측정
    recent_tension_avg = sum(tension_history[-100:]) / 100.0

    # 손실 함수나 클리핑의 외적 개입 없이, 오직 위상차 복원력에 의해 스스로 궤도 평형점에 수렴(Tension Gap -> 0)했는지 검증
    # 소수점 이하 매우 미소한 평형 수치(0.01 이하)에 도달함을 확인
    assert recent_tension_avg < 0.01, f"Failed to Phase-Lock! Average tension gap remains high: {recent_tension_avg:.4e}"

    print(f"\n[Test 3 Complete] Orbital Resonance achieved! Tension Gap decayed from {last_tension_gap:.4f} to {recent_tension_avg:.4e} without manual loss functions.")
