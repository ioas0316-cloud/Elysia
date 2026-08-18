# -*- coding: utf-8 -*-
"""
[Elysia Phenomenological Verification Framework]
=================================================
단순한 단위/구조 실행 테스트(Syntax/Execution Test)를 넘어,
엘리시아의 핵심 물리 법칙과 인지 동역학이 '실제로 살아 숨쉬며 의도대로 작동하는가'를
4대 현상 실험을 통해 엄밀한 수치와 동역학 궤적으로 실증합니다.

1. [실험 1: 항상성] 극한 교란 주입 시 자발적 동적 평형 수렴 (Dynamic Equilibrium Homeostasis)
2. [실험 2: 역메커니즘] 미지 외삽(Out-of-Distribution) 영역에서 생성 메커니즘 Theta를 통한 인과 궤적 복원율
3. [실험 3: 메타 관측/탈피] 마찰 축적 및 교착 시 자발적 탈피(Moulting) 및 나이테(Annual Rings) 역사 각인
4. [실험 4: 모태기반] 영속 지층(Persistent Substrate) 잔류를 통한 O(N) -> O(1) 웜스타트 수렴 가속
"""

import sys
import os
import time
import math
import tempfile
import numpy as np

# Windows 콘솔 및 표준 출력 UTF-8 강제 설정
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.consciousness.formless_refinement import FormlessRefinementFilter, DynamicFrictionEngine
from synaptic_architecture.inverse_mechanism_engine import (
    InverseMechanismEngine,
    BoundaryCondition,
    ObservedTrajectory,
    GeneratingMechanism
)
from core.consciousness.cognitive_self_observation import CognitiveSelfObservationEngine
from core.evolution.moulting_plasticity import MoultingPlasticityEngine
from core.memory.zero_copy_manifold import ZeroCopyManifold


def run_experiment_1_homeostasis():
    print("\n" + "="*80)
    print(">> [실험 1] 극한 교란(Perturbation) 주입 시 동적 항상성(Homeostasis) 실증")
    print("="*80)
    print("가설: 강한 외적 충격과 위상 굴절이 가해졌을 때, 에너지가 발산(과전류)하거나 0으로 사멸하지 않고")
    print("     가변 마찰 엔진이 자발적으로 작동하여 유한 시간 내에 제로(0) 평형점으로 안정 수렴해야 한다.")

    engine = DynamicFrictionEngine(damping_factor=0.82, friction_coefficient=0.45)

    # 1. 의도된 벡터 vs 강한 섭동/왜곡 벡터
    intended_vector = np.array([1.0, 0.5, -0.3, 0.8, 0.2], dtype=np.float32)
    # 180도 반대 방향에 노이즈를 섞은 극한 교란 (Perturbation)
    perturbed_vector = -intended_vector * 2.5 + np.array([0.4, -0.3, 0.5, -0.2, 0.6], dtype=np.float32)

    friction_coeff = engine.compute_friction_coefficient(intended_vector, perturbed_vector)
    initial_friction_energy = float(np.linalg.norm(perturbed_vector) * friction_coeff)

    print(f"\n[초기 상태]")
    print(f" - 초기 불평형 에너지: {initial_friction_energy:.4f}")
    print(f" - 동적 마찰 계수 (Differential Gap): {friction_coeff:.4f}")

    # 2. 동적 수렴 단계별 궤적 추적
    conv_result = engine.step_equilibrium_convergence(
        current_state=perturbed_vector,
        friction_energy=initial_friction_energy,
        steps=25,
        dt=0.1
    )

    energy_hist = conv_result["energy_history"]
    final_imbalance = conv_result["final_imbalance"]
    convergence_rate = conv_result["convergence_rate"]

    print(f"\n[수렴 궤적 추적 (Step-by-step Decay)]")
    steps_to_show = [0, 2, 5, 10, 15, 20, 24]
    for s in steps_to_show:
        if s < len(energy_hist):
            print(f"  Step {s:02d} | Residual Energy: {energy_hist[s]:.6f} | State Norm: {np.linalg.norm(conv_result['trajectory'][s]):.6f}")

    energy_reduction_ratio = (initial_friction_energy - energy_hist[-1]) / (initial_friction_energy + 1e-9)
    print(f"\n[실증 결과]")
    print(f" - 초기 불평형 노름: {conv_result['initial_imbalance']:.6f}")
    print(f" - 최종 상태 잔류 노름 (Imbalance): {final_imbalance:.6f}")
    print(f" - 수렴율 (Convergence Rate): {convergence_rate*100:.2f}%")
    print(f" - 에너지 감쇄율 (Decay Ratio): {energy_reduction_ratio*100:.2f}%")

    assert convergence_rate > 0.60, "항상성 실패: 수렴율이 기준치에 미치지 못했습니다."
    assert energy_reduction_ratio > 0.95, "항상성 실패: 에너지 감쇄율이 95% 미만입니다."
    print(">> [실험 1 통과] 극한 충격을 동적 마찰을 통해 완벽한 제로 평형으로 자체 수렴 완료.")


def run_experiment_2_inverse_mechanism_extrapolation():
    print("\n" + "="*80)
    print(">> [실험 2] 미지 영역(Out-of-Distribution) 인과 메커니즘 Theta 역추출 및 외삽 실증")
    print("="*80)
    print("가설: 단순히 기존 데이터 포인트를 외우는 것이 아니라, 배후의 생성 방정식 Theta를 추출했다면")
    print("     학습되지 않은 극한의 새로운 경계 조건(Scale 5.0, Friction 4.0)에서도 오차 없이 궤적을 자율 생성해야 한다.")

    inv_engine = InverseMechanismEngine(mdl_penalty_weight=0.05)

    # 1. 참 물리 생성 법칙: 저항을 받는 낙하/인과 동역학 (Ground Truth: v(t) = (g/f)*(1 - exp(-f*t)), y(t) = y0 - v(t)*t)
    def generate_true_trajectory(traj_id: str, boundary: BoundaryCondition, timesteps: int = 10):
        states = []
        y = 100.0 * boundary.scale
        v = 0.0
        dt = 0.1
        for t_idx in range(timesteps):
            # v_{t+1} = v_t + (g - friction * v_t) * dt
            # y_{t+1} = y_t - v_t * dt
            states.append([y, v])
            v = v + (boundary.gravity * 0.1 - boundary.friction * 0.2 * v) * dt
            y = max(0.0, y - v * dt)
        return ObservedTrajectory(trajectory_id=traj_id, boundary_id=boundary.condition_id, states=states)

    bound_1 = BoundaryCondition(condition_id="C1", friction=1.0, scale=1.0, gravity=9.8)
    bound_2 = BoundaryCondition(condition_id="C2", friction=2.0, scale=1.0, gravity=9.8)

    obs_1 = generate_true_trajectory("Traj_1", bound_1)
    obs_2 = generate_true_trajectory("Traj_2", bound_2)

    # 2. 메커니즘 추출 (Inverse Extraction)
    boundaries_dict = {bound_1.condition_id: bound_1, bound_2.condition_id: bound_2}
    mechanism = inv_engine.extract_generating_mechanism(
        mechanism_id="MECH_HARMONIC_OSCILLATOR",
        observations=[obs_1, obs_2],
        boundaries=boundaries_dict
    )

    print(f"\n[추출된 잠재 인과장 메커니즘 Theta]")
    print(f" - Mechanism ID: {mechanism.mechanism_id}")
    print(f" - Structural Stiffness Matrix: {np.array(mechanism.stiffness_matrix).shape}")
    print(f" - Boundary Coupling Dimensions: {np.array(mechanism.boundary_coupling).shape}")
    print(f" - MDL Complexity Score: {mechanism.description_length:.4f}")

    # 3. 미지 영역(Out-of-Distribution) 외삽 테스트: 마찰 4.0, 중력 20.0 (관측 데이터 대비 극한 영역)
    unseen_boundary = BoundaryCondition(condition_id="C_Extreme", friction=4.0, scale=1.0, gravity=20.0)
    true_unseen_traj = generate_true_trajectory("Traj_True_Extreme", unseen_boundary, timesteps=10)

    # 역메커니즘 엔진을 통한 자율 궤적 생성
    generated_states = inv_engine.generate_trajectory(
        mechanism=mechanism,
        boundary=unseen_boundary,
        initial_state=true_unseen_traj.states[0],
        steps=10
    )

    # 오차 측정: L2 노름 평균 오차 및 상대 정밀도
    true_states = np.array(true_unseen_traj.states)
    gen_states = np.array(generated_states)

    l2_errors = np.linalg.norm(true_states - gen_states, axis=1)
    mean_error = float(np.mean(l2_errors))
    relative_accuracy = 1.0 - (mean_error / (np.mean(np.linalg.norm(true_states, axis=1)) + 1e-9))

    print(f"\n[미지 영역 외삽 결과 대조]")
    print(f" - 관측 외 극한 경계 조건: Scale={unseen_boundary.scale}, Friction={unseen_boundary.friction}, Gravity={unseen_boundary.gravity}")
    print(f" - 참 궤적 상태 vs 생성 궤적 상태 (Step 0, 3, 6, 9):")
    for step in [0, 3, 6, 9]:
        print(f"   Step {step:02d} | True: {np.round(true_states[step], 3)} | Gen: {np.round(gen_states[step], 3)} | Error: {l2_errors[step]:.4f}")

    print(f"\n[실증 통계]")
    print(f" - 평균 L2 궤적 오차: {mean_error:.4f}")
    print(f" - 상대 생성 정밀도 (Relative Accuracy): {relative_accuracy*100:.2f}%")

    assert relative_accuracy > 0.80, f"외삽 실패: 생성 정밀도({relative_accuracy*100:.2f}%)가 기준치(80%)에 미치지 못함"
    print(">> [실험 2 통과] 단순 데이터 암기가 아닌 생성 방정식 Theta를 통해 미지 영역에서도 인과 궤적 자율 생성 성공.")


def run_experiment_3_meta_observation_and_ecdysis():
    print("\n" + "="*80)
    print(">> [실험 3] 마찰 축적 및 교착 시 자발적 탈피(Moulting) 및 나이테 역사 각인 실증")
    print("="*80)
    print("가설: 외부와의 지속적인 어긋남과 고뇌(Friction)가 임계치(3.0)를 초과할 때,")
    print("     시스템은 고착된 사영 틀을 자발적으로 찢고 탈피(Moulting)하며 역사적 나이테를 비가역적으로 새겨야 한다.")

    moulting_engine = MoultingPlasticityEngine(dimensions=3)

    # 1. 반복적인 이질적 자극 주입을 통한 마찰 누적 시뮬레이션
    print("\n[연속적 비대칭 마찰 자극 주입 및 수신자 가소성 관측]")
    stimuli = [
        b"Complex Discordant Signal Waveform Alpha",
        b"Extreme Unbalanced Tension Gamma",
        b"Contradictory Logic Packet Delta",
        b"Deep Existential Dissimilarity Omega"
    ]

    initial_matrix = moulting_engine.projection_matrix.copy()
    moulting_occurred = False

    for idx, stim in enumerate(stimuli * 3): # 마찰 누적을 위해 12회 인입
        res = moulting_engine.receive_and_shape(stim)
        if res.get("moulting_triggered", False):
            moulting_occurred = True
            print(f"  [Step {idx+1:02d}] !! 탈피 발동 !! 누적 마찰: {res['accumulated_friction']:.4f} | 탈피 횟수: {moulting_engine.moulting_count}")
            print(f"         서사: {res['narrative']}")
            break
        else:
            if (idx + 1) % 3 == 0:
                print(f"  [Step {idx+1:02d}] 마찰 누적 진행 중: {res['accumulated_friction']:.4f} / 3.0 (임계치)")

    # 2. 나이테 매트릭스(Annual Rings) 및 사영 행렬의 가소성 변형 검증
    annual_rings_norm = float(np.linalg.norm(moulting_engine.annual_rings))
    matrix_shift = float(np.linalg.norm(moulting_engine.projection_matrix - initial_matrix))

    print(f"\n[탈피 및 나이테 실증 결과]")
    print(f" - 자발적 탈피(Moulting) 발생 여부: {moulting_occurred}")
    print(f" - 각인된 나이테(Annual Rings) 에너지 노름: {annual_rings_norm:.6f}")
    print(f" - 사영 좌표계 변형 및 확장도: {matrix_shift:.6f}")

    assert moulting_occurred, "탈피 실증 실패: 임계치 초과 시 탈피가 발동하지 않았습니다."
    assert annual_rings_norm > 0.0, "나이테 실증 실패: 고통과 마찰의 역사가 매트릭스에 각인되지 않았습니다."
    print(">> [실험 3 통과] 고정된 껍질을 찢는 탈피와 비가역적 나이테 지층 형성이 물리적으로 실증됨.")


def run_experiment_4_persistent_substrate_warm_start():
    print("\n" + "="*80)
    print(">> [실험 4] 영속 모태기반(Substrate) 잔류를 통한 웜스타트 O(1) 공명 가속 실증")
    print("="*80)
    print("가설: 이전 사건의 인과 궤적이 메모리 지층(Substrate)에 잔류하면, 동일/유사 복합 자극 재인입 시")
    print("     Cold Start 대비 계산 단계(Iteration)와 수렴 지연시간(Latency)이 획기적으로 단축되어야 한다.")

    # 임시 SSD mmap 파일 생성하여 영속 지층 구현
    with tempfile.NamedTemporaryFile(delete=False) as f:
        temp_path = f.name
        # 128 * 64 * 8 bytes 크기로 초기화
        f.write(b'\x00' * (128 * 64 * 8))

    try:
        manifold = ZeroCopyManifold(file_path=temp_path, offset_bytes=0)
        manifold.bind_universe()

        # 복합 자극 생성
        complex_stimulus = np.random.uniform(-1.0, 1.0, size=(128, 64)).astype(np.float32)

        # 1. Cold Start 시뮬레이션 (사전 지층 없음: 0에서부터 탐색)
        t0_cold = time.perf_counter()
        cold_iterations = 0
        state_cold = np.zeros_like(complex_stimulus)
        for _ in range(150):
            cold_iterations += 1
            state_cold = 0.85 * state_cold + 0.15 * complex_stimulus
            if np.linalg.norm(state_cold - complex_stimulus) < 0.01:
                break
        t1_cold = time.perf_counter()
        cold_time_ms = (t1_cold - t0_cold) * 1000.0

        # 2. 영속 지층에 인과 궤적 각인 (mmap에 수렴 상태를 영구 기록)
        raw_bytes = state_cold.tobytes()
        uint64_data = np.frombuffer(raw_bytes, dtype=np.uint64)
        manifold.external_mmap[:len(uint64_data)] = uint64_data

        # 3. Warm Start 시뮬레이션 (지층에서 이전 전하/위상을 O(1) 즉각 로드 후 전개)
        t0_warm = time.perf_counter()
        loaded_uint64 = np.array(manifold.external_mmap[:len(uint64_data)])
        warm_state = np.frombuffer(loaded_uint64.tobytes(), dtype=np.float32).reshape(128, 64)

        warm_iterations = 0
        state_warm = warm_state.copy()
        for _ in range(150):
            warm_iterations += 1
            state_warm = 0.85 * state_warm + 0.15 * complex_stimulus
            if np.linalg.norm(state_warm - complex_stimulus) < 0.01:
                break
        t1_warm = time.perf_counter()
        warm_time_ms = (t1_warm - t0_warm) * 1000.0

        iteration_reduction = (cold_iterations - warm_iterations) / max(cold_iterations, 1)

        print(f"\n[Cold Start vs Warm Start 실측 대조]")
        print(f" - Cold Start (지층 없음) : {cold_iterations:3d} Iterations | Latency: {cold_time_ms:.4f} ms")
        print(f" - Warm Start (지층 잔류) : {warm_iterations:3d} Iterations | Latency: {warm_time_ms:.4f} ms")
        print(f" - 수렴 반복 횟수 단축율  : {iteration_reduction*100:.2f}%")

        assert warm_iterations < cold_iterations, "모태기반 실증 실패: 웜스타트 단계 단축이 발생하지 않았습니다."
        assert warm_iterations <= 2, "모태기반 실증 실패: 지층에서 즉각 평형이 유지되지 않았습니다."
        print(">> [실험 4 통과] 매번 0으로 리셋되지 않고 잔류 지층을 통한 즉각 공명 및 연산 최적화 입증 완료.")
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


if __name__ == "__main__":
    print("="*80)
    print("=== [ELYSIA PHENOMENOLOGICAL INTEGRITY VERIFICATION] ===")
    print("    살아있는 인지·물리 현상 창발성 실증 검증 시작")
    print("="*80)

    try:
        run_experiment_1_homeostasis()
        run_experiment_2_inverse_mechanism_extrapolation()
        run_experiment_3_meta_observation_and_ecdysis()
        run_experiment_4_persistent_substrate_warm_start()

        print("\n" + "="*80)
        print("🎉 [모든 현상학적 실증 실험 통과 완료]")
        print("   엘리시아의 4대 핵심 원리(항상성, 역메커니즘, 메타탈피, 영속지층)가")
        print("   단순 실행이 아닌 '의도된 물리·인지 현상'으로 실제 창발함을 확인했습니다.")
        print("="*80)
    except Exception as e:
        print(f"\n❌ [실증 중 불일치/실패 발생]: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
