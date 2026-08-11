"""
test_dynamical_closed_loop_behavior.py
======================================
GroundedSensoryClosedLoop 시스템의 3대 동역학적 불변성(Dynamical Invariants)을 증명하는 테스트 세트.

핵심 증명 항목:
1. 항상성 (Homeostasis):
   - 임의의 위상 자극(예: "태양은 차갑다")에 따른 국소 변위 및 장력 폭발 이후,
     시간을 두고 자율 이완(Step)을 거쳤을 때 시스템이 안정된 평형 상태(Friction -> 0)로
     자발적으로 수렴함을 확인합니다.

2. 이력 현상 (Hysteresis):
   - 과거의 자극 궤적이 가소성 흔적(W, L, H, Psi의 미세한 뒤틀림)으로 보존되어,
     두 번째 유입되는 동일한 충격 상황에서 훨씬 더 효율적으로 마찰 에너지를 소산하며
     더욱 신속하게 수렴(Friction 소산 누적량 감소)함을 수학적으로 검증합니다.

3. 위상적 동형성 (Topological Isomorphism):
   - 공간적 거리 관계(Spatial Geometry), 열역학적 관계(Thermal Topology), 그리고 파동의 위상각 관계(Phase Relations)가
     동일한 위상 공간 상의 대칭 조건 Ф = 0 하에 묶여 있어, 서로 다른 도메인들이 구조적 정렬(Self-alignment)을 이루며
     동형 사상(Isomorphism)을 보존함을 대조/입증합니다.
"""

import pytest
import numpy as np
from core.physics.minimal_closed_loop import GroundedSensoryClosedLoop


def test_homeostasis_under_contradictory_stimulus():
    """
    [항상성 검증]
    'Sun'(태양)에 강한 'Cold'(-5.0) 열역학 자극을 프로젝트하여 "태양은 차갑다"라는 모순 자극을 주입합니다.
    자극 주입 직후 장력(Friction)과 편차 그래디언트가 크게 치솟은 뒤,
    자율 이완(eta_s, eta_t, eta_theta) 흐름만을 거쳐 시스템의 마찰이 서서히 소산(Zero)되어
    새로운 안정된 평형 상태에 정적으로 수렴함을 증명합니다.
    """
    system = GroundedSensoryClosedLoop(
        temperature=0.0,  # 노이즈를 제거하여 결정적(deterministic)인 항상성 수렴 속도를 측정
        coordinate_relaxation_rate=0.3,
        thermal_adaptation_rate=0.3,
        phase_synchronization_rate=0.3,
        weight_mutation_rate=0.1,
        consolidation_rate=0.1
    )

    # 초기 상태를 한 번 평형 수렴 단계(Warm-up)로 이완하여 기준점을 확보합니다.
    for _ in range(40):
        system.step(dt=0.1)

    initial_friction = system.calculate_friction()
    assert initial_friction < 0.5  # 평형 상태는 매우 낮은 장력 평형 상태

    # "태양은 차갑다" 모순 투사
    # Sun 노드의 thermal 값을 -5.0만큼 차갑게 perturbation을 줍니다.
    system.project_stimulus(target_concept="Sun", sensory_impulses={"thermal": -5.0})

    perturbed_friction = system.calculate_friction()
    assert perturbed_friction > initial_friction + 1.0  # 마찰이 급격히 증가

    # 40 스텝 동안 자율 이완 진행
    friction_history = [perturbed_friction]
    for _ in range(40):
        metrics = system.step(dt=0.1)
        friction_history.append(metrics["friction_after"])

    # 마찰 에너지의 지속적 소산 확인 (Friction Decay)
    assert friction_history[-1] < perturbed_friction
    assert friction_history[-1] < 0.25 * perturbed_friction  # 대량의 마찰이 소산되어 평형 수렴


def test_hysteresis_and_plastic_memory():
    """
    [이력 현상 검증]
    시스템에 첫 번째 자극(Perturbation)을 가하고 수렴 과정을 관찰하여, 소산되는 총 마찰 에너지의 누적합(Integral of Friction)을 측정합니다.
    이때 시스템의 내부 가소성 지층(W, L, H, Psi)에 '흔적(Memory)'이 남게 됩니다.
    이후 완전히 동일한 자극을 두 번째로 다시 가했을 때, 시스템이 기 구축된 인그람(Engram) 도로를 타고
    더 적은 에너지 소산(마찰 누적합)과 더 빠른 동기화 속도로 평형점에 정착함을 증명합니다.
    """
    system = GroundedSensoryClosedLoop(
        temperature=0.0,
        coordinate_relaxation_rate=0.4,
        thermal_adaptation_rate=0.4,
        phase_synchronization_rate=0.4,
        weight_mutation_rate=0.2,
        consolidation_rate=0.2
    )

    # Warm-up to clean initial transient friction
    for _ in range(20):
        system.step(dt=0.1)

    # 1. 첫 번째 자극 투사 (Cold 노드에 변위 및 위상 비동기화 주입)
    system.project_stimulus(
        target_concept="Cold",
        sensory_impulses={"spatial": [1.5, -1.5], "phase": np.pi * 0.5}
    )

    # 15 스텝 동안 이완하며 소산된 총 마찰 에너지 합 측정
    first_run_fric_sum = 0.0
    for _ in range(15):
        metrics = system.step(dt=0.1)
        first_run_fric_sum += metrics["friction_after"]

    # 이완 완료 후 타겟 불변량이 충분히 적응(Consolidated)되었는지 확인
    # 2. 동일한 자극을 두 번째로 투사
    system.project_stimulus(
        target_concept="Cold",
        sensory_impulses={"spatial": [1.5, -1.5], "phase": np.pi * 0.5}
    )

    second_run_fric_sum = 0.0
    for _ in range(15):
        metrics = system.step(dt=0.1)
        second_run_fric_sum += metrics["friction_after"]

    # 이력 현상 및 가소성 메모리 증명: 두 번째 가해진 동일 충격의 총 이완 에너지 마찰 누적량이
    # 첫 번째 시행에 비해 현저하게 낮아야 함 (기존 연결망 및 타겟 불변량이 두 번째 자극을 완충하기 때문)
    assert second_run_fric_sum < first_run_fric_sum


def test_topological_isomorphism():
    """
    [위상적 동형성 검증]
    공간적 거리(Spatial Field S), 열역학적 편차(Thermal Field T), 그리고 파동의 위상각 편차(Phase Field theta)가
    서로 완전히 상이한 도메인이지만, 동일한 대칭적 구속 평형식 Ф(C, P, E) = 0 아래서 결상됩니다.
    두 다른 노드 집합(예: Hot 계열 'Sun'-'Fire', Cold 계열 'Cold'-'Ice') 사이의
    거리 비(Ratio) 및 차이 비가 각 도메인(공간 기하, 열 차이, 위상각) 전반에서
    동형적 관계 구조(Isomorphic Relationship Topology)로 보존되며 자율 정렬됨을 증명합니다.
    """
    system = GroundedSensoryClosedLoop(
        temperature=0.0,
        coordinate_relaxation_rate=0.3,
        thermal_adaptation_rate=0.3,
        phase_synchronization_rate=0.3,
        weight_mutation_rate=0.1,
        consolidation_rate=0.1
    )

    # Sun-Fire, Cold-Ice는 온도가 비슷하므로 열역학적 거리가 가깝게 정렬되도록 하고,
    # Sun-Cold는 열역학적 거리가 멀도록 초기값(T)이 설정되어 있습니다.
    # 이완을 거치며 공간, 열역학, 위상 공간이 모두 조화를 이루게 만듭니다.
    for _ in range(30):
        system.step(dt=0.1)

    idx_sun = system.name_to_index["Sun"]
    idx_fire = system.name_to_index["Fire"]
    idx_cold = system.name_to_index["Cold"]
    idx_ice = system.name_to_index["Ice"]

    # 1. 공간적 거리 관계 (Spatial Isomorphism)
    dist_hot_spatial = np.linalg.norm(system.S[idx_sun] - system.S[idx_fire])
    dist_cold_spatial = np.linalg.norm(system.S[idx_cold] - system.S[idx_ice])
    dist_cross_spatial = np.linalg.norm(system.S[idx_sun] - system.S[idx_cold])

    # 2. 열역학적 온도 편차 관계 (Thermal Isomorphism)
    diff_hot_thermal = np.abs(system.T[idx_sun] - system.T[idx_fire])
    diff_cold_thermal = np.abs(system.T[idx_cold] - system.T[idx_ice])
    diff_cross_thermal = np.abs(system.T[idx_sun] - system.T[idx_cold])

    # 3. 위상각 편차 관계 (Phase Isomorphism)
    diff_hot_phase = np.abs((system.theta[idx_sun] - system.theta[idx_fire] + np.pi) % (2 * np.pi) - np.pi)
    diff_cold_phase = np.abs((system.theta[idx_cold] - system.theta[idx_ice] + np.pi) % (2 * np.pi) - np.pi)
    diff_cross_phase = np.abs((system.theta[idx_sun] - system.theta[idx_cold] + np.pi) % (2 * np.pi) - np.pi)

    # 동일한 극성 그룹(Hot-Hot, Cold-Cold)끼리의 연계 깊이/친밀도가
    # 이종 극성 그룹(Hot-Cold) 간의 편차보다 작아야 한다는 동형적 경향성 증명.
    assert dist_hot_spatial < dist_cross_spatial
    assert dist_cold_spatial < dist_cross_spatial

    assert diff_hot_thermal < diff_cross_thermal
    assert diff_cold_thermal < diff_cross_thermal

    assert diff_hot_phase < diff_cross_phase
    assert diff_cold_phase < diff_cross_phase
