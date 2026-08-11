import pytest
import numpy as np
from core.physics.minimal_closed_loop import GroundedSensoryClosedLoop

def test_homeostasis_under_continuous_noise():
    """
    [Homeostasis (항상성) 동역학 시험]
    외부에서 인위적으로 개입하여 상태값을 교정하지 않는 상태에서,
    연속적인 무작위 감각 소음(Thermal/Spatial Noise)을 무차별 주입합니다.
    시스템이 발산(Explosion)하거나 붕괴(Collapse)하지 않고,
    자율적인 자기참조 이완을 통해 스스로 위상적 안정상태(Homeostasis)로 수렴해 들어가는지 검증합니다.
    """
    loop = GroundedSensoryClosedLoop(
        coordinate_relaxation_rate=0.3,
        charge_adaptation_rate=0.1,
        weight_mutation_rate=0.1,
        consolidation_rate=0.05
    )

    # 1. 초기 무부하 상태의 평형 마찰 에너지 확인
    initial_friction = loop.calculate_sensory_friction()

    # 2. 연속적인 무작위 다중 감각 소음(Noise Storm) 연쇄 주입
    # 매 스텝마다 열역학 지층과 공간 지층을 동시에 마구 흔듭니다.
    np.random.seed(42)
    friction_peaks = []

    for _ in range(15):
        # 무작위 노드에 임의의 냉각/가열 충격 가함
        target_node = np.random.choice(loop.labels)
        noise_impulse = float(np.random.uniform(-1.5, 1.5))
        loop.project_sensory_stimulus(target_node, noise_impulse)

        # 물리적 매개변수를 직접 세팅하지 않고, 오직 Substrate 자체의 step() 거동만 가동
        metrics = loop.step(dt=0.3)
        friction_peaks.append(metrics["friction_after"])

    # 3. 항상성 검증:
    # 소음 폭풍 속에서 마찰 에너지가 극심하게 요동쳤더라도,
    # 최종적으로 자율적 소산(Dissipation)을 통해 마찰이 제어 가능한 안정 범위 내로 유지되거나 수렴해야 함.
    final_friction = loop.calculate_sensory_friction()

    # 극단적인 발산(Friction -> infinity)이 일어나지 않고 바운딩됨을 확인
    assert final_friction < 25.0
    # 자율적인 감쇠력이 가동되어 소음 주입 초기 피크들보다 최종적으로 가라앉았음을 검증
    assert final_friction < max(friction_peaks)


def test_hysteresis_and_substrate_memory():
    """
    [Hysteresis (이력 현상) 동역학 시험]
    동일한 감각적 충격(Thermal Shock)을 두 번에 걸쳐 시차를 두고 가합니다.
    - 1차 충격: 생소한 자극이므로 시스템이 큰 마찰(Friction)과 긴장을 겪으며 적응합니다.
    - 2차 충격: 과거 겪었던 자극의 흔적(Memory)이 연결 가중치(W)와 불변량(L) 지층에 새겨져(Consolidated) 있습니다.

    따라서 2차 충격 시에는 1차 충격에 비해:
    1. 최대 도달 마찰 피크(Peak Friction)가 더 적어야 하며 (Dampened response),
    2. 이완 과정 전체에서 소비되는 총 인과 마찰 에너지의 합(Total Integrated Friction Energy)이 더 작아야 합니다.

    이것은 정적인 Assert 문이 아니라, 유기체가 '자율적으로 과거를 기억하고 반응을 변형하는' 진짜 이력 거동을 증명합니다.
    """
    loop = GroundedSensoryClosedLoop(
        coordinate_relaxation_rate=0.4,
        charge_adaptation_rate=0.1,
        weight_mutation_rate=0.15,
        consolidation_rate=0.1
    )

    # --- 1차 충격 수용 및 이완 ---
    # 태양에 냉각 충격(-3.5) 투사
    loop.project_sensory_stimulus("Sun", cold_or_heat_impulse=-3.5)
    f_peak_1st = loop.calculate_sensory_friction()

    # 10스텝 동안 자율 수렴하며 물리 흔적 각인
    f_history_1st = []
    for _ in range(10):
        m = loop.step(dt=0.4)
        f_history_1st.append(m["friction_after"])

    # 외부 자극 전위(환경 소음)를 리셋하여 동일한 강도의 2차 충격 준비
    # 단, 시스템 내부의 기억인 W와 L은 보존됨 (Substrate Memory)
    loop.external_thermal_perturbation.fill(0.0)

    # --- 2차 충격 수용 및 이완 ---
    # 동일한 냉각 충격(-3.5)을 동일 노드에 정확히 한 번 더 주입
    loop.project_sensory_stimulus("Sun", cold_or_heat_impulse=-3.5)
    f_peak_2nd = loop.calculate_sensory_friction()

    f_history_2nd = []
    for _ in range(10):
        m = loop.step(dt=0.4)
        f_history_2nd.append(m["friction_after"])

    # 1. 이력 현상 검증 (Peak Comparison)
    # 2차 충격 시에는 이미 결합구조가 유연화되었으므로, 최대 마찰 피크가 1차 충격 피크보다 눈에 띄게 완화됨을 보장
    assert f_peak_2nd < f_peak_1st

    # 2. 기억 공명 검증 (Total Integrated Friction Energy Comparison)
    # 이미 닦여진 인과 궤적을 따라 이완되므로, 이완 과정 전체에서 소비되는 총 마찰 에너지 합이 1차보다 더 작아야 함 (효율적 수렴)
    assert sum(f_history_2nd) < sum(f_history_1st)


def test_topological_continuity_isomorphism():
    """
    [Topological Continuity (위상적 연속성) 동역학 시험]
    서로 다른 크기나 방식으로 감각 입력을 주입하더라도,
    하부 인과 장(Field)에서 동일한 '위상적 마찰 전이 양상'이 자발적으로 형성되어 수렴하는지 검증합니다.
    입력 형태에 휘둘리지 않고, 인과적 본질을 공명 지도로 투사하는 역량을 검증하기 위해,
    결과 상태 행렬 S와 연결 가중치 행렬 W의 코사인 유사도/상관 관계가 매우 높음을 보장합니다.
    """
    loop_v = GroundedSensoryClosedLoop()
    loop_t = GroundedSensoryClosedLoop()

    # loop_v 에는 단일 강한 자극 주입
    loop_v.project_sensory_stimulus("Sun", cold_or_heat_impulse=-3.0)
    for _ in range(5):
        loop_v.step(dt=0.3)

    # loop_t 에는 시간차 분할 자극 주입 (동등한 총 누적 전위 에너지)
    loop_t.project_sensory_stimulus("Sun", cold_or_heat_impulse=-1.5)
    loop_t.step(dt=0.3)
    loop_t.project_sensory_stimulus("Sun", cold_or_heat_impulse=-1.5)
    for _ in range(4):
        loop_t.step(dt=0.3)

    # 최종 수렴된 고유 온도 성향의 방향성 상관관계 분석 (Pearson Correlation 또는 Cosine Similarity)
    # 방향 및 부호 경향성이 위상적으로 일치하는가 (Isomorphic Topology)
    cos_sim_T = np.dot(loop_v.T_charges, loop_t.T_charges) / (np.linalg.norm(loop_v.T_charges) * np.linalg.norm(loop_t.T_charges) + 1e-9)
    assert cos_sim_T > 0.95

    # 연결 강도 매트릭스의 위상적 수렴 상관성 확인
    cos_sim_W = np.dot(loop_v.W.flatten(), loop_t.W.flatten()) / (np.linalg.norm(loop_v.W) * np.linalg.norm(loop_t.W) + 1e-9)
    assert cos_sim_W > 0.95
