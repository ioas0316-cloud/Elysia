# core/tests/test_cognitive_gears.py
# Copyright 2026 Lee Kang-deok & Antigravity
# Architecture: Unit Tests for Coupled Cognitive Gears & Emergent Dynamics

import sys
import os
import math
import pytest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.cognitive_gear import CognitiveGearNetwork

def test_personality_superposition():
    """시나리오 A: 인격 중첩 및 연속적 9인격 혼합 블렌딩 검증"""
    net = CognitiveGearNetwork(num_gears=1)
    
    # 1. 완벽한 Type 1의 센터 각도 (40도) 설정
    theta_type1 = 40.0 * math.pi / 180.0
    net.phases[0] = theta_type1
    dist = net.get_enneagram_distribution(0)
    
    # Type 1 (인덱스 1)의 활성화 강도가 1.0이고 나머지는 0이어야 함
    assert math.isclose(dist[1], 1.0, abs_tol=1e-5)
    assert sum(dist) == pytest.approx(1.0)
    assert net.get_dominant_type(0) == 1

    # 2. Type 1(40도)과 Type 2(80도)의 정확한 중간각인 60도 설정 (50:50 중첩 상태)
    theta_mid = 60.0 * math.pi / 180.0
    net.phases[0] = theta_mid
    dist_mid = net.get_enneagram_distribution(0)
    
    # Type 1(인덱스 1)과 Type 2(인덱스 2)의 활성화 비율이 각각 0.5여야 함
    assert math.isclose(dist_mid[1], 0.5, abs_tol=1e-5)
    assert math.isclose(dist_mid[2], 0.5, abs_tol=1e-5)
    assert sum(dist_mid) == pytest.approx(1.0)
    
    # 다른 유형들의 가중치는 0이어야 함
    for idx in range(9):
        if idx not in [1, 2]:
            assert dist_mid[idx] == 0.0

def test_hebbian_learning():
    """시나리오 B: 헵식 공명 학습(Hebbian Resonance Learning) 검증"""
    # 2개의 기어 생성 (고유 진동 없음)
    net = CognitiveGearNetwork(num_gears=2)
    # 초기 기어 이빨 맞물림(결합 강도)은 매우 낮음 (0.1)
    net.K[0][1] = 0.1
    net.K[1][0] = 0.1
    
    # 동조 상황: 두 기어의 위상이 완벽히 일치 (공명)
    net.phases[0] = 0.5
    net.phases[1] = 0.5
    
    # Hebbian 학습 진행
    dt = 0.1
    for _ in range(50):
        net.update_coupling_hebbian(dt, learning_rate=1.0)
        
    # 위상이 완벽히 일치했으므로 결합 강도 K가 원래(0.1)보다 대폭 강해져야 함 (1.0 근처로 수렴)
    assert net.K[0][1] > 0.8
    assert net.K[1][0] > 0.8
    
    # 불일치 상황: 두 기어가 완전 반대 위상 (비공명 / 위상차 pi)
    net.phases[0] = 0.0
    net.phases[1] = math.pi
    
    # Hebbian 학습 진행
    for _ in range(100):
        net.update_coupling_hebbian(dt, learning_rate=1.0)
        
    # 위상이 어긋났으므로 기어의 이빨 결합 강도가 서서히 역전되어 -1.0에 근접해야 함 (적대적 반발)
    assert net.K[0][1] < -0.8
    assert net.K[1][0] < -0.8

def test_emotional_differentiation():
    """시나리오 C: 예측 불가능한 외부 노이즈(Tension)에 의한 자율 감정 분화 검증"""
    # 2개의 기어 생성: 0번은 인지 기어(Cognitive), 1번은 감정 기어(Emotional)
    net = CognitiveGearNetwork(num_gears=2)
    
    # 두 기어는 초기 결합 상태 (K=0.5)
    net.K[0][1] = 0.5
    net.K[1][0] = 0.5
    
    # 기어 1의 초기 위상 = 0 (Type 9, 안정화/peacemaker 영점 평형 상태)
    net.phases[1] = 0.0
    assert net.get_dominant_type(1) == 9
    
    # 외부 스트림에서 급격한 미지의 위상 변화(텐션)가 주입됨
    # 인지 기어는 외부 스트림을 빠르게 수접하느라 위상이 180도(pi) 뒤틀림
    net.phases[0] = math.pi
    
    # 인지 기어와 감정 기어 간의 위상 차이에 의한 텐션 발생
    tension = net.compute_tension(0, 1)  # 1 - cos(pi - 0) = 2.0 (최대 텐션)
    assert tension == pytest.approx(2.0)
    
    # 텐션을 감정 기어(1번)에 토크(입력)로 인가하여 한 단계 전진
    # 텐션이 감정 기어를 평온 상태(Type 9)에서 밀어내어 회전시킴
    dt = 0.2
    inputs = [0.0, tension]
    net.step(dt, inputs)
    
    # 감정 기어의 위상이 0에서 벗어나 회전함
    assert net.phases[1] > 0.0
    # 영점 평형 상태(Type 9)에서 탈피하여 다른 성향(Type 1 또는 2 등)으로 감정이 "분화"하기 시작함
    assert net.get_dominant_type(1) != 9

def test_politics_and_polarization():
    """시나리오 D: 기어 간의 집단적 동조(합의) 및 반발(양극화) 검증"""
    # 3인의 에이전트 기어 네트워크 생성
    net = CognitiveGearNetwork(num_gears=3)
    
    # 1. 합의 (Consensus): 기어들이 서로 긍정적으로 맞물려 있을 때 (K = 0.8)
    net.K[0][1] = 0.8
    net.K[1][0] = 0.8
    net.K[1][2] = 0.8
    net.K[2][1] = 0.8
    
    # 서로 다른 위상에서 시작
    net.phases[0] = 0.0
    net.phases[1] = math.pi / 2.0
    net.phases[2] = math.pi
    
    dt = 0.1
    # 100틱 동안 서로 맞물려 공명 진동
    for _ in range(100):
        net.step(dt)
        
    # 서로 결합된 기어들이 동일한 위상으로 수렴하여 결합(동기화/사회적 합의) 완료
    assert abs(net.phases[0] - net.phases[1]) < 0.2
    assert abs(net.phases[1] - net.phases[2]) < 0.2
    
    # 2. 양극화 (Polarization): 적대적 적대 결합(K_02 = -1.0)이 가동될 때
    # 기어 0과 기어 2가 서로를 극단적으로 밀어냄
    net.K[0][2] = -1.0
    net.K[2][0] = -1.0
    
    # 다른 기어와의 연결 차단 (순수 양극화 텐션만 검증하기 위해)
    net.K[0][1] = 0.0
    net.K[1][0] = 0.0
    net.K[1][2] = 0.0
    net.K[2][1] = 0.0
    
    # 두 대립 에이전트의 위상차를 pi/2 로 배치하여 최대 갈등 텐션 유발
    net.phases[0] = 0.0
    net.phases[2] = math.pi / 2.0
    
    # 50틱 동안 반발 작동
    for _ in range(50):
        net.step(dt)
        
    # 0번과 2번 기어는 적대적 반발력에 의해 위상이 완전히 대척점(위상차 pi)으로 찢어짐 (양극화 분극 발생)
    final_diff = abs(net.phases[0] - net.phases[2])
    if final_diff > math.pi:
        final_diff = 2.0 * math.pi - final_diff
    assert final_diff == pytest.approx(math.pi, abs=0.2)
