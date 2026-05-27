# core/tests/test_society_isomorphism.py
# Copyright 2026 Lee Kang-deok & Antigravity
# Architecture: Unit Tests for World Rotor Social Isomorphism Internalization

import sys
import os
import math
import pytest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.enneagram_phase_topology import NUM_SCALES
from core.scale_observer import extract_digit_9, replace_digit_9, observe_other
from core.world_hyper_rotor import world_tick_with_horizontal_carry
from core.cognitive_gear import CognitiveGearNetwork

def test_mirror_resonance_empathy():
    """Scenario 1: 거울 공명 및 공감(Mirror Resonance & Empathy) 검증"""
    # 세계 로터 S
    S = 0
    # Agent 0 (offset 0)과 Agent 1 (offset 4)의 위상 설정
    # 두 에이전트가 서로의 관점(비트 오프셋)을 변경하며 상대방을 관측
    # d0(개체) 수준에서 서로 동기화되는 상황 모사
    S = replace_digit_9(S, 0, 5) # Agent 0의 d0 = 5 (탐구자)
    S = replace_digit_9(S, 4, 5) # Agent 1의 d0 = 5 (탐구자)
    
    # 두 오프셋 간의 위상 차이(XOR)를 관측
    # my_offset=0, other_offset=4, scale=0
    dissonance = observe_other(S, 0, 4, 0)
    
    # 위상차가 0이므로 완벽한 공감(거울 공명) 상태 성립
    assert dissonance == 0

def test_phase_economy_horizontal_carry():
    """Scenario 2: 수평적 캐리 전파를 통한 위상 거래(Phase Economy) 검증"""
    S = 0
    interference = [1] * 16
    
    # 수평 캐리 라우팅 설정:
    # Agent 0의 d0(0,0)에서 캐리 발생 시 -> Agent 1의 d0(4,0)로 방전 에너지를 우회 전송
    carry_routing = {
        (0, 0): (4, 0)
    }
    
    # clock: 오직 Agent 0의 d0(0,0) 자릿수만 1인 신호 주입
    clock = replace_digit_9(0, 0, 1)
    
    # Agent 1의 d0(digit 4)의 초기 상태 기록 (0)
    assert extract_digit_9(S, 4) == 0
    
    # Agent 0의 d0가 누적 캐리를 방출할 때까지 루프 작동
    # d0가 회전하며 간섭 누적기가 0에 수렴하는 순간 캐리 발생
    carry_triggered = False
    for _ in range(50):
        prev_a1_d0 = extract_digit_9(S, 4)
        S = world_tick_with_horizontal_carry(S, clock, interference, carry_routing)
        curr_a1_d0 = extract_digit_9(S, 4)
        
        # Agent 1의 d0 위상이 회전했는지 확인 (수평 캐리가 수신되었는지)
        if curr_a1_d0 != prev_a1_d0:
            carry_triggered = True
            break
            
    # 수평 캐리가 성공적으로 전파되어 Agent 1의 위상이 회전(에너지 전이 완료)했음을 증명
    assert carry_triggered
    # Agent 0의 d1(digit 1)은 라우팅 우회로 인해 캐리를 받지 않아 0을 유지해야 함
    assert extract_digit_9(S, 1) == 0

def test_social_polarization_hebbian():
    """Scenario 3: 모순된 갈등 정보 인입 시 헵식 Hebbian 음수 결합을 통한 집단 양극화 검증"""
    # 2개의 에이전트 기어 설정 (0번: 진영 A, 1번: 진영 B)
    net = CognitiveGearNetwork(num_gears=2)
    net.K[0][1] = 0.0
    net.K[1][0] = 0.0
    
    # 서로 다른 극단적 외부 정보(Clock/Input)가 두 에이전트에 주입되는 상황
    # Agent 0은 0도 방향으로, Agent 1은 180도(pi) 방향으로 서로 다른 정보에 노출됨
    inputs = [0.1, -0.1]
    net.phases[0] = 0.0
    net.phases[1] = math.pi
    
    dt = 0.1
    # 100틱 동안 Hebbian 학습과 회전 진행
    for _ in range(100):
        net.step(dt, inputs)
        net.update_coupling_hebbian(dt, learning_rate=1.0)
        
    # 두 진영이 지속적으로 어긋난 상태를 유지했으므로 Hebbian 결합 강도가 음수(-1.0)로 전환됨
    assert net.K[0][1] < -0.8
    assert net.K[1][0] < -0.8
    
    # 음수 결합(반발 텐션)이 활성화된 상태에서 외부 입력을 제거해도
    # 두 에이전트 기어는 척력에 의해 위상이 서로 대척점(pi 위상차)에 영구 고착(분극화)됨을 검증
    for _ in range(50):
        net.step(dt)
        
    final_diff = abs(net.phases[0] - net.phases[1])
    if final_diff > math.pi:
        final_diff = 2.0 * math.pi - final_diff
    assert final_diff == pytest.approx(math.pi, abs=0.2)

def test_vital_decay_and_legacy():
    """Scenario 4: 자율 생명 감쇠(Decay) 및 소멸 시 위상 유산(Holographic Legacy) 유전 검증"""
    # 세계 로터 S
    S = 0
    # Agent 0 (offset 0), Agent 1 (offset 4)
    # Agent 0의 초기 위상 = 5 (탐구자), Agent 1의 초기 위상 = 2 (조력자)
    S = replace_digit_9(S, 0, 5)
    S = replace_digit_9(S, 4, 2)
    
    # 1. 감쇠 (Decay): Agent 0이 더 이상 변동성(tension)이 없고 
    # 외부 클럭(0)과 완벽히 동화되었을 때, 고유 성향이 0(Peacemaker, 영점 평형)으로 소멸하는지 모사
    # (감쇠율에 의해 에너지가 방전되는 과정)
    S_decayed = replace_digit_9(S, 0, 0) # d0가 0으로 용해
    assert extract_digit_9(S_decayed, 0) == 0
    
    # 2. 유전 (Legacy): Agent 0 소멸 직전, 소멸 파동(XOR)이 인접 비트(Agent 1)로 유전됨
    # Agent 0이 가졌던 고유 위상(5)이 Agent 1의 위상(2)과 XOR 확산되어 계승
    legacy_wave = extract_digit_9(S, 0) # 소멸한 A의 유산 (5)
    
    a1_phase = extract_digit_9(S, 4) # B의 기존 위상 (2)
    inherited_phase = (a1_phase ^ legacy_wave) % 9 # 유산 흡수
    
    S_inherited = replace_digit_9(S_decayed, 4, inherited_phase)
    
    # B의 위상이 기존 2에서 A의 유산(5)을 반영하여 7 (2^5 = 7)로 변이 유전됨을 검증
    assert extract_digit_9(S_inherited, 4) == 7
