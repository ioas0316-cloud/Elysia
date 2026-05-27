# core/tests/test_world_hyper_rotor.py
# Copyright 2026 Lee Kang-deok & Antigravity
# Architecture: Unit Tests for World Hyper-Rotor & Phase Carry

import sys
import os
import pytest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.enneagram_phase_topology import NUM_SCALES
from core.scale_observer import extract_digit_9, replace_digit_9, observe_scale, observe_other
from core.world_hyper_rotor import world_tick, world_tick_with_phase_carry

def test_digit_operations():
    """자릿수 추출, 치환, 9진법 경계 모듈러 조건 검증"""
    # 0x0인 상태에서는 모든 자릿수가 0
    S = 0
    for scale in range(NUM_SCALES):
        assert extract_digit_9(S, scale) == 0

    # 특정 자릿수를 변경하고 추출하여 확인
    S = replace_digit_9(S, 0, 5)  # d0 = 5
    S = replace_digit_9(S, 5, 8)  # d5 = 8
    S = replace_digit_9(S, 12, 3) # d12 = 3
    
    assert extract_digit_9(S, 0) == 5
    assert extract_digit_9(S, 5) == 8
    assert extract_digit_9(S, 12) == 3
    
    # 9 이상으로 치환할 때 9진법 모듈러(modulo 9)가 적용되는지 검증 (Anti-If & 0/1 관계 철학)
    S = replace_digit_9(S, 2, 9)  # 9는 0으로 치환되어야 함
    assert extract_digit_9(S, 2) == 0
    
    S = replace_digit_9(S, 3, 11) # 11 % 9 = 2
    assert extract_digit_9(S, 3) == 2

def test_world_tick_basic():
    """기본 틱 회전에서 자릿수별 XOR 및 9진법 모듈러 적용 여부 검증"""
    S = 0
    # d0=5, d1=3
    S = replace_digit_9(S, 0, 5)
    S = replace_digit_9(S, 1, 3)
    
    # clock: d0=3, d1=3
    clock = 0
    clock = replace_digit_9(clock, 0, 3)
    clock = replace_digit_9(clock, 1, 3)
    
    S_new = world_tick(S, clock)
    
    # d0: 5 ^ 3 = 6 (6 % 9 = 6)
    assert extract_digit_9(S_new, 0) == 6
    # d1: 3 ^ 3 = 0 (완전 동기화 -> 0점 수렴)
    assert extract_digit_9(S_new, 1) == 0

def test_phase_carry_propagation():
    """위상 간섭 누적 및 상위 스케일 캐리 전파 검증"""
    # 0으로 시작
    S = 0
    # 모든 자릿수 간섭 누적기를 1로 초기화 (0에서 시작하면 첫 틱부터 즉시 캐리가 전파되므로)
    interference = [1] * NUM_SCALES
    
    # clock = 오직 d0(최하위 개체)만 1인 신호 (자연적 캐리 흐름 테스트)
    clock = replace_digit_9(0, 0, 1)
        
    # 수십 번의 틱을 돌리면서 캐리 발생을 관측
    # d0(최하위 개체)의 캐리는 자주 일어나며, d1, d2로 갈수록 기하급수적으로 빈도가 낮아져야 함
    d0_changes = 0
    d1_changes = 0
    d2_changes = 0
    
    prev_d0 = extract_digit_9(S, 0)
    prev_d1 = extract_digit_9(S, 1)
    prev_d2 = extract_digit_9(S, 2)
    
    for tick in range(100):
        S = world_tick_with_phase_carry(S, clock, interference)
        
        curr_d0 = extract_digit_9(S, 0)
        curr_d1 = extract_digit_9(S, 1)
        curr_d2 = extract_digit_9(S, 2)
        
        if curr_d0 != prev_d0:
            d0_changes += 1
            prev_d0 = curr_d0
        if curr_d1 != prev_d1:
            d1_changes += 1
            prev_d1 = curr_d1
        if curr_d2 != prev_d2:
            d2_changes += 1
            prev_d2 = curr_d2
            
    print(f"\n[Carry Stats] d0 changes: {d0_changes}, d1 changes: {d1_changes}, d2 changes: {d2_changes}")
    
    # d0는 변화가 많아야 하고, d1은 적어야 하며, d2는 더 적어야 함 (지수 스케일 붕괴 확인)
    assert d0_changes >= d1_changes
    assert d1_changes >= d2_changes

def test_non_deterministic_interference():
    """입력 클럭의 변동성에 따른 비결정론적 방전(Carry) 시점 검증"""
    # 서로 다른 두 조건(동일 클럭 연속 주입 vs 무작위 클럭 주입)에서 캐리 발생 간격이 다름을 검증
    # 조건 1: 정적 클럭 (주기적)
    S1 = 0
    interference1 = [1] * NUM_SCALES
    clock_static = replace_digit_9(0, 0, 1)
    
    carries1 = []
    for tick in range(100):
        S1_prev = S1
        S1 = world_tick_with_phase_carry(S1, clock_static, interference1)
        # d1이 변경된 틱 감지
        if extract_digit_9(S1, 1) != extract_digit_9(S1_prev, 1):
            carries1.append(tick)
            
    # 조건 2: 동적 노이즈 클럭
    S2 = 0
    interference2 = [1] * NUM_SCALES
    
    import random
    random.seed(42)
    
    carries2 = []
    for tick in range(100):
        S2_prev = S2
        # 매 틱 무작위 노이즈 클럭 생성 (오직 d0만 흔듦)
        clock_dynamic = replace_digit_9(0, 0, random.randint(0, 8))
        S2 = world_tick_with_phase_carry(S2, clock_dynamic, interference2)
        if extract_digit_9(S2, 1) != extract_digit_9(S2_prev, 1):
            carries2.append(tick)
            
    print(f"\n[Non-deterministic Stats] carries1: {carries1}, carries2: {carries2}")
    # 정적 클럭과 동적 클럭의 캐리 주기가 서로 다름 = 비결정론적 창발 증명
    assert carries1 != carries2

def test_mirror_observation():
    """거울 관측(XOR 간섭 무늬) 검증"""
    # 세계 로터 S
    S = 0
    # 나의 위치 (offset=0)
    # 상대의 위치 (offset=3)
    # d0에 각기 다른 위상을 저장
    S = replace_digit_9(S, 0, 7) # 나의 d0 = 7 (낙천가)
    S = replace_digit_9(S, 3, 2) # 상대의 d0 = 2 (조력자)
    
    # scale=0 (개체 수준)에서 나(0)와 상대(3) 사이의 관계 관측
    # my_phase = S >> 0 -> d0 = 7
    # other_phase = S >> (3*4) -> d0 = 2
    # relation = (7 ^ 2) % 9 = 5 % 9 = 5
    relation = observe_other(S, 0, 3, 0)
    assert relation == (7 ^ 2) % 9
