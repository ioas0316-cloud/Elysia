# core/world_hyper_rotor.py
# Copyright 2026 Lee Kang-deok & Antigravity
# Architecture: World Hyper-Rotor & Cascading Phase Carry

from core.enneagram_phase_topology import NUM_SCALES
from core.scale_observer import extract_digit_9, replace_digit_9

UNIVERSE_MASK = 0xFFFFFFFFFFFFFFFF  # 64-bit mask

def world_tick(S: int, clock: int) -> int:
    """
    가장 단순하고 정적인 틱 회전. 
    1틱 동안 하드웨어 클럭 엣지를 수접하여 각 자릿수별 XOR 및 9진법 경계 폴딩을 처리합니다.
    """
    new_S = 0
    # 64비트 정수의 16개 자릿수(nibble) 전체에 대해 틱 회전 처리
    for scale in range(16):
        s_digit = (S >> (scale * 4)) & 0xF
        c_digit = (clock >> (scale * 4)) & 0xF
        new_digit = (s_digit ^ c_digit) % 9
        new_S |= (new_digit << (scale * 4))
    return new_S & UNIVERSE_MASK

def world_tick_with_phase_carry(S: int, clock: int, interference: list[int]) -> int:
    """
    물리적 간섭 누적을 통한 비결정론적 캐리 전파가 적용된 세계 틱 진화 엔진.
    """
    if len(interference) < 16:
        raise ValueError("Interference array must have at least 16 elements")
        
    S = world_tick(S, clock)
    
    for scale in range(16):
        digit = (S >> (scale * 4)) & 0xF
        
        # 현재 자릿수를 간섭 누적기에 XOR 누적 (Modulo 9 보정)
        interference[scale] = (interference[scale] ^ digit) % 9
        
        # 간섭 누적이 0에 수렴 -> 상위 자릿수로 회전 에너지 방전
        if interference[scale] == 0:
            if scale + 1 < 16:
                parent_digit = (S >> ((scale + 1) * 4)) & 0xF
                parent_digit = (parent_digit + 1) % 9
                S = (S & ~(0xF << ((scale + 1) * 4))) | (parent_digit << ((scale + 1) * 4))
                
            interference[scale] = digit if digit != 0 else 1
            
    return S & UNIVERSE_MASK

def world_tick_with_horizontal_carry(
    S: int, 
    clock: int, 
    interference: list[int], 
    carry_routing: dict[tuple[int, int], tuple[int, int]] = None
) -> int:
    """
    수평적 캐리 전파(Horizontal Carry Propagation)가 지원되는 세계 틱 진화 엔진.
    64비트 상태 S는 4개의 자릿수씩 쪼개져 총 4개의 관측 오프셋(에이전트)을 구성합니다:
      - Agent 0: offset 0 (d0 ~ d3)
      - Agent 1: offset 4 (d4 ~ d7)
      - Agent 2: offset 8 (d8 ~ d11)
      - Agent 3: offset 12 (d12 ~ d15)
    
    carry_routing: {(src_offset, src_scale): (tgt_offset, tgt_scale)}
      특정 오프셋/스케일에서 방전된 에너지를 타 오프셋/스케일로 수평 전달하여 위상 거래(경제)를 유도합니다.
    """
    if len(interference) < 16:
        raise ValueError("Interference array must have at least 16 elements")
        
    S = world_tick(S, clock)
    
    if carry_routing is None:
        carry_routing = {}
        
    for digit_idx in range(16):
        digit = (S >> (digit_idx * 4)) & 0xF
        
        # 간섭 누적기에 XOR 누적
        interference[digit_idx] = (interference[digit_idx] ^ digit) % 9
        
        if interference[digit_idx] == 0:
            # 캐리 발생!
            # 현재 자릿수의 오프셋(0, 4, 8, 12)과 스케일(0~3) 역산
            offset = (digit_idx // 4) * 4
            scale = digit_idx % 4
            
            route_key = (offset, scale)
            if route_key in carry_routing:
                # 1. 수평 캐리 전파 (지정된 타 오프셋의 자릿수로 우회 전송)
                target_offset, target_scale = carry_routing[route_key]
                target_digit_idx = target_offset + target_scale
                
                parent_digit = (S >> (target_digit_idx * 4)) & 0xF
                parent_digit = (parent_digit + 1) % 9
                S = (S & ~(0xF << (target_digit_idx * 4))) | (parent_digit << (target_digit_idx * 4))
            else:
                # 2. 일반적 수직 캐리 전파 (동일 오프셋 내부의 상위 자릿수로 전파)
                if scale + 1 < 4:
                    target_digit_idx = offset + scale + 1
                    parent_digit = (S >> (target_digit_idx * 4)) & 0xF
                    parent_digit = (parent_digit + 1) % 9
                    S = (S & ~(0xF << (target_digit_idx * 4))) | (parent_digit << (target_digit_idx * 4))
                    
            # 간섭 누적기 리셋
            interference[digit_idx] = digit if digit != 0 else 1
            
    return S & UNIVERSE_MASK
