# core/scale_observer.py
# Copyright 2026 Lee Kang-deok & Antigravity
# Architecture: Multi-scale Base-9 Enneagram Observer

from core.enneagram_phase_topology import ENNEAGRAM_TYPES, SCALE_NAMES, NUM_SCALES

def extract_digit_9(S: int, scale: int) -> int:
    """
    64비트 세계 로터 S에서 특정 스케일(0~12)의 9진법 자릿수를 추출합니다.
    각 자릿수는 4비트 nibble로 매핑되어 있으며, 0~8 범위를 유지하도록 9로 모듈러 연산합니다.
    """
    if scale < 0 or scale >= NUM_SCALES:
        raise ValueError(f"Scale must be between 0 and {NUM_SCALES - 1}")
    
    digit = (S >> (scale * 4)) & 0xF
    return digit % 9

def replace_digit_9(S: int, scale: int, new_digit: int) -> int:
    """
    세계 로터 S의 특정 스케일 자릿수를 새로운 값으로 치환하여 반환합니다.
    """
    if scale < 0 or scale >= NUM_SCALES:
        raise ValueError(f"Scale must be between 0 and {NUM_SCALES - 1}")
        
    shift = scale * 4
    # 64비트 마스크 적용하여 부호 유지 및 오버플로우 방지
    mask = ~(0xF << shift) & 0xFFFFFFFFFFFFFFFF
    sanitized_digit = (new_digit % 9) & 0xF
    return (S & mask) | (sanitized_digit << shift)

def observe_scale(S: int, scale: int) -> dict:
    """
    특정 스케일에서 세계 로터 S를 관측하여 해당하는 애니어그램 인격 및 스케일 메타데이터를 반환합니다.
    """
    digit = extract_digit_9(S, scale)
    type_info = ENNEAGRAM_TYPES[digit]
    
    return {
        "scale": scale,
        "scale_name": SCALE_NAMES.get(scale, f"Scale {scale}"),
        "type": type_info["type"],
        "name": type_info["name"],
        "description": type_info["description"]
    }

def observe_other(S: int, my_offset: int, other_offset: int, scale: int) -> int:
    """
    나(my_offset)의 관점(오프셋)에서 타자(other_offset)의 위상을 관측합니다.
    세계를 타자의 오프셋에서 읽되, 나의 위상과의 XOR 차이(간섭 무늬)를 반환합니다.
    """
    # 각 오프셋에서 세계를 관측하여 해당 스케일의 위상을 추출
    my_shifted = S >> (my_offset * 4)
    other_shifted = S >> (other_offset * 4)
    
    my_phase = extract_digit_9(my_shifted, scale)
    other_phase = extract_digit_9(other_shifted, scale)
    
    # 두 위상의 차이 = 관계의 텐션
    return (my_phase ^ other_phase) % 9
