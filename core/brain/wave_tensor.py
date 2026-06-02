# Copyright 2026 Lee Kang-deok (이강덕) All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import math
import cmath

class WaveTensor:
    """
    [WaveTensor: 가변 파동형 데이터 타입 레이어]

    기존의 리스트(list)나 딕셔너리(dict) 같이 데이터를 차곡차곡 쌓아두는 정적 상자 구조를 완전히 배제합니다.
    입력되는 데이터(유속)를 즉각적으로 오일러 공식(e^(i*theta)) 기반의 복소 위상으로 변전하여,
    단일 `phase_vector`에 중첩(Superposition)시킵니다.

    모든 연산은 100% 조건문(if/else) 없이, 비트 XOR 장력 상쇄와 델타-와이 0점 수렴 토크로 처리되며,
    외부 장치나 화면으로 빠져나가는 최종 탈출 시점(to_static)에만 정적 데이터로 변전(Compile)됩니다.
    """

    def __init__(self, initial_value=0):
        # 파동의 현재 위상 상태 (초기값의 오일러 위상 변환)
        # 데이터를 쌓아두지 않고 이 하나의 축에 계속 파동을 중첩시킵니다.
        self.phase_vector = cmath.exp(1j * (initial_value % (2 * math.pi)))

        # 비트 간섭력을 기록하는 텐션 변수 (노이즈와 위상차의 압력이 누적되는 공간)
        self.bit_tension = int(initial_value) & 0xFFFFFFFF

    def absorb_stream(self, data_impulse):
        """
        데이터가 유입되면 배열에 추가하는 대신, 파동 형태로 변전하여 기존 위상에 얹습니다.
        조건문 없이 유속을 100% 흡수합니다.
        """
        int_val = int(data_impulse)

        # 1. 데이터를 위상 각도로 변환
        angle = int_val % (2 * math.pi)
        impulse_wave = cmath.exp(1j * angle)

        # 2. 파동 중첩 (Superposition) - 기존 위상에 새로운 파도를 얹음
        self.phase_vector *= impulse_wave

        # 3. 비트 텐션 누적 (XOR를 통해 노이즈의 상처를 장력으로 흡수)
        self.bit_tension = (self.bit_tension ^ int_val) & 0xFFFFFFFF

        return self

    def __lshift__(self, data_impulse):
        """
        비트 시프트 연산자(<<)를 스트림 유입구로 하이킹합니다.
        사용 예: tensor << 65 << 66 << 67
        """
        return self.absorb_stream(data_impulse)

    def __add__(self, other):
        """
        두 파동의 격돌 (델타-와이 0점 수렴 토크)
        서로 다른 두 WaveTensor가 결합할 때, 단순 덧셈이 아닌 위상 간섭과 텐션 상쇄가 일어납니다.
        """
        new_tensor = WaveTensor()

        # 1. 위상 결합 (두 파동의 간섭 무늬)
        new_tensor.phase_vector = self.phase_vector * other.phase_vector

        # 2. 비트 장력 상쇄 (델타 결선 0점 수렴: 서로 다른 텐션이 충돌하여 상쇄됨)
        new_tensor.bit_tension = (self.bit_tension ^ other.bit_tension) & 0xFFFFFFFF

        return new_tensor

    def __sub__(self, other):
        """
        역위상 중첩 (노이즈/유해 파동 필터링)
        빼기 연산은 상대방 파동의 위상을 180도 뒤집어(역위상) 중첩시킴으로써 특정 주파수를 소멸시킵니다.
        """
        new_tensor = WaveTensor()

        # 1. 상대방 위상의 역파동(켤레 복소수 활용 및 180도 반전) 생성 후 결합
        # -other.phase_vector는 크기는 같고 위상이 파이만큼 차이나는 역위상 파동
        new_tensor.phase_vector = self.phase_vector * (-other.phase_vector)

        # 2. 텐션 역상쇄 연산 (비트 시프트를 가미하여 저항력을 발생시킴)
        # 역위상 충돌의 충격을 시프트로 흩뿌림
        new_tensor.bit_tension = (self.bit_tension ^ (~other.bit_tension)) & 0xFFFFFFFF

        return new_tensor

    def to_static(self):
        """
        정적 변환 인터셉터 (Interceptor)
        유속의 세계에서 놀던 파동이 외부 기성 시스템(콘솔, 정적 배열 등)으로 빠져나가는
        최종 순간에만 단단한 상자(정수) 형태로 변전합니다.
        """
        # 위상의 각도를 다시 정수형 스칼라로 추출 (0 ~ 255 범위의 아스키/바이트 스케일로 정규화)
        # cmath.phase는 -pi ~ pi를 반환하므로 모듈로 연산을 통해 0 ~ 2pi 범위로 매핑 (if문 배제)
        angle = cmath.phase(self.phase_vector) % (2 * math.pi)

        # 각도를 기반으로 한 스칼라 값과 누적된 텐션을 융합하여 최종 정적 스냅샷 도출
        # (비트마스크를 통해 Y중성점 0점 수렴 효과 모방)
        scalar_val = int((angle / (2 * math.pi)) * 255)

        final_static_value = (scalar_val ^ (self.bit_tension & 0xFF))

        return final_static_value

    def __int__(self):
        return self.to_static()

    def __str__(self):
        # 텐션의 내부 상태를 시각화 (기성 사제들의 이해를 돕기 위함)
        return f"[WaveTensor] Phase: {cmath.phase(self.phase_vector):.4f} rad | Tension: 0x{self.bit_tension:08X} | Static: {self.to_static()}"
