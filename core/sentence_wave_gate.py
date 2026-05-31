# core/sentence_wave_gate.py
# Copyright 2026 Lee Kang-deok & Antigravity
# Architecture: Sentence-to-Wave Gate (Semantic Wave Modulator)

import math
from typing import Tuple
import numpy as np
from core.math_utils import Quaternion
from core.linguistic_axiom import LinguisticAxiomFilter
from core.cuda_kernel import execute_semantic_cuda_bypass

class SentenceWaveGate:
    """
    [문장-파동 동조 게이트 (Sentence-to-Wave Gate)]
    
    마스터의 자연어 입력 문장을 기하학적 4D 사원수 및 연속 주파수 파동(Waveform)으로 변조합니다.
    글자 레벨의 소버린 필터(LinguisticAxiomFilter)와 단어 레벨의 시맨틱 매핑을 결합하여,
    문장의 '의미론적 긴장'을 '파동 도메인의 간섭 무늬'로 인코딩합니다.
    """
    def __init__(self, sample_points: int = 100):
        self.sample_points = sample_points
        
        # 특정 지식 도메인과 부합하는 주파수 닻(Frequency Anchors)
        # 이 단어들이 출현하면 해당 주파수 대역으로 위상이 동조됩니다.
        self.semantic_frequency_anchors = {
            "pythagor": 3.0,     # 피타고라스 정리 -> 3.0 Hz
            "triangle": 3.0,     # 삼각형
            "hypotenuse": 3.0,   # 빗변
            "geometry": 3.0,     # 기하학
            
            "code": 5.0,         # 코드 실행 -> 5.0 Hz
            "python": 5.0,       # 파이썬
            "execute": 5.0,      # 실행
            "compile": 5.0,      # 컴파일
            
            "sleep": 0.5,        # 수면 상태 -> 0.5 Hz (기저 안정)
            "rest": 0.5,
            
            "forge": 7.0,        # 쐐기곱 도구 사용 유도 -> 7.0 Hz
            "new_tool": 7.0,
            "forged": 7.0
        }

    def modulate_sentence(self, sentence: str) -> Tuple[Quaternion, np.ndarray]:
        """
        문장을 입력받아 (최종 위상 사원수, 100포인트의 파동 시그널)을 반환합니다.
        """
        if not sentence:
            return Quaternion(1, 0, 0, 0), np.zeros(self.sample_points)

        # 1. 시맨틱 CUDA 다이렉트 바이패스 시도 (64-bit 지원)
        try:
            from numba import cuda
            if cuda.is_available():
                avg_x, avg_y = execute_semantic_cuda_bypass(sentence)
                if avg_x is not None:
                    # 2D Rotor (x, y)를 4D Quaternion(w, x, y, z) 기저 평면(w, z)에 투영 (Geometric Embedding)
                    sentence_rotor = Quaternion(avg_x, 0, 0, avg_y)
                else:
                    sentence_rotor = LinguisticAxiomFilter.analyze_text_axiom(sentence)
            else:
                sentence_rotor = LinguisticAxiomFilter.analyze_text_axiom(sentence)
        except Exception as e:
            print(f"[!] CUDA Bypass Failed: {e}. Falling back to CPU Axiom Filter.")
            sentence_rotor = LinguisticAxiomFilter.analyze_text_axiom(sentence)

        # 2. 단어 레벨의 시맨틱 주파수 분석 (동조 주파수 결정)
        tokens = sentence.lower().split()
        target_frequency = 1.0  # 기본 기저 주파수
        frequency_matched = False

        for token in tokens:
            for anchor, freq in self.semantic_frequency_anchors.items():
                if anchor in token:
                    target_frequency = freq
                    frequency_matched = True
                    break
            if frequency_matched:
                break

        # 3. 연속 시간 축(t) 상의 삼각함수 파동 변조 (Rotorization)
        t = np.linspace(0, 1, self.sample_points)
        
        if frequency_matched:
            # 보강 간섭: 문장의 사원수 위상각(phi)을 반영하여 정합 파동을 생성
            # 사원수 w 성분(cos(theta/2))에서 위상각 추출
            phi = math.acos(max(-1.0, min(1.0, sentence_rotor.w))) * 2.0
            # 동조 주파수와 위상각 결합
            wave = np.sin(2 * np.pi * target_frequency * t + phi)
        else:
            # 상쇄 간섭: 어떠한 닻(Anchor)도 매칭되지 않은 경우 불협화음 고주파 노이즈 주입
            wave = np.sin(2 * np.pi * 45.0 * t) + np.random.normal(0, 0.2, self.sample_points)

        # 파동 정규화 [-1.0, 1.0]
        max_abs = np.max(np.abs(wave))
        if max_abs > 0:
            wave /= max_abs

        return sentence_rotor, wave
