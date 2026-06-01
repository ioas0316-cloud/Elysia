import math
from typing import Tuple
import numpy as np
from core.math_utils import Quaternion
from core.linguistic_axiom import LinguisticAxiomFilter
from core.cuda_kernel import execute_semantic_cuda_bypass

class SentenceWaveGate:
    """
    [Phase 96] 자생적 문장-파동 동조 게이트 (Autopoietic Sentence-to-Wave Gate)
    
    기존의 하드코딩된 '주파수 닻(Frequency Anchors)'을 완벽히 폐기하고,
    엘리시아의 홀로그램 기억망(HologramMemory)에 맺힌 프랙탈 로터들의 텐션(Tau)을
    동적으로 참조하여 스스로 문장의 의미론적 주파수(Context Frequency)를 창발시킵니다.
    """
    def __init__(self, memory=None, sample_points: int = 100):
        self.memory = memory  # [Phase 96] 하드코딩 딕셔너리 대신 기억망 자체를 참조
        self.sample_points = sample_points

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

        # 2. [Phase 96] 프랙탈 기억망에 기반한 자생적 주파수 분석 (Autopoietic Frequency Emergence)
        tokens = sentence.lower().split()
        target_frequency = 1.0  # 기본 기저 주파수
        frequency_matched = False

        if self.memory and hasattr(self.memory, 'ui_concept_map'):
            for token in tokens:
                for word, node in self.memory.ui_concept_map.items():
                    # 단어가 엘리시아의 기억망에 존재한다면, 해당 노드의 텐션(Tau)을 주파수로 승화
                    if word.lower() in token:
                        target_frequency = max(0.5, 1.0 + abs(node.tau) * 0.3)
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
            # 상쇄 간섭: 아는 단어가 없어 매칭되지 않은 경우 (혼란의 고주파)
            wave = np.sin(2 * np.pi * 45.0 * t) + np.random.normal(0, 0.2, self.sample_points)

        # 파동 정규화 [-1.0, 1.0]
        max_abs = np.max(np.abs(wave))
        if max_abs > 0:
            wave /= max_abs

        return sentence_rotor, wave
