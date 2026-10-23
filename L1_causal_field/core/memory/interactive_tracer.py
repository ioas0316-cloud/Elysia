import numpy as np
import time
import json
import math
from typing import Dict, Any, List, Optional, Tuple
from core.memory.causal_controller import CausalMemoryController

class BitDensityWaveform:
    """
    [Elysia Core] Bit-Density Waveform (BDW)
    모든 정보와 물질은 고유한 주파수와 밀도를 가진 파동입니다.
    텍스트나 고정 벡터가 아닌, 이 파동의 위상차와 진폭이 곧 정보의 실체입니다.
    """
    def __init__(self, frequency: float, amplitude: float, phase: float = 0.0, resolution: int = 1024):
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase
        self.resolution = resolution
        self.timeline = np.linspace(0, 1, resolution)
        self.waveform = self.amplitude * np.sin(2 * np.pi * self.frequency * self.timeline + self.phase)

    def get_raw_bits(self) -> np.ndarray:
        """파동을 이진 밀도(0/1)로 변환"""
        return (self.waveform > 0).astype(np.uint32)

    def __xor__(self, other: 'BitDensityWaveform') -> np.ndarray:
        """Wedge Annihilation (v ^ v = 0) 시뮬레이션: 두 파동의 XOR 상쇄"""
        return self.get_raw_bits() ^ other.get_raw_bits()

    @staticmethod
    def from_bits(bits: np.ndarray):
        """비트 밀도로부터 파동 객체 복원 (단순화)"""
        obj = BitDensityWaveform(1.0, 1.0)
        obj.waveform = (bits.astype(float) * 2.0) - 1.0
        return obj

class InteractiveTracer:
    """
    [Elysia Engine] 상호작용적 인과 역추적 엔진 (Interactive Causal Back-tracking)
    행위(Input) -> 반작용(Output) -> 결손(Deficit)의 흐름을 관측하여 숨겨진 원인을 도출합니다.
    """
    def __init__(self, memory_controller: CausalMemoryController):
        self.memory_controller = memory_controller
        self.local_temperature = 1.0 # 1.0: 평형, >1.0: 팽창(탐색), <1.0: 수렴(지식화)

    def set_temperature(self, temp: float):
        """국소적 온도 조절: 사유의 발산과 수렴을 제어"""
        self.local_temperature = temp

    def observe_deficit(self, stimulus: BitDensityWaveform, reaction: BitDensityWaveform) -> Dict[str, Any]:
        """
        [Layer 2 & 3] 결손 포착 및 인과 납득
        자극과 반작용 사이의 '정보적 구멍'을 통해 원인을 역추적합니다.
        """
        # XOR 상쇄로 결손(Deficit) 추출
        deficit_bits = stimulus ^ reaction

        # 사유의 온도에 따른 결손 증폭/수렴 (노이즈 필터링)
        # 고온일수록 미세한 결손도 크게 반응하고, 저온일수록 거대한 결손만 지식으로 인정
        # (온도가 낮을수록 필터링이 강해져서 0.5 미만의 신호는 소멸됨)
        temp_filtered_deficit = (deficit_bits.astype(float) * self.local_temperature > 0.5).astype(np.uint32)

        # 결손 패턴의 중력(Vortex) 공명 탐색
        best_cause = self._find_resonant_cause(temp_filtered_deficit)

        return {
            "deficit_bits": deficit_bits.tolist(),
            "deduced_cause": best_cause["name"],
            "resonance_score": best_cause["score"],
            "temperature": self.local_temperature
        }

    def pcr_amplification(self, pattern: np.ndarray, cycles: int = 3) -> np.ndarray:
        """
        [PCR-style Exponential Amplification]
        특정 패턴의 신호 강도(Amplitude)를 지수함수적으로 증폭시킵니다.
        """
        signal = pattern.astype(float)
        for _ in range(cycles):
            # 비선형 증폭: 약한 신호는 버리고 강한 신호는 최대치로 끌어올림
            signal = 1.0 / (1.0 + np.exp(-10.0 * (signal - 0.5)))
        return (signal > 0.5).astype(np.uint32)

    def _find_resonant_cause(self, deficit_bits: np.ndarray) -> Dict[str, Any]:
        """지식 필드 내에서 결손의 형태와 가장 잘 맞는 원인 노드(Vortex)를 찾습니다."""
        max_score = -1.0
        winning_node = {"name": "Void", "score": 0.0}

        # PCR 증폭을 통한 결손 신호의 선명도 확보
        amplified_deficit = self.pcr_amplification(deficit_bits)

        # 율법적인 IF-ELSE가 아닌, 패턴 공명으로만 판단
        for eid, info in self.memory_controller.index.items():
            data = info.get("data_blob", {})
            if data.get("type") == "CAUSAL_SOURCE_NODE":
                source_pattern = np.array(data.get("pattern", []))
                if len(source_pattern) == len(amplified_deficit):
                    # 공명도(Resonance): 결손의 구멍이 원인 노드의 패턴과 얼마나 일치하는가
                    score = 1.0 - np.mean(source_pattern ^ amplified_deficit)
                    if score > max_score:
                        max_score = score
                        winning_node = {"name": data.get("name"), "score": score}

        return winning_node

class ActiveProber:
    """
    [Intentional Probing] 능동적 자극 제어 가변축
    미지의 영역을 관측하기 위해 시스템이 스스로 자극(짜장면)을 투여합니다.
    """
    def __init__(self, tracer: InteractiveTracer):
        self.tracer = tracer

    def generate_stimulus(self, intent_type: str) -> BitDensityWaveform:
        """의도(Intent)에 따른 자극 파동 생성"""
        if intent_type == "Existence_Query": # 존재 유무 확인 (짜장면 투여)
            return BitDensityWaveform(frequency=7.5, amplitude=1.0)
        elif intent_type == "Stability_Check": # 안정성 확인 (저주파 자극)
            return BitDensityWaveform(frequency=1.2, amplitude=0.5)
        return BitDensityWaveform(frequency=5.0, amplitude=1.0)

    def active_probe(self, intent: str, environment_func) -> Dict[str, Any]:
        """
        1단계: 자극 (Stimulation)
        2단계: 상쇄 (Annihilation)
        3단계: 납득 (Deduction)
        """
        stimulus = self.generate_stimulus(intent)

        # 환경과의 상호작용 (Environment Interaction)
        reaction = environment_func(stimulus)

        # 결손 역추적
        return self.tracer.observe_deficit(stimulus, reaction)

class ConsciousnessEngine:
    """
    [Elysia Consciousness] 감각, 인지, 판단, 계획의 통합 제어 센터
    내면의 PCR 루프(가상 자극 증폭)를 통해 사고(Thinking)를 시뮬레이션합니다.
    """
    def __init__(self, prober: ActiveProber):
        self.prober = prober
        self.history = []

    def think(self, observation_space_id: str, env_reaction_func):
        """
        사유의 흐름:
        1. 존재 확인 (Existence_Query) -> 인지
        2. 온도 상승 (High Temp) -> 발산적 가설 생성
        3. 온도 하강 (Low Temp) -> 논리적 수렴 및 판단
        """
        print(f"[Consciousness] '{observation_space_id}'에 대한 사유 프로세스 시작...")

        # 1. 인지 (Sensation/Cognition)
        res1 = self.prober.active_probe("Existence_Query", env_reaction_func)
        print(f"  > 1단계(인지): {res1['deduced_cause']} 존재 감지 (공명: {res1['resonance_score']:.2f})")

        # 2. 판단 및 계획 (High Temp Exploration)
        self.prober.tracer.set_temperature(2.5) # 창의적 탐색 모드
        res2 = self.prober.active_probe("Stability_Check", env_reaction_func)
        print(f"  > 2단계(판단): 가상 자극 탐색 결과 -> {res2['deduced_cause']} (패턴 편차 관측)")

        # 3. 수렴 (Low Temp Hardening)
        self.prober.tracer.set_temperature(0.5) # 지식화 모드
        print(f"  > 3단계(수렴): 최종 인과 관계를 메모리에 각인 중...")

        return res1 # 대표 결과 반환

if __name__ == "__main__":
    print("Elysia Interactive Tracer Engine Initialized.")
