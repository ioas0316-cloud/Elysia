"""
Elysia Core Architecture: Bidirectional Phase Negotiation (Sensory Grounding Protocol)

This module implements the continuous phase negotiation loop between internal prediction
waves and external sensory streams, computing phase alignment errors (q_err) and
dynamically adjusting boundary impedance and predictive resonance to achieve Sensory Grounding.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class SensoryWaveStream:
    """외부 자극 스트림 및 내부 예측 파동 상태"""
    frequency: float = 1.0
    phase: float = 0.0
    amplitude: float = 1.0
    chromatic_vector: List[float] = field(default_factory=lambda: [0.33, 0.33, 0.34])  # Flux, Order, Entropy


class BidirectionalPhaseNegotiator:
    """
    내부 예측 파동과 외부 스트림 간의 쌍방향 위상 협상 (Sensory Grounding) 엔진.
    마우스 클릭, 소리, 유체 저항 등의 외부 스트림을 단순한 트리거가 아닌
    내부 격자와 부딪히며 위상 오차(q_err)를 최소화하는 인과적 공명 프로토콜로 다룸.
    """

    def __init__(
        self,
        learning_rate: float = 0.15,
        coupling_strength: float = 0.8,
        damping_factor: float = 0.05
    ):
        self.learning_rate = learning_rate
        self.coupling_strength = coupling_strength
        self.damping_factor = damping_factor

        # 내부 파동 상태
        self.internal_wave = SensoryWaveStream(frequency=1.0, phase=0.0, amplitude=1.0)
        # 위상 오차 및 공명도 기록
        self.q_err: float = 0.0
        self.resonance_score: float = 0.0
        self.phase_history: List[float] = []

    def compute_phase_error(self, external_wave: SensoryWaveStream) -> float:
        """
        내부 파동과 외부 파동 간의 위상 차이(q_err) 계산 (-pi ~ +pi 사이로 정규화)
        """
        diff = (external_wave.phase - self.internal_wave.phase) % (2.0 * math.pi)
        if diff > math.pi:
            diff -= 2.0 * math.pi
        self.q_err = diff
        return self.q_err

    def step_negotiation(
        self,
        external_wave: SensoryWaveStream,
        time_delta: float = 0.1
    ) -> Dict[str, float]:
        """
        한 스텝의 쌍방향 위상 협상 실행.
        1) 위상 오차(q_err) 산출
        2) 내부 예측 파동의 주파수 및 위상 미세 조정
        3) 공명 에너지 효율 계산 (R = cos(q_err) * exp(-damping * |q_err|))
        """
        # 1. 위상 오차 계산 (현재 상태에서의 차이)
        q_err = self.compute_phase_error(external_wave)

        # 2. 주파수 및 위상 업데이트 (Kuramoto 모델 기반 인과적 위상 끌림/동기화)
        freq_diff = external_wave.frequency - self.internal_wave.frequency
        self.internal_wave.frequency += self.learning_rate * freq_diff

        # 위상 피드백 조정
        phase_adjustment = self.coupling_strength * math.sin(q_err)
        self.internal_wave.phase += (self.internal_wave.frequency * time_delta) + phase_adjustment
        self.internal_wave.phase %= (2.0 * math.pi)

        # 재계산 후의 오차 파악
        self.compute_phase_error(external_wave)

        # 진폭 적응 (외부와 내부 진폭 맞춤)
        amp_diff = external_wave.amplitude - self.internal_wave.amplitude
        self.internal_wave.amplitude += self.learning_rate * amp_diff * time_delta

        # 3. 공명 효율 점수 계산 (1.0 = 완벽한 공명, 0.0 = 비공명)
        self.resonance_score = max(0.0, math.cos(self.q_err)) * math.exp(-self.damping_factor * abs(self.q_err))
        self.phase_history.append(self.resonance_score)
        if len(self.phase_history) > 100:
            self.phase_history.pop(0)

        return {
            "q_err": self.q_err,
            "resonance_score": self.resonance_score,
            "internal_phase": self.internal_wave.phase,
            "internal_freq": self.internal_wave.frequency,
            "internal_amplitude": self.internal_wave.amplitude,
        }

    def is_phase_locked(self, threshold: float = 0.15, min_history_len: int = 10) -> bool:
        """
        위상 고정(Phase-Lock) 상태 달성 여부 확인.
        최근 위상 오차가 threshold 이하로 일정 기간 유지되었는가?
        """
        if len(self.phase_history) < min_history_len:
            return False
        recent_scores = self.phase_history[-min_history_len:]
        avg_resonance = sum(recent_scores) / len(recent_scores)
        return avg_resonance >= (1.0 - threshold)
