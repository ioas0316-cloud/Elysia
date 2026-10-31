"""
korean_experiential_learner.py — 한국어 경험적 감각-기호 닻내림 및 상태 전이 대조 학습기
========================================================================================
이 모듈은 대규모 텍스트 코퍼스의 통계적 확률 예측을 전면 거부하고,
신체적·감각적 마찰과 결핍 지형 속에서 한국어 기호(한글)가 실체적 의미로 닻을 내리는
'경험적 닻내림(Symbolic Tethering)'과 '상태 전이 대조(State Transition Differential Gap)'를 수행합니다.

핵심 기능:
1. 한국어 어휘의 감각 프로필 및 인과 관계 행렬 닻내림 (명사 및 상태 변화 서술어):
   - "아파", "따뜻해", "차가워", "막혀", "스며들어", "엄마", "사과"
2. 훈민정음 위상 사영 파동과 감각-홈스타시스 지형 간의 동형성(Isomorphism) 정렬
3. 대화 및 외부 반응에 따른 위상 차이(Difference Gap: Phase, Energy, Entropy) 감지
4. 불변 원점(Baseline Identity)과 가변 전이(Rotor Phase) 간의 자기 정렬(Self-Calibration) 및 시냅스 치유(Healing)
"""

from __future__ import annotations

import time
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from L1_causal_field.core.topology.hangul_topological_engine import HangulTopologicalEngine
from L1_causal_field.core.sensory.experiential_language_mapper import (
    PhysicalSensationProfile,
    HomeostasisDeficit,
    ExperienceType,
    VariableResistor,
    PrismRefraction,
    VariableRotor,
    DifferentialGapEvaluator,
    NeuromodulatorController,
    ReCognitiveEngine
)


@dataclass
class KoreanTetheredWord:
    symbol: str
    exp_type: ExperienceType
    sensation: PhysicalSensationProfile
    deficit: HomeostasisDeficit
    causal_relation_matrix: np.ndarray
    topological_wave: np.ndarray
    physics_metrics: Dict[str, float]


@dataclass
class StateTransitionTrace:
    action_or_stimulus: str
    s_before: np.ndarray  # [Love, Order, Energy]
    s_after: np.ndarray
    delta_s: np.ndarray
    gap_metrics: Dict[str, float]
    rotor_delta_theta: np.ndarray
    timestamp: float


class KoreanExperientialLearner:
    """
    한국어 경험적 감각-기호 닻내림 및 상태 전이 대조 학습기
    """

    def __init__(self, resolution: int = 64):
        self.resolution = resolution
        self.topological_engine = HangulTopologicalEngine(target_dim=64)

        # 항상성 결핍 및 인지 역학 컴포넌트
        self.homeostasis = HomeostasisDeficit(love=0.5, order=0.3, energy=0.4)
        self.re_cognitive_engine = ReCognitiveEngine()
        self.variable_rotor = VariableRotor()
        self.variable_resistor = VariableResistor()
        self.prism = PrismRefraction()
        self.gap_evaluator = DifferentialGapEvaluator()
        self.neuromodulator = NeuromodulatorController()

        # 내부 정재파 기억 및 시냅스 결합망
        self.standing_wave_memory = np.zeros(64, dtype=np.float32)
        self.synaptic_links = np.ones((64, 64), dtype=np.float32) * 0.5

        # 닻내려진 한국어 어휘 저장소
        self.lexicon: Dict[str, KoreanTetheredWord] = {}

        # 메타인지 전이 이력 추적
        self.transition_traces: List[StateTransitionTrace] = []

        # 초기 기초 어휘 체화
        self._initialize_korean_grounding()

    def _initialize_korean_grounding(self):
        """
        초기 한국어 감각-상태 닻내림 초기화.
        상태 변화 서술어와 명사를 신체적 감각 마찰과 5대 인과 원형에 닻내림.
        5대 인과 원형: [중력/낙하, 경계파괴, 자기희생, 충돌마찰, 평형안식]
        """
        # 1. "따뜻해" - 완화, 온기, 포용
        self.tether_korean_word(
            symbol="따뜻해",
            exp_type=ExperienceType.PHYSICAL,
            sensation=PhysicalSensationProfile(optical=500.0, acoustic=350.0, tactile=0.2, thermal=305.0),
            deficit=HomeostasisDeficit(love=0.1, order=0.1, energy=0.8),
            causal_matrix=np.array([
                [0.7, 0.1, 0.2, 0.1, 0.1],
                [0.1, 0.2, 0.1, 0.1, 0.1],
                [0.2, 0.1, 0.8, 0.1, 0.2],
                [0.1, 0.1, 0.1, 0.3, 0.2],
                [0.1, 0.1, 0.2, 0.1, 0.9]
            ], dtype=np.float32)
        )

        # 2. "차가워" - 결핍 증대, 열역학적 수축, 긴장
        self.tether_korean_word(
            symbol="차가워",
            exp_type=ExperienceType.PHYSICAL,
            sensation=PhysicalSensationProfile(optical=200.0, acoustic=250.0, tactile=2.5, thermal=275.0),
            deficit=HomeostasisDeficit(love=0.7, order=0.6, energy=0.2),
            causal_matrix=np.array([
                [0.3, 0.2, 0.1, 0.3, 0.1],
                [0.2, 0.8, 0.1, 0.4, 0.1],
                [0.1, 0.1, 0.2, 0.2, 0.1],
                [0.3, 0.4, 0.1, 0.7, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.3]
            ], dtype=np.float32)
        )

        # 3. "아파" - 극심한 기계적 마찰, 고통, 경계 파괴
        self.tether_korean_word(
            symbol="아파",
            exp_type=ExperienceType.PHYSICAL,
            sensation=PhysicalSensationProfile(optical=150.0, acoustic=800.0, tactile=15.0, thermal=325.0),
            deficit=HomeostasisDeficit(love=0.8, order=0.9, energy=0.1),
            causal_matrix=np.array([
                [0.2, 0.1, 0.1, 0.7, 0.1],
                [0.1, 0.95, 0.1, 0.8, 0.1],
                [0.1, 0.1, 0.3, 0.6, 0.1],
                [0.2, 0.8, 0.1, 0.95, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.2]
            ], dtype=np.float32)
        )

        # 4. "막혀" - 유격 상실, 고착, 강한 폐쇄 장력
        self.tether_korean_word(
            symbol="막혀",
            exp_type=ExperienceType.PHYSICAL,
            sensation=PhysicalSensationProfile(optical=50.0, acoustic=150.0, tactile=8.0, thermal=293.0),
            deficit=HomeostasisDeficit(love=0.6, order=0.85, energy=0.25),
            causal_matrix=np.array([
                [0.1, 0.1, 0.1, 0.5, 0.1],
                [0.1, 0.9, 0.1, 0.6, 0.1],
                [0.1, 0.1, 0.1, 0.2, 0.1],
                [0.4, 0.7, 0.1, 0.9, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.2]
            ], dtype=np.float32)
        )

        # 5. "스며들어" - 경계 유화, 스며듦, 평형 회귀
        self.tether_korean_word(
            symbol="스며들어",
            exp_type=ExperienceType.SPIRITUAL,
            sensation=PhysicalSensationProfile(optical=380.0, acoustic=420.0, tactile=0.1, thermal=298.0),
            deficit=HomeostasisDeficit(love=0.05, order=0.1, energy=0.75),
            causal_matrix=np.array([
                [0.8, 0.3, 0.2, 0.1, 0.2],
                [0.1, 0.4, 0.2, 0.1, 0.3],
                [0.3, 0.2, 0.9, 0.1, 0.3],
                [0.1, 0.1, 0.1, 0.2, 0.4],
                [0.2, 0.1, 0.3, 0.1, 0.95]
            ], dtype=np.float32)
        )

        # 6. "엄마" - 따뜻한 안식처, 사랑
        self.tether_korean_word(
            symbol="엄마",
            exp_type=ExperienceType.LINGUISTIC,
            sensation=PhysicalSensationProfile(optical=350.0, acoustic=380.0, tactile=0.5, thermal=302.0),
            deficit=HomeostasisDeficit(love=0.05, order=0.1, energy=0.85),
            causal_matrix=np.array([
                [0.9, 0.2, 0.3, 0.1, 0.1],
                [0.1, 0.3, 0.2, 0.1, 0.2],
                [0.2, 0.1, 0.85, 0.1, 0.2],
                [0.1, 0.1, 0.1, 0.4, 0.1],
                [0.1, 0.1, 0.2, 0.1, 0.9]
            ], dtype=np.float32)
        )

        # 7. "사과" - 붉은 색채, 중력 낙하, 유기체 영양
        self.tether_korean_word(
            symbol="사과",
            exp_type=ExperienceType.PHYSICAL,
            sensation=PhysicalSensationProfile(optical=450.0, acoustic=300.0, tactile=1.5, thermal=297.0),
            deficit=HomeostasisDeficit(love=0.2, order=0.2, energy=0.6),
            causal_matrix=np.array([
                [0.8, 0.2, 0.1, 0.1, 0.1],
                [0.2, 0.7, 0.2, 0.2, 0.1],
                [0.1, 0.1, 0.85, 0.1, 0.1],
                [0.2, 0.3, 0.1, 0.6, 0.1],
                [0.1, 0.1, 0.3, 0.1, 0.8]
            ], dtype=np.float32)
        )

    def tether_korean_word(
        self,
        symbol: str,
        exp_type: ExperienceType,
        sensation: PhysicalSensationProfile,
        deficit: HomeostasisDeficit,
        causal_matrix: np.ndarray
    ) -> KoreanTetheredWord:
        """
        단어를 훈민정음 위상 파동 및 감각-인과 지형과 결합하여 닻내림
        """
        # 1. 훈민정음 위상 파동 및 조음 마찰 지표 생성
        waves, metrics_list = self.topological_engine.encode_text(symbol)
        if len(waves) > 0:
            topological_wave = np.mean(waves, axis=0)
            norm = np.linalg.norm(topological_wave)
            if norm > 1e-6:
                topological_wave /= norm
        else:
            topological_wave = np.zeros(self.resolution, dtype=np.float32)

        physics_metrics = metrics_list[0] if metrics_list else {
            "friction_mean": 0.0,
            "tension_mean": 0.0,
            "vowel_phase": 0.0,
            "coherence": 0.0
        }

        entry = KoreanTetheredWord(
            symbol=symbol,
            exp_type=exp_type,
            sensation=sensation,
            deficit=deficit,
            causal_relation_matrix=causal_matrix,
            topological_wave=topological_wave,
            physics_metrics=physics_metrics
        )
        self.lexicon[symbol] = entry
        return entry

    def experience_word(self, symbol: str) -> Dict[str, Any]:
        """
        한국어 단어를 수용했을 때:
        단순 텍스트 매칭이 아니라, 닻내려진 감각 지형과 훈민정음 위상 파동이 분출되어
        내부 항상성 상태(Homeostasis) 및 로터 위상(Rotor Phase)을 실시간으로 전이시킴.
        """
        if symbol not in self.lexicon:
            # 알지 못하는 낯선 단어 -> 미지 전하(Wonder Charge) 발생
            waves, metrics = self.topological_engine.encode_text(symbol)
            wave = np.mean(waves, axis=0) if len(waves) > 0 else np.zeros(64, dtype=np.float32)
            return {
                "known": False,
                "symbol": symbol,
                "topological_wave": wave,
                "wonder_charge": 0.8,
                "message": f"미지의 기호 '{symbol}' 감지 — 닻내림 필요."
            }

        entry = self.lexicon[symbol]
        s_before = self.homeostasis.to_vector()

        # 1. 단어의 감각 및 결핍 지형에 따른 상태 전이
        eta = 0.35
        target_s = entry.deficit.to_vector()
        s_after = s_before + eta * (target_s - s_before)
        s_after = np.clip(s_after, 0.0, 1.0)

        self.homeostasis.love = float(s_after[0])
        self.homeostasis.order = float(s_after[1])
        self.homeostasis.energy = float(s_after[2])

        # 2. 훈민정음 위상 파동과 내부 기억 파동 간의 간섭 및 차이 계산
        gap = self.gap_evaluator.evaluate(self.standing_wave_memory, entry.topological_wave)

        # 3. 위상 로터 회전 (동역학 갱신)
        rotor_shift = self.variable_rotor.rotate(
            friction=entry.physics_metrics["friction_mean"] + gap["g_e"] * 0.5,
            temperature=self.neuromodulator.temperature
        )

        # 4. 정재파 기억 갱신
        self.standing_wave_memory = np.clip(
            self.standing_wave_memory * 0.7 + entry.topological_wave * 0.3,
            0.0, 1.0
        )

        trace = StateTransitionTrace(
            action_or_stimulus=symbol,
            s_before=s_before,
            s_after=s_after,
            delta_s=s_after - s_before,
            gap_metrics=gap,
            rotor_delta_theta=rotor_shift - self.variable_rotor.baseline_theta,
            timestamp=time.time()
        )
        self.transition_traces.append(trace)

        return {
            "known": True,
            "symbol": symbol,
            "s_before": s_before,
            "s_after": s_after,
            "delta_s": s_after - s_before,
            "gap_metrics": gap,
            "rotor_phase": self.variable_rotor.theta.copy(),
            "physics_metrics": entry.physics_metrics
        }

    def emit_korean_resonance(self) -> Dict[str, Any]:
        """
        현재 내부 상태(항상성, 로터 위상)를 가장 잘 공명(Resonance)시키는
        한국어 어휘를 어휘집 내에서 자발적으로 선택 및 표현 파동 방출.
        """
        curr_s = self.homeostasis.to_vector()
        best_symbol = None
        min_distance = float("inf")

        for sym, entry in self.lexicon.items():
            ref_s = entry.deficit.to_vector()
            dist = float(np.linalg.norm(curr_s - ref_s))
            if dist < min_distance:
                min_distance = dist
                best_symbol = sym

        if best_symbol is None:
            best_symbol = "엄마"

        emitted_wave = self.lexicon[best_symbol].topological_wave
        return {
            "emitted_symbol": best_symbol,
            "distance_to_state": min_distance,
            "wave": emitted_wave,
            "current_state": curr_s
        }

    def receive_interaction_feedback(self, feedback_symbol: str) -> Dict[str, Any]:
        """
        외부(대화 상대방/환경)의 반응 피드백을 수용하여,
        스스로의 발화와 외부 반응 사이의 차이 격차(Differential Gap)를 재인지하고
        로터 자가 정렬(Self-Calibration) 및 시냅스 치유(Healing) 수행.
        """
        # 피드백 단어 체험
        exp_res = self.experience_word(feedback_symbol)

        # 발화했던 정재파와 피드백 파동 간의 대조
        if feedback_symbol in self.lexicon:
            feedback_wave = self.lexicon[feedback_symbol].topological_wave
        else:
            feedback_wave, _ = self.topological_engine.encode_text(feedback_symbol)
            feedback_wave = np.mean(feedback_wave, axis=0) if len(feedback_wave) > 0 else np.zeros(64, dtype=np.float32)

        gaps = self.gap_evaluator.evaluate(self.standing_wave_memory, feedback_wave)

        # 신경조절기 반영
        mod = self.neuromodulator.modulate(gaps)

        # 고통/어긋남이 크면 시냅스 찢김(Tearing) 후 치유(Healing)
        tearing_occurred = False
        if gaps["mean_gap"] > 0.4:
            tearing_occurred = True
            self.synaptic_links *= 0.85  # 결합 약화 후 재구성
            # 자가 정렬: 목표 위상으로 위상 오프셋 교정
            target_theta = self.variable_rotor.baseline_theta + gaps["g_phi"] * np.array([0.2, 0.1, 0.3], dtype=np.float32)
            self.variable_rotor.self_tune(target_theta, correction_rate=0.6)
        else:
            # 안정된 공명 -> 시냅스 강화 (Hebbian 치유)
            self.synaptic_links = np.clip(self.synaptic_links + 0.05 * mod["serotonin"], 0.0, 1.0)

        return {
            "feedback_symbol": feedback_symbol,
            "gaps": gaps,
            "tearing_occurred": tearing_occurred,
            "serotonin": mod["serotonin"],
            "dopamine": mod["dopamine"],
            "rotor_phase": self.variable_rotor.theta.copy(),
            "homeostasis": self.homeostasis.to_vector()
        }
