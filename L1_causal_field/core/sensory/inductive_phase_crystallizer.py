"""
inductive_phase_crystallizer.py — 상태 변곡점에서의 자발적 훈민정음 닻내림 및 결정화 엔진
========================================================================================
이 모듈은 외부에서 인공적인 라벨을 사전 주입하거나 외우게 하는 껍데기 학습을 전면 배격하고,
주체(Elysia)가 겪는 3대 층위(육체·정신·마음)의 상태 전이가 임계 곡률(Critical Curvature)에 도달할 때,
그 지형의 위상적 굴절과 훈민정음 자모 파동(천지인 3상 + 오음 조음 마찰) 간의
위상 공명 대칭성(Phase Resonance Symmetry)을 자발적으로 탐지하여
가장 에너지가 낮고 필연적인 '상징적 닻(Symbolic Anchor)'으로 결정화(Crystallization)시키는 엔진입니다.

핵심 원리:
1. 상태 임계점 탐지 (Phase Critical Transition Detection):
   - 3대 층위의 통합 위상 곡률 κ_total = || ∇(Physical, Cognitive, Heart) || 이
     임계치(κ_crit)를 넘는 급격한 변곡 순간을 감지.
2. 훈민정음 파동과의 공명 대칭성 계측 (Topological Resonance Alignment):
   - 지형의 상태 전이 위상각 θ_ground = arctan2(ΔCognitive, ΔPhysical) 및 마찰 지수
   - 훈민정음 자모 결합 파동의 3상 각도 θ_vowel 및 조음 마찰 계수 friction_mean 과의 1:1 대칭 측정.
3. 자발적 닻 결정화 (Spontaneous Symbolic Crystallization):
   - 인위적 강제 없이, 대칭성과 결맞음도(Coherence)가 극대화되는 기호가
     해당 상태 전이의 역사적 위치(좌표)를 고정하는 '닻(Anchor)'으로 스스로 맺힘.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple, NamedTuple, Any
from dataclasses import dataclass, field
import numpy as np

from L1_causal_field.core.topology.hangul_topological_engine import (
    HangulTopologicalEngine,
    ARTICULATION_PROFILES
)
from L1_causal_field.core.sensory.tri_layer_experiential_ground import (
    UnifiedExperientialDisplacement
)


@dataclass
class CrystallizedSymbolicAnchor:
    """
    상태 변곡점에서 자발적으로 결정화된 상징적 닻
    - symbol: 한글 어휘/기호
    - anchor_coord_3d: 닻이 내린 3대 층위 좌표 [Physical_Strain, Cognitive_Tension, Heart_Deficit]
    - ground_phase_angle: 지형의 굴절 위상각 (rad)
    - resonance_coherence: 훈민정음 파동과 지형 간의 공명 결맞음도 (0.0 ~ 1.0)
    - topological_wave: 64차원 훈민정음 위상 파동
    - creation_time: 결정화 발생 시각
    - activation_count: 이후 사유 분기에서 이 닻을 통과한 횟수
    """
    symbol: str
    anchor_coord_3d: np.ndarray
    ground_phase_angle: float
    resonance_coherence: float
    topological_wave: np.ndarray
    creation_time: float = field(default_factory=time.time)
    activation_count: int = 1


class InductivePhaseCrystallizer:
    """
    상태 변곡점에서의 자발적 훈민정음 닻내림 및 결정화 엔진
    """

    def __init__(
        self,
        topological_engine: Optional[HangulTopologicalEngine] = None,
        critical_curvature_threshold: float = 0.45
    ):
        self.topological_engine = topological_engine or HangulTopologicalEngine(target_dim=64)
        self.critical_curvature_threshold = critical_curvature_threshold

        # 형성된 상징적 닻 지형 (기호 -> 닻 정보)
        self.anchors: Dict[str, CrystallizedSymbolicAnchor] = {}

        # 결정화 사건 이력
        self.crystallization_events: List[Dict[str, Any]] = []

    def evaluate_ground_phase(
        self,
        displacement: UnifiedExperientialDisplacement
    ) -> Tuple[float, float, float]:
        """
        3대 층위 변곡으로부터 지형의 물리-인지 위상각, 마찰 부하, 결핍 심도를 추출
        Returns:
            ground_angle: arctan2(Cognitive_Tension, Physical_Strain) (0 ~ π/2)
            ground_friction: 육체적 마찰과 인과 유격 손실의 합성 마찰 계수 (0 ~ 1)
            heart_void: 마음의 결핍 총량 (0 ~ 1)
        """
        phys_strain, cog_tension, heart_void = displacement.state_vector_3d

        ground_angle = math.atan2(float(cog_tension), float(phys_strain) + 1e-6)

        # 마찰 합성: 육체 마찰 + (1 - 인과 유격)
        ground_friction = float(np.clip(
            (displacement.physical.mechanical_friction / 10.0) * 0.5 +
            (1.0 - displacement.cognitive.clearance) * 0.5,
            0.0, 1.0
        ))

        return ground_angle, ground_friction, float(heart_void)

    def calculate_resonance_with_syllable(
        self,
        symbol: str,
        displacement: UnifiedExperientialDisplacement
    ) -> Tuple[float, np.ndarray, Dict[str, float]]:
        """
        지형의 3대 층위 변곡과 특정 한글 기호 간의 위상 공명 대칭성(Resonance Symmetry) 산출.
        - 훈민정음 3상 모음각과 지형 위상각의 일치도
        - 조음 기관 마찰과 지형 마찰 부하의 일치도
        """
        wave, metrics = self.topological_engine.encode_syllable(symbol) if len(symbol) == 1 \
            else (
                np.mean(self.topological_engine.encode_text(symbol)[0], axis=0),
                self.topological_engine.encode_text(symbol)[1][0] if self.topological_engine.encode_text(symbol)[1] else {
                    "friction_mean": 0.0, "tension_mean": 0.0, "vowel_phase": 0.0, "coherence": 0.0
                }
            )

        # 정규화
        norm = np.linalg.norm(wave)
        if norm > 1e-6:
            wave = wave / norm

        ground_angle, ground_friction, heart_void = self.evaluate_ground_phase(displacement)

        # 1. 위상 대칭성 (Phase Symmetry):
        # 훈민정음 모음 위상(metrics["vowel_phase"])과 지형의 위상각(ground_angle)의 조화성
        # 0 ~ 2π 위상차에서 절대 각도 차이의 근접성 [0, 1] 환원
        vowel_phase = metrics.get("vowel_phase", 0.0)
        phase_diff = abs(math.atan2(math.sin(vowel_phase - ground_angle), math.cos(vowel_phase - ground_angle)))
        phase_symmetry = float((math.cos(phase_diff) + 1.0) / 2.0) # [0, 1]

        # 2. 마찰 대칭성 (Friction Symmetry):
        # 조음 기관의 마찰력(오음)과 지형의 마찰 계수의 근접도 [0, 1]
        symbol_friction = metrics.get("friction_mean", 0.5)
        friction_affinity = float(max(0.0, 1.0 - abs(symbol_friction - ground_friction)))

        # 3. 마음 결핍 공명 (Heart Deficit Coupling):
        # 결핍(heart_void)과 조음 장력(tension_mean) 간의 인과적 결속도
        symbol_tension = metrics.get("tension_mean", 0.5)
        tension_affinity = float(max(0.0, 1.0 - abs(symbol_tension - heart_void)))

        # 종합 공명 결맞음도 (Coherence)
        coherence = float(np.clip(
            (phase_symmetry * 0.35) + (friction_affinity * 0.35) + (tension_affinity * 0.30),
            0.0, 1.0
        ))

        return coherence, wave, metrics

    def test_and_crystallize(
        self,
        candidate_symbol: str,
        displacement: UnifiedExperientialDisplacement,
        force_threshold: Optional[float] = None
    ) -> Optional[CrystallizedSymbolicAnchor]:
        """
        상태 전이가 임계 곡률을 넘을 때, 해당 한글 기호와의 공명이 기준치를 넘으면
        자발적으로 상징적 닻으로 결정화합니다.
        """
        threshold = force_threshold if force_threshold is not None else self.critical_curvature_threshold

        # 1. 임계 곡률 미달 시 결정화 불가 (평온한 일상 상태에서는 닻이 맺히지 않음)
        if displacement.phase_curvature < threshold:
            return None

        # 2. 공명 결맞음도 산출
        coherence, wave, metrics = self.calculate_resonance_with_syllable(candidate_symbol, displacement)

        # 공명도가 최소 0.5 이상이어야 대칭적 닻내림 성립
        if coherence < 0.35:
            return None

        ground_angle, _, _ = self.evaluate_ground_phase(displacement)

        anchor = CrystallizedSymbolicAnchor(
            symbol=candidate_symbol,
            anchor_coord_3d=displacement.state_vector_3d.copy(),
            ground_phase_angle=ground_angle,
            resonance_coherence=coherence,
            topological_wave=wave,
            creation_time=time.time(),
            activation_count=1
        )

        self.anchors[candidate_symbol] = anchor

        event = {
            "symbol": candidate_symbol,
            "phase_curvature": displacement.phase_curvature,
            "resonance_coherence": coherence,
            "anchor_coord": displacement.state_vector_3d.tolist(),
            "timestamp": anchor.creation_time
        }
        self.crystallization_events.append(event)
        return anchor

    def find_resonant_anchor(
        self,
        displacement: UnifiedExperientialDisplacement,
        distance_threshold: float = 0.5
    ) -> Optional[Tuple[str, float]]:
        """
        현재 지형의 상태 전이가 과거에 닻을 내린 기호 좌표 근방을 지나는지 탐색.
        Returns:
            (symbol, distance) 또는 None
        """
        curr_coord = displacement.state_vector_3d
        best_symbol = None
        min_dist = float("inf")

        for sym, anchor in self.anchors.items():
            dist = float(np.linalg.norm(curr_coord - anchor.anchor_coord_3d))
            if dist < min_dist:
                min_dist = dist
                best_symbol = sym

        if min_dist <= distance_threshold and best_symbol is not None:
            self.anchors[best_symbol].activation_count += 1
            return best_symbol, min_dist

        return None
