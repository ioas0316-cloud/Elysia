"""
Elysia Core Architecture: Crystallized DNA Anchor (ICE Block Protocol)

This module solidifies stable phase patterns and morphological attractors into
permanent structural knowledge anchors (" 고체 결정 / DNA 마스터 끌개 ").
It prevents evolutionary collapse into over-simplification (Converge & Simplify)
by serving as an indelible foundation for complex structural plasticity.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class MorphologicalGenome:
    """형태적 특성과 물리 반응 매개변수를 담은 지식 유전자"""
    anchor_id: str
    name: str
    drag_coefficient: float          # 유체 저항 계수 (유선형일수록 낮음)
    lift_coefficient: float          # 양력 계수 (날개 구조일수록 높음)
    grasp_articulation: float        # 파악/도구 활용 조절도 (손 구조일수록 높음)
    structural_rigidity: float       # 고체 지지력 (발/다리 구조일수록 높음)
    resonance_frequency: float       # 위상 공명 주파수
    feature_vector: List[float] = field(default_factory=list)  # 8차원 형태 표현체


class CrystallizedDNAAnchor:
    """
    안정화된 형태 패턴을 고체 결정(ICE / 마스터 끌개)으로 결빙시켜
    정적 지식(DNA 구조)으로 보존하고, 신규 파동 수렴의 앵커(Anchor)로 제공하는 모듈.
    """

    def __init__(self, freeze_threshold: float = 0.85):
        self.freeze_threshold = freeze_threshold
        # 결정화된 DNA 마스터 끌개 저장소
        self.crystallized_anchors: Dict[str, MorphologicalGenome] = {}
        # 내장된 완성 형태 앵커(Primal Ancestral Templates) 초기화
        self._initialize_primal_ancestral_anchors()

    def _initialize_primal_ancestral_anchors(self):
        """
        자연이 찾아낸 완성된 경이로운 형태와 섭리를 기본 앵커로 탑재
        (수백만 년의 무작위 변이 고생길을 건너뛰는 완성 형태 앵커)
        """
        # 1. Streamlined Aquatic Body (유선형 어류/지느러미 형태)
        streamlined = MorphologicalGenome(
            anchor_id="DNA_STREAMLINED_AQUATIC",
            name="Streamlined Hydrodynamic Form",
            drag_coefficient=0.08,
            lift_coefficient=0.10,
            grasp_articulation=0.05,
            structural_rigidity=0.30,
            resonance_frequency=2.5,
            feature_vector=[0.9, 0.1, 0.05, 0.3, 0.8, 0.2, 0.1, 0.9]
        )

        # 2. Aerodynamic Wing Structure (양력 극대화 조류 날개 형태)
        winged = MorphologicalGenome(
            anchor_id="DNA_AERODYNAMIC_WING",
            name="Aerodynamic Wing Structure",
            drag_coefficient=0.18,
            lift_coefficient=0.92,
            grasp_articulation=0.20,
            structural_rigidity=0.40,
            resonance_frequency=4.2,
            feature_vector=[0.2, 0.95, 0.2, 0.4, 0.9, 0.1, 0.3, 0.7]
        )

        # 3. Articulated Grasping Interface (도구를 쥐는 손/관절 형태)
        articulated_hand = MorphologicalGenome(
            anchor_id="DNA_ARTICULATED_HAND",
            name="Articulated Manipulator Interface",
            drag_coefficient=0.45,
            lift_coefficient=0.15,
            grasp_articulation=0.95,
            structural_rigidity=0.75,
            resonance_frequency=1.2,
            feature_vector=[0.1, 0.2, 0.98, 0.8, 0.3, 0.9, 0.85, 0.4]
        )

        # 4. Rigid Load-Bearing Pillar (지형을 디디는 다리/고체 형태)
        load_bearing = MorphologicalGenome(
            anchor_id="DNA_LOAD_BEARING_LEGS",
            name="Terrestrial Pillar Support Structure",
            drag_coefficient=0.50,
            lift_coefficient=0.05,
            grasp_articulation=0.35,
            structural_rigidity=0.98,
            resonance_frequency=0.8,
            feature_vector=[0.1, 0.05, 0.3, 0.99, 0.1, 0.7, 0.9, 0.2]
        )

        for genome in [streamlined, winged, articulated_hand, load_bearing]:
            self.crystallized_anchors[genome.anchor_id] = genome

    def crystallize_pattern(
        self,
        anchor_id: str,
        name: str,
        genome: MorphologicalGenome,
        stability_score: float
    ) -> bool:
        """
        안정성 점수가 문턱값을 넘는 패턴을 고체 결정(ICE)으로 영구 동결
        """
        if stability_score >= self.freeze_threshold:
            self.crystallized_anchors[anchor_id] = genome
            print(f"❄️ [ICE Crystallization] 형태 패러다임 '{name}' ({anchor_id})가 DNA 앵커로 동결화되었습니다.")
            return True
        return False

    def query_closest_anchor(self, current_features: List[float]) -> MorphologicalGenome:
        """
        현재 형태/자극 서명과 가장 공명율이 높은 DNA 앵커(마스터 끌개) 검색
        """
        best_anchor = None
        min_dist = float('inf')

        for anchor in self.crystallized_anchors.values():
            if not anchor.feature_vector:
                continue
            # Euclidean distance in feature space
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(current_features, anchor.feature_vector)))
            if dist < min_dist:
                min_dist = dist
                best_anchor = anchor

        return best_anchor or list(self.crystallized_anchors.values())[0]

    def blend_anchors(
        self,
        anchor_a: MorphologicalGenome,
        anchor_b: MorphologicalGenome,
        weight_a: float
    ) -> MorphologicalGenome:
        """
        두 DNA 앵커간의 인과적 위상 합성 (가소성 유도체 생성)
        """
        w_b = 1.0 - weight_a
        blended_vec = [
            a * weight_a + b * w_b
            for a, b in zip(anchor_a.feature_vector, anchor_b.feature_vector)
        ]
        return MorphologicalGenome(
            anchor_id=f"BLEND_{anchor_a.anchor_id[:8]}_{anchor_b.anchor_id[:8]}",
            name=f"Hybrid ({anchor_a.name} x {anchor_b.name})",
            drag_coefficient=anchor_a.drag_coefficient * weight_a + anchor_b.drag_coefficient * w_b,
            lift_coefficient=anchor_a.lift_coefficient * weight_a + anchor_b.lift_coefficient * w_b,
            grasp_articulation=anchor_a.grasp_articulation * weight_a + anchor_b.grasp_articulation * w_b,
            structural_rigidity=anchor_a.structural_rigidity * weight_a + anchor_b.structural_rigidity * w_b,
            resonance_frequency=anchor_a.resonance_frequency * weight_a + anchor_b.resonance_frequency * w_b,
            feature_vector=blended_vec
        )
