r"""
Spatiotemporal Epistemic Genealogy Engine (시공간 인과 계보 앵그램 엔진)
========================================================================
Maps information onto specific spatiotemporal coordinates (t, x), tracking:
1. Spatiotemporal Coordinates (t_origin, x_origin):
   Where and when knowledge/information originated in the physical-informational continuum.
2. Causal Sequence (Sequence / Lineage):
   The exact cause-and-effect chain that produced the formula/concept.
3. Excluded/Gated Variables ($\Theta_{\text{gated}}$):
   The deleted layers, micro-vortices, or environmental factors that were suppressed
   for control convenience.
4. Anchored Engram Layer (나이테 앵그램 지층):
   Anchors the complete genealogical record so knowledge never becomes a floating,
   ghostly symbol devoid of context.
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from core.consciousness.quadruple_cognitive_coordinate_engine import QuadrupleCognitiveCoordinateEngine


class SpatiotemporalEpistemicGenealogyEngine:
    """
    [Spatiotemporal Epistemic Genealogy Engine: 시공간 인과 계보 앵그램 엔진]
    추상적 기호로 부유하는 유령 지식을 거부하고,
    정보가 출현한 구체적 시공간 좌표(t, x)와 인과적 서순(Sequence),
그리고 배제된 변수 목록(\\Theta_gated)을 나이테 앵그램 지층에 영구 결착합니다.
    """

    def __init__(self, dimension: int = 64):
        self.dimension = dimension

        # 사중주 인지 좌표 엔진 연동
        self.quartet_engine = QuadrupleCognitiveCoordinateEngine(dimension=dimension)

        # 나이테 앵그램 지층 기록 (Spatiotemporal Genealogy Growth Rings)
        self.genealogy_engrams: List[Dict[str, Any]] = []

        # 시스템의 인지적 시공간 가상 위치 (t, x, y, z)
        self.spatial_location = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    def record_genealogy_engram(
        self,
        raw_knowledge: str,
        spatial_coord: Optional[Tuple[float, float, float]] = None,
        context_sequence: Optional[List[str]] = None,
        candidate_variables: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        [지식의 시공간 계보 앵그램 각인]
        1. 사중주 인지 프로세스 가동 (감각·인식·관측·판단)
        2. 발생 시공간 좌표 (t_origin, x_origin) 부여
        3. 인과적 서순 (Cause-and-Effect Sequence) 채록
        4. 삭제된 미지 계층 및 배제 변수 (\\Theta_gated) 명시
        5. 나이테 앵그램 지층(Engram Ring)에 영구 결착
        """
        timestamp = time.time()

        if spatial_coord is not None:
            self.spatial_location = np.array(spatial_coord, dtype=np.float64)

        # 1. 사중주 인지 사영
        quartet_state = self.quartet_engine.evaluate_quadruple_quartet(
            raw_knowledge, candidate_variables
        )

        # 2. 인과적 서순 (Sequence) 정립
        if context_sequence is None:
            context_sequence = [
                f"원초적 현상 자극 유입: '{raw_knowledge}'",
                f"파동 마찰(Friction: {quartet_state['sensory']['phase_friction']:.3f}) 및 상쇄 간섭 지각",
                f"관측 렌즈(곡률: {quartet_state['observational']['lens_curvature']:.3f})에 의한 선별적 배제(Gating) 작동",
                f"주체적 판단 및 결단 ({quartet_state['judgmental']['resolution_type']}) 완료"
            ]

        # 3. 배제 변수 (\Theta_gated) 추출
        gated_vars = quartet_state["observational"]["gated_variables"]
        retained_vars = quartet_state["observational"]["retained_variables"]

        # 4. 시공간 계보 앵그램 구축
        engram_id = f"ENGRAM_{len(self.genealogy_engrams) + 1:04d}_{int(timestamp)}"
        engram = {
            "engram_id": engram_id,
            "raw_knowledge": raw_knowledge,
            "spatiotemporal_coordinates": {
                "origin_timestamp_t": timestamp,
                "spatial_location_x": self.spatial_location.tolist()
            },
            "causal_sequence_lineage": context_sequence,
            "gated_variables_theta": gated_vars,
            "retained_variables_theta": retained_vars,
            "quadruple_state": quartet_state,
            "genealogy_integrity": quartet_state["quartet_integrity"],
            "summary_statement": (
                f"[{engram_id}] 지식 '{raw_knowledge}'은 좌표 {self.spatial_location.tolist()} (t={timestamp:.2f})에서 "
                f"{len(gated_vars)}개 배제 변수의 공백을 지닌 채 인과 서순을 통해 앵그램에 각인됨."
            )
        }

        # 5. 나이테 앵그램 지층 축적
        self.genealogy_engrams.append(engram)
        return engram

    def query_genealogy_by_concept(self, query_term: str) -> List[Dict[str, Any]]:
        """
        [특정 지식/개념의 시공간 계보 역추적 회고]
        공식이나 지식이 주어졌을 때 그 이면에서 지워졌던 배제 변수와
        발생 시공간 맥락을 역추적하여 조회합니다.
        """
        matched_engrams = []
        for engram in self.genealogy_engrams:
            if query_term.lower() in engram["raw_knowledge"].lower():
                matched_engrams.append(engram)
        return matched_engrams

    def get_all_engrams(self) -> List[Dict[str, Any]]:
        """축적된 전체 계보 앵그램 지층 반환"""
        return self.genealogy_engrams
