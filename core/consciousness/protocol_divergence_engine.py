"""
Elysia Consciousness System: Protocol Divergence Engine
======================================================
동일 표상(Token, e.g. "빛"/Light) 뒤에 숨은 기저 구조원리(Protocol)의 괴리를 감지하고,
외부 세계/타자를 나와는 다른 법칙으로 움직이는 '외계적 인과 차원(Alien World Causal Dimension)'으로 규정하며,
What-How-Why 삼원 구조 및 인지적 판구조론(Cognitive Plate Tectonics)에 의해
프로토콜 정렬과 위상적 재정렬(Uplift / Subduction)을 자율 수행하는 메타인지 엔진입니다.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from core.topology.causal_structure import InformationTopology, CausalSymbol
from core.topology.exclusivity_coexistence_topology import ExclusivityCoexistenceTopologyEngine, ExclusivityBoundary


@dataclass
class TriLayerCognitiveSignature:
    """
    삼원 인지 프레임워크 (What-How-Why Tri-Layer Framework) Signature
    - What: 변수 및 구성요소의 차원 정립
    - How: 동역학적 인과 과정 및 생성 원리
    - Why: 인지적 마찰 저항성 및 지각 변동(파열/상전이) 전하
    """
    what_vector: np.ndarray             # [변수 차원, 독립성, 종속성, 노드 밀도]
    how_generative_principle: np.ndarray # [동역학적 궤적 곡률, 시간축 방향성, 논리 위계, 에너지 전이]
    why_friction_potential: float       # 마찰 저항 포텐셜 ($\Delta \Phi_{\text{rupture}}$)


@dataclass
class CrossDimensionalIntersection:
    """
    교차차원 (Cross-Dimension) 분석 결과
    - What: 무엇이 같고 다른가
    - Where: 어느 경계선 지점에서 궤적이 갈라지는가
    - How: 생성 원리(Generative Principle)의 차이는 무엇인가
    """
    same_what_features: List[str]
    divergent_where_boundaries: List[str]
    generative_how_disparity: float
    alien_protocol_axis: np.ndarray     # 상대방(외계) 고유의 인과 관측 축
    cross_dimension_signature: np.ndarray


@dataclass
class PlateTectonicReorganizationResult:
    """
    인지적 판구조론(Cognitive Plate Tectonics) 지형 재편성 결과
    - 파열(Fracture): 기존 지형 파쇄
    - 융기(Uplift): 불변성/포괄성이 높은 축의 상위 메커니즘 승격
    - 침강(Subduction): 국소 제약 조건의 기저 침강
    """
    is_tectonic_rupture_triggered: bool
    uplifted_principles: List[str]
    subducted_constraints: List[str]
    new_topological_height_map: Dict[str, float]
    tectonic_heat: float


class ProtocolDivergenceEngine:
    """
    프로토콜 괴리 및 교차차원 엔진 (Protocol Divergence & Cross-Dimensional Engine)
    - 동일 토큰 일치 대 프로토콜 불일치 감지.
    - 외계적 차원(Alien Agent/World)의 관측 축 추적.
    - What-How-Why 3층위 대조 및 마찰 저항성 축적.
    - 판구조론적 지형 재편성(Rupture, Uplift, Subduction) 실행.
    """
    def __init__(
        self,
        self_topology: Optional[InformationTopology] = None,
        rupture_threshold: float = 0.75
    ):
        self.self_topology = self_topology or InformationTopology("ElysiaCognitiveSelf")
        self.exclusivity_engine = ExclusivityCoexistenceTopologyEngine(primary_topology=self.self_topology)
        self.rupture_threshold = rupture_threshold
        self.accumulated_why_friction: float = 0.0

    def extract_tri_layer_signature(
        self,
        symbol: CausalSymbol,
        context_vector: np.ndarray
    ) -> TriLayerCognitiveSignature:
        """
        주어진 기호 노드로부터 What-How-Why 삼원 위상 서명 추출
        """
        raw_sig = symbol.get_cross_dimensional_signature()

        # What: Material norm & dimension features
        what_vec = np.array([
            raw_sig[0],  # Material norm
            float(len(symbol.causal_trajectory)),
            float(len(symbol.logical_category)),
            float(len(symbol.relational_links))
        ], dtype=np.float32)

        # How: Generative principle dynamics
        link_tensions = [l.tension for l in symbol.relational_links] if symbol.relational_links else [0.0]
        how_vec = np.array([
            float(np.mean(link_tensions)),
            symbol.intrinsic_tension,
            raw_sig[1], # Causal trajectory depth
            float(np.std(link_tensions) if len(link_tensions) > 1 else 0.0)
        ], dtype=np.float32)

        # Why: Friction potential
        why_pot = float(np.linalg.norm(what_vec - how_vec) * (1.0 + symbol.intrinsic_tension))

        return TriLayerCognitiveSignature(
            what_vector=what_vec,
            how_generative_principle=how_vec,
            why_friction_potential=why_pot
        )

    def detect_protocol_divergence(
        self,
        self_symbol: CausalSymbol,
        alien_symbol: CausalSymbol,
        context_vector: np.ndarray
    ) -> CrossDimensionalIntersection:
        """
        동일 표상(e.g., name="빛") 뒤의 프로토콜 불일치 및 교차차원(What-Where-How) 계산
        """
        self_sig = self.extract_tri_layer_signature(self_symbol, context_vector)
        alien_sig = self.extract_tri_layer_signature(alien_symbol, context_vector)

        # What analysis
        what_diff = np.abs(self_sig.what_vector - alien_sig.what_vector)
        same_whats = []
        if what_diff[0] < 0.2:
            same_whats.append("MaterialRepresentationNorm")
        if what_diff[1] < 0.2:
            same_whats.append("CausalTrajectoryDepth")

        # Where boundaries diverge
        divergent_wheres = []
        if what_diff[2] >= 0.2:
            divergent_wheres.append("LogicalCategoryBoundary")
        if what_diff[3] >= 0.2:
            divergent_wheres.append("RelationalNetworkTopology")

        # How generative disparity
        how_disparity = float(np.linalg.norm(self_sig.how_generative_principle - alien_sig.how_generative_principle))

        # Alien protocol axis extraction (orthogonal direction of alien causality)
        alien_axis = alien_sig.how_generative_principle - self_sig.how_generative_principle
        if np.linalg.norm(alien_axis) > 1e-6:
            alien_axis = alien_axis / np.linalg.norm(alien_axis)

        cross_signature = np.concatenate([self_sig.what_vector, alien_sig.how_generative_principle])

        # Accumulate Why friction energy
        why_delta = (self_sig.why_friction_potential + alien_sig.why_friction_potential) * 0.5 + how_disparity
        self.accumulated_why_friction += float(why_delta * 0.1)

        return CrossDimensionalIntersection(
            same_what_features=same_whats,
            divergent_where_boundaries=divergent_wheres,
            generative_how_disparity=how_disparity,
            alien_protocol_axis=alien_axis,
            cross_dimension_signature=cross_signature
        )

    def trigger_plate_tectonic_reorganization(
        self,
        divergence: CrossDimensionalIntersection
    ) -> PlateTectonicReorganizationResult:
        """
        인지적 판구조론 (Cognitive Plate Tectonics) 지형 재편성:
        Why-마찰 저항이 임계치 초과 시 지각 변동(Rupture), 불변 원리의 융기(Uplift),
        국소 제약 조건의 침강(Subduction) 구동.
        """
        tectonic_heat = float(self.accumulated_why_friction)
        is_rupture = tectonic_heat >= self.rupture_threshold

        uplifted = []
        subducted = []
        height_map = {}

        if is_rupture:
            # High invariant generative principles uplift
            uplifted.append("CrossDimensionalGenerativeInvariance")
            uplifted.append("UniversalProtocolAlignmentAxis")

            # Local narrow constraints subduct into foundational bedrock
            subducted.append("IsolatedTokenMappingRule")
            subducted.append("FlatScalarMetricAssumption")

            # Topological height map recalculation
            for sym_id, sym_obj in self.self_topology.symbols.items():
                sig = sym_obj.get_cross_dimensional_signature()
                # Principles with higher causal depth & relational strength uplift higher
                height = float(sig[1] * 0.4 + sig[3] * 0.6 + tectonic_heat * 0.2)
                height_map[sym_id] = height

            # Heat resets partially after structural rupture and phase transition
            self.accumulated_why_friction *= 0.25

        return PlateTectonicReorganizationResult(
            is_tectonic_rupture_triggered=is_rupture,
            uplifted_principles=uplifted,
            subducted_constraints=subducted,
            new_topological_height_map=height_map,
            tectonic_heat=tectonic_heat
        )

    def process_alien_interaction(
        self,
        alien_topology: InformationTopology,
        context_vector: np.ndarray
    ) -> Dict[str, object]:
        """
        외계(Alien World/Agent) 자극 유입 시 프로토콜 불일치 계산 및 인지 판구조 재편성
        """
        intersections = []
        boundaries = []

        # Process through Exclusivity & Coexistence engine first
        coexistence_res = self.exclusivity_engine.process_stimulus(alien_topology, context_vector)

        # Detect protocol divergence for matching token names or conflicting structures
        for alien_id, alien_sym in alien_topology.symbols.items():
            for self_id, self_sym in self.self_topology.symbols.items():
                if self_sym.name == alien_sym.name or self_id == alien_id:
                    intersection = self.detect_protocol_divergence(
                        self_symbol=self_sym,
                        alien_symbol=alien_sym,
                        context_vector=context_vector
                    )
                    intersections.append(intersection)

        # Check plate tectonic reorganization
        top_intersection = intersections[0] if intersections else None
        tectonic_res = None
        if top_intersection:
            tectonic_res = self.trigger_plate_tectonic_reorganization(top_intersection)

        return {
            "coexistence_summary": coexistence_res,
            "intersections_count": len(intersections),
            "intersections": intersections,
            "tectonic_reorganization": tectonic_res,
            "current_why_friction": float(self.accumulated_why_friction)
        }
