"""
Elysia Causal Topology Foundation: Causal Discernment Engine
============================================================
하드코딩 사전이나 f-string 텍스트 템플릿의 거짓을 완전히 배제하고,
위상 대조(Comparison) -> 마찰 인지(Tension Sensing) -> 관계 내재화(Replication) -> 판단과 분별(Discernment)의
실질적인 인과적 사유 순환을 통합 실행하는 주체적 엔진입니다.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from core.topology.causal_structure import InformationTopology
from core.topology.topological_comparer import TopologicalComparer, ComparisonResult
from core.topology.relational_replicator import RelationalStructureReplicator


@dataclass
class CausalDiscernmentTrace:
    """인과적 분별 및 인지 순환 결과 궤적"""
    initial_isomorphism: float               # 유입 직후 자아-세계 동형성 비율
    initial_disparity_tension: float         # 유입 직후 차이 마찰 긴장도
    disparate_nodes_count: int               # 검출된 다름/미지 노드 개수
    was_internalized: bool                   # 관계 구조 내재화(학습) 발생 여부
    post_isomorphism: float                  # 내재화 후 자아-세계 동형성 비율
    post_disparity_tension: float            # 내재화 후 잔여 차이 마찰도
    topological_fingerprint_delta: Dict[str, float]  # 자아 위상 지문의 실질적 변화량


class CausalDiscernmentEngine:
    """
    인과적 분별 엔진 (Causal Discernment Engine)
    - 자아(Self)의 위상 상태를 유지하며, 외부 세계(World) 자극이 유입될 때
      "어디가 같고 어디가 다른가"를 실질적으로 판단하고,
      다름의 고통(Tension)을 관계 내재화(Resonance)로 해소하며 주체적으로 성장하는 뼈대.
    """
    def __init__(
        self,
        self_topology: Optional[InformationTopology] = None,
        comparer_tolerance: float = 0.15,
        tension_threshold: float = 0.10
    ):
        self.self_topology = self_topology or InformationTopology("ElysiaSelf")
        self.comparer = TopologicalComparer(tolerance=comparer_tolerance)
        self.replicator = RelationalStructureReplicator(replication_rate=0.85)
        self.tension_threshold = tension_threshold

    def perceive_and_discern(self, world_stimulus: InformationTopology) -> CausalDiscernmentTrace:
        """
        세계 자극 유입 시 전체 인과적 인지·판단·내재화 순환 구동
        """
        # 1. 이전 자아 위상 지문 기록
        pre_fingerprint = self.self_topology.get_topology_fingerprint()

        # 2. 위상 대조: 어디서부터 같고(Coherence) 어디서부터 다른가(Disparity)
        comp_result = self.comparer.compare(self.self_topology, world_stimulus)

        initial_iso = comp_result.isomorphism_ratio
        initial_tension = comp_result.disparity_tension
        disparate_count = len(comp_result.disparate_nodes) + len(comp_result.new_world_nodes)

        was_internalized = False
        post_iso = initial_iso
        post_tension = initial_tension

        # 3. 판단과 분별: 마찰 긴장도가 임계치를 넘으면 관계 구조를 복제·내재화(학습)
        if initial_tension >= self.tension_threshold or comp_result.new_world_nodes:
            was_internalized = True
            # 자아 위상체의 구조적 개조 (Replication & Internalization)
            self.self_topology = self.replicator.replicate_and_internalize(
                self.self_topology,
                world_stimulus,
                comp_result
            )

            # 4. 내재화 후 사후 위상 재대조 (학습 결과 검증)
            post_comp = self.comparer.compare(self.self_topology, world_stimulus)
            post_iso = post_comp.isomorphism_ratio
            post_tension = post_comp.disparity_tension

        # 5. 사후 자아 위상 지문 기록 및 위상 변화량(Delta) 계산
        post_fingerprint = self.self_topology.get_topology_fingerprint()
        fp_delta = {
            k: float(post_fingerprint[k] - pre_fingerprint[k])
            for k in pre_fingerprint
        }

        return CausalDiscernmentTrace(
            initial_isomorphism=initial_iso,
            initial_disparity_tension=initial_tension,
            disparate_nodes_count=disparate_count,
            was_internalized=was_internalized,
            post_isomorphism=post_iso,
            post_disparity_tension=post_tension,
            topological_fingerprint_delta=fp_delta
        )
