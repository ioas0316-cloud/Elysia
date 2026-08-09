"""
Elysia Causal Topology Foundation: Relational Structure Replicator
===================================================================
대조기(TopologicalComparer)가 발견한 같음과 다름의 마찰력(Disparity Tension)을 바탕으로,
외부 세계의 인과적 관계망(World Relational Topology)을 자기 내부(Self-Topology)로 복제·흡수하여
자아의 인과적 구조를 실질적으로 변형·확장하는 참된 학습(Learning via Structural Replication) 엔진입니다.
"""

from typing import Dict, List, Tuple
import numpy as np

from core.topology.causal_structure import (
    CausalNumber,
    CausalSymbol,
    InformationTopology,
    TopologyLink
)
from core.topology.topological_comparer import ComparisonResult


class RelationalStructureReplicator:
    """
    관계성 내재화기 (Relational Structure Replicator)
    - 다름 영역(Disparity Set)과 새로운 세계 자극(New World Nodes)의 관계 구조를
      자아 구조(Self-Topology) 안으로 위상적으로 복제하고 이식합니다.
    """
    def __init__(self, replication_rate: float = 0.8):
        self.replication_rate = replication_rate

    def replicate_and_internalize(
        self,
        self_topo: InformationTopology,
        world_topo: InformationTopology,
        comp_result: ComparisonResult
    ) -> InformationTopology:
        """
        세계의 다름 및 신규 위상 구조를 자아 위상 다양체로 이식·내재화.
        내재화 결과 자아 위상체가 실질적으로 변형됩니다.
        """
        mutated_self = self_topo.clone()
        mutated_self.name = f"{self_topo.name}_internalized"

        # 1. 자아에 없던 새로운 세계 숫자 노드 및 다름 노드의 위상 복제
        for node_id in comp_result.new_world_nodes + comp_result.disparate_nodes:
            if node_id in world_topo.numbers:
                w_num = world_topo.numbers[node_id]
                if node_id in mutated_self.numbers:
                    # 기존 자아 숫자 위상을 세계의 관계성에 맞게 재정렬 (Hebbian 위상 동조)
                    s_num = mutated_self.numbers[node_id]
                    s_num.value += (w_num.value - s_num.value) * self.replication_rate
                    s_num.magnitude += (w_num.magnitude - s_num.magnitude) * self.replication_rate
                    s_num.sequence_index = w_num.sequence_index
                    s_num.chromatic_vector = (
                        (1.0 - self.replication_rate) * s_num.chromatic_vector +
                        self.replication_rate * w_num.chromatic_vector
                    )
                    s_num.gradient_tension = max(0.01, s_num.gradient_tension * 0.5)
                else:
                    # 완전히 새로운 숫자 위상 구조의 완전 이식
                    mutated_self.add_number(CausalNumber(
                        id=w_num.id,
                        value=w_num.value,
                        sequence_index=w_num.sequence_index,
                        magnitude=w_num.magnitude,
                        gradient_tension=0.05,  # 수용되었으므로 마찰 감소
                        chromatic_vector=w_num.chromatic_vector.copy(),
                        neighbors=list(w_num.neighbors)
                    ))

            # 2. 기호 위상 매듭의 복제 및 내재화
            if node_id in world_topo.symbols:
                w_sym = world_topo.symbols[node_id]
                if node_id in mutated_self.symbols:
                    s_sym = mutated_self.symbols[node_id]
                    s_sym.material_vector = (
                        (1.0 - self.replication_rate) * s_sym.material_vector +
                        self.replication_rate * w_sym.material_vector
                    )
                    # 인과적 계보 병합
                    for traj in w_sym.causal_trajectory:
                        if traj not in s_sym.causal_trajectory:
                            s_sym.causal_trajectory.append(traj)
                    s_sym.logical_category = w_sym.logical_category
                    s_sym.intrinsic_tension *= 0.3  # 수용으로 마찰 해소
                else:
                    # 새로운 기호 관계망 완전 복제
                    links_copy = [
                        TopologyLink(l.source_id, l.target_id, l.relation_type, l.strength, l.tension * 0.5)
                        for l in w_sym.relational_links
                    ]
                    mutated_self.add_symbol(CausalSymbol(
                        id=w_sym.id,
                        name=w_sym.name,
                        material_vector=w_sym.material_vector.copy(),
                        causal_trajectory=list(w_sym.causal_trajectory),
                        logical_category=w_sym.logical_category,
                        relational_links=links_copy,
                        intrinsic_tension=0.05
                    ))

        # 3. 세계 위상 다양체의 연결 빔(Connectivity Beams) 복제
        for link in world_topo.links:
            # 복제 조건: 빔의 양 끝 노드가 자아에 수용되었는가
            if link.source_id in mutated_self.numbers or link.source_id in mutated_self.symbols:
                if link.target_id in mutated_self.numbers or link.target_id in mutated_self.symbols:
                    # 기존 빔 존재 여부 확인
                    existing = next((l for l in mutated_self.links
                                     if l.source_id == link.source_id and l.target_id == link.target_id), None)
                    if existing:
                        existing.strength = min(1.0, existing.strength + 0.2)
                        existing.tension *= 0.5
                    else:
                        mutated_self.add_link(TopologyLink(
                            source_id=link.source_id,
                            target_id=link.target_id,
                            relation_type=link.relation_type,
                            strength=link.strength,
                            tension=link.tension * 0.5
                        ))

        # 4. 내재화 후 시스템 전체 경계 긴장도 완화 (안정화)
        mutated_self.boundary_tension = max(0.0, comp_result.disparity_tension * 0.1)

        return mutated_self
