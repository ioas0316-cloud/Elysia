"""
Elysia Causal World Surgery Engine
==================================
바이너리 인과 격자 G = (V, E, C)에 주디아 펄(Judea Pearl)의 do-operator를 이식하여,
전체 파일을 재컴파일하거나 재인코딩하지 않고도 특정 인과 궤적과 노드를 물리적으로 수술하고,
훼손된 비트(Bit-rot)를 인과 제약 조건에 의해 역복원하는 자가 치유(Self-Healing) 엔진.
"""

from typing import Dict, List, Set, Tuple, Any, Optional, Callable
import copy

from core.topology.causal_stem_branch_engine import (
    CausalGraph, CausalNode, CausalEdge, NodeType
)


class CausalWorldSurgeryEngine:
    """
    바이너리 인과 격자에 대한 외과적 수술(Graph Surgery) 및 자가 치유를 수행하는 엔진.
    """

    def apply_do_surgery(
        self,
        graph: CausalGraph,
        target_node_id: str,
        new_val: int
    ) -> CausalGraph:
        """
        do(V_target = new_val) 인과 수술:
        1. 대상 노드의 원본 인과 궤적을 끊지 않고 불변 상태를 강제 변이.
        2. 페이로드 및 불변 서명을 정합성 있게 갱신.
        3. 맥락 제약(C)에 수술 기록(Surgery Log)을 보존.
        """
        cloned_graph = graph.clone()

        if target_node_id not in cloned_graph.nodes:
            raise KeyError(f"Target node '{target_node_id}' not found in causal graph.")

        node = cloned_graph.nodes[target_node_id]
        old_val = node.payload.get("val", 0)

        # do-operator 개입 적용
        node.payload["val"] = new_val
        node.payload["hex"] = hex(new_val)
        node.payload["surgically_modified"] = True
        node.payload["original_val"] = old_val

        # C(맥락 제약)에 개입 흔적 기록
        cloned_graph.context_constraints[target_node_id].add(
            f"do_surgery:mutated_from_{old_val}_to_{new_val}"
        )

        return cloned_graph

    def inject_bit_rot(self, graph: CausalGraph, target_node_id: str, corrupted_val: int):
        """인위적인 비트 오염(Bit-rot) 발생 (테스트 및 검증용)"""
        if target_node_id in graph.nodes:
            graph.nodes[target_node_id].payload["val"] = corrupted_val
            graph.nodes[target_node_id].payload["corrupted"] = True

    def self_heal_bit_rot(
        self,
        graph: CausalGraph,
        corrupted_node_id: str,
        invariant_parity_sum: int
    ) -> int:
        """
        인과적 자가 치유 (Self-Healing):
        주변 노드들의 상태와 인과 보존 법칙(Parity / Conservation Invariant) 제약(C)을 바탕으로,
        훼손된 노드의 본래 비트값을 수식 탐색 없이 정직하게 대수적으로 복원.
        """
        if corrupted_node_id not in graph.nodes:
            raise KeyError(f"Corrupted node '{corrupted_node_id}' not found.")

        # 모든 정상 노드들의 합 계산
        current_sum = 0
        for nid, node in graph.nodes.items():
            if nid != corrupted_node_id:
                current_sum += node.payload.get("val", 0)

        # 보존 법칙: Total Sum = invariant_parity_sum
        healed_val = (invariant_parity_sum - current_sum) % 256
        if healed_val < 0:
            healed_val += 256

        # 치유 적용
        node = graph.nodes[corrupted_node_id]
        node.payload["val"] = healed_val
        node.payload["hex"] = hex(healed_val)
        node.payload["corrupted"] = False
        node.payload["healed"] = True

        graph.context_constraints[corrupted_node_id].add(f"healed_from_rot:{healed_val}")
        return healed_val

    def transcode_stream(
        self,
        graph: CausalGraph,
        transform_fn: Callable[[int], int]
    ) -> CausalGraph:
        """
        위상 보존 트랜스코딩:
        파일 전체를 재인코딩하지 않고, 불변 줄기(Stem)의 각 노드에 변환을 가하여
        새로운 인과 궤적을 도출.
        """
        cloned_graph = graph.clone()
        for node in cloned_graph.nodes.values():
            if node.node_type == NodeType.STEM and "val" in node.payload:
                new_v = transform_fn(node.payload["val"]) % 256
                node.payload["val"] = new_v
                node.payload["hex"] = hex(new_v)
        return cloned_graph
