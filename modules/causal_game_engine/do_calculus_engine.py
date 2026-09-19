"""
do_calculus_engine.py
=====================
Elysia Causal Engine - Pearl's Do-Calculus Intervention & Graph Surgery Engine
Implements Structural Causal Models (SCM), Graph Surgery, do(X = x) operators,
and Average Causal Effect (ACE) calculations.
"""

from dataclasses import dataclass, field
import copy
from typing import Dict, List, Set, Tuple, Any


@dataclass
class CausalNode:
    """인과 그래프 상의 개별 노드"""
    node_id: str
    value: float = 0.0
    parents: Set[str] = field(default_factory=set)
    children: Set[str] = field(default_factory=set)


class StructuralCausalModel:
    """
    구조적 인과 모델 (SCM) 및 Graph Surgery 연산 엔진
    """

    def __init__(self):
        self.nodes: Dict[str, CausalNode] = {}
        self.edge_weights: Dict[Tuple[str, str], float] = {}

    def add_causal_edge(
        self, parent_id: str, child_id: str, weight: float
    ) -> None:
        """인과 엣지 추가 (Parent -> Child)"""
        if parent_id not in self.nodes:
            self.nodes[parent_id] = CausalNode(node_id=parent_id, value=0.0)
        if child_id not in self.nodes:
            self.nodes[child_id] = CausalNode(node_id=child_id, value=0.0)

        self.nodes[parent_id].children.add(child_id)
        self.nodes[child_id].parents.add(parent_id)
        self.edge_weights[(parent_id, child_id)] = weight

    def apply_do_intervention(
        self, target_node_id: str, intervention_value: float
    ) -> "StructuralCausalModel":
        """
        do(X = x) 연산 수행: Target Node로 들어오는 모든 부모 엣지를 절단(Graph Surgery)한
        사본 SCM 반환
        """
        surgered_scm = copy.deepcopy(self)
        if target_node_id not in surgered_scm.nodes:
            surgered_scm.nodes[target_node_id] = CausalNode(node_id=target_node_id, value=intervention_value)

        target_node = surgered_scm.nodes[target_node_id]

        # 1. 부모 노드들과의 연결 절단 (Incoming Edges Severing)
        for parent_id in list(target_node.parents):
            surgered_scm.nodes[parent_id].children.remove(target_node_id)
            if (parent_id, target_node_id) in surgered_scm.edge_weights:
                del surgered_scm.edge_weights[(parent_id, target_node_id)]
        target_node.parents.clear()

        # 2. 강제 값 할당
        target_node.value = intervention_value

        # 3. 하류 노드(Downstream)로 인과 효과 전파 연산
        surgered_scm._propagate_causal_effects(start_node_id=target_node_id)

        return surgered_scm

    def _propagate_causal_effects(self, start_node_id: str) -> None:
        """절단된 그래프 상에서 순방향 인과 전파"""
        queue = [start_node_id]
        visited = set()

        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)

            current_node = self.nodes[current_id]

            for child_id in current_node.children:
                child_node = self.nodes[child_id]

                # 하위 노드 값 업데이트: 자식 값 = sum(부모 값 * 가중치)
                new_value = sum(
                    self.nodes[p].value * self.edge_weights.get((p, child_id), 0.0)
                    for p in child_node.parents
                )
                child_node.value = new_value
                queue.append(child_id)

    def calculate_average_causal_effect(
        self, target_id: str, outcome_id: str, val_a: float, val_b: float
    ) -> float:
        """
        평균 인과 효과 (ACE) 연산:
        ACE = E[Y | do(X = val_a)] - E[Y | do(X = val_b)]
        """
        scm_a = self.apply_do_intervention(target_id, val_a)
        scm_b = self.apply_do_intervention(target_id, val_b)

        y_a = scm_a.nodes[outcome_id].value if outcome_id in scm_a.nodes else 0.0
        y_b = scm_b.nodes[outcome_id].value if outcome_id in scm_b.nodes else 0.0

        return y_a - y_b

    def get_current_state(self) -> Dict[str, float]:
        """현재 모든 노드의 상태 값 반환"""
        return {node_id: node.value for node_id, node in self.nodes.items()}
