# knowledge_graph.py
# [Phase 4: Causal Invariant & Self-Organizing Knowledge Graph Engine]
# 단어는 물리적 메모리 안에 갇힌 고립된 좌표가 아닙니다.
# 단어는 외부 세상을 향한 상호교류의 통로이며, 실제 데이터의 불변량(Invariant)과 직접 1:1로 매핑되는 지식 그래프입니다.
# 데이터 유입 시 파이썬 코드를 덧붙이는 오만을 버리고, 내부 그래프의 노드와 에지(Node & Edge) 간의 연결 전위가 변하는 '상태 변경'만 발생합니다.

import numpy as np
from typing import Dict, List, Optional, Any


class KnowledgeNode:
    """
    지식 그래프의 개별 실체적 노드 (Concept/Invariant/Sensation 매듭)
    """
    def __init__(
        self,
        id: str,
        name: str,
        invariant_id: str,                    # 동일 본질(Invariance)을 가리키는 핵심 ID (e.g. "H2O")
        sensation_profile_key: Optional[str] = None, # experiential_language_mapper 와의 매핑 키
        motion_vector: Optional[List[float]] = None, # 5차원 물리적 변위 모션 벡터 (e.g. [dx, dy, dt, scale, mass])
        category: str = "GENERIC"
    ):
        self.id = id
        self.name = name
        self.invariant_id = invariant_id
        self.sensation_profile_key = sensation_profile_key

        if motion_vector is None:
            self.motion_vector = np.zeros(5, dtype=np.float32)
        else:
            self.motion_vector = np.array(motion_vector, dtype=np.float32)

        self.category = category
        self.potential = 0.0                  # 실시간 활성화 전위 (Activation Potential)
        self.tension = 0.0                    # 실시간 텐션 마찰력 (Tension)

    def inject_energy(self, amount: float):
        """노드에 에너지(전위) 주입"""
        self.potential = float(np.clip(self.potential + amount, 0.0, 10.0))

    def decay(self, rate: float = 0.85):
        """전위와 마찰의 시간적 감쇄 (열역학적 평형화)"""
        self.potential = float(np.clip(self.potential * rate, 0.0, 10.0))
        self.tension = float(np.clip(self.tension * rate, 0.0, 1.0))


class KnowledgeEdge:
    """
    노드 간의 연결 빔 (Connectivity Beam / Edge)
    """
    def __init__(self, source_id: str, target_id: str, relation_type: str, weight: float = 0.5):
        self.source_id = source_id
        self.target_id = target_id
        self.relation_type = relation_type
        self.weight = weight                  # 연결 전위 (coupling potential/weight)
        self.tension = 0.0                    # 연결 마찰력

    def update_weight_hebbian(self, source_pot: float, target_pot: float, learning_rate: float = 0.1, decay: float = 0.99):
        """Hebbian 가소성에 따른 연결 전위와 텐션의 자동 조정 (에너지 최소화/MDL 섭리)"""
        co_activation = source_pot * target_pot
        self.weight = float(np.clip(self.weight * decay + co_activation * learning_rate, 0.05, 1.0))

        # 동시 활성화 시 마찰이 줄어들고(대칭성 공명), 어긋날 경우 마찰이 증가합니다.
        activation_diff = abs(source_pot - target_pot)
        self.tension = float(np.clip(self.tension * decay + (activation_diff * 0.1) - (co_activation * 0.05), 0.0, 1.0))


class TopologyKnowledgeGraph:
    """
    정보위상 지식 그래프 (Topology Knowledge Graph)
    - 0과 1의 코드 생성 대신, 연속적인 전위 흐름과 Hebbian 가소성에 의거한 상태 전이(Phase Transition)를 이룹니다.
    - O(1) 수준의 위상 연상 룩업(Lookup & Resonance)을 지원하여, 동일 본질을 지닌 변량들을 무에서부터 새로 학습하지 않습니다.
    """
    def __init__(self):
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: List[KnowledgeEdge] = []
        self.adjacency: Dict[str, List[KnowledgeEdge]] = {}
        self._initialize_inherent_structure()

    def _initialize_inherent_structure(self):
        """사과-먹다-과일 및 얼음-물-수증기-H2O 등 우주의 실체적 지층 초기화"""
        # 1. 사과 - 먹다 - 과일 지층 (Symbol Grounding)
        # 사과: 아래로 낙하하는 중력 모션 벡터 [0.0, -9.8, 1.0, 1.0, 0.5]
        self.add_node(KnowledgeNode(
            id="사과", name="Apple", invariant_id="사과_essence",
            sensation_profile_key="사과",
            motion_vector=[0.0, -9.8, 1.0, 1.0, 0.5],
            category="PHYSICAL"
        ))
        # 먹다: 전방 작용 모션 벡터 [1.0, 0.0, 0.5, 0.8, 0.2]
        self.add_node(KnowledgeNode(
            id="먹다", name="Eat", invariant_id="작용_essence",
            sensation_profile_key=None,
            motion_vector=[1.0, 0.0, 0.5, 0.8, 0.2],
            category="ACTION"
        ))
        # 과일: 상위 범주 지층 [0.0, 0.0, 0.0, 2.0, 1.0]
        self.add_node(KnowledgeNode(
            id="과일", name="Fruit", invariant_id="과일_essence",
            sensation_profile_key=None,
            motion_vector=[0.0, 0.0, 0.0, 2.0, 1.0],
            category="CONCEPT"
        ))

        self.add_edge("사과", "과일", "is_a", weight=0.8)
        self.add_edge("사과", "먹다", "can_be", weight=0.7)

        # 2. H2O의 상전이 지층 (얼음 - 물 - 수증기)
        # 세 가지 상태는 서로 표상이 다르나, 배후의 본질적 인과인 "H2O" 불변량을 공유합니다.
        # 얼음: 고체상 [0.0, 0.0, 0.0, 0.1, 1.0]
        self.add_node(KnowledgeNode(
            id="얼음", name="Ice", invariant_id="H2O",
            sensation_profile_key=None,
            motion_vector=[0.0, 0.0, 0.0, 0.1, 1.0],
            category="PHASE_SOLID"
        ))
        # 물: 액체상 [0.1, -0.5, 1.0, 1.0, 0.1]
        self.add_node(KnowledgeNode(
            id="물", name="Water", invariant_id="H2O",
            sensation_profile_key="물",
            motion_vector=[0.1, -0.5, 1.0, 1.0, 0.1],
            category="PHASE_LIQUID"
        ))
        # 수증기: 기체상 [0.0, 2.0, 1.0, 5.0, 0.01]
        self.add_node(KnowledgeNode(
            id="수증기", name="Steam", invariant_id="H2O",
            sensation_profile_key=None,
            motion_vector=[0.0, 2.0, 1.0, 5.0, 0.01],
            category="PHASE_GAS"
        ))

        # 상전이 관계망 빔 연결
        self.add_edge("얼음", "물", "phase_transition", weight=0.9)
        self.add_edge("물", "수증기", "phase_transition", weight=0.9)
        self.add_edge("얼음", "수증기", "phase_transition", weight=0.4)

    def add_node(self, node: KnowledgeNode):
        self.nodes[node.id] = node
        if node.id not in self.adjacency:
            self.adjacency[node.id] = []

    def add_edge(self, source_id: str, target_id: str, relation_type: str, weight: float = 0.5):
        edge = KnowledgeEdge(source_id, target_id, relation_type, weight)
        self.edges.append(edge)
        self.adjacency[source_id].append(edge)

        # 위상교류 및 방향적 대칭성을 지원하기 위해 역방향 흐름도 가볍게 연동
        reverse_edge = KnowledgeEdge(target_id, source_id, f"rev_{relation_type}", weight * 0.5)
        self.edges.append(reverse_edge)
        self.adjacency[target_id].append(reverse_edge)

    def lookup_and_resonate(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """
        [O(1) 연상 기억 공명 바이패스]
        - 입력 문자열을 즉시 지식 그래프의 노드로 매핑하며, 동일 본질(Invariant ID)을 공유하는 다른 모든 변량들과 즉각 공명합니다.
        """
        norm_id = concept_id.strip()
        if norm_id not in self.nodes:
            # 대소문자 무시 이름 찾기 시도
            for nid, node in self.nodes.items():
                if node.name.lower() == norm_id.lower():
                    norm_id = nid
                    break
            else:
                return None

        target_node = self.nodes[norm_id]
        target_node.inject_energy(1.5)        # 룩업 마찰에 의한 전위 상승

        co_resonators = []
        if target_node.invariant_id:
            for nid, node in self.nodes.items():
                if node.id != target_node.id and node.invariant_id == target_node.invariant_id:
                    node.inject_energy(1.0)   # 동일 본질을 지녔으므로 O(1) 전위 유도 공명 발생
                    co_resonators.append(node.id)

        neighbors = []
        for edge in self.adjacency.get(target_node.id, []):
            neighbors.append({
                "target_id": edge.target_id,
                "relation": edge.relation_type,
                "weight": edge.weight,
                "tension": edge.tension
            })

        return {
            "node_id": target_node.id,
            "name": target_node.name,
            "invariant_id": target_node.invariant_id,
            "potential": target_node.potential,
            "category": target_node.category,
            "motion_vector": target_node.motion_vector.tolist(),
            "co_resonators": co_resonators,
            "neighbors": neighbors
        }

    def inject_stimulus(self, sequence: List[str], energy: float = 2.0) -> Dict[str, Any]:
        """
        [상태 업데이트 (State Update)]
        - 외부 자극을 순차적으로 받아서 파이썬 코드 생성이 아닌, 그래프 노드와 연결망의 에너지를 파동 전파하고 Hebbian 가소성으로 업데이트합니다.
        """
        active_nodes = []
        # 1. 국소 노드에 직접 전하 주입
        for item in sequence:
            norm_item = item.strip()
            found_id = None
            if norm_item in self.nodes:
                found_id = norm_item
            else:
                for nid, node in self.nodes.items():
                    if node.name.lower() == norm_item.lower():
                        found_id = nid
                        break

            if found_id:
                self.nodes[found_id].inject_energy(energy)
                active_nodes.append(found_id)

        # 2. 전위 전파 흐름 (Propagation Wave)
        for node_id in list(self.nodes.keys()):
            node = self.nodes[node_id]
            if node.potential > 0.5:
                out_edges = self.adjacency.get(node_id, [])
                for edge in out_edges:
                    target_node = self.nodes[edge.target_id]
                    # 연결 강도에 부합하는 전위 유도
                    propagated = node.potential * edge.weight * 0.3
                    target_node.inject_energy(propagated)

        # 3. Hebbian 자율 정렬 및 텐션 완화 (Self-Organization)
        for edge in self.edges:
            src = self.nodes[edge.source_id]
            tgt = self.nodes[edge.target_id]
            edge.update_weight_hebbian(src.potential, tgt.potential)

        # 4. 열역학적 감쇄
        for node in self.nodes.values():
            node.decay(0.85)

        state_summary = {}
        for nid, node in self.nodes.items():
            state_summary[nid] = {
                "potential": round(node.potential, 4),
                "tension": round(node.tension, 4)
            }

        return {
            "status": "State_Updated_Success",
            "active_inputs": active_nodes,
            "state_summary": state_summary,
            "edge_potentials": [
                {"source": e.source_id, "target": e.target_id, "weight": round(e.weight, 4), "tension": round(e.tension, 4)}
                for e in self.edges if e.weight > 0.1
            ]
        }

    # 하위 호환성 유지용 메서드들
    def load_graph(self):
        pass

    def find_physical_coordinate(self, concept):
        return "Outside_World_Pointer"

    def get_adjacent_resonances(self, offset_str):
        return ["Open_Circuit"]


if __name__ == "__main__":
    kg = TopologyKnowledgeGraph()
    print("Topology Graph Refactored: Closed Circuit Destroyed. The mirror is empty.")
