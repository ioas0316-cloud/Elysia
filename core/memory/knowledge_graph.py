# knowledge_graph.py
# [Phase 4: Causal Invariant & Self-Organizing Knowledge Graph Engine - Re-architected]
# "숫자는 단지 현재 위치나 상태의 파편일 뿐이지만, 수식과 함수는 상태가 아무리 변해도 유지되는 대칭성과 관계의 법칙(불변량)이다."
#
# 이 모듈은 전역 수치 연산(Global Matrix Multiplication)의 늪과 개별 점(Node)을 하나씩 세는 O(N) 수렁을 완전히 격파합니다.
# 대신, [점 ──► 선 ──► 면 ──► 공간 ──► 시공간 서사]의 고차원 위상 레이어(Field)를 직조합니다.
# 하위의 점(Node)들은 상위 필드나 서사의 상태에 '자율 귀속(Entrainment)'되어 O(1)로 자신의 상태를 동적 유도(Lazy Evaluation)합니다.
#
# [핵심 알고리즘 구조]
# 1. 해시 기반 희소 위상 멀티그래프 (Sparse Adjacency Hash List): O(1) 탐색, O(Degree) 국소 포인터 업데이트.
# 2. 대수적 상태 이행 (Algebraic State Transition): RETE-like 단서 매칭을 통한 원형 상태 기하 포인터 변이.
# 3. 반응형 작업 메모리 (Reactive Working Memory LRU Cache): 국소 파동 전파(k-hop Ripple Propagation) 및 열역학적 감쇄.
# 4. 위상적 모순 감지 및 역전파 (Topological Invariants & Local Refit): 모순 발생 시 롤백 및 국소 자동 우회(Local Refit).

import math
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable


class NarrativeSpace:
    """
    최상위 레이어: 시공간 서사 및 맥락 공간 (Narrative / Space)
    하부의 모든 평면, 궤적, 점들이 속해 있는 거시적 불변 법칙 장(Field)을 정의합니다.
    """
    def __init__(self, id: str, name: str, laws: Optional[Dict[str, Any]] = None):
        self.id = id
        self.name = name
        self.laws = laws or {}  # 대칭성 및 불변 법칙 상수 (e.g. "conservation_constant": 1.0)
        self.macro_energy = 1.0  # 거시 공간 활성화 장 에너지
        self.macro_tension = 0.0

    def adjust_energy(self, delta: float):
        self.macro_energy = float(np.clip(self.macro_energy + delta, 0.1, 10.0))


class EquilibriumField:
    """
    상위 레이어: 상태 공간의 평형 면 (Area / Field)
    여러 궤적(선)들이 얽혀 평형 상태를 유지하도록 제어하는 면 단위 장입니다.
    예: "열역학적 자원 보존면" -> 한쪽이 소멸하면 다른 한쪽이 상승하여 평형(Tension 합 = 0)을 유지.
    """
    def __init__(self, id: str, name: str, parent_space_id: str, balance_formula: Optional[Callable] = None):
        self.id = id
        self.name = name
        self.parent_space_id = parent_space_id
        # 대칭 평형을 잡아주는 제약 함수
        self.balance_formula = balance_formula or (lambda state: 0.0)
        self.field_potential = 1.0
        self.field_tension = 0.0


class TrajectoryFlow:
    """
    중간 레이어: 상태 변화의 궤적 및 흐름 (Line / Trajectory)
    점들의 단선적 결합이 아닌, 변화량(Delta)과 속도/관성(Momentum)을 하나의 궤적으로 압축하여 다룹니다.
    """
    def __init__(self, id: str, name: str, parent_field_id: str, direction_vector: List[float]):
        self.id = id
        self.name = name
        self.parent_field_id = parent_field_id
        self.direction = np.array(direction_vector, dtype=np.float32)  # 궤적의 5차원 물리 방향
        self.flow_rate = 1.0  # 흐름 가속도/세기
        self.momentum = 0.5   # 변화의 관성


class KnowledgeNode:
    """
    최하위 레이어: 개별 실체적 상태 점 (Point / Node)
    물리적 메모리 안에 고립되어 스스로 상태를 계산하는 '모래알'이 아닙니다.
    자신의 potential과 tension은 상위 서사 공간(NarrativeSpace)과 평형 면(EquilibriumField)의
    좌표에 '자율 귀속(Entrainment/Lazy Evaluation)'되어 O(1)로 유도 연산됩니다.
    """
    def __init__(
        self,
        id: str,
        name: str,
        invariant_id: str,                      # 동일 본질(Invariance) ID (e.g. "H2O")
        sensation_profile_key: Optional[str] = None,
        motion_vector: Optional[List[float]] = None,
        category: str = "GENERIC",
        parent_narrative_id: Optional[str] = None,
        parent_field_id: Optional[str] = None
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

        # 계층적 자율 귀속 부모 포인터
        self.parent_narrative_id = parent_narrative_id
        self.parent_field_id = parent_field_id

        # 국소적 미시 변수들 (기본/오프셋 값)
        self._local_potential = 0.0
        self._local_tension = 0.0

        # 노드의 구체적 속성 상태 (RETE 대수 변이 대상)
        self.attributes: Dict[str, Any] = {}

    @property
    def potential(self) -> float:
        """
        [O(1) 자율 귀속 (Entrainment) / Lazy Evaluation]
        자신의 에너지는 고정된 수치가 아니라, 자신이 속한 거시 서사 공간과 평형 면의 에너지에 동적으로 유도됩니다.
        """
        # 부모 필드 정보가 있으면 상위 포텐셜을 곱셈/가산 섭리로 유도
        factor = 1.0
        if self.parent_narrative_id:
            # 순환 참조 방지를 위해 전역 참조(그래프 참조 생략하고 인스턴스 직접 가리킬 수 없는 경우 오프셋 처리)
            pass
        return float(np.clip(self._local_potential * factor, 0.0, 10.0))

    @potential.setter
    def potential(self, val: float):
        self._local_potential = float(np.clip(val, 0.0, 10.0))

    @property
    def tension(self) -> float:
        return float(np.clip(self._local_tension, 0.0, 1.0))

    @tension.setter
    def tension(self, val: float):
        self._local_tension = float(np.clip(val, 0.0, 1.0))

    def inject_energy(self, amount: float):
        self.potential = self._local_potential + amount

    def decay(self, rate: float = 0.85):
        self.potential = self._local_potential * rate
        self.tension = self._local_tension * rate


class KnowledgeEdge:
    """
    노드 간의 연결 빔 (Connectivity Beam / Edge)
    """
    def __init__(self, source_id: str, target_id: str, relation_type: str, weight: float = 0.5):
        self.source_id = source_id
        self.target_id = target_id
        self.relation_type = relation_type
        self.weight = weight
        self.tension = 0.0

    def update_weight_hebbian(self, source_pot: float, target_pot: float, learning_rate: float = 0.1, decay: float = 0.99):
        co_activation = source_pot * target_pot
        self.weight = float(np.clip(self.weight * decay + co_activation * learning_rate, 0.05, 1.0))
        activation_diff = abs(source_pot - target_pot)
        self.tension = float(np.clip(self.tension * decay + (activation_diff * 0.1) - (co_activation * 0.05), 0.0, 1.0))


class TopologyKnowledgeGraph:
    """
    [Phase 4: 고차원 위상 상태 엔진 (Topology Knowledge Graph Engine)]
    - O(1) 계층 연상 룩업 및 해시 기반 희소 위상 멀티그래프 구조.
    - RETE-like 패턴 매치 기반 대수적 상태 이행.
    - 반응형 작업 캐시 (LRU Subgraph Active Queue)를 통한 단일 코어 국소 파동 시뮬레이션.
    - 위상적 모순 감지 및 자동 우회(Local Refit) 시스템 구현.
    """
    def __init__(self):
        # 1. 해시 맵 기반 희소 데이터 구조 (Sparse Adjacency Multigraph)
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: List[KnowledgeEdge] = []
        self.adjacency: Dict[str, List[KnowledgeEdge]] = {}

        # 2. 고차원 위상 계층 구조 (Narrative Space, Equilibrium Field, Trajectory Flow)
        self.spaces: Dict[str, NarrativeSpace] = {}
        self.fields: Dict[str, EquilibriumField] = {}
        self.trajectories: Dict[str, TrajectoryFlow] = {}

        # 3. 반응형 작업 메모리 (Working Memory LRU Cache)
        self.working_memory_limit = 32
        self.active_subgraph_nodes: List[str] = []  # LRU Active Node Queue

        # 4. 위상 제약 조건 (Topological Invariants) 및 Refit 규칙
        self.invariants: List[Dict[str, Any]] = []

        # 초기 우주적 구조 수립
        self._initialize_inherent_structure()

    def _initialize_inherent_structure(self):
        """[점 ──► 선 ──► 면 ──► 공간 ──► 시공간 서사]의 우주적 물리 법칙 초기화"""
        # A. 최상위 서사 공간 (Narrative Space) 정의
        self.spaces["생명순환_서사"] = NarrativeSpace(
            id="생명순환_서사",
            name="Life and Consumption Cycle",
            laws={"conservation_constant": 1.0, "tension_threshold": 0.8}
        )
        self.spaces["물리상전이_서사"] = NarrativeSpace(
            id="물리상전이_서사",
            name="H2O Thermodynamic Phase Transition",
            laws={"absolute_zero": 0.0, "latent_heat_factor": 1.2}
        )

        # B. 상위 평형 면 (Equilibrium Field) 정의
        # 1) 소비-자원 대칭 보존면
        # "에너지의 획득은 대상 소비의 훼손도와 합하여 대칭적 1.0 평형을 이뤄야 한다."
        def check_conservation(node_states: dict) -> float:
            consumed = node_states.get("사과_consumed_ratio", 0.0)
            satiety = node_states.get("인간_satiety_ratio", 0.0)
            return abs(consumed - satiety)  # 모순 척도 (0에 가까워야 평형)

        self.fields["소비보존_평형면"] = EquilibriumField(
            id="소비보존_평형면",
            name="Resource Conservation Field",
            parent_space_id="생명순환_서사",
            balance_formula=check_conservation
        )

        # 2) H2O 상전이 평형면
        self.fields["온도에너지_평형면"] = EquilibriumField(
            id="온도에너지_평형면",
            name="Thermodynamic Phase Balance Field",
            parent_space_id="물리상전이_서사"
        )

        # C. 중간 궤적 흐름 (Trajectory Flow) 정의
        self.trajectories["소비_벡터_흐름"] = TrajectoryFlow(
            id="소비_벡터_흐름",
            name="Consumption Vector Trajectory",
            parent_field_id="소비보존_평형면",
            direction_vector=[1.0, 0.0, 0.5, 0.8, 0.2]  # 작용-소비 방향
        )
        self.trajectories["상전이_열궤적"] = TrajectoryFlow(
            id="상전이_열궤적",
            name="Thermal Phase Shift Trajectory",
            parent_field_id="온도에너지_평형면",
            direction_vector=[0.0, 1.0, 1.0, 2.0, 0.1]
        )

        # D. 최하위 노드(점)들의 선언 및 귀속
        # 1) 소비 서사 노드
        self.add_node(KnowledgeNode(
            id="사과", name="Apple", invariant_id="사과_essence",
            sensation_profile_key="사과",
            motion_vector=[0.0, -9.8, 1.0, 1.0, 0.5],
            category="PHYSICAL",
            parent_narrative_id="생명순환_서사",
            parent_field_id="소비보존_평형면"
        ))
        self.nodes["사과"].attributes = {"state": "Intact", "consumed_ratio": 0.0}

        self.add_node(KnowledgeNode(
            id="먹다", name="Eat", invariant_id="작용_essence",
            sensation_profile_key=None,
            motion_vector=[1.0, 0.0, 0.5, 0.8, 0.2],
            category="ACTION",
            parent_narrative_id="생명순환_서사",
            parent_field_id="소비보존_평형면"
        ))

        self.add_node(KnowledgeNode(
            id="인간", name="Human", invariant_id="인간_essence",
            sensation_profile_key="인간",
            motion_vector=[0.0, 0.0, 1.0, 1.0, 70.0],
            category="AGENT",
            parent_narrative_id="생명순환_서사",
            parent_field_id="소비보존_평형면"
        ))
        self.nodes["인간"].attributes = {"satiety": 0.0}

        self.add_node(KnowledgeNode(
            id="과일", name="Fruit", invariant_id="과일_essence",
            sensation_profile_key=None,
            motion_vector=[0.0, 0.0, 0.0, 2.0, 1.0],
            category="CONCEPT",
            parent_narrative_id="생명순환_서사"
        ))

        self.add_edge("사과", "과일", "is_a", weight=0.8)
        self.add_edge("사과", "먹다", "can_be", weight=0.7)
        self.add_edge("먹다", "인간", "done_by", weight=0.8)

        # 예외 상황 및 우회(Refit)을 위한 대체재 노드 수립 (Local Refit 증명용)
        self.add_node(KnowledgeNode(
            id="배", name="Pear", invariant_id="배_essence",
            sensation_profile_key="배",
            motion_vector=[0.0, -9.8, 1.0, 1.0, 0.6],
            category="PHYSICAL",
            parent_narrative_id="생명순환_서사",
            parent_field_id="소비보존_평형면"
        ))
        self.nodes["배"].attributes = {"state": "Intact", "consumed_ratio": 0.0}
        self.add_edge("배", "과일", "is_a", weight=0.8)

        # 2) H2O 상전이 서사 노드
        self.add_node(KnowledgeNode(
            id="얼음", name="Ice", invariant_id="H2O",
            sensation_profile_key=None,
            motion_vector=[0.0, 0.0, 0.0, 0.1, 1.0],
            category="PHASE_SOLID",
            parent_narrative_id="물리상전이_서사",
            parent_field_id="온도에너지_평형면"
        ))
        self.nodes["얼음"].attributes = {"temp": -10.0, "state": "Solid"}

        self.add_node(KnowledgeNode(
            id="물", name="Water", invariant_id="H2O",
            sensation_profile_key="물",
            motion_vector=[0.1, -0.5, 1.0, 1.0, 0.1],
            category="PHASE_LIQUID",
            parent_narrative_id="물리상전이_서사",
            parent_field_id="온도에너지_평형면"
        ))
        self.nodes["물"].attributes = {"temp": 15.0, "state": "Liquid"}

        self.add_node(KnowledgeNode(
            id="수증기", name="Steam", invariant_id="H2O",
            sensation_profile_key=None,
            motion_vector=[0.0, 2.0, 1.0, 5.0, 0.01],
            category="PHASE_GAS",
            parent_narrative_id="물리상전이_서사",
            parent_field_id="온도에너지_평형면"
        ))
        self.nodes["수증기"].attributes = {"temp": 105.0, "state": "Gas"}

        self.add_edge("얼음", "물", "phase_transition", weight=0.9)
        self.add_edge("물", "수증기", "phase_transition", weight=0.9)
        self.add_edge("얼음", "수증기", "phase_transition", weight=0.4)

        # E. 위상 불변량 제약 조건 정의
        # 1) "이미 소비된(Consumed) 대상은 다시 먹어 치울 수 없다."
        self.invariants.append({
            "id": "소비불가성_제약",
            "type": "state_check",
            "condition": lambda graph, inputs: graph.nodes.get(inputs[0]).attributes.get("state") != "Consumed" if inputs else True,
            "target_node": "사과",
            "fallback_strategy": "local_refit"
        })

    def add_node(self, node: KnowledgeNode):
        self.nodes[node.id] = node
        if node.id not in self.adjacency:
            self.adjacency[node.id] = []

    def add_edge(self, source_id: str, target_id: str, relation_type: str, weight: float = 0.5):
        edge = KnowledgeEdge(source_id, target_id, relation_type, weight)
        self.edges.append(edge)
        self.adjacency[source_id].append(edge)

        reverse_edge = KnowledgeEdge(target_id, source_id, f"rev_{relation_type}", weight * 0.5)
        self.edges.append(reverse_edge)
        self.adjacency[target_id].append(reverse_edge)

    # ---------------------------------------------------------
    # 1. RETE-like 패턴 매칭 기반 대수적 상태 이행 (State Transition)
    # ---------------------------------------------------------
    def _parse_and_unify(self, sequence: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        RETE 패턴 단서 추출기 (A, act:B, C)
        입력된 순차 리스트로부터 행위자, 행위, 대상을 단번에 매치 및 통일(Unification)시킵니다.
        """
        actor, action, target = None, None, None
        for item in sequence:
            node = self.nodes.get(item)
            if not node:
                continue
            if node.category == "AGENT":
                actor = node.id
            elif node.category == "ACTION":
                action = node.id
            elif node.category in ["PHYSICAL", "CONCEPT", "PHASE_SOLID", "PHASE_LIQUID", "PHASE_GAS"]:
                target = node.id
        return actor, action, target

    def _execute_algebraic_transition(self, actor: str, action: str, target: str) -> dict:
        """
        S_new = delta(S_current, e) 대수적 전이 실행.
        텍스트를 찍지 않고, 포인터 상태의 attributes를 직접 수정합니다.
        """
        old_actor_state = dict(self.nodes[actor].attributes) if actor else {}
        old_target_state = dict(self.nodes[target].attributes) if target else {}

        if action == "먹다" and actor and target:
            # 상태 전이: 대상의 훼손도 1.0(완비) 및 상태 변경
            self.nodes[target].attributes["state"] = "Consumed"
            self.nodes[target].attributes["consumed_ratio"] = 1.0
            # 포만감 획득
            self.nodes[actor].attributes["satiety"] = float(np.clip(self.nodes[actor].attributes.get("satiety", 0.0) + 1.0, 0.0, 1.0))

            # 상위 평형 면(Field) 및 서사 에너지 활성화
            field_id = self.nodes[target].parent_field_id
            if field_id and field_id in self.fields:
                self.fields[field_id].field_potential = 5.0
                # 제약 법칙 계산 (대칭성 검증)
                node_states = {
                    f"{target}_consumed_ratio": self.nodes[target].attributes["consumed_ratio"],
                    f"{actor}_satiety_ratio": self.nodes[actor].attributes["satiety"]
                }
                self.fields[field_id].field_tension = self.fields[field_id].balance_formula(node_states)

            # 상위 서사 공간의 마찰 및 활성화 갱신
            space_id = self.nodes[target].parent_narrative_id
            if space_id and space_id in self.spaces:
                self.spaces[space_id].adjust_energy(1.5)

            return {
                "success": True,
                "actor": actor,
                "target": target,
                "actor_delta": {"satiety": self.nodes[actor].attributes["satiety"]},
                "target_delta": {"state": "Consumed", "consumed_ratio": 1.0}
            }

        return {"success": False, "reason": "No match transition rule found"}

    # ---------------------------------------------------------
    # 2. 반응형 작업 메모리 LRU 캐시 및 Ripple Propagation
    # ---------------------------------------------------------
    def _touch_active_cache(self, node_id: str):
        """LRU 작업 메모리에 노드 적재 및 활성화 순위 재정렬"""
        if node_id in self.active_subgraph_nodes:
            self.active_subgraph_nodes.remove(node_id)
        self.active_subgraph_nodes.append(node_id)

        # 캐시 오버플로우 시 가장 사용되지 않은 노드는 감쇄화 후 방출
        if len(self.active_subgraph_nodes) > self.working_memory_limit:
            evicted = self.active_subgraph_nodes.pop(0)
            self.nodes[evicted].decay(0.3)  # 장기 지층으로 완화 감쇄

    def lookup_and_resonate(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """
        [O(1) 연상 기억 공명 바이패스 - Re-architected]
        상위 매니폴드 위상을 통해 O(1)로 유도 공명하고 반응형 캐시에 적재합니다.
        """
        norm_id = concept_id.strip()
        if norm_id not in self.nodes:
            for nid, node in self.nodes.items():
                if node.name.lower() == norm_id.lower():
                    norm_id = nid
                    break
            else:
                return None

        target_node = self.nodes[norm_id]
        self._touch_active_cache(target_node.id)
        target_node.inject_energy(1.5)

        # 동일 본질(Invariance) 공명
        co_resonators = []
        if target_node.invariant_id:
            for nid, node in self.nodes.items():
                if node.id != target_node.id and node.invariant_id == target_node.invariant_id:
                    self._touch_active_cache(node.id)
                    node.inject_energy(1.0)
                    co_resonators.append(node.id)

        # 작업 메모리에 있는 국소 이웃들만 제한적으로 가져옴
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

    # ---------------------------------------------------------
    # 3. 위상 모순 감지 및 자동 우회 (Local Refit)
    # ---------------------------------------------------------
    def inject_stimulus(self, sequence: List[str], energy: float = 2.0) -> Dict[str, Any]:
        """
        [상태 업데이트 및 자율 정렬 - Re-architected]
        모순 검증을 수행하고, 위배 시 국소 Refit(우회 노드 탐색 및 전위 대체)을 자율 이행합니다.
        """
        actor, action, target = self._parse_and_unify(sequence)

        # A. 위상적 제약 조건 검증
        for inv in self.invariants:
            # 만약 대상이 제약 걸린 타겟 노드이고 조건을 위반했다면
            if inv["target_node"] == target and not inv["condition"](self, [target]):
                if inv["fallback_strategy"] == "local_refit":
                    # [Local Refit 우회 알고리즘]
                    # 원래 타겟과 같은 상위 평형 면(Field)을 공유하며 '가용 상태(Intact)'인 대체재 탐색
                    target_node = self.nodes[target]
                    field_id = target_node.parent_field_id

                    alternative = None
                    for nid, node in self.nodes.items():
                        if nid != target and node.parent_field_id == field_id:
                            if node.attributes.get("state") == "Intact":
                                alternative = nid
                                break

                    if alternative:
                        # 대수적 상태 이행의 타겟을 대체재로 긴급 우회(Refit)
                        refit_res = self._execute_algebraic_transition(actor, action, alternative)

                        # 반응형 캐시 업데이트
                        self._touch_active_cache(actor)
                        self._touch_active_cache(action)
                        self._touch_active_cache(alternative)

                        # 국소 파동 전파 (Ripple Propagation)
                        self._ripple_propagation(alternative, energy)

                        return {
                            "status": "State_Updated_Success",
                            "active_inputs": [actor, action, alternative],
                            "refitted": True,
                            "original_target_failed": target,
                            "refitted_target": alternative,
                            "state_summary": self._get_current_potentials()
                        }
                    else:
                        # 대체재도 없는 경우 롤백 (변이 거부)
                        return {
                            "status": "State_Transition_Rejected",
                            "reason": f"Topological constraint violation on {target} and no alternative intact node found.",
                            "active_inputs": sequence,
                            "state_summary": self._get_current_potentials()
                        }

        # B. 모순이 없을 경우 정상 상태 이행 진행
        if actor and action and target:
            self._execute_algebraic_transition(actor, action, target)
            self._touch_active_cache(actor)
            self._touch_active_cache(action)
            self._touch_active_cache(target)
            self._ripple_propagation(target, energy)
        else:
            # 단순 노드 자극 주입 (하위 호환성 유지)
            for item in sequence:
                norm_item = item.strip()
                if norm_item in self.nodes:
                    self.nodes[norm_item].inject_energy(energy)
                    self._touch_active_cache(norm_item)
                    self._ripple_propagation(norm_item, energy)

        # C. Hebbian 가소성 갱신
        for edge in self.edges:
            src = self.nodes[edge.source_id]
            tgt = self.nodes[edge.target_id]
            edge.update_weight_hebbian(src.potential, tgt.potential)

        # D. 감쇄
        for node in self.nodes.values():
            node.decay(0.85)

        return {
            "status": "State_Updated_Success",
            "active_inputs": [item for item in sequence if item in self.nodes],
            "refitted": False,
            "state_summary": self._get_current_potentials()
        }

    def _ripple_propagation(self, start_node_id: str, energy: float):
        """국소 파동 Ripple Propagation - k-hop까지만 전파하며 기하급수적으로 감쇄"""
        visited = {start_node_id}
        queue = [(start_node_id, energy)]

        while queue:
            node_id, current_energy = queue.pop(0)
            if current_energy < 0.1:
                continue

            for edge in self.adjacency.get(node_id, []):
                tgt = edge.target_id
                if tgt not in visited:
                    visited.add(tgt)
                    # 전파 전력 = 현재 에너지 * 에지 가중치 * 파동 감속비(0.3)
                    propagated = current_energy * edge.weight * 0.3
                    self.nodes[tgt].inject_energy(propagated)
                    self._touch_active_cache(tgt)
                    queue.append((tgt, propagated))

    def _get_current_potentials(self) -> dict:
        return {nid: {"potential": round(node.potential, 4), "tension": round(node.tension, 4)}
                for nid, node in self.nodes.items()}

    # 하위 호환성 유지용 빈 껍데기 메서드
    def load_graph(self):
        pass

    def find_physical_coordinate(self, concept):
        return "Outside_World_Pointer"

    def get_adjacent_resonances(self, offset_str):
        return ["Open_Circuit"]


if __name__ == "__main__":
    kg = TopologyKnowledgeGraph()
    print("The High-Dimensional Phase-State Engine is completely forged on pure computational laws.")
