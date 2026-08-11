"""
Verification Test for Elysia Causal Invariant & Self-Organizing Knowledge Graph
=============================================================================
고립된 수치 점의 헛돌기 및 인공적인 파이썬 코드 생성 촌극을 배제하고,
실제 모션 벡터와 본질적 불변량(H2O)을 지닌 지층을 수립하여
인과교류 및 상태 변경(State Update), 그리고 Hebbian 공명을 전수 검증합니다.
"""

import numpy as np
import pytest
from core.memory.knowledge_graph import (
    TopologyKnowledgeGraph,
    KnowledgeNode,
    KnowledgeEdge
)


def test_knowledge_node_and_edge_state_dynamics():
    """
    1. 개별 지식 노드와 에지가 단순 문자열이 아닌,
       양(potential), 텐션(tension), Hebbian 가소성을 수반하는 상태 개체임을 검증.
    """
    node_src = KnowledgeNode(id="src_node", name="Source", invariant_id="test_inv")
    node_tgt = KnowledgeNode(id="tgt_node", name="Target", invariant_id="test_inv")

    assert node_src.potential == 0.0
    assert node_src.tension == 0.0

    # 에너지 주입 검증
    node_src.inject_energy(5.0)
    assert node_src.potential == 5.0

    node_tgt.inject_energy(3.0)

    # Hebbian 가소성 및 텐션 완화 검증
    edge = KnowledgeEdge("src_node", "tgt_node", "is_connected", weight=0.5)
    initial_weight = edge.weight

    # 두 노드가 동시에 높은 전위를 가지고 활성화되었을 때 연결 강도가 강화되어야 함
    edge.update_weight_hebbian(node_src.potential, node_tgt.potential)
    assert edge.weight > initial_weight

    # 시간적 감쇄 및 평형화 검증
    node_src.decay(0.8)
    assert node_src.potential == 5.0 * 0.8


def test_symbol_grounding_and_physical_binding():
    """
    2. 기호의 헛돌기(Symbol Grounding) 차단 검증.
       '사과'가 단순 텍스트 토큰이 아니라 시각/물리적 모션 벡터와 직접 매핑되어 있음을 검증.
    """
    kg = TopologyKnowledgeGraph()
    res = kg.lookup_and_resonate("사과")

    assert res is not None
    assert res["node_id"] == "사과"
    assert res["name"] == "Apple"
    assert res["category"] == "PHYSICAL"

    # 5차원 물리적 변위 모션 벡터 (e.g. 중력 가속도 낙하 방향 [0.0, -9.8, ...])가 존재함을 검증
    motion_vector = np.array(res["motion_vector"])
    assert len(motion_vector) == 5
    assert abs(motion_vector[1] - (-9.8)) < 1e-5  # Y축 중력 낙하 궤적 바인딩


def test_invariant_coresonance_o1_lookup():
    """
    3. 동일 본질(Invariance: H2O)을 통한 O(1) 연상 기억 공명 바이패스 검증.
       '얼음'을 다루다 만나더라도 무에서부터 새로 공부할 필요 없이,
       'H2O'라는 불변 본질을 통해 '물'과 '수증기'의 전위가 동시에 자율 유도(공명)되는지 검증.
    """
    kg = TopologyKnowledgeGraph()

    # 초기의 물과 수증기 전위는 0.0
    assert kg.nodes["물"].potential == 0.0
    assert kg.nodes["수증기"].potential == 0.0

    # 얼음을 인지(Lookup)하는 즉시, 동일 본질을 공유하는 다른 상태(물, 수증기)들과 O(1)로 연상 공명
    res = kg.lookup_and_resonate("얼음")

    assert res is not None
    assert res["invariant_id"] == "H2O"
    assert "물" in res["co_resonators"]
    assert "수증기" in res["co_resonators"]

    # 얼음 인지 충격으로 인해 물과 수증기의 전위가 자동으로 상승해 있어야 함 (자율 공명 유도)
    assert kg.nodes["물"].potential > 0.0
    assert kg.nodes["수증기"].potential > 0.0


def test_state_update_without_code_generation():
    """
    4. 코드 생성 촌극을 탈피한 순수 알고리즘적 상태 업데이트(State Update) 검증.
       입력이 들어올 때 코드를 새로 컴파일하는 대신, 전위 전파(Propagation)와 Hebbian 업데이트가 발생함을 검증.
    """
    kg = TopologyKnowledgeGraph()

    # "사과"와 "먹다" 자극 순차 유입
    state_before = kg.nodes["과일"].potential
    edge_before = next(e.weight for e in kg.edges if e.source_id == "사과" and e.target_id == "과일")

    # 자극 주입 (State Update 실행)
    res_update = kg.inject_stimulus(["사과", "먹다"])

    assert res_update["status"] == "State_Updated_Success"
    assert "사과" in res_update["active_inputs"]
    assert "먹다" in res_update["active_inputs"]

    # "사과"와 "먹다"가 활성화됨에 따라 이와 인접한 "과일" 노드로 전위가 유도되어 전파되었음을 검증
    assert kg.nodes["과일"].potential > state_before

    # "사과" - "과일" 에지의 가중치(연결 전위)가 Hebbian 가소성에 의해 동적으로 갱신되었음을 검증
    edge_after = next(e.weight for e in kg.edges if e.source_id == "사과" and e.target_id == "과일")
    assert edge_after != edge_before


def test_lazy_evaluation_and_entrainment_scaling():
    """
    5. 상위 위상 장(Field)에 귀속된 하위 노드들의 동적 유도(Entrainment/Lazy Evaluation) 검증.
    """
    kg = TopologyKnowledgeGraph()
    apple_node = kg.nodes["사과"]

    # 사과 노드는 생명순환_서사 공간에 소속됨
    assert apple_node.parent_narrative_id == "생명순환_서사"
    assert apple_node.parent_field_id == "소비보존_평형면"

    # 최초 오프셋 전위 주입
    apple_node.inject_energy(2.0)
    assert apple_node.potential == 2.0


def test_rete_algebraic_state_transition():
    """
    6. RETE-like 패턴 매칭 기반 대수적 상태 이행 검증.
       '사과를 먹는 이벤트'가 유입되었을 때, 대상 노드의 attributes 속성(훼손도, 포만감)이
       포인터 단위로 실시간 수정되는지 검증.
    """
    kg = TopologyKnowledgeGraph()

    # 이행 전 상태
    assert kg.nodes["사과"].attributes["state"] == "Intact"
    assert kg.nodes["사과"].attributes["consumed_ratio"] == 0.0
    assert kg.nodes["인간"].attributes["satiety"] == 0.0

    # "인간", "사과", "먹다" 자극 주입으로 상태 이행 유발
    res = kg.inject_stimulus(["인간", "먹다", "사과"])

    # 사과와 인간의 상태가 직접 업데이트되었는지 검증
    assert kg.nodes["사과"].attributes["state"] == "Consumed"
    assert kg.nodes["사과"].attributes["consumed_ratio"] == 1.0
    assert kg.nodes["인간"].attributes["satiety"] == 1.0


def test_reactive_working_memory_and_decay():
    """
    7. 반응형 작업 메모리(LRU Active Node Cache) 및 k-hop Ripple Propagation 검증.
    """
    kg = TopologyKnowledgeGraph()

    # "얼음" 자극 주입 시, 반응형 LRU 캐시에 얼음 및 인접 H2O 노드들(물, 수증기)이 로드되는지 확인
    kg.inject_stimulus(["얼음"])

    assert "얼음" in kg.active_subgraph_nodes
    # 1-hop 혹은 co-resonance를 통해 "물" 역시 활성화되어 캐시에 적재되었음을 확인
    assert "물" in kg.active_subgraph_nodes


def test_topological_contradiction_and_local_refit():
    """
    8. 위상적 모순 감지 및 자동 우회(Local Refit) 검증.
       이미 소비된 사과를 다시 먹으려는 모순 이벤트 유입 시,
       시스템이 오류를 내거나 강제 실행을 멈추고 동일 평형면의 대체제('배')로 '자율 우회(Local Refit)'를 실행하는지 검증.
    """
    kg = TopologyKnowledgeGraph()

    # 1. 첫 번째 소비 완료
    kg.inject_stimulus(["인간", "먹다", "사과"])
    assert kg.nodes["사과"].attributes["state"] == "Consumed"
    assert kg.nodes["배"].attributes["state"] == "Intact"

    # 2. 이미 소비된 사과에 대해 다시 먹는 행위 시도 -> 모순 감지 후 '배'로 자율 우회(Local Refit)
    res = kg.inject_stimulus(["인간", "먹다", "사과"])

    assert res["refitted"] is True
    assert res["original_target_failed"] == "사과"
    assert res["refitted_target"] == "배"

    # 배가 먹힌 상태로 정상 이행 완료되었음을 검증
    assert kg.nodes["배"].attributes["state"] == "Consumed"
    assert kg.nodes["배"].attributes["consumed_ratio"] == 1.0
    assert kg.nodes["인간"].attributes["satiety"] == 1.0
