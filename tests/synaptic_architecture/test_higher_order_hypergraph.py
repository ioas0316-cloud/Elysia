"""
[Tests for Higher-Order Hypergraph & Traversal Engine]
재귀적 하이퍼그래프 데이터 구조, 1급 과정(First-Class Process), 지연 평가, 맥락 주입,
동적 영향력 재조정, 포텐셜 이완 연산, 및 [햇빛 -> 바나나 곡률] 상위 위상 결합 시나리오 검증.
"""

import pytest
import numpy as np
from synaptic_architecture.higher_order_hypergraph import (
    HyperEntity,
    AtomicNode,
    ProcessNode,
    HigherOrderHypergraphEngine,
    PotentialRelaxationEngine
)
from synaptic_architecture.inverse_mechanism_engine import (
    BoundaryCondition,
    ObservedTrajectory,
    InverseMechanismEngine
)


def test_first_class_entities_and_composition():
    """1급 개체(HyperEntity) 및 AtomicNode / ProcessNode 재귀 구성 테스트"""
    engine = HigherOrderHypergraphEngine()

    atom_a = AtomicNode(name="Sunlight_Intensity", value=100.0)
    atom_b = AtomicNode(name="Gravity_Field", value=9.81)

    # Process 1: [햇빛 -> 바나나 곡률 변이 P1]
    p1_curvature = ProcessNode(
        name="Banana_Curvature_Process_P1",
        causal_operator="ADD",
        boundary_conditions={"scale": 1.2, "friction": 1.0}
    )
    engine.connect_process(p1_curvature, inputs=[atom_a], outputs=[])

    # Process 2: P1 과정 자체를 입력으로 받는 상위 과정 P2 [식물 생리학 메커니즘]
    p2_physiology = ProcessNode(
        name="Plant_Physiology_Meta_Process_P2",
        causal_operator="MULTIPLY",
        boundary_conditions={"scale": 0.9}
    )
    engine.connect_process(p2_physiology, inputs=[p1_curvature, atom_b], outputs=[])

    # 엔티티 등록 및 구조 확인
    assert engine.get_entity(atom_a.id) is not None
    assert engine.get_entity(p1_curvature.id) is not None
    assert engine.get_entity(p2_physiology.id) is not None
    assert p1_curvature.id in p2_physiology.inputs


def test_lazy_recursive_evaluation_and_context_injection():
    """지연된 재귀 평가 (Lazy Recursive Resolution) 및 맥락 주입 (Context Injection) 검증"""
    engine = HigherOrderHypergraphEngine()

    atom_input = AtomicNode(name="Input_Energy", value=50.0)
    result_atom = AtomicNode(name="Result_State", value=0.0)

    # 하위 프로세스 P1
    p1 = ProcessNode(
        name="Sub_Process_P1",
        causal_operator="ADD",
        boundary_conditions={"scale": 1.0, "friction": 1.0}
    )
    engine.connect_process(p1, inputs=[atom_input])

    # 상위 프로세스 P2 (P1의 연산 결과를 받아 처리)
    p2 = ProcessNode(
        name="Super_Process_P2",
        causal_operator="ADD",
        boundary_conditions={"scale": 2.0, "friction": 1.0}
    )
    engine.connect_process(p2, inputs=[p1], outputs=[result_atom])

    # 지연 평가 수행
    eval_res = engine.evaluate_lazy(result_atom.id)
    assert result_atom.value == eval_res
    assert eval_res == 100.0  # (50.0 * 1.0) * 2.0 = 100.0
    assert p2.id in result_atom.causal_trace

    # 맥락 주입 (Context Override)을 통한 동적 재평가
    # P1 scale=1.0, P2 override scale=3.0 -> 50.0 * 1.0 * 3.0 = 150.0
    eval_res_override = engine.evaluate_lazy(result_atom.id, context_override={"scale": 3.0})
    # context_override 에 의해 scale 이 오버라이드됨
    assert eval_res_override == 450.0  # 둘 다 scale=3.0 적용시 (50 * 3) * 3 = 450.0


def test_contextual_recalibration():
    """상태 전이의 동적 영향력 (Contextual Re-calibration) 검증"""
    engine = HigherOrderHypergraphEngine()

    node_x = AtomicNode(name="Node_X", value=10.0)
    p_connected = ProcessNode(
        name="Process_Connected",
        causal_operator="IDENTITY",
        boundary_conditions={"friction": 1.0, "temperature": 0.5},
        stiffness=1.0
    )
    engine.connect_process(p_connected, inputs=[node_x])

    # 새로운 관계/변이가 연결망에 주입되었을 때 연쇄 반응 트리거
    recal_result = engine.trigger_contextual_recalibration(
        trigger_entity_id=node_x.id,
        new_boundary_delta={"friction": 2.0, "gravity": 12.0}
    )

    assert recal_result["recalibrated_count"] >= 1
    assert p_connected.id in recal_result["affected_processes"]
    assert p_connected.boundary_conditions["gravity"] == 12.0
    assert p_connected.stiffness == 0.5  # 1.0 * (1.0 / 2.0)


def test_potential_relaxation_engine():
    """포텐셜 이완 연산기 (Potential Relaxation Engine) 수렴 테스트"""
    relax_engine = PotentialRelaxationEngine(state_dim=2)

    # 의도: (3.0, 4.0) 위치에 도달하고 싶음
    target = np.array([3.0, 4.0])
    relax_engine.set_intent(lambda s: float(np.sum((s - target) ** 2)))

    # 경계 제약: x + y <= 5.0
    relax_engine.add_boundary_constraint(
        constraint_fn=lambda s: float(np.maximum(0.0, (s[0] + s[1]) - 5.0)),
        penalty_weight=500.0
    )

    eq_state = relax_engine.relax_to_equilibrium(initial_state=[0.0, 0.0], steps=300, lr=0.01)

    # 제약 만족 및 포텐셜 수렴 확인 (x + y ~ 5.0 부근)
    assert eq_state[0] + eq_state[1] <= 5.1
    assert abs(eq_state[0] - eq_state[1]) < 1.0  # 균등 이완


def test_banana_curvature_meta_topology_scenario():
    """
    [햇빛 -> 바나나 곡률 변이] 상위 메타 위상 결합 시나리오 검증:
    [식물 생리학], [지구 중력장], [인간 인지 패턴] 프로세스가 재귀 결합되어 지연 평가되는 통합 시나리오
    """
    engine = HigherOrderHypergraphEngine()

    # 1. 관측 궤적 및 역메커니즘 Θ 역추출
    obs1 = ObservedTrajectory("traj1", "bc1", [[0.0, 0.0], [1.0, 0.5], [2.0, 2.0]])
    obs2 = ObservedTrajectory("traj2", "bc2", [[0.0, 0.0], [1.0, 0.8], [2.0, 3.0]])
    bc1 = BoundaryCondition("bc1", friction=1.0, scale=1.0)
    bc2 = BoundaryCondition("bc2", friction=1.5, scale=1.2)

    inv_engine = InverseMechanismEngine()
    mechanism_theta = inv_engine.extract_generating_mechanism("banana_curvature_mech", [obs1, obs2], {"bc1": bc1, "bc2": bc2})

    # 2. 하이퍼그래프 노드 형성
    sunlight_atom = AtomicNode(name="Sunlight_Vector", value=[1.0, 2.0])

    # P1: [햇빛 -> 바나나 곡률 변이] 과정 자체
    p1_curvature = ProcessNode(
        name="Banana_Curvature_Process_P1",
        causal_operator="MECHANISM_EXTRAPOLATION",
        mechanism=mechanism_theta,
        boundary_conditions={"friction": 1.0, "scale": 1.0, "steps": 4}
    )
    engine.connect_process(p1_curvature, inputs=[sunlight_atom])

    # P2: [지구 중력장 환경] 포텐셜 이완 연산
    p2_gravity = ProcessNode(
        name="Gravity_Field_Process_P2",
        causal_operator="RELAXATION",
        boundary_conditions={
            "state_dim": 2,
            "intent_target": np.array([2.0, 2.0]),
            "barrier": 3.0,
            "relaxation_steps": 100
        }
    )
    engine.connect_process(p2_gravity, inputs=[])

    # P_meta: [식물 생리학 & 인지 패턴 인식] 상위 결합 과정
    # P1의 궤적과 P2의 중력 평형 상태를 입력으로 받아 상위 인과 결합
    result_meta_node = AtomicNode(name="Meta_Integrated_Cognitive_Field")

    def meta_integration_fn(inputs, boundary):
        # inputs[0]: P1의 궤적 (list of states)
        # inputs[1]: P2의 평형 상태 (list)
        traj_p1 = inputs[0]
        eq_p2 = inputs[1]

        # 상위 위상적 결합: 궤적에 중력 평형 벡터를 투영
        coupled_states = []
        if isinstance(traj_p1, list):
            for st in traj_p1:
                coupled = [st[d] + eq_p2[d] * boundary.get("coupling_coeff", 0.5) for d in range(min(len(st), len(eq_p2)))]
                coupled_states.append(coupled)
        return {"coupled_trajectory": coupled_states, "meta_status": "INTEGRATED"}

    p_meta = ProcessNode(
        name="Meta_Cognitive_Integration_P_Meta",
        causal_operator="CUSTOM",
        custom_fn=meta_integration_fn,
        boundary_conditions={"coupling_coeff": 0.5}
    )
    engine.connect_process(p_meta, inputs=[p1_curvature, p2_gravity], outputs=[result_meta_node])

    # 3. 메타 시스템 지연 평가 수행
    final_output = engine.evaluate_lazy(result_meta_node.id)

    assert final_output["meta_status"] == "INTEGRATED"
    assert "coupled_trajectory" in final_output
    assert len(final_output["coupled_trajectory"]) == 4
    assert result_meta_node.value == final_output
    assert p_meta.id in result_meta_node.causal_trace
