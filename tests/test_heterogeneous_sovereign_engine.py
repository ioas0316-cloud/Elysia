"""
tests/test_heterogeneous_sovereign_engine.py
=============================================
Comprehensive pytest suite for Heterogeneous Sovereign Engine (이질적 주권 및 위상 도약 엔진).
Verifies:
1. Heterogeneous 3-tier separation vs homogeneous scaling law comparison.
2. Top-down topological structural constraints (L1 Joint -> L0 Fluid degrees of freedom).
3. Bottom-up friction rupture (L0 Q_heat -> L1 Emergent Axis branching).
4. Controllability boundary partitioning (Controllable internal agency vs Uncontrollable external forces).
5. Social cross-validation field ("Me, You, Us") detecting deception, single-track hallucination, and uncanniness alarm.
6. Feasibility ("Can it be?") and sovereignty attribution ("Whose Sovereignty?") verification scenarios.
7. Master coordinator integrated causal flow.
"""

import pytest
import numpy as np

from L1_causal_field.core.sovereignty.heterogeneous_sovereign_engine import (
    L0FluidState,
    L0SubstrateFluidTier,
    MechanicalJointConstraint,
    L1CellularMechanicalTier,
    CausalProposalNode,
    L2SocialSovereigntyTier,
    HeterogeneousSovereignEngine
)


def test_1_heterogeneous_3tier_separation_vs_homogeneous_scaling():
    """
    [검증 1] 동일 연산 단순 무한 반복(Homogeneous Scaling)과 이질적 3계층 원리의 차별성 검증.
    L0 유체 원리만으로는 L1 관절 회전 각도나 L2 주권 평가를 처리할 수 없음을 명확히 증명.
    """
    l0_tier = L0SubstrateFluidTier(viscosity=1.5)
    fluid_state = L0FluidState(velocity_vector=np.array([2.0, 0.0, 0.0]), channel_radius=1.0)

    l0_res = l0_tier.compute_fluid_dynamics(fluid_state)
    assert "q_heat" in l0_res
    assert l0_res["v_norm"] == 2.0

    # Verify L0 tier alone has no joint or sovereignty logic (Heterogeneous separation)
    assert not hasattr(l0_tier, "cross_validate_social_field")
    assert not hasattr(l0_tier, "partition_controllability")


def test_2_top_down_structural_constraint():
    """
    [검증 2] 하향식 구조적 구속 (Top-Down Structural Constraint).
    상위 L1 관절의 회전 각도 및 근결 방향이 하위 L0 유체의 흐름 방향과 채널 반경을 종속시킴.
    """
    engine = HeterogeneousSovereignEngine(rupture_threshold=5.0)

    # Register joint constraint with 90 degree rotation and muscle fiber orientation along Y axis
    joint = MechanicalJointConstraint(
        joint_id="elbow_joint_01",
        rotation_axis=np.array([0.0, 0.0, 1.0]),
        max_angle_rad=np.pi / 2,
        current_angle_rad=np.pi / 4,  # Halfway bent
        muscle_fiber_orientation=np.array([0.0, 1.0, 0.0])  # Y direction
    )
    engine.l1_tier.add_joint_constraint(joint)

    initial_l0 = L0FluidState(velocity_vector=np.array([5.0, 0.0, 0.0]), channel_radius=1.0)

    # Execute flow with top-down constraint
    log = engine.execute_heterogeneous_causal_flow(initial_l0, active_joint_id="elbow_joint_01")

    # Velocity direction should now be constrained to Y axis (0, 1, 0) * speed(5.0)
    assert np.allclose(initial_l0.velocity_vector, np.array([0.0, 5.0, 0.0]))
    # Channel radius should be pinched due to joint bending angle
    assert initial_l0.channel_radius < 1.0


def test_3_bottom_up_friction_rupture_emergence():
    """
    [검증 3] 상향식 파열 창발 (Bottom-Up Friction Rupture Emergence).
    L0 마찰열(Q_heat)이 임계값(rupture_threshold)을 초과할 때, 상위 계층에서 새로운 위상 축(Emergent Axis) 자가 발아.
    """
    engine = HeterogeneousSovereignEngine(rupture_threshold=3.0)

    high_drag_l0 = L0FluidState(velocity_vector=np.array([10.0, 0.0, 0.0]), channel_radius=0.2)

    log = engine.execute_heterogeneous_causal_flow(high_drag_l0, external_resistance=5.0)

    assert log["q_heat"] > 3.0
    assert log["emergent_axis_triggered"] is True
    assert log["emergent_axis_data"]["origin_tier"] == "L1_CellularMechanical"
    assert "Heterogeneous Topological Recalibration" in log["emergent_axis_data"]["emergent_rule"]


def test_4_controllability_boundary_partitioning():
    """
    [검증 4] 통제 가능 vs 통제 불가능 영역 경계 분할 (Controllability Boundary Partitioning).
    내부 위상/개입 주권(Controllable)과 외부 절대적 외력/타인의 의도(Uncontrollable) 구분.
    """
    l2_tier = L2SocialSovereigntyTier()

    node_internal = CausalProposalNode(
        node_id="internal_phase_tuning",
        description=" 내면 장력 필드 조율 및 내 개입 축 수정",
        proposed_causal_sequence=["focus_attention", "adjust_tension"],
        is_internal_phase=True
    )

    node_external = CausalProposalNode(
        node_id="external_other_intention",
        description=" 상대방의 기분 및 외부 시장 가격 즉시 변경",
        proposed_causal_sequence=["force_market_change"],
        is_internal_phase=False
    )

    partitioned = l2_tier.partition_controllability([node_internal, node_external])

    assert len(partitioned["controllable_internal_agency"]) == 1
    assert partitioned["controllable_internal_agency"][0].node_id == "internal_phase_tuning"
    assert len(partitioned["uncontrollable_external_constraints"]) == 1
    assert partitioned["uncontrollable_external_constraints"][0].node_id == "external_other_intention"


def test_5_social_cross_validation_deception_alarm():
    """
    [검증 5] 사회적 다중 스케일 교차 검증 필드 ("나와 너, 그리고 우리").
    언어축(+1.0), 미세표정축(-0.9), 맥락장력축(-0.8)의 모순 감지 시 기만 경보 발령.
    """
    l2_tier = L2SocialSovereigntyTier(uncanniness_sensitivity=1.2)

    # Deceptive claim scenario
    res = l2_tier.cross_validate_social_field(
        me_speech_signal=1.0,        # Speech: "Guaranteed 100x return, zero risk!"
        you_expression_signal=-0.9,  # Micro-expression: Anxiety, pupil tremor
        us_field_tension=-0.8        # Context: High relational friction
    )

    assert res["deception_alarm"] is True
    assert res["field_verdict"] == "SINGLE_TRACK_DECEPTIVE_HALLUCINATION_DETECTED"
    assert res["q_heat_spike"] > 1.5

    # Harmonious scenario
    harmonious_res = l2_tier.cross_validate_social_field(
        me_speech_signal=0.8,
        you_expression_signal=0.7,
        us_field_tension=0.8
    )
    assert harmonious_res["deception_alarm"] is False
    assert harmonious_res["field_verdict"] == "HARMONIOUS_CROSS_VALIDATED_ALIGNMENT"


def test_6_feasibility_and_sovereignty_attribution_audit():
    """
    [검증 6] 실현가능성("Can it be?") 및 주권 귀속("Whose Sovereignty?") 검증 시나리오.
    사기꾼의 기만적 제안(마찰 대가는 '나', 열매는 '타인', sensory grounding ~ 0) 정밀 차단.
    """
    engine = HeterogeneousSovereignEngine()

    # Deceptive proposal node (Scam / Exploitative trap)
    scam_proposal = CausalProposalNode(
        node_id="scam_investment_trap",
        description="허황된 무위험 고수익 투자 제안",
        proposed_causal_sequence=["transfer_capital", "wait_for_miracle"],
        sensory_grounding_score=0.1,  # Floating words, zero physical friction backing
        friction_cost_bearer="me",     # I bear the risk and loss
        fruit_recipient="other",       # Other takes the profit
        is_internal_phase=False
    )

    audit_res = engine.audit_proposal_sovereignty_and_feasibility(scam_proposal)

    assert audit_res["is_authentic_and_sovereign"] is False
    assert audit_res["feasibility_assessment"]["can_it_be"] is False
    assert audit_res["sovereignty_assessment"]["exploitation_detected"] is True
    assert audit_res["verdict"] == "REJECTED_UNFOUNDED_OR_EXPLOITATIVE_HALLUCINATION"

    # Authentic sovereign action proposal
    sovereign_proposal = CausalProposalNode(
        node_id="authentic_skill_building",
        description="직접 마찰을 견디며 기술과 능력을 연마하는 과정",
        proposed_causal_sequence=["practice_action", "experience_friction", "master_skill"],
        sensory_grounding_score=0.95, # Directly grounded in physical practice
        friction_cost_bearer="me",
        fruit_recipient="me",
        is_internal_phase=True
    )

    authentic_audit = engine.audit_proposal_sovereignty_and_feasibility(sovereign_proposal)

    assert authentic_audit["is_authentic_and_sovereign"] is True
    assert authentic_audit["feasibility_assessment"]["can_it_be"] is True
    assert authentic_audit["sovereignty_assessment"]["is_sovereign_aligned"] is True
    assert authentic_audit["verdict"] == "VERIFIED_AUTHENTIC_SOVEREIGN_ACTION"


def test_7_integrated_heterogeneous_sovereign_flow():
    """
    [검증 7] 통합 마스터 코디네이터 이질적 인과 흐름 전체 루프 검증.
    """
    engine = HeterogeneousSovereignEngine(rupture_threshold=2.0)

    joint = MechanicalJointConstraint(
        joint_id="shoulder_joint",
        rotation_axis=np.array([1.0, 0.0, 0.0]),
        muscle_fiber_orientation=np.array([1.0, 0.0, 0.0])
    )
    engine.l1_tier.add_joint_constraint(joint)

    l0_state = L0FluidState(velocity_vector=np.array([3.0, 0.0, 0.0]), channel_radius=1.0)

    log = engine.execute_heterogeneous_causal_flow(
        l0_state=l0_state,
        active_joint_id="shoulder_joint",
        external_resistance=1.0,
        social_signals=(1.0, -0.9, -0.8)
    )

    assert log["emergent_axis_triggered"] is True
    assert log["l2_cross_validation"]["deception_alarm"] is True
    assert len(engine.execution_logs) == 1
