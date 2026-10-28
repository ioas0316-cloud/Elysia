"""
tests/test_heterogeneous_sovereign_engine.py
=============================================
Comprehensive pytest suite for Heterogeneous Sovereign Engine (이질적 주권 및 다차원 삼위일체 위상 도약 엔진).
Verifies:
1. Heterogeneous 3-tier separation vs homogeneous scaling law comparison.
2. Top-down topological structural constraints (L1 Joint -> L0 Fluid degrees of freedom).
3. Bottom-up friction rupture (L0 Q_heat -> L1 Emergent Axis branching).
4. Environmental Axis (gravity, temperature, impact drag) interaction with L0 fluid substrate.
5. Mind tier micro synaptic heat, macro topological tension field, and deficit-driven tool sprouting.
6. Heart tier Action-Grounded Truth ("Faith/Love with Action" principle) vs floating speech hallucinations.
7. Controllability boundary partitioning and social cross-validation field ("Me, You, Us").
8. Integrated Multi-Scale Trinitarian (Body - Mind - Heart) master flow.
"""

import pytest
import numpy as np

from L1_causal_field.core.sovereignty.heterogeneous_sovereign_engine import (
    EnvironmentalAxis,
    L0FluidState,
    L0SubstrateFluidTier,
    MechanicalJointConstraint,
    L1CellularMechanicalTier,
    SynapticState,
    MindCognitionTier,
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

    joint = MechanicalJointConstraint(
        joint_id="elbow_joint_01",
        rotation_axis=np.array([0.0, 0.0, 1.0]),
        max_angle_rad=np.pi / 2,
        current_angle_rad=np.pi / 4,  # Halfway bent
        muscle_fiber_orientation=np.array([0.0, 1.0, 0.0])  # Y direction
    )
    engine.l1_tier.add_joint_constraint(joint)

    initial_l0 = L0FluidState(velocity_vector=np.array([5.0, 0.0, 0.0]), channel_radius=1.0)

    log = engine.execute_heterogeneous_causal_flow(initial_l0, active_joint_id="elbow_joint_01")

    assert np.allclose(initial_l0.velocity_vector, np.array([0.0, 5.0, 0.0]))
    assert initial_l0.channel_radius < 1.0


def test_3_environmental_axis_and_bottom_up_rupture_emergence():
    """
    [검증 3] 환경의 축 (Environmental Axis) 연동 및 상향식 파열 창발.
    중력, 물리적 충격 및 마찰 항력이 Q_heat를 상승시켜 상위 축(Emergent Axis) 자가 발아 유도.
    """
    engine = HeterogeneousSovereignEngine(rupture_threshold=3.0)

    env = EnvironmentalAxis(
        gravity=9.81,
        external_drag_coefficient=0.8,
        impact_force_vector=np.array([5.0, 0.0, 0.0])
    )

    high_drag_l0 = L0FluidState(velocity_vector=np.array([4.0, 0.0, 0.0]), channel_radius=0.5)

    log = engine.execute_heterogeneous_causal_flow(high_drag_l0, env_axis=env)

    assert log["q_env"] > 0.0
    assert log["q_heat"] > 3.0
    assert log["emergent_axis_triggered"] is True
    assert log["emergent_axis_data"]["origin_tier"] == "L1_CellularMechanical"


def test_4_mind_tier_synaptic_tension_and_tool_sprouting():
    """
    [검증 4] 정신 (Mind) 다차원 스케일 검증:
    1. 시냅스 전위 마찰열과 감각 불일치(Delta)에 의한 위상 장력 및 위화감 경보.
    2. 신체적 결핍 자각 시 박쥐/외부 현상 관측을 통한 도구/레이더 축 자가 덧대기.
    """
    mind_tier = MindCognitionTier(deficit_threshold=2.5)

    synaptic = SynapticState(potential_difference_mv=80.0, neurotransmitter_flux=2.0, synaptic_heat_dissipation=0.2)
    tension_res = mind_tier.compute_topological_tension_field(synaptic_state=synaptic, sensory_mismatch_delta=1.5)

    assert tension_res["tension_field"] > 2.0
    assert tension_res["uncanniness_alarm"] is True

    tool_sprout = mind_tier.evaluate_deficit_awareness_and_tool_sprouting(
        physical_limitation_deficit=4.0,
        observed_external_entity="bat_radar_echolocation"
    )

    assert tool_sprout is not None
    assert tool_sprout["extension_type"] == "RADAR_OR_LANGUAGE_CAUSAL_AXIS"
    assert tool_sprout["status"] == "HYPER_MACRO_CAUSAL_EXTENSION_ACTIVE"


def test_5_action_grounded_truth_and_faith_love_with_action():
    """
    [검증 5] 마음 (Heart) 계층: 행함 기반 진실 판별 ("행함이 없는 말/믿음은 죽은 기호").
    말만 있고 마찰(Action friction) 및 궤적이 0인 경우 Floating Dead Symbol Hallucination으로 판별.
    실제 마찰과 희생을 견딘 행함이 깊은 에너지 우물(Energy Well)을 형성함을 검증.
    """
    l2_tier = L2SocialSovereigntyTier()

    # Talk-only / Empty promise
    talk_only_proposal = CausalProposalNode(
        node_id="talk_only_love",
        description="말로만 하는 사랑과 약속",
        verbal_claim="I love you and promise everything",
        sensory_grounding_score=0.0,
        action_friction_heat=0.0,
        structural_time_spent=0.0
    )

    talk_res = l2_tier.evaluate_action_grounded_truth(talk_only_proposal)
    assert talk_res["is_action_grounded"] is False
    assert talk_res["verdict"] == "FLOATING_DEAD_SYMBOL_HALLUCINATION"
    assert talk_res["uncanniness_score"] == 0.95

    # Action-grounded true love / sacrifice
    action_grounded_proposal = CausalProposalNode(
        node_id="action_grounded_sacrifice",
        description="시간을 내어주고 손해와 마찰을 감수한 실체적 행함",
        verbal_claim="I support you through thick and thin",
        sensory_grounding_score=0.8,
        action_friction_heat=2.5,
        structural_time_spent=1.5
    )

    action_res = l2_tier.evaluate_action_grounded_truth(action_grounded_proposal)
    assert action_res["is_action_grounded"] is True
    assert action_res["verdict"] == "LIVING_TRUTH_REALITY_CARVED_ENERGY_WELL"
    assert action_res["energy_well_depth"] > 5.0


def test_6_controllability_and_social_cross_validation():
    """
    [검증 6] 통제 영역 분할 및 사회적 다중 스케일 교차 검증 ("나, 너, 우리").
    """
    l2_tier = L2SocialSovereigntyTier(uncanniness_sensitivity=1.2)

    # Deceptive claim scenario
    res = l2_tier.cross_validate_social_field(
        me_speech_signal=1.0,        # Speech: "Guaranteed profit!"
        you_expression_signal=-0.9,  # Micro-expression tremor
        us_field_tension=-0.8        # High relational friction
    )

    assert res["deception_alarm"] is True
    assert res["field_verdict"] == "SINGLE_TRACK_DECEPTIVE_HALLUCINATION_DETECTED"


def test_7_integrated_trinitarian_multi_scale_flow():
    """
    [검증 7] 통합 다차원 삼위일체 (육체 - 정신 - 마음) 마스터 흐름 검증.
    """
    engine = HeterogeneousSovereignEngine(rupture_threshold=2.0, deficit_threshold=2.5)

    joint = MechanicalJointConstraint(
        joint_id="shoulder_joint",
        rotation_axis=np.array([1.0, 0.0, 0.0]),
        muscle_fiber_orientation=np.array([1.0, 0.0, 0.0])
    )
    engine.l1_tier.add_joint_constraint(joint)

    l0_state = L0FluidState(velocity_vector=np.array([3.0, 0.0, 0.0]), channel_radius=1.0)
    synaptic_state = SynapticState(potential_difference_mv=75.0, neurotransmitter_flux=1.5)

    proposal = CausalProposalNode(
        node_id="trinitarian_authentic_action",
        description="신체적 마찰과 내적 통제 및 사회적 진실성에 기반한 행함",
        verbal_claim="I will act with physical devotion",
        sensory_grounding_score=0.9,
        friction_cost_bearer="me",
        fruit_recipient="me",
        action_friction_heat=2.0,
        structural_time_spent=1.0,
        is_internal_phase=True
    )

    trinitarian_res = engine.execute_trinitarian_multi_scale_flow(
        l0_state=l0_state,
        synaptic_state=synaptic_state,
        proposal_node=proposal,
        active_joint_id="shoulder_joint",
        sensory_mismatch_delta=1.2,
        physical_deficit=3.5,
        observed_external_entity="bat_radar_echolocation",
        social_signals=(0.8, 0.8, 0.8)
    )

    assert trinitarian_res["body_tier"]["q_heat"] > 0.0
    assert trinitarian_res["mind_tier"]["sprouted_tool_axis"] is not None
    assert trinitarian_res["heart_tier"]["is_action_grounded"] is True
    assert trinitarian_res["heart_tier"]["sovereignty_verdict"] == "VERIFIED_AUTHENTIC_SOVEREIGN_ACTION"
