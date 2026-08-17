"""
[Unit Test: Causal Topology Discernment Test]
인과적 정보 구조론(Causal Information Topology)의 3대 핵심 원리를
synaptic_architecture/topological_axiomatic_engine.py 상에서 종합 검증하는 단위 테스트.

1. 상위 동형성 (Isomorphic Equivalence): O(1) 공통 불변량 (I_red, I_loss 등) 포착
2. 맥락적 기하학 차이 (Differential Curvature & Lineage DAG): 생성 궤적 및 위상학적 차이 분별 (S_apple vs S_blood, S_family_loss ~ S_job_loss)
3. 무지성 계산 소멸 및 자율 추론 (Zero Bypass & Reasoning): I_meta 경계 조건 이완 시 하부 텐서 연산 자율 소멸
"""

import pytest
from synaptic_architecture.topological_axiomatic_engine import (
    TopologicalAxiomaticEngine,
    CausalLineageDAG,
    CausalNode,
    MetaMechanismSignature,
)
from synaptic_architecture.non_tensor_meta_boundary import (
    TypeConstraint,
    StaticBypassManager,
    SymmetryState,
)


def build_engine_with_color_and_grief_domains() -> TopologicalAxiomaticEngine:
    """테스트용 인과 위상 엔진 구축 및 샘플 시그니처 등록"""
    engine = TopologicalAxiomaticEngine()

    # ==========================================
    # 1. 색상 도메인: I_red (사과의 빨강 vs 피의 빨강)
    # ==========================================
    # 1-1. S_apple (사과의 빨간색) Lineage DAG
    dag_apple = CausalLineageDAG(
        dag_id="dag_apple_red",
        root_invariant_id="I_red",
        topological_classification="ORGANIC_FRUIT_REFLECTANCE"
    )
    dag_apple.add_node(CausalNode("apple_tree", "Botany", "ROOT_AXIS"))
    dag_apple.add_node(CausalNode("fruit_peel", "Botany", "LOOP_FEEDBACK"))
    dag_apple.add_node(CausalNode("apple_reflectance", "Optics", "INTERFACE"))
    dag_apple.add_edge("apple_tree", "fruit_peel")
    dag_apple.add_edge("fruit_peel", "apple_reflectance")

    sig_apple = engine.extract_meta_signature_from_axioms(
        signature_id="S_apple",
        symmetry_group="SU(2)_OPTICAL",
        axioms=["I_red", "I_organic_synthesis"],
        dag={"apple_tree": ["fruit_peel"], "fruit_peel": ["apple_reflectance"]},
        transitions=[("STATE_INIT", "STATE_REFLECTING"), ("STATE_REFLECTING", "STATE_OBSERVED")],
        lineage_dag=dag_apple
    )

    # 1-2. S_blood (피의 빨간색) Lineage DAG
    dag_blood = CausalLineageDAG(
        dag_id="dag_blood_red",
        root_invariant_id="I_red",
        topological_classification="CIRCULATORY_HEMOGLOBIN_SIGNAL"
    )
    dag_blood.add_node(CausalNode("hemoglobin", "Biology", "ROOT_AXIS"))
    dag_blood.add_node(CausalNode("circulatory_system", "Biology", "INTERACTION_NET"))
    dag_blood.add_node(CausalNode("wound_warning", "Survival", "INTERFACE"))
    dag_blood.add_edge("hemoglobin", "circulatory_system")
    dag_blood.add_edge("circulatory_system", "wound_warning")

    sig_blood = engine.extract_meta_signature_from_axioms(
        signature_id="S_blood",
        symmetry_group="SU(2)_BIOLOGICAL",
        axioms=["I_red", "I_life_support"],
        dag={"hemoglobin": ["circulatory_system"], "circulatory_system": ["wound_warning"]},
        transitions=[("STATE_INIT", "STATE_OXYGENATING"), ("STATE_OXYGENATING", "STATE_OBSERVED")],
        lineage_dag=dag_blood
    )

    # ==========================================
    # 2. 감정/슬픔 도메인: I_loss (가족, 친구, 반려동물, 직장 상실)
    # ==========================================
    # 2-1. S_family_loss (가족 상실)
    dag_family = CausalLineageDAG(
        dag_id="dag_family_loss",
        root_invariant_id="I_loss",
        topological_classification="AXIS_COLLAPSE"
    )
    dag_family.add_node(CausalNode("ontological_root", "Existential", "ROOT_AXIS"))
    dag_family.add_node(CausalNode("identity_anchor", "Psychology", "ROOT_AXIS"))
    dag_family.add_edge("ontological_root", "identity_anchor")

    engine.extract_meta_signature_from_axioms(
        signature_id="S_family_loss",
        symmetry_group="U(1)_EXISTENTIAL",
        axioms=["I_loss", "I_ontological_anchor"],
        dag={"ontological_root": ["identity_anchor"]},
        transitions=[("STATE_INTACT", "STATE_BROKEN")],
        lineage_dag=dag_family
    )

    # 2-2. S_friend_loss (친구 상실)
    dag_friend = CausalLineageDAG(
        dag_id="dag_friend_loss",
        root_invariant_id="I_loss",
        topological_classification="NETWORK_SEVERANCE"
    )
    dag_friend.add_node(CausalNode("social_network", "Social", "INTERACTION_NET"))
    dag_friend.add_node(CausalNode("peer_node", "Social", "INTERACTION_NET"))
    dag_friend.add_edge("social_network", "peer_node")

    engine.extract_meta_signature_from_axioms(
        signature_id="S_friend_loss",
        symmetry_group="U(1)_SOCIAL",
        axioms=["I_loss", "I_horizontal_bond"],
        dag={"social_network": ["peer_node"]},
        transitions=[("STATE_INTACT", "STATE_BROKEN")],
        lineage_dag=dag_friend
    )

    # 2-3. S_pet_loss (반려동물 상실)
    dag_pet = CausalLineageDAG(
        dag_id="dag_pet_loss",
        root_invariant_id="I_loss",
        topological_classification="LOOP_PARALYSIS"
    )
    dag_pet.add_node(CausalNode("daily_care_routine", "Behavioral", "LOOP_FEEDBACK"))
    dag_pet.add_node(CausalNode("unconditional_bonding", "Emotional", "LOOP_FEEDBACK"))
    dag_pet.add_edge("daily_care_routine", "unconditional_bonding")

    engine.extract_meta_signature_from_axioms(
        signature_id="S_pet_loss",
        symmetry_group="U(1)_ROUTINE",
        axioms=["I_loss", "I_care_loop"],
        dag={"daily_care_routine": ["unconditional_bonding"]},
        transitions=[("STATE_INTACT", "STATE_BROKEN")],
        lineage_dag=dag_pet
    )

    # 2-4. S_job_loss (직장 상실)
    dag_job = CausalLineageDAG(
        dag_id="dag_job_loss",
        root_invariant_id="I_loss",
        topological_classification="INTERFACE_BLOCK"
    )
    dag_job.add_node(CausalNode("resource_pipeline", "Infrastructure", "INTERFACE"))
    dag_job.add_node(CausalNode("social_efficacy", "Role", "INTERFACE"))
    dag_job.add_edge("resource_pipeline", "social_efficacy")

    engine.extract_meta_signature_from_axioms(
        signature_id="S_job_loss",
        symmetry_group="U(1)_FUNCTIONAL",
        axioms=["I_loss", "I_external_interface"],
        dag={"resource_pipeline": ["social_efficacy"]},
        transitions=[("STATE_INTACT", "STATE_BROKEN")],
        lineage_dag=dag_job
    )

    return engine


def test_isomorphic_equivalence_O1():
    """1. 상위 동형성 (Isomorphic Equivalence) O(1) 포착 검증"""
    engine = build_engine_with_color_and_grief_domains()

    # I_red 포착: S_apple과 S_blood가 즉시 포착됨
    red_equivalences = engine.identify_isomorphic_equivalence("I_red")
    assert "S_apple" in red_equivalences
    assert "S_blood" in red_equivalences
    assert len(red_equivalences) == 2

    # I_loss 포착: 4가지 슬픔/상실이 즉시 포착됨
    loss_equivalences = engine.identify_isomorphic_equivalence("I_loss")
    assert "S_family_loss" in loss_equivalences
    assert "S_friend_loss" in loss_equivalences
    assert "S_pet_loss" in loss_equivalences
    assert "S_job_loss" in loss_equivalences
    assert len(loss_equivalences) == 4


def test_differential_curvature_lineage_dag():
    """2. 맥락적 기하학 차이 (Differential Curvature & Lineage DAG) 분별 검증"""
    engine = build_engine_with_color_and_grief_domains()

    # 사과의 빨강 vs 피의 빨강 인과 궤적 분별
    disc_apple = engine.discriminate_differential_curvature("S_apple")
    disc_blood = engine.discriminate_differential_curvature("S_blood")

    assert disc_apple["topological_classification"] == "ORGANIC_FRUIT_REFLECTANCE"
    assert disc_blood["topological_classification"] == "CIRCULATORY_HEMOGLOBIN_SIGNAL"
    assert disc_apple["minimal_geodesic_route"] != disc_blood["minimal_geodesic_route"]

    # 4가지 슬픔의 위상학적 구조 차이 분별
    disc_family = engine.discriminate_differential_curvature("S_family_loss")
    disc_friend = engine.discriminate_differential_curvature("S_friend_loss")
    disc_pet = engine.discriminate_differential_curvature("S_pet_loss")
    disc_job = engine.discriminate_differential_curvature("S_job_loss")

    assert disc_family["topological_classification"] == "AXIS_COLLAPSE"
    assert disc_friend["topological_classification"] == "NETWORK_SEVERANCE"
    assert disc_pet["topological_classification"] == "LOOP_PARALYSIS"
    assert disc_job["topological_classification"] == "INTERFACE_BLOCK"


def test_zero_bypass_and_reasoning():
    """3. 무지성 계산 소멸 및 자율 추론 (Zero Bypass & Reasoning) 검증"""
    engine = build_engine_with_color_and_grief_domains()

    tensor_called = False

    def dummy_tensor_calc():
        nonlocal tensor_called
        tensor_called = True
        return "HEAVY_VRAM_CALCULATION_RESULT"

    # 시나리오 A: I_meta 경계 조건이 만족된 상태 (I_meta balanced) -> 텐서 연산 100% 소멸 (Bypass)
    proof, is_bypassed, result = engine.resolve_with_zero_bypass(
        signature_id="S_apple",
        current_transition=("STATE_INIT", "STATE_REFLECTING"),
        active_tension=0.0,
        i_meta_boundary_balanced=True,
        tensor_callback=dummy_tensor_calc
    )

    assert is_bypassed is True
    assert result is None
    assert tensor_called is False
    assert proof.symmetry_state == SymmetryState.PRESERVED

    # 시나리오 B: I_meta 경계 조건 불균형 및 장력 스파이크 발생 -> 비동기 텐서 연산 유발
    tensor_called = False
    proof_b, is_bypassed_b, result_b = engine.resolve_with_zero_bypass(
        signature_id="S_blood",
        current_transition=("STATE_INIT", "STATE_OXYGENATING"),
        active_tension=5.0,  # > threshold
        i_meta_boundary_balanced=False,
        tensor_callback=dummy_tensor_calc
    )

    assert is_bypassed_b is False
    assert result_b == "HEAVY_VRAM_CALCULATION_RESULT"
    assert tensor_called is True
    assert proof_b.symmetry_state == SymmetryState.SPIKED
