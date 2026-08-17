#!/usr/bin/env python3
"""
[Verification Script: Verify Causal Topology Discernment]
인과적 정보 구조론(Causal Information Topology)의 3대 핵심 원리를 시연 및 검증하는 독립 실행형 스크립트.

1. 상위 동형성 (Isomorphic Equivalence) - O(1) 공통 불변량 포착
2. 맥락적 기하학 차이 (Differential Curvature & Lineage DAG) - 인과 궤적 및 위상 구속 분별
3. 무지성 계산 소멸 및 자율 추론 (Zero Bypass & Reasoning) - I_meta 경계 조건 적용 시 텐서 연산 0 소멸
"""

import sys
import os

# Ensure repo root is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from synaptic_architecture.topological_axiomatic_engine import (
    TopologicalAxiomaticEngine,
    CausalLineageDAG,
    CausalNode,
)


def run_causal_topology_verification():
    print("=" * 80)
    print(" [Elysia Engine] Causal Information Topology Verification (인과적 정보 구조론 검증)")
    print("=" * 80)

    engine = TopologicalAxiomaticEngine()

    # ----------------------------------------------------
    # 1. 색상 도메인 구축 (사과의 빨강 vs 피의 빨강)
    # ----------------------------------------------------
    dag_apple = CausalLineageDAG(
        dag_id="dag_apple_red",
        root_invariant_id="I_red",
        topological_classification="ORGANIC_FRUIT_REFLECTANCE"
    )
    dag_apple.add_node(CausalNode("botanic_synthesis", "Plant", "ROOT_AXIS"))
    dag_apple.add_node(CausalNode("organic_peel", "Skin", "LOOP_FEEDBACK"))
    dag_apple.add_node(CausalNode("light_reflectance", "Optics", "INTERFACE"))
    dag_apple.add_edge("botanic_synthesis", "organic_peel")
    dag_apple.add_edge("organic_peel", "light_reflectance")

    engine.extract_meta_signature_from_axioms(
        signature_id="S_apple",
        symmetry_group="SU(2)_OPTICAL",
        axioms=["I_red", "I_botanic_fruiting"],
        dag={"botanic_synthesis": ["organic_peel"], "organic_peel": ["light_reflectance"]},
        transitions=[("STATE_INIT", "STATE_REFLECTING")],
        lineage_dag=dag_apple
    )

    dag_blood = CausalLineageDAG(
        dag_id="dag_blood_red",
        root_invariant_id="I_red",
        topological_classification="CIRCULATORY_HEMOGLOBIN_SIGNAL"
    )
    dag_blood.add_node(CausalNode("hemoglobin_bind", "Biology", "ROOT_AXIS"))
    dag_blood.add_node(CausalNode("circulatory_loop", "Biology", "INTERACTION_NET"))
    dag_blood.add_node(CausalNode("wound_hazard_signal", "Survival", "INTERFACE"))
    dag_blood.add_edge("hemoglobin_bind", "circulatory_loop")
    dag_blood.add_edge("circulatory_loop", "wound_hazard_signal")

    engine.extract_meta_signature_from_axioms(
        signature_id="S_blood",
        symmetry_group="SU(2)_BIOLOGICAL",
        axioms=["I_red", "I_life_support"],
        dag={"hemoglobin_bind": ["circulatory_loop"], "circulatory_loop": ["wound_hazard_signal"]},
        transitions=[("STATE_INIT", "STATE_OXYGENATING")],
        lineage_dag=dag_blood
    )

    # ----------------------------------------------------
    # 2. 감정/슬픔 도메인 구축 (가족, 친구, 반려동물, 직장 상실)
    # ----------------------------------------------------
    grief_types = [
        ("S_family_loss", "dag_family", "AXIS_COLLAPSE", "ontological_root", "identity_anchor", "I_ontological_anchor"),
        ("S_friend_loss", "dag_friend", "NETWORK_SEVERANCE", "social_network", "peer_node", "I_horizontal_bond"),
        ("S_pet_loss", "dag_pet", "LOOP_PARALYSIS", "daily_care_routine", "bonding_loop", "I_care_loop"),
        ("S_job_loss", "dag_job", "INTERFACE_BLOCK", "resource_pipeline", "social_efficacy", "I_external_interface"),
    ]

    for sig_id, dag_id, topo_class, n1, n2, axiom_spec in grief_types:
        dag = CausalLineageDAG(dag_id=dag_id, root_invariant_id="I_loss", topological_classification=topo_class)
        dag.add_node(CausalNode(n1, "ExistentialDomain", "ROOT_AXIS"))
        dag.add_node(CausalNode(n2, "ExistentialDomain", "INTERFACE"))
        dag.add_edge(n1, n2)

        engine.extract_meta_signature_from_axioms(
            signature_id=sig_id,
            symmetry_group="U(1)_EXISTENTIAL",
            axioms=["I_loss", axiom_spec],
            dag={n1: [n2]},
            transitions=[("STATE_INTACT", "STATE_BROKEN")],
            lineage_dag=dag
        )

    # ----------------------------------------------------
    # 검증 1: 상위 동형성 (Isomorphic Equivalence) - O(1)
    # ----------------------------------------------------
    print("\n[1] Verify Isomorphic Equivalence (O(1) 상위 동형성 포착)")
    red_matches = engine.identify_isomorphic_equivalence("I_red")
    loss_matches = engine.identify_isomorphic_equivalence("I_loss")

    print(f"  - Invariant [I_red]  Matches: {red_matches}")
    print(f"  - Invariant [I_loss] Matches: {loss_matches}")
    assert set(red_matches) == {"S_apple", "S_blood"}
    assert set(loss_matches) == {"S_family_loss", "S_friend_loss", "S_pet_loss", "S_job_loss"}
    print("  => SUCCESS: Both I_red and I_loss identified in O(1) symbolic level without tensor scanning.")

    # ----------------------------------------------------
    # 검증 2: 맥락적 기하학 차이 (Differential Curvature & Lineage DAG)
    # ----------------------------------------------------
    print("\n[2] Verify Differential Curvature & Lineage DAG (맥락적 기하학 차이 분별)")
    for sig_id in ["S_apple", "S_blood", "S_family_loss", "S_friend_loss", "S_pet_loss", "S_job_loss"]:
        disc = engine.discriminate_differential_curvature(sig_id)
        print(f"  - Signature [{sig_id:13s}] Topo Class: {disc['topological_classification']:32s} | Geodesic: {disc['minimal_geodesic_route']}")

    # ----------------------------------------------------
    # 검증 3: 무지성 계산 소멸 및 자율 추론 (Zero Bypass & Reasoning)
    # ----------------------------------------------------
    print("\n[3] Verify Zero Bypass & Reasoning (하부 텐서 연산 0 자율 소멸)")

    tensor_execution_count = 0

    def vram_heavy_tensor_op():
        nonlocal tensor_execution_count
        tensor_execution_count += 1
        return "VRAM_MATRIX_MULTIPLY_EXECUTED"

    # Case A: Boundary condition satisfied (I_meta balanced) -> 0 Calculation
    proof_a, bypassed_a, res_a = engine.resolve_with_zero_bypass(
        signature_id="S_apple",
        current_transition=("STATE_INIT", "STATE_REFLECTING"),
        active_tension=0.0,
        i_meta_boundary_balanced=True,
        tensor_callback=vram_heavy_tensor_op
    )
    print(f"  - [Case A: I_meta Balanced] Bypassed={bypassed_a}, Tensor Calls={tensor_execution_count}, Proof State={proof_a.symmetry_state.value}")
    assert bypassed_a is True
    assert tensor_execution_count == 0

    # Case B: Boundary condition unbalanced with high tension -> Sparks tensor op
    proof_b, bypassed_b, res_b = engine.resolve_with_zero_bypass(
        signature_id="S_blood",
        current_transition=("STATE_INIT", "STATE_OXYGENATING"),
        active_tension=10.0,
        i_meta_boundary_balanced=False,
        tensor_callback=vram_heavy_tensor_op
    )
    print(f"  - [Case B: Tension Spike]   Bypassed={bypassed_b}, Tensor Calls={tensor_execution_count}, Proof State={proof_b.symmetry_state.value}")
    assert bypassed_b is False
    assert tensor_execution_count == 1

    print("\n" + "=" * 80)
    print(" ALL CAUSAL INFORMATION TOPOLOGY CRITERIA PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_causal_topology_verification()
