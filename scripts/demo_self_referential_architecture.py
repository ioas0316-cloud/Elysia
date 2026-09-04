"""
Elysia Self-Referential Information Architecture Interactive Demonstration
==========================================================================
외부 라벨러 없이 데이터 스스로의 내부 구조적 제약(Self-Reference)으로 스스로를 분별,
정의, 경계 설정, 의지적 기하 로터 탐구, 교차차원화, 자율 차원 분화($N \to N+1$),
및 외부 라벨 역공학/자가 납득(Self-Assimilation / Meta-Causality)을 실행하는 종합 데모 스크립트입니다.
"""

import sys
import os
import numpy as np

# Root path setup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.topology.self_referential_architecture import (
    PrimitiveDiscernmentEngine,
    SelfReferentialLanguageEngine,
    SelfReferentialVideoEngine,
    MetaOperator,
    MetaDefinition,
    RecursiveCausalityLoop,
    VolitionalGeometricRotorEngine,
    MetaCognitiveDomainLensSwitcher,
    FoundationalArchetypeDecodingEngine,
    DimensionalCircuit,
    MetaResonanceBus,
    DynamicDimensionSelfDifferentiationEngine,
    LabelSelfAssimilationEngine,
    SelfReferentialArchitectureEngine,
)


def print_section(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def main():
    print_section("Elysia Self-Referential Information Architecture Engine Demonstration")

    # 1. 0차 원리 분별 ('1' vs '2')
    print_section("1. 0차 원리 분별 (Primitive Discernment: '1' vs '2')")
    prim_engine = PrimitiveDiscernmentEngine()
    unity_1 = prim_engine.create_unity("1", dim=4)
    expanded_2 = prim_engine.expand_structure(unity_1, "+1")

    discern_res = prim_engine.discern_difference(unity_1, expanded_2)
    print(f"[*] '1' 단일 불변량 경계 Rank: {unity_1.dimension_rank}")
    print(f"[*] '2' 구조적 결합 경계 Rank: {expanded_2.dimension_rank} (내면에 '1' 원형 사영 보유)")
    print(f"[*] 공유 원형 코히어런스: {discern_res['shared_coherence']}")
    print(f"[*] 차원 차이: {discern_res['rank_difference']}")
    print(f"[*] 위상차 (Phase Discrepancy): {discern_res['phase_discrepancy']:.4f}")
    print(f"[*] 분별 성명: {discern_res['discernment_statement']}")

    # 2. 4대 자기-참조적 정보 작동 방식
    print_section("2. 4대 자기-참조적 정보 작동 방식 (Self-Referential Media Dynamics)")

    lang_engine = SelfReferentialLanguageEngine()
    lang_res = lang_engine.redefine_term_in_context(
        term="자아",
        context_a=["의지", "인과", "결핍"],
        context_b=["시스템", "연산", "메모리"]
    )
    print(f"[A. 언어 정의] 용어 '{lang_res['term']}' 문맥 간 의미론적 마찰(Friction): {lang_res['semantic_friction']:.4f}")
    print(f"             자율 재정의 값: {lang_res['redefined_value']:.4f} ({lang_res['status']})")

    vid_engine = SelfReferentialVideoEngine(max_fingers_allowed=5)
    sample_video_6finger = {"digit_count": 6, "kinematic_stress": 0.8, "optical_inconsistency": 0.4}
    vid_res = vid_engine.verify_and_reject_anomaly(sample_video_6finger)
    print(f"[B. 영상 정의] 검출 손가락 개수: {vid_res['digit_count']}개 | 구조적 위상 마찰: {vid_res['structural_friction']:.4f}")
    print(f"             자율 기각 여부 (Self-Rejection): {vid_res['is_rejected']}")
    print(f"             기각 사유: {vid_res['rejection_reason']}")

    op_plus = MetaOperator("+", binding_power=1.0, transformation_kernel="additive")
    op_mult = MetaOperator("*", binding_power=2.0, transformation_kernel="multiplicative")
    meta_op = op_plus.apply_meta_transformation(op_mult, causal_constraint=0.3)
    print(f"[C. Meta-Operator] 신규 고차 결합 연산자 기호: {meta_op.symbol}")
    print(f"                  재조합 결합력: {meta_op.binding_power:.4f} | 커널: {meta_op.transformation_kernel}")

    meta_def = MetaDefinition(
        intention=np.array([1.0, 0.8, -0.4, 0.6]),
        constraints=np.array([0.3, 0.5, 0.2, 0.4])
    )
    def_res = meta_def.execute_causal_process()
    print(f"[D. Meta-Definition] 의도 크기: {def_res['intention_norm']:.4f} | 제약 저항: {def_res['constraint_resistance']:.4f}")
    print(f"                  과정 마찰: {def_res['process_friction']:.4f}")
    print(f"                  정의의 실체: {def_res['definition_meaning']}")

    # 3. 독립 제약 회로, 위상 전이 커플링 & 메타 공명 버스
    print_section("3. 독립 제약 회로, 위상 전이 커플링 & 메타 공명 버스")
    lang_c = DimensionalCircuit("Language", lambda s: float(np.std(s) * 0.5))
    math_c = DimensionalCircuit("Math", lambda s: float(abs(np.sum(s) - 1.0)))
    phys_c = DimensionalCircuit("Physics", lambda s: float(np.max(np.abs(s)) * 0.2))
    bus = MetaResonanceBus()

    global_intent = np.array([0.9, 0.6, 0.3, 0.1])
    s_lang, r_lang = lang_c.step(global_intent)
    s_math, r_math = bus.phase_shift_coupling(s_lang, r_lang, math_c)
    s_phys, r_phys = bus.phase_shift_coupling(s_math, r_math, phys_c)

    total_friction = bus.compute_total_resonance_friction({"Language": r_lang, "Math": r_math, "Physics": r_phys})
    print(f"[*] [언어 회로 마찰 R_Lang]: {r_lang:.4f}")
    print(f"[*] [수학 회로 위상 전이 후 마찰 R_Math]: {r_math:.4f}")
    print(f"[*] [물리 회로 위상 전이 후 마찰 R_Phys]: {r_phys:.4f}")
    print(f"[*] [메타 공명 버스 전체 마찰 R_total]: {total_friction:.4f}")

    # 4. 동적 차원 자가 분화 ($N \to N+1$)
    print_section("4. 동적 차원 자가 분화 엔진 (Dynamic Dimension Self-Differentiation: N -> N+1)")
    sprouter = DynamicDimensionSelfDifferentiationEngine(unmapped_threshold=0.5)
    unmapped_friction = 0.82
    print(f"[*] 해소되지 않은 미지 잔여 마찰 (Unmapped Residual Friction): {unmapped_friction:.4f}")
    new_circuit = sprouter.evaluate_and_sprout(unmapped_friction, "Emotional_Resonance")
    if new_circuit:
        print(f"[*] [차원 자가 분화 발화] 신규 $N+1$번째 지식 차원 회로 생성됨: {new_circuit.name}")
        print(f"[*] 신규 차원 초기 상태: {new_circuit.S.round(3)}")

    # 5. 외부 라벨 역공학 및 자가 납득 (Meta-Causality)
    print_section("5. 외부 라벨 역공학 및 자가 납득 (Label Reverse-Engineering & Self-Assimilation)")
    label_engine = LabelSelfAssimilationEngine()
    label_res = label_engine.reverse_engineer_label(
        external_label="로그 (Logarithm)",
        observed_phenomenon={
            "constraints": np.array([2.5, 1.8, 1.2]),
            "effect_trajectory": np.array([0.4, 0.2, 0.1])
        }
    )
    print(f"[*] 외부 라벨 증상: '{label_res['external_label']}'")
    print(f"[*] 메타-인과 역추적 비율 (Meta-Causality Ratio): {label_res['meta_causality_ratio']:.4f}")
    print(f"[*] 자가 납득 증명: {label_res['self_assimilation_proof']}")
    print(f"[*] 내적 지식 전환 상태: {label_res['status']}")

    # 6. 재귀적 인과 피드백 루프
    print_section("6. 재귀적 인과 피드백 루프 (Recursive Causality Loop)")
    causal_loop = RecursiveCausalityLoop(initial_boundary=np.array([1.0, 1.0, 1.0, 1.0]))
    raw_cause = np.array([0.9, 0.4, 0.7, 0.2])
    value_ground = np.array([0.1, 0.1, 0.1, 0.1])

    c1 = causal_loop.execute_cycle(raw_cause, value_ground)
    print(f"[*] [Cycle 1] 마찰: {c1['trajectory_friction']:.4f} -> 신규 선험적 경계 제약: {c1['updated_boundary_constraint'].round(3)}")

    c2 = causal_loop.execute_cycle(raw_cause, value_ground)
    print(f"[*] [Cycle 2] (갱신된 경계 내재화 후) 마찰: {c2['trajectory_friction']:.4f} -> 신규 선험적 경계 제약: {c2['updated_boundary_constraint'].round(3)}")

    # 7. 의지적 기하 로터 엔진 & 자율 탐구
    print_section("7. 의지적 기하 로터 엔진 & 자율 탐구 (Volitional Geometric Rotor Engine)")
    rotor_engine = VolitionalGeometricRotorEngine()
    intention = np.array([1.0, 1.0, 0.0, 0.0])
    current_state = np.array([0.2, 0.4, 0.0, 0.0])
    volition_vec = rotor_engine.compute_volition_vector(intention, current_state)

    cand_a = np.array([0.8, 0.6, 0.0, 0.0])
    cand_b = np.array([0.0, 0.0, 1.0, 0.0])
    constraints = np.array([0.1, 0.1, 0.1, 0.0])

    rotor_res = rotor_engine.compare_rotor_trajectories(volition_vec, cand_a, cand_b, constraints)
    print(f"[*] 의지 벡터 (Volition Vector): {volition_vec.round(3)}")
    print(f"[*] Candidate A 마찰: {rotor_res['friction_a']:.4f} vs Candidate B 마찰: {rotor_res['friction_b']:.4f}")
    print(f"[*] 마찰 최소화 선택 궤적: {rotor_res['chosen_trajectory']}")
    print(f"[*] 자율 탐구 가상 회전축: {rotor_res['hypothetical_rotation_axis'].round(3)}")
    print(f"[*] 자율 탐구 질의: {rotor_res['self_directed_query']}")

    # 8. 상위 인지 도메인 렌즈 스위처 & 교차차원화
    print_section("8. 상위 인지 도메인 렌즈 스위처 & 교차차원화 ('빛' 5대 렌즈 투영)")
    lens_switcher = MetaCognitiveDomainLensSwitcher()
    cross_dim = lens_switcher.project_cross_dimensions("빛 (Light)")
    print(f"[*] 원형 본질: {cross_dim['core_archetype']}")
    for domain, text in cross_dim["projections"].items():
        print(f"    - {domain:10s} 렌즈: {text}")
    print(f"[*] 추출된 동형성 (Isomorphism): {cross_dim['isomorphism']}")
    print(f"[*] 추출된 이질성 (Heterogeneity): {cross_dim['heterogeneity']}")

    # 9. 통합 오케스트레이션 풀 사이클
    print_section("9. 통합 오케스트레이션 풀 사이클 (Full Orchestration Cycle)")
    full_arch = SelfReferentialArchitectureEngine()
    full_output = full_arch.run_full_self_referential_cycle({
        "video_data": {"digit_count": 6},
        "unmapped_friction": 0.88,
        "external_label": "Gravity"
    })
    print("[*] 통합 자율 분별 회로 풀 사이클 실행 완료.")
    print(f"    0차 원리 분별: {full_output['0th_primitive_discernment']['discernment_statement']}")
    print(f"    영상 자율 기각 사유: {full_output['video_self_rejection']['rejection_reason']}")
    print(f"    메타 공명 버스 전체 마찰: {full_output['total_resonance_friction']:.4f}")
    print(f"    자가 분화된 $N+1$차원 회로: {full_output['sprouted_dimension']}")
    print(f"    외부 라벨 자가 납득 성명: {full_output['label_self_assimilation']['self_assimilation_proof']}")

    print_section("Demonstration Successfully Completed.")


if __name__ == "__main__":
    main()
