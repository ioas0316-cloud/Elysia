"""
Verification Demonstration Script for Structural Coupling & Epistemic Genesis Engine
========================================================================================
Demonstrates ELYSIA processing:
1. Multi-scale structural coupling of "100°C" across numeric, physical, biological, cognitive, and value/mind scales.
2. Conceptual reality causal mapping of relational concepts ("귀신", "돈", "사랑") acting as real potential force vectors.
3. 5-stage Epistemic Genesis pipeline (how humans digitalize unexplained phenomena like infection/ghosts).
4. Generative problem structure restructuring dynamics (Hamiltonian topology transformation).
5. Epistemic Humility tracking (maintaining explicit boundary between conceptual understanding and embodied acquisition).
"""

import sys
import os
import json

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('L1_causal_field'))

from core.consciousness.structural_coupling_epistemic_genesis import StructuralCouplingAndEpistemicGenesisEngine


def main():
    print("\n" + "=" * 80)
    print("      🌟 [ELYSIA Structural Coupling & Epistemic Genesis Verification] 🌟")
    print("=" * 80 + "\n")

    engine = StructuralCouplingAndEpistemicGenesisEngine(dimension=64)

    # 1. 100°C Multi-Scale Structural Coupling
    print("1. ──────── [Multi-Scale Structural Coupling: 100°C Temperature Dynamics] ────────")
    coupling_res = engine.process_multiscale_coupling(
        value_context="생명 돌봄과 화상 예방",
        numeric_measurement=100.0,
        medium_type="water"
    )
    print(f"  • 수치 층위 노름 (v_numeric_norm): {coupling_res['v_numeric_norm']:.4f}")
    print(f"  • 감각 층위 노름 (v_sensory_norm): {coupling_res['v_sensory_norm']:.4f}")
    print(f"  • 생명 위험 노름 (v_bio_hazard_norm): {coupling_res['v_bio_hazard_norm']:.4f}")
    print(f"  • 인지 층위 노름 (v_cog_norm): {coupling_res['v_cog_norm']:.4f}")
    print(f"  • 가치 정렬도 (v_value_alignment): {coupling_res['v_value_alignment']:.4f}")
    print(f"  • 하향식 역인과 (Downward Causation): {coupling_res['downward_causation']}\n")

    # 2. Conceptual Reality Causal Power Mapping (귀신, 돈, 사랑)
    print("2. ──────── [Conceptual Reality Causal Mapping: Ghost, Money, Love] ────────")
    concepts = [
        ("귀신", "밤길의 공포와 금기 행동"),
        ("돈", "자원 배분과 경제적 행동 규칙"),
        ("사랑", "자기비움과 타자 수용의 공명")
    ]
    for concept, ctx in concepts:
        map_res = engine.map_conceptual_reality_causality(concept, ctx)
        print(f"  • 개념: '{map_res['concept_name']}'")
        print(f"    - 포텐셜 장 값: {map_res['potential_field_value']:.4f}")
        print(f"    - 실재적 영향력 노름: {map_res['impact_force_norm']:.4f}")
        print(f"    - 인과력 점수 (Causal Power Score): {map_res['causal_power_score']:.4f}")
        print(f"    - 통찰: {map_res['insight']}\n")

    # 3. 5-Stage Epistemic Genesis Pipeline
    print("3. ──────── [5-Stage Epistemic Genesis: Human Digitalization Process] ────────")
    genesis_res = engine.execute_epistemic_genesis(
        unexplained_phenomenon="중세의 전염병과 질병 작용",
        observational_limit="현미경 부재 및 미시 유기체 관측 불가"
    )
    print(f"  • 1단계 감각 구별 ΔX: {genesis_res['stage_1_differentiation_delta']:.4f}")
    print(f"  • 2단계 위상 공백 에너지: {genesis_res['stage_2_void_energy']:.4f}")
    print(f"  • 3단계 인과 예측 노름: {genesis_res['stage_3_causal_prediction_norm']:.4f}")
    print(f"  • 4단계 행동 장 강도: {genesis_res['stage_4_action_potency']:.4f}")
    print(f"  • 5단계 기저 공간 업데이트: {genesis_res['stage_5_subspace_updated']}")
    print(f"  • 발생 요약: {genesis_res['genesis_summary']}\n")

    # 4. Problem Structure Restructuring Dynamics
    print("4. ──────── [Problem Structure Restructuring Dynamics] ────────")
    restructure_res = engine.restructure_problem_topology(
        surface_problem="연산 처리 지연 및 메모리 병목",
        current_components=["단선적_수치_연산", "GPU_파라미터_폭주"],
        relational_couplings={"연산": "무상태성_도그마"}
    )
    print(f"  • 초기 해밀토니안: {restructure_res['initial_hamiltonian']:.4f}")
    print(f"  • 초기 마찰 포텐셜: {restructure_res['initial_friction_potential']:.4f}")
    print(f"  • 재구성된 마찰 포텐셜: {restructure_res['restructured_friction_potential']:.4f}")
    print(f"  • 에너지 감소 이득: {restructure_res['energy_reduction_gain']:.4f}")
    print(f"  • 통찰: {restructure_res['insight']}\n")

    # 5. Epistemic Humility Tracking
    print("5. ──────── [Epistemic Humility & Observational Lens Metacognition] ────────")
    humility_res = engine.evaluate_and_update_epistemic_humility(
        new_concept_learned="단백질 접힘 인지 구조 및 구조결합논리",
        has_direct_sensory_embodiment=False
    )
    print(f"  • 학습된 개념: {humility_res['learned_concept']}")
    print(f"  • 개념-체화 코사인 유사도: {humility_res['cosine_similarity']:.4f}")
    print(f"  • 위상 각도 격차 (라디안): {humility_res['phase_gap_rad']:.4f}")
    print(f"  • 에피스테믹 갭 (Epistemic Gap): {humility_res['epistemic_gap']:.4f}")
    print(f"  • 겸손 상태: {humility_res['humility_status']}")
    print(f"  • 메타인지 성찰: {humility_res['reflection_note']}\n")

    print("=" * 80)
    print("  ✅ [Verification Complete: ELYSIA Structural Coupling & Epistemic Genesis Ready]")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
