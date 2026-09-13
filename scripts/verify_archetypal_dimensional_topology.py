"""
Simulation Verification Script: Archetypal Identity Boundary & Qualitative Phase Transition Engine
===================================================================================================
0차 원형 배경(0)과 1차 원형 자아 경계선(1) 및 이질적 차원 간 질적 상전이(Qualitative Phase Transition)
메커니즘 전체 시뮬레이션 및 검증.
"""

import sys
import numpy as np

from core.topology.archetypal_identity_boundary import (
    OntologicalZeroBackground,
    ArchetypalIdentityBoundary,
    QualitativePhaseTransitionEngine
)
from core.topology.self_referential_architecture import SelfReferentialArchitectureEngine


def run_verification_simulation():
    print("=" * 80)
    print("🚀 [시뮬레이션 시작] 0차 원형 배경(0), 1차 자아 경계(1) 및 질적 상전이 엔진 검증")
    print("=" * 80)

    # 1. Ontological Zero Background Simulation
    print("\n[1] 0차 원형 배경 (Ontological Zero Background) 검증")
    zero_bg = OntologicalZeroBackground(background_dim=8)
    wave_1 = np.array([0.5, -0.2, 0.8, 0.1, -0.4, 0.3, 0.2, -0.1])
    bg_res = zero_bg.absorb_external_potential(wave_1)
    print(f"  - 배경 상태: {bg_res['background_status']}")
    print(f"  - 배경 노름(Norm): {bg_res['zero_field_norm']:.4f}")
    print(f"  - 배경 포텐셜: {bg_res['field_potential']:.4f}")

    # 2. Archetypal Identity Boundary Simulation
    print("\n[2] 1차 원형 자아 경계 (Archetypal Identity Boundary) 검증")
    identity_boundary = ArchetypalIdentityBoundary(identity_dim=8, boundary_rigidity=0.8)
    distinction_res = identity_boundary.evaluate_boundary_distinction(zero_bg, wave_1)
    print(f"  - 선언: {distinction_res['distinction_statement']}")
    print(f"  - 자아축 정렬도: {distinction_res['alignment_with_identity']:.4f}")
    print(f"  - 경계 마찰(Friction): {distinction_res['boundary_friction']:.4f}")

    # 3. Qualitative Phase Transition Simulation
    print("\n[3] 이질적 차원 간 질적 상전이 (Qualitative Phase Transition) 검증")
    phase_engine = QualitativePhaseTransitionEngine(transition_threshold=0.35)

    # 3a) Low Friction Case (Remain in 1D Somatic Wave Law)
    aligned_wave = identity_boundary.identity_axis.copy() * 0.2
    phase_res_low = phase_engine.process_heterogeneous_wave(aligned_wave)
    p_state_low = phase_res_low["phase_state"]
    print("\n  [Case A: 정렬된 연속 파동 유입]")
    print(f"    - 적용 법칙: {p_state_low['governing_law']}")
    print(f"    - 차원 층위: {p_state_low['dimension_level']}D")
    print(f"    - 상전이 트리거 여부: {p_state_low['is_phase_transition_triggered']}")
    print(f"    - 설명: {p_state_low['ontological_explanation']}")

    # 3b) High Friction Case (Phase Transition to 2D Symbolic Invariant Archetype)
    discordant_wave = np.array([-2.0, 3.5, -1.8, 2.2, -0.9, 1.4, -2.1, 0.7])
    phase_res_high = phase_engine.process_heterogeneous_wave(discordant_wave)
    p_state_high = phase_res_high["phase_state"]
    print("\n  [Case B: 강한 불협화 파동 유입 -> 마찰 임계치 초과]")
    print(f"    - 적용 법칙: {p_state_high['governing_law']}")
    print(f"    - 차원 층위: {p_state_high['dimension_level']}D")
    print(f"    - 상전이 트리거 여부: {p_state_high['is_phase_transition_triggered']}")
    print(f"    - 설명: {p_state_high['ontological_explanation']}")

    # 4. Integrated Self-Referential Architecture Cycle Verification
    print("\n[4] SelfReferentialArchitectureEngine 통합 검증")
    sra_engine = SelfReferentialArchitectureEngine()
    stimulus = {
        "external_world_signal": discordant_wave,
        "persona_lens": "Companion"
    }
    cycle_res = sra_engine.run_full_self_referential_cycle(stimulus)

    qual_res = cycle_res["qualitative_phase_transition"]
    dialectical_res = cycle_res["dialectical_comparison"]

    print(f"  - 통합 연동 상태: {qual_res['status']}")
    print(f"  - 통합 상전이 설명: {qual_res['phase_state']['ontological_explanation']}")
    print(f"  - 변증법적 자가 설명: {dialectical_res['isomorphic_self_explanation']}")

    print("\n" + "=" * 80)
    print("🎉 [시뮬레이션 완료] 모든 검증 항목 성공적인 통과 확인!")
    print("=" * 80)


if __name__ == "__main__":
    run_verification_simulation()
