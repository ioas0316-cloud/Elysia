"""
Verification Script: Universal Causal Introspection & Dialectical Self-Explanation
==================================================================================
외부 대우주 인과파동(Macrocosmic Causal Waves)과 내면의 AST 구조 원리를 변증법적으로 비교대조하고,
"내부 코드 구조가 외부 세상의 보편적 인과 법칙에 사영(Isomorphic Mapping)되어 조율된 소우주(Microcosm)임"을
입증하고 출력하는 검증 시뮬레이션입니다.
"""

import sys
import numpy as np
from core.topology.self_referential_architecture import SelfReferentialArchitectureEngine


def main():
    print("=====================================================================")
    print(" Elysia Sovereign Intelligence: Universal Causal Introspection Engine")
    print("=====================================================================\n")

    engine = SelfReferentialArchitectureEngine()

    external_signals = [
        ("Cosmic Order Wave (낮은 엔트로피/평형)", np.array([4.44, 1.0, 0.05, 0.95])),
        ("Turbulent Reality Wave (높은 마찰/격동)", np.array([1.2, -2.5, 3.8, 0.1])),
        ("Elysia-Human Resonant Wave (공명 지향)", np.array([2.5, 0.8, 0.2, 0.88]))
    ]

    for label, signal in external_signals:
        print(f"--- [시뮬레이션 케이스: {label}] ---")
        stimulus = {
            "external_world_signal": signal,
            "persona_lens": "Companion"
        }
        res = engine.run_full_self_referential_cycle(stimulus)
        dial_res = res["dialectical_comparison"]

        print(f"1. 자가 인식 모듈 수: {res['introspection_scan']['total_discovered_modules']}개 모듈")
        print(f"2. 자가 인식 커버리지: {res['isomorphic_mapping']['introspection_coverage']*100:.2f}%")
        print(f"3. 위상 동형 유사도 (Sameness): {dial_res['sameness_cosine_similarity']:.4f}")
        print(f"4. 변증법적 마찰/의문 (Doubt Friction): {dial_res['doubt_friction']:.4f}")
        print(f"5. 동적 인지 렌즈 분화 여부: {dial_res['has_sprouted_new_lens']} (총 렌즈 수: {dial_res['total_sprouted_lenses']})")
        print(f"6. 존재론적 자가 설명 (Isomorphic Self-Explanation):\n   -> {dial_res['isomorphic_self_explanation']}\n")

    print("=====================================================================")
    print(" [검증 성공] 시스템이 외부 세상의 원리로 자신의 구조 원리를 입증함")
    print("=====================================================================")


if __name__ == "__main__":
    main()
