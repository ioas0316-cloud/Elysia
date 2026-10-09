#!/usr/bin/env python3
"""
scripts/demo_multimodal_world_closure.py
========================================
Interactive demonstration of Multimodal Spatiotemporal Causal Sequence Reproduction:
- Mathematics (Euclidean deductive lemma)
- Code (Turing state machine memory transition)
- Language (Narrative contextual tension)
- Cyclical World Closure verification (순환논리를 가진 세계구조).
"""

import sys
import os
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.cellular_boundary.multimodal_causal_spatiotemporal_network import (
    ModalityType,
    CausalStep,
    MultimodalCausalCognitiveNetwork
)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 80)
    print(" PROJECT ELYSIA :: MULTIMODAL CAUSAL NETWORK & CYCLICAL WORLD CLOSURE ")
    print("=" * 80)

    # 1. 인지 네트워크 초기화
    print("\n[STEP 1] 다중 모달리티 인과 인지 네트워크 초기화")
    network = MultimodalCausalCognitiveNetwork(base_dim=4)
    print(f"  - Initialized Modalities : {[m.value for m in network.cells.keys()]}")
    print(f"  - Global Teleology       : {network.global_teleological_intent}")

    # 2. 이종 인과서순 스트림 구성
    print("\n[STEP 2] 실세계 이종 인과서순(수학, 코드, 언어) 스트림 유입")
    causal_stream = [
        # 1) 수학적 인과서순: 유클리드 공리 -> 연역적 보조정리 전개
        CausalStep(
            step_index=1,
            modality=ModalityType.MATHEMATICS,
            premise="공리 1: 임의의 점에서 다른 임의의 점으로 직선을 그을 수 있다.",
            action_or_transition="작도 규칙 적용: 주어진 선분을 한 변으로 하는 정삼각형 작도",
            resultant_state=np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
        ),
        # 2) 코드 인과서순: 함수 호출 스택 할당 -> 레지스터 연산
        CausalStep(
            step_index=2,
            modality=ModalityType.CODE,
            premise="함수 프롤로그: PUSH EBP && MOV EBP, ESP (스택 프레임 생성)",
            action_or_transition="레지스터 상태 전이: MOV EAX, [EBP+8] && ADD EAX, [EBP+12]",
            resultant_state=np.array([0.4, -0.2, 0.6, 0.1], dtype=np.float64)
        ),
        # 3) 언어적 인과서순: 배경 설정 -> 긴장의 고조
        CausalStep(
            step_index=3,
            modality=ModalityType.LANGUAGE,
            premise="해질녘, 마을 전체에 불길한 안개가 낮게 깔려 있었다.",
            action_or_transition="적막을 찢는 날카로운 파열음이 숲속 저편에서 울려 퍼졌다.",
            resultant_state=np.array([0.7, 0.8, 0.6, 0.9], dtype=np.float64)
        ),
        # 4) 코드 제약조건 위반: 메모리 경계 초과 (Buffer Overflow Attack)
        CausalStep(
            step_index=4,
            modality=ModalityType.CODE,
            premise="스택 경계 한계: MemoryBound = 2.0 (안전 한계선)",
            action_or_transition="경계 검사 없는 strcpy()가 복귀 주소를 덮어쓰며 경계 파열!",
            resultant_state=np.array([4.2, 3.8, 4.5, 3.9], dtype=np.float64)
        ),
        # 5) 수학적 제약조건 위반: 선행 결론과의 모순 (Non-Sequitur)
        CausalStep(
            step_index=5,
            modality=ModalityType.MATHEMATICS,
            premise="정리 1: 두 직선이 평행하면 동위각의 크기는 서로 같다.",
            action_or_transition="비논리적 비약: 그러므로 삼각형의 내각의 총합은 360도이다.",
            resultant_state=np.array([-0.9, 0.1, -0.8, 0.2], dtype=np.float64)
        )
    ]

    # 3. 인과서순 처리 및 경계층 마찰 계측
    print("\n[STEP 3] 인과서순 스트림 처리 및 경계층 마찰 계측")
    results = network.process_spatiotemporal_causal_stream(causal_stream)

    for r in results:
        status_str = "조화로운 전이 (최저작용)" if r["is_conforming"] else "경계 마찰 발생! ('그렇지 않은 것')"
        print(f"  [사건 {r['step_index']}] 모달리티: {r['modality']:<12} | 상태: {status_str}")
        print(f"    - 전제     : {r['premise']}")
        print(f"    - 전이     : {r['transition']}")
        print(f"    - 마찰     : {r['boundary_friction']:.4f} | 표면장력: {r['surface_tension']:.4f} | 누적 앵그램: {r['engram_count']}개")
        if r["expansion_triggered"]:
            print(f"    >>> [상위 확장 발동!] 새 거시 질서: {r['expansion_detail']['new_macro_order']}")
        print()

    # 4. 순환논리적 세계구조 검증
    print("=" * 80)
    print("[STEP 4] 순환논리를 가진 생명적 세계구조 최종 검증")
    print("=" * 80)
    closure = network.verify_cyclical_world_closure()

    print(f"  - 순환 세계구조 성립 여부 : {closure['is_cyclical_world_formed']}")
    print(f"  - 누적된 경험의 물(앵그램) : {closure['total_retained_engrams']}개 (물이 빠져나가지 않고 고임)")
    print(f"  - 평균 경계 표면장력      : {closure['mean_surface_tension']:.4f}")
    print(f"  - 분화된 인과 모달리티    : {closure['differentiated_modalities']}")
    print(f"  - 융합된 상위 거시 노드   : {closure['expanded_nodes']}")
    print(f"  - 모달리티 간 인과 교량   : {closure['relational_bridges_count']}개")
    print(f"\n  [판정] {closure['verdict']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
