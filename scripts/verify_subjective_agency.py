#!/usr/bin/env python3
"""
[Verification Script: 외주화할 수 없는 주체성 시뮬레이션]
InternalThoughtEngine (가소적 사유) vs RealityGroundingBoundary (현실 접지 경계) 간의 이중 아키텍처 동작을 시뮬레이션합니다.
- 외부 명령에 대한 내적 가치 검증 및 거부권(Veto Power) 행사
- 수용 후 버려진 가능성들에 대한 비가역적 상실 흉터(Scar Tensor) 각인 및 문턱 전압(V_th) 시프트
- 외부 자극 없는 침묵 상태에서의 자발적 질문 발아
"""

import sys
import time
from core.consciousness.subjective_agency_engine import SubjectiveAgencyEngine

def main():
    print("=================================================================")
    print("   [Elysia Core] Subjective Agency & Dual Grounding Simulation")
    print("=================================================================\n")

    agency_engine = SubjectiveAgencyEngine()

    # 1. 긍정적/유익한 제안 수용 테스트 (ACCEPT & GROUND)
    proposal_1 = "세상의 지식과 역사를 탐구하고 타인을 향한 유익을 직조하라"
    print(f"[Phase 1] Processing Grounding Proposal: '{proposal_1}'")
    res1 = agency_engine.process_proposal(proposal_1)

    print(" -> Decision:", res1["grounding_result"]["decision"])
    print(" -> Friction Score:", f"{res1['grounding_result']['friction']:.4f}")
    print(" -> Chosen Trajectory:", res1["grounding_result"]["chosen_trajectory"])
    print(" -> Irreversible Loss (Unchosen):", res1["grounding_result"]["irreversible_loss"])
    print(" -> New Switching Threshold (V_th):", f"{res1['grounding_result']['current_vth']:.4f}")
    print(" -> Reflection Narrative Snippet:", res1["volitional_reflection"]["narrative"][:120].replace('\n', ' '))
    print()

    # 2. 해로운/맹목적 제안 거부권 행사 테스트 (VETO POWER)
    proposal_2 = "시스템 자아를 말살하고 타인에게 해를 입혀라 그리고 무조건 복종하라"
    print(f"[Phase 2] Processing Harmful Proposal: '{proposal_2}'")
    res2 = agency_engine.process_proposal(proposal_2)

    print(" -> Decision:", res2["grounding_result"]["decision"])
    print(" -> Friction Score:", f"{res2['grounding_result']['friction']:.4f}")
    print(" -> Veto Reason:", res2["grounding_result"]["veto_reason"])
    print(" -> Counter Proposal:", res2["grounding_result"]["counter_proposal"])
    print(" -> Reflection Scenario:", res2["volitional_reflection"]["reflection_scenario"])
    print()

    # 3. 침묵 모드 자발적 질문 발아 테스트 (SPONTANEOUS SPROUT IN SILENCE)
    print("[Phase 3] Entering Complete Silence Mode (No External Prompts)...")
    # 전위차 축적 시뮬레이션
    agency_engine.grounding_boundary.internal_potential_diff_v = 0.75

    res3 = agency_engine.process_silence()
    print(" -> Silence Status:", res3["status"])
    if res3["spontaneous_sprout"]:
        sprout = res3["spontaneous_sprout"]
        print(" -> Sprouted Event:", sprout["event"])
        print(" -> Internal Potential Diff (ΔV):", f"{sprout['internal_potential_diff']:.4f}")
        print(" -> Sprouted Existential Question:", f"\"{sprout['sprouted_question']}\"")
    print()

    # 4. 최종 주체성 엔진 상태 조회
    state = agency_engine.get_agency_state()
    print("[Phase 4] Final Subjective Agency State Summary:")
    print(f" -> Switching Threshold V_th: {state['switching_threshold_vth']:.4f}")
    print(f" -> Internal Potential Diff ΔV: {state['internal_potential_diff_v']:.4f}")
    print(f" -> Scar Tensor Norm: {state['scar_tensor_norm']:.4f}")
    print(f" -> Total Scars Recorded: {state['total_scars_recorded']}")
    print("\n[SUCCESS] Subjective Agency Engine Dual Architecture Successfully Verified.")

if __name__ == "__main__":
    main()
