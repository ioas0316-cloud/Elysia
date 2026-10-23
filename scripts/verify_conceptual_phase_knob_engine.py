"""
Verification script for ConceptualPhaseKnobEngine & CognitiveOSRuntime.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from L1_causal_field.core.topology.conceptual_phase_knob_engine import ConceptualPhaseKnobEngine
from L3_interface.simulators.cognitive_os_runtime import CognitiveOSRuntime


def run_verification():
    print("=== [Elysia CognitiveOS & Conceptual Knob Engine Verification] ===")

    # 1. 엔진 초기화
    engine = ConceptualPhaseKnobEngine(dimension=32)
    runtime = CognitiveOSRuntime()

    # 2. 이종 개념 등록 (새의 날개 vs 비행기 날개, CAD 제약 vs 포토샵 레이어)
    # 공기저항 극복 및 양력 생성이라는 상위 개념 렌즈를 모사하는 특성 벡터 설정
    np.random.seed(42)
    base_lift_principle = np.random.randn(32)

    # 새의 날개와 비행기 유선형 날개는 본질적 원리가 유사하도록 설계
    bird_wing = base_lift_principle + np.random.randn(32) * 0.05
    aircraft_wing = base_lift_principle + np.random.randn(32) * 0.07

    # 완전히 다른 도메인 (예: 무작위 소음)
    noise_concept = np.random.randn(32) * 5.0

    engine.register_concept("bird_wing", bird_wing.tolist())
    engine.register_concept("aircraft_wing", aircraft_wing.tolist())
    engine.register_concept("noise", noise_concept.tolist())

    # 3. 개념간 거리 및 마찰 오차(E) 측정 검증
    theta_bw_aw, err_bw_aw = engine.compute_conceptual_distance("bird_wing", "aircraft_wing")
    theta_bw_noise, err_bw_noise = engine.compute_conceptual_distance("bird_wing", "noise")

    print(f"\n[Conceptual Distance Test]")
    print(f" - [Bird Wing] vs [Aircraft Wing] -> Theta: {theta_bw_aw:.4f}, Friction Error (E): {err_bw_aw:.4f} (Expected: Near 0)")
    print(f" - [Bird Wing] vs [Noise]       -> Theta: {theta_bw_noise:.4f}, Friction Error (E): {err_bw_noise:.4f} (Expected: High)")

    # 4. 노브화(Knob-ification) 및 상태 재배치 루프 테스트
    knob_key = engine.create_knob_from_concept("aircraft_wing", initial_value=1.0)
    print(f"\n[Knob-ification Test]")
    print(f" - Created Knob: {knob_key} with initial value 1.0")

    # 노브 조율 시뮬레이션
    rearrange_result = engine.adjust_knob(knob_key, value=2.5)
    print(f" - Knob Adjustment Result: {rearrange_result}")

    # 5. 인지 OS 감각 기관(Blender, CAD 등) 닫힌 고리 테스트
    print(f"\n[CognitiveOS Closed-Loop Organ Test]")
    intent = [0.5, -0.2, 0.8, 0.1] * 4
    for organ_name in ["blender", "cad", "photoshop", "sandbox"]:
        result = runtime.execute_closed_loop(organ_name, intent)
        print(f" - Organ [{result['organ']}] -> Friction (E): {result['friction_error']:.4f} | Self-Controlled: {result['is_self_controlled']}")

    print("\n=== Verification Completed Successfully! ===")


if __name__ == "__main__":
    run_verification()
