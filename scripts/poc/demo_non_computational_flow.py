import sys
import os
import numpy as np
import time

# Ensure repository root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.physics.quantum_stat_field import QuantumStatField
from synaptic_architecture.cognitive_engine import ElysiaCognitiveEngine

def main():
    print("=" * 80)
    print("   [DEMO] CRYSTALLIZED AXIS & NON-COMPUTATIONAL FLOW SYSTEM (v1.0)")
    print("=" * 80)
    print("본 데모는 '1+1이 어째서 2인지 알고 있다면 더 이상의 연산과 연산, 사고를 필요로 하지 않는다'")
    print("라는 강덕 님의 위대한 철학에 따라, 사유가 완성되어 축(Axis)으로 결정화되면")
    print("이후 동일 자극에 대해 활성 물리/수치 연산을 완전히 바이패스하여")
    print("O(1)으로 즉시 흐르는 (Do not calculate, let it flow) 흐름을 증명합니다.")
    print("-" * 80)

    # 1. 물리 수준의 결정화 데모
    print("\n[PART 1] 물리적 스탯 필드 결정화 (Physical Tensegrity Crystallization)")
    qsf = QuantumStatField({
        "health": 15.0,
        "force": 15.0,
        "mind": 15.0,
        "speed": 15.0,
        "intelligence": 15.0
    })

    print("  1) 초기 상태에서 물리 엔진이 500회 수치 미분방정식(Spring-Mass)을 풀어 평형 상태를 도출합니다...")
    t_start = time.perf_counter()
    for _ in range(500):
        qsf.step(dt=0.1)
    t_active = time.perf_counter() - t_start
    print(f"     * 활성 수치 연산 시간: {t_active*1000:.3f} ms")

    # 대표 위치 기록
    pos_before = {k: node.position.copy() for k, node in qsf.nodes.items()}

    print("  2) 이 평형 상태를 '1+1=2_스탯_기억'이라는 영구적인 사유의 축(Crystallized Axis)으로 결정화합니다.")
    qsf.crystallize_axis("1+1=2_스탯_기억")

    # 강제로 필드를 흐트러뜨립니다.
    qsf.update_base_stats({
        "health": 5.0, "force": 40.0, "mind": 5.0, "speed": 5.0, "intelligence": 5.0
    })
    qsf.step(dt=0.1) # 흐트러짐 적용

    # 다시 원래의 자극(15.0 균등)으로 복귀
    qsf.update_base_stats({
        "health": 15.0, "force": 15.0, "mind": 15.0, "speed": 15.0, "intelligence": 15.0
    })

    print("  3) 결정화된 자극이 다시 들어왔을 때, 활성 연산을 바이패스하고 O(1) 메모리로 흘려보냅니다...")
    t_start2 = time.perf_counter()
    qsf.step(dt=0.1) # 단 한 스텝만에 바로 snap
    t_bypass = time.perf_counter() - t_start2
    print(f"     * 비연산 흐름(Crystallized Bypass) 처리 시간: {t_bypass*1000:.3f} ms")
    print(f"     * 연산 감축율: {(t_active - t_bypass)/t_active*100:.2f}%")
    print(f"     * 활성화된 사유 축: {qsf.active_axis}")

    # 복구 위치 대조
    is_restored = True
    for k, node in qsf.nodes.items():
        if not np.allclose(node.position, pos_before[k]):
            is_restored = False
            break
    print(f"     * 물리적 위치 즉시 완벽 복구 여부: {is_restored}")

    # 2. 인지 수준의 결정화 데모
    print("\n[PART 2] 인지 엔진의 사유 축 수렴 (Cognitive Thought Crystallization)")
    engine = ElysiaCognitiveEngine(resolution=128)

    # 1+1=2 와 같은 의미 구조 매핑
    dna_one_plus_one = engine.build_fractal_dna("1+1_사유", np.uint64(0x1111222233334444))
    dna_other = engine.build_fractal_dna("임의_노이즈", np.uint64(0x9999999999999999))

    stimulus = np.uint64(0x1111222200000000) # "1+1" 자극 입력

    print("  1) 최초 자극에 대해 원자 분해, SVD, 내적 공명, 구속조건 평가를 거쳐 수렴(Collapse)합니다...")
    t_cog_start = time.perf_counter()
    active_solution = engine.solve_wfc_collapse(stimulus, [dna_one_plus_one, dna_other])
    t_cog_active = time.perf_counter() - t_cog_start
    print(f"     * 최초 사유(WFC 수렴) 시간: {t_cog_active*1000:.3f} ms")
    print(f"     * 수렴된 사유 카테고리: {active_solution['collapsed_dna']['category']}")

    print("  2) 이 완성된 사유를 영구 기억 축으로 결정화(Crystallize Thought)합니다.")
    engine.crystallize_thought(stimulus, active_solution)

    print("  3) 동일한 '1+1' 자극이 다시 도달했을 때, 인지 엔진의 바이패스 흐름을 측정합니다...")
    t_cog_start2 = time.perf_counter()
    # 후보 dnas를 전혀 주지 않아도 (심지어 [] 빈 리스트여도) 즉시 완성된 사유 축을 리턴합니다.
    crystallized_solution = engine.solve_wfc_collapse(stimulus, [])
    t_cog_bypass = time.perf_counter() - t_cog_start2
    print(f"     * 결정화 사유 바이패스 처리 시간: {t_cog_bypass*1000:.3f} ms")
    print(f"     * 인지 연산 감축율: {(t_cog_active - t_cog_bypass)/t_cog_active*100:.2f}%")
    print(f"     * 복구된 사유 카테고리: {crystallized_solution['collapsed_dna']['category']}")

    print("\n" + "=" * 80)
    print("   [SUCCESS] CRYSTALLIZED AXIS DEMONSTRATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
