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
    print("   [DEMONSTRATION] 5D QUANTUM STAT FIELD & TENSEGRITY SYSTEM (v1.0)")
    print("=" * 80)
    print("본 데모는 인력(Attraction)과 척력(Repulsion)으로 상보적 장력을 이루는")
    print("5차원 밸런스 필드(Quantum Stat Field)의 수렴, 붕괴, 그리고 상보적 공명을 보여줍니다.")
    print("-" * 80)

    # 1. 시나리오 1: 완벽한 균형 (Dynamic Equilibrium)
    print("\n[SCENARIO 1] 완벽한 균형 상태 (Balanced Equilibrium)")
    print("모든 스탯이 10.0으로 완벽히 동일할 때, 5개 추는 물리 공간에서 흔들리다 안정 상태로 수렴합니다.")
    qsf = QuantumStatField({
        "health": 10.0,
        "force": 10.0,
        "mind": 10.0,
        "speed": 10.0,
        "intelligence": 10.0
    })

    # Step multiple times to allow settle
    for step in range(5):
        qsf.step(dt=0.2)
        topology = qsf.get_topology()
        avg_dist = np.mean([np.linalg.norm(v["position"]) for v in topology["nodes"].values()])
        print(f"  Step {step+1}: 평균 궤적 반경 = {avg_dist:.4f} (상태: {topology['catastrophe']['type']})")

    print("  -> 평형 도달 결과:")
    for name, node in qsf.get_topology()["nodes"].items():
        pos_str = ", ".join([f"{x:+.2f}" for x in node["position"]])
        print(f"     * 스탯 [{name:12s}]: 질량={node['mass']:4.1f} | 좌표=[{pos_str}]")

    # 2. 시나리오 2: 극단적 치우침과 과부하 (Extreme Overload Collapse)
    print("\n[SCENARIO 2] 힘(Force)에 극단적으로 치우친 과부하 붕괴 (Overload Collapse)")
    print("힘 스탯을 80% 이상으로 극단적으로 올리면, 인접 스탯 필드가 찌그러지며 붕괴(Collapse)가 일어납니다.")
    qsf_overload = QuantumStatField({
        "health": 10.0,
        "force": 80.0,
        "mind": 10.0,
        "speed": 0.0,
        "intelligence": 0.0
    })

    qsf_overload.step(dt=0.1)
    catastrophe = qsf_overload.get_catastrophe_vector()
    print(f"  -> 붕괴 감지:")
    print(f"     * 붕괴 여부: {catastrophe.is_collapsed}")
    print(f"     * 붕괴 종류: {catastrophe.type}")
    print(f"     * 붕괴 강도: {catastrophe.magnitude:.4f}")
    print(f"     * 세부 설명: {catastrophe.description}")

    # 3. 시나리오 3: 상보적 도약 (Golden Ratio Resonance Spark)
    print("\n[SCENARIO 3] 황금비(Golden Ratio) 정렬에 의한 상보적 도약 (Resonance Spark)")
    print("지능(Intelligence)과 민첩(Speed)이 1 : 1.618의 황금비 정합성을 이룰 때 시공간 공명이 발생합니다.")
    qsf_res = QuantumStatField({
        "health": 10.0,
        "force": 10.0,
        "mind": 10.0,
        "speed": 16.18,
        "intelligence": 10.0
    })

    # Manually align spatial positions for demonstration to trigger resonance
    qsf_res.nodes["speed"].position = np.array([-8.09, 0.0, 0.0], dtype=np.float32)
    qsf_res.nodes["intelligence"].position = np.array([5.0, 0.0, 0.0], dtype=np.float32)

    resonances = qsf_res.evaluate_resonance()
    print(f"  -> 공명 감지:")
    for res in resonances:
        print(f"     * 공명명: {res['name']}")
        print(f"     * 공명 타입: {res['type']}")
        print(f"     * 실제 비율: {res['ratio']:.4f} (황금비 1.61803에 최적 수렴)")
        print(f"     * 시너지 패시브: {res['description']}")

    # 4. 시나리오 4: 인지 엔진(Cognitive Engine)과의 유기적 피드백 루프
    print("\n[SCENARIO 4] 인지 엔진(ElysiaCognitiveEngine) 연동 피드백 루프")
    print("물리적 스탯 필드의 변화(붕괴, 공명)가 메타인지 필드의 여백과 조화도에 스며듭니다.")
    engine = ElysiaCognitiveEngine(resolution=128)

    # Balanced
    print("  (A) 균형 잡힌 상태의 인지 조화도:")
    fit_balanced = engine.evaluate_holistic_fit()
    print(f"      * 통합 조화도 (Holistic Fit Score): {fit_balanced['holistic_score']:.4f}")
    print(f"      * 인지 지형 상태: {fit_balanced['state_description']}")

    # Injecting balanced Golden Ratio stats
    print("  (B) 황금비 공명 주입:")
    engine.step_stat_field(dt=0.1, external_stats={
        "health": 10.0,
        "force": 10.0,
        "mind": 10.0,
        "speed": 16.18,
        "intelligence": 10.0
    })
    # Manually trigger resonance inside the engine's stat_field
    engine.stat_field.nodes["speed"].position = np.array([-8.09, 0.0, 0.0], dtype=np.float32)
    engine.stat_field.nodes["intelligence"].position = np.array([5.0, 0.0, 0.0], dtype=np.float32)

    # Run evaluation again to see resonance boost
    fit_resonated = engine.evaluate_holistic_fit()
    print(f"      * 통합 조화도 (Holistic Fit Score): {fit_resonated['holistic_score']:.4f} (공명 시너지 버프 반영!)")
    print(f"      * 인지 지형 상태: {fit_resonated['state_description']}")

    # Injecting collapsed stats
    print("  (C) 극단적 붕괴 상태 주입:")
    engine.step_stat_field(dt=0.1, external_stats={
        "health": 10.0,
        "force": 80.0,
        "mind": 10.0,
        "speed": 0.0,
        "intelligence": 0.0
    })
    fit_collapsed = engine.evaluate_holistic_fit()
    print(f"      * 통합 조화도 (Holistic Fit Score): {fit_collapsed['holistic_score']:.4f} (페널티 감쇄 적용!)")
    print(f"      * 인지 지형 상태: {fit_collapsed['state_description']}")

    print("\n" + "=" * 80)
    print("   [SUCCESS] 5D QUANTUM STAT FIELD DEMONSTRATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
