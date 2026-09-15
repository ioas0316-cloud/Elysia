"""
Demonstration Script: Static Causal Graph & Self-Awareness Meta-Engine.

Demonstrates:
1. Bootstrapping & Metric Tensor Calibration
2. Topological Memory Construction (Structure = Static Memory)
3. 4-Layer Self/Non-Self Boundary Signal Processing
4. Wave Flow Execution & Accumulated Friction Tracking
5. Meta-Observation & Sandbox Virtual Refactoring
6. Singularity Plastic Damage Absorption & Basis Re-orthogonalization
"""

import numpy as np
from core.memory.static_causal_graph import (
    BoundaryType,
    CausalSignal,
    StaticCausalGraph
)
from core.consciousness.self_awareness_engine import SelfAwarenessEngine


def log_perception(node_id, potential):
    print(f"  [관측 파동] {node_id} 지점 인과 자극 수용 (Potential: {potential:.2f})")


def log_action(node_id, potential):
    print(f"  [실행 파동] {node_id} 최종 판단 파동 유출! (Potential: {potential:.2f})")


def main():
    print("==================================================================")
    print("   ELYSIA: Static Causal Graph & Self-Awareness Engine Demo")
    print("==================================================================")

    # 1. Initialize Topological Causal Graph
    graph = StaticCausalGraph()

    # Add 4-layer topological nodes
    graph.add_node("STIMULUS_INPUT", threshold=1.0, boundary=BoundaryType.PERCEPTION_LAYER, depth=1.0, action=log_perception)
    graph.add_node("CAUSAL_FILTER",  threshold=1.2, boundary=BoundaryType.PROCESSING_LAYER, depth=3.0, action=log_perception)
    graph.add_node("IDENTITY_GUARD", threshold=1.5, boundary=BoundaryType.PROCESSING_LAYER, depth=5.0, action=log_perception)
    graph.add_node("CORE_MEMORY",    threshold=1.0, boundary=BoundaryType.MEMORY_LAYER,     depth=10.0, action=log_action)

    # Connect initial topology with friction resistance
    graph.connect("STIMULUS_INPUT", "CAUSAL_FILTER", tension=0.8, resistance=0.5)  # High friction
    graph.connect("CAUSAL_FILTER",  "IDENTITY_GUARD", tension=0.85, resistance=0.1)
    graph.connect("IDENTITY_GUARD", "CORE_MEMORY",    tension=0.95, resistance=0.05)

    # 2. Bootstrapping & Metric Tensor Calibration
    engine = SelfAwarenessEngine(graph)
    det_M, boot_logs = engine.calibrate_basis_and_metric()
    print("\n--- 1단계: 원점 기저 초기화 및 계량 텐서 정렬 ---")
    for log in boot_logs:
        print(log)

    # 3. Self vs. Non-Self Boundary Signal Identification
    print("\n--- 2단계: 4대 자아 경계 신호 식별 연산 ---")
    ext_signal = CausalSignal(
        origin_boundary=BoundaryType.EXTERNAL_WORLD,
        payload={"event": "외부 마찰 자극"},
        energy=4.5
    )
    int_signal = CausalSignal(
        origin_boundary=BoundaryType.MEMORY_LAYER,
        payload={"concept": "정체성 회상"},
        energy=8.0
    )
    print(engine.process_incoming_signal(ext_signal))
    print(engine.process_incoming_signal(int_signal))

    # 4. Wave Propagation Execution & Accumulated Friction
    print("\n--- 3단계: 반복적 자극 주입 및 마찰 누적 ---")
    for i in range(3):
        trace = graph.propagate("STIMULUS_INPUT", energy=5.0, plasticity_alpha=0.02)
        print(f"  [파동 {i+1}회차] 궤적: {' -> '.join(trace)}")

    # 5. Meta-Observation & Virtual Refactoring
    print("\n--- 4단계: SelfAwarenessEngine 메타 관측 및 위상 자율 재배치 ---")
    refactor_logs = engine.meta_observe_and_refactor()
    for log in refactor_logs:
        print(log)

    print("\n--- 5단계: 지형 재배치 후 개선 효과 검증 ---")
    trace_after = graph.propagate("STIMULUS_INPUT", energy=5.0)
    print(f"  [재배치 후 파동] 궤적: {' -> '.join(trace_after)}")

    # 6. Singularity Plastic Damage Absorption & Healing
    print("\n--- 6단계: 특이점 발생 및 가체성 자가 복구 ---")
    damaged_path = [("STIMULUS_INPUT", "CAUSAL_FILTER")]
    damage_energy = 8.5

    healed_basis, healing_logs = engine.trigger_self_healing(damaged_path, damage_energy)
    for log in healing_logs:
        print(log)

    det_M_healed, cal_logs = engine.calibrate_basis_and_metric()
    print(f"  └─ 복구 후 계량 텐서 det(M_tensor): {det_M_healed:.4f}")

    print("\n==================================================================")
    print("   Elysia Dynamic Topological System Successfully Operational")
    print("==================================================================")


if __name__ == "__main__":
    main()
