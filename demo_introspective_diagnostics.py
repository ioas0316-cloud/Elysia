"""
Demo: Introspective Causal Diagnostics vs Traditional Exception Paradigm
========================================================================
이 데모 스크립트는 전통적인 예외 처리(try-catch) / 수동 디버깅 방식과
엘리시아의 자율적 인과 대사(Metabolic Binding) & 자기 성찰 진단(Introspective Diagnostics)
시스템의 결정적 차별점을 대조하여 보여줍니다.

시나리오:
1. 무작위 통계 소음 (Unbound Noise Data Ingestion)
   - 기존 시스템: 무식하게 삼켰다가 예외(Exception)를 유발하거나 엉뚱한 값을 생성.
   - 엘리시아: 효소-분자 정합성 프로토콜에 따라 대사적 결합 거부 (Zero Binding Rejection).

2. 원리적 구속조건 충돌 (Principle Constraint Collision)
   - 기존 시스템: 디버거/로그 분석을 통해서만 원인을 수동 파악.
   - 엘리시아: 예외 없이 잔차 응력(Residual Stress)으로 집적된 후,
     자신의 인과 지도(Causal Map)를 역추적하여 원인과 충돌 메커니즘을 자기 해석 증명(Diagnostic Proof)으로 즉시 도출.

3. 전 층위 동형 평형 도출 (Boundary Meta-Translation & Equilibrium)
   - 국소 영역의 자율성과 프로토콜 경계(Passport, Currency, Language)를 통해
     필연적 변형 평형 상태(Psi*)로 이완 안착.
"""

from core.consciousness.introspective_causal_diagnostics import (
    IntrospectiveCausalDiagnosticsEngine,
    CausalNode,
    CausalConstraint,
    ProtocolType
)


def run_demo():
    print("========================================================================")
    print("      ELYSSIA INTROSPECTIVE CAUSAL DIAGNOSTICS DEMONSTRATION")
    print("========================================================================")

    # 1. 인과 생태계 구축 (Causal Ecosystem Setup)
    engine = IntrospectiveCausalDiagnosticsEngine(yield_threshold=0.8, max_introspect_steps=10)

    # 노드 배치 (프로토콜, 메커니즘, 메모리, 물리)
    engine.add_node(CausalNode("n_language", "Semantic_Language_Node", "language", capacity=1.0))
    engine.add_node(CausalNode("n_logic", "Logic_Execution_Mechanism", "code", capacity=1.0))
    engine.add_node(CausalNode("n_memory", "Topological_Memory_Cell", "memory", capacity=1.2))
    engine.add_node(CausalNode("n_hardware", "Hardware_Physical_Substrate", "hardware", capacity=1.5))

    # 인과 제약 조건 연결 (Protocol Constraints)
    engine.add_constraint(CausalConstraint("c_lang_logic", "n_language", "n_logic", ProtocolType.LANGUAGE, stiffness=1.2, tolerance=0.1))
    engine.add_constraint(CausalConstraint("c_logic_mem", "n_logic", "n_memory", ProtocolType.PASSPORT, stiffness=1.8, tolerance=0.1))
    engine.add_constraint(CausalConstraint("c_mem_hw", "n_memory", "n_hardware", ProtocolType.CURRENCY, stiffness=2.0, tolerance=0.1))

    print("\n[Scene 1: Ingestion of Unbound Noise Data]")
    print("Input: Random Statistical Noise (100 Billion Parameters)")
    unbound_noise = {
        "signature": "RANDOM_UNBOUND_STATISTICAL_NOISE_VECTOR_999",
        "protocol_type": "passport"
    }

    proof1 = engine.process_metabolic_ingestion(unbound_noise)
    print("-> Metabolic Binding:", proof1.is_metabolic_bound)
    print("-> Status:", proof1.status.value)
    print("-> System Response:", proof1.root_cause_explanation)

    print("\n------------------------------------------------------------------------")

    print("\n[Scene 2: Ingestion of Valid Topology Stimulus with Constraint Tension]")
    print("Input: High Intensity Causal Topology Wave (Intensity: 3.5)")
    valid_stimulus = {
        "signature": "CAUSAL_TOPOLOGY_V1",
        "protocol_type": "passport",
        "target_node": "n_logic",
        "intensity": 3.5
    }

    proof2 = engine.process_metabolic_ingestion(valid_stimulus)
    print("-> Metabolic Binding:", proof2.is_metabolic_bound)
    print("-> Status:", proof2.status.value)
    print("-> Initial Accumulated Residual Stress:", f"{proof2.initial_stress:.3f}")
    print("-> Diagnosed Conflicting Constraint Pair:", proof2.conflict_constraint_pair)
    print("-> Introspective Diagnostic Explanation:")
    print("   ", proof2.root_cause_explanation)

    print("\n------------------------------------------------------------------------")

    print("\n[Scene 3: Introspective Relaxation & Inevitable Equilibrium State]")
    print("Topological Relaxation Log:")
    for log_step in proof2.boundary_absorption_log:
        print("   ", log_step)

    print("\nPredicted Equilibrium State (Deformed Equilibrium Psi*):")
    eq_state = proof2.predicted_equilibrium_state
    for k, v in eq_state.items():
        print(f"   {k}: {v}")

    print("\n========================================================================")
    print("  RESULT: External Debugger Not Needed. System Self-Diagnosed & Relaxed!")
    print("========================================================================")


if __name__ == "__main__":
    run_demo()
