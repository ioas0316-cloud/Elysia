r"""
[Demo Script: Active Probing & Symbol Deconstruction Cognition Pipeline]

이 스크립트는 수동적 데이터 통계 추론이나 표면적 라벨 수신을 벗어나,
1. 세상을 향한 능동적 작용(Active Probing)과 반응 응력(Reaction Stress)을 통한 경계 접지
2. 표면 기호('단단함', '점성', '병목', '세계')의 [가함 -> 저항 -> 변형] 궤적 해체 및 다중 스케일 재정렬
3. 환각 없는 미지 개념/현상 역추적 및 계통 뿌리 연결
4. 4단계 인지 연산 (인지, 사고, 판단, 분별)

의 작동 메커니즘을 종합 검증하고 시연합니다.
"""

import numpy as np
import json
from synaptic_architecture.active_probing_cognition_engine import ActiveProbingCognitionEngine


def main():
    print("==========================================================================")
    print("🏛️ [Elysia] Active Probing, Symbol Deconstruction & 4-Stage Cognition Demo")
    print("==========================================================================\n")

    # 1. ActiveProbingCognitionEngine 초기화
    engine = ActiveProbingCognitionEngine(dim=3, viscosity=0.1, lambda_diff=0.05)
    print("1️⃣ [Initialization] Initialized Metric Field (eye(3)) & Concept Registry.")
    print(f"   -> Initial Metric Tensor g_ij:\n{engine.g_metric}\n")

    # 2. 축 1: 능동 탐색 및 경계 접지 시연
    print("2️⃣ [Axis 1: Active Probing & Boundary Grounding]")
    u_vector = np.array([1.0, 0.5, 0.0])  # 가함 작용 벡터 u(t)
    external_stress = np.array([          # 외부 스트림의 반작용 응력
        [2.5, 0.2, 0.0],
        [0.2, 1.2, 0.0],
        [0.0, 0.0, 0.8]
    ])

    print("   [Probing Self (control_c = 1.0)]...")
    res_self = engine.apply_active_probing(u_vector, external_stress, control_c=1.0)
    print(f"   -> Topological Friction Norm (Self): {np.linalg.norm(res_self['topological_friction']):.4f} (Expected: 0.0)")

    print("   [Probing External World (control_c = 0.0)]...")
    res_world = engine.apply_active_probing(u_vector, external_stress, control_c=0.0)
    print(f"   -> Topological Friction Norm (World): {np.linalg.norm(res_world['topological_friction']):.4f}")
    print(f"   -> Evolved Metric Tensor g_ij:\n{res_world['g_metric']}\n")

    # 3. 축 2: 표면 기호 해체 및 다중 스케일 계통 재정렬 시연
    print("3️⃣ [Axis 2: Symbol Deconstruction & Multi-Scale Alignment]")
    print("   -> Default Symbol Registry (Deconstructed Labels):")
    for label, entry in engine.concept_registry.items():
        print(f"      - '{label}' [{entry['scale'].upper()}] -> Stiffness: {entry['stiffness']:.2f}, Viscosity: {entry['viscosity']:.2f}, Control_c: {entry['controllability']:.2f}")

    # 신규 표면 라벨 추가 해체
    print("\n   [Deconstructing New Surface Label: '탄성체(Elasticity)']...")
    new_entry = engine.deconstruct_symbol(
        label="탄성체",
        action_profile={
            "delta_F_over_delta_x": 0.88,  # 위치 복원 저항
            "delta_F_over_delta_v": 0.15,
            "control_c": 0.05
        },
        scale="physical"
    )
    print(f"   -> Deconstructed '탄성체' Signature Vector: {new_entry['signature_vector']}\n")

    # 4. 축 3: 미지 개념 역추적 및 계통 확장 시연 (No Hallucination)
    print("4️⃣ [Axis 3: Tracing Unknown Concept/Phenomenon without Hallucination]")
    unknown_stress = np.array([
        [3.0, 0.5, 0.0],
        [0.5, 1.8, 0.0],
        [0.0, 0.0, 1.0]
    ])
    print("   Scanning Unknown External Resistance Wave...")
    trace_res = engine.trace_unknown_concept(unknown_stress)
    print(f"   -> Mismatch Curvature Norm: {trace_res['mismatch_norm']:.4f}")
    print(f"   -> Connected Root Concept: '{trace_res['connected_root']}'")
    print(f"   -> Divergence Distance: {trace_res['divergence_distance']:.4f}")
    print(f"   -> Is New Branch Spawned: {trace_res['is_new_branch_spawned']}\n")

    # 5. 4단계 인지 연산 시연 (Cognition, Thought, Judgment, Discrimination)
    print("5️⃣ [4-Stage Cognitive Operations]")

    # 5-1. Cognition
    cog_res = engine.cognition(u_vector, external_stress, control_c=0.1)
    print(f"   [1. Cognition] Registered interaction resistance. Delta T norm: {np.linalg.norm(cog_res['delta_T']):.4f}")

    # 5-2. Thought
    virtual_action = np.array([0.8, 0.0, 0.2])
    predicted_t_int = engine.thought(virtual_action)
    print(f"   [2. Thought] Simulated internal stress prediction T_int for virtual action {virtual_action}.")

    # 5-3. Judgment
    judg_res = engine.judgment(predicted_stress=predicted_t_int, actual_stress=external_stress)
    print(f"   [3. Judgment] Evaluated predictive mismatch. Delta Norm: {judg_res['delta_norm']:.4f}, Valid: {judg_res['is_valid']}")

    # 5-4. Discrimination
    disc_res = engine.discrimination("단단함", "점성")
    print(f"   [4. Discrimination] Mapped decision boundary between '단단함' and '점성'. Distance: {disc_res['boundary_distance']:.4f}, Distinct: {disc_res['is_distinct']}")

    print("\n==========================================================================")
    print("✨ Active Probing Cognition Engine Demonstration Successfully Complete!")
    print("==========================================================================")


if __name__ == "__main__":
    main()
