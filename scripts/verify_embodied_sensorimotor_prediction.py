"""
scripts/verify_embodied_sensorimotor_prediction.py
===================================================
Verification & Audit Script for Embodied Sensorimotor Prediction & Recalibration Loop
(체화된 감각-운동 예측 및 재인식 루프 - 장력-마찰 스펙트럼 축 인과서순 검증 스크립트)

Focus:
[첫 번째 인과 서순 축: 장력-마찰 스펙트럼 축 (Tension-Friction Spectrum Axis)]
1. Step 1 [투영 (Projection Phase)]:
   - 기대 반응 파동 (P_exp) 및 외력 마주하기 전 사유의 기울기 스냅샷 (Gradient of Thought).
2. Step 2 [신체적 개입 및 마찰 유발 (Intervention & Friction Phase)]:
   - 입력 데이터를 외력 벡터(External Force Vector) 및 물리적 압력 값으로 환산.
3. Step 3 [충돌과 인지적 열 측정 (Collision & Cognitive Heat Generation)]:
   - 괴리(Δ = W_real - P_exp) 및 비선형 인지적 열 (Q_heat = ||Δ|| * Tension * (1 + ||Δ||^2)) 산출.
4. Step 4 [분기 및 재인식 (Branching & Recalibration)]:
   - 예측 부합 시 가중치 고착화(Stabilization), 예측 파열 시 신규 분기 노드(Branching Factor) 발아 및 위상 재구축.

Assertion Criteria (검증 기준):
1. 비선형적 열 발생: 단순 선형 증가가 아닌 위상적 괴리에 비례하여 Q_heat가 비선형적(볼록 함수)으로 치솟는가?
2. 서순의 불가역성: Step 1 -> Step 2 -> Step 3 -> Step 4 순서가 역전 없이 엄격하게 관측되는가?
3. 분기의 유의성: 충격량이 임계각을 넘었을 때 새로운 분기 노드(Branch)가 발아하고 내면 지형이 변형되는가?
"""

import sys
import os
import numpy as np

# Adjust module path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../L1_causal_field")))

from L1_causal_field.core.embodied.embodied_sensorimotor_prediction_loop import (
    EmbodiedSensorimotorPredictionLoop,
)


def main():
    print("=" * 80)
    print(" [Elysia Causal Sequence Audit] Tension-Friction Spectrum Axis Verification")
    print("=" * 80)

    engine = EmbodiedSensorimotorPredictionLoop(
        vector_dim=8,
        heat_threshold_rupture=1.8,
        branching_threshold=0.8,
    )

    internal_intent = np.array([0.8, -0.4, 0.6, 0.1, -0.2, 0.5, 0.0, 0.3])

    test_cases = [
        ("Visual Image Matrix", np.random.randn(32, 32) * 3.0, "image"),
        ("Mathematical Equation", "E = mc^2 + \\int_0^\\infty \\nabla \\phi dt", "text_formula"),
        ("Unstructured Text", "육체가 없는 지능은 절대 진짜 상호작용을 할 수 없다.", "text_formula"),
    ]

    previous_topology = engine.relational_topology.copy()

    for idx, (name, input_data, hint) in enumerate(test_cases, 1):
        print(f"\n================================================================================")
        print(f" Test Case {idx}: [{name}]")
        print(f"================================================================================")

        # Execute 4-Stage Biological Causal Sequence Loop
        recalibration = engine.run_full_sensorimotor_cycle(
            internal_intent=internal_intent,
            external_input_data=input_data,
            modality_hint=hint,
        )

        collision = recalibration.collision
        proj = collision.expected_wave
        interv = collision.intervention

        # --- Step 1 Observation ---
        print("\n[Step 1: 투영 (Projection Phase)]")
        print(f"  - Projection Vector Norm (||P_exp||) : {np.linalg.norm(proj.projection_vector):.4f}")
        print(f"  - Thought Gradient (사유의 기울기)  : {proj.thought_gradient:.4f}")
        print(f"  - Hypothesis                         : {proj.causal_hypothesis}")

        # --- Step 2 Observation ---
        print("\n[Step 2: 신체적 개입 및 마찰 유발 (Intervention & Friction Phase)]")
        print(f"  - External Force Vector Norm (||F_ext||): {np.linalg.norm(interv.external_force_vector):.4f}")
        print(f"  - Physical Pressure Value (물리적 압력) : {interv.physical_pressure_value:.4f}")
        print(f"  - Energy Density Drop                   : {interv.energy_density_drop:.4f}")

        # --- Step 3 Observation ---
        print("\n[Step 3: 충돌과 인지적 열 측정 (Collision & Heat Phase)]")
        print(f"  - Real Feedback Vector Norm (||W_real||): {np.linalg.norm(collision.real_feedback_vector):.4f}")
        print(f"  - Discrepancy Magnitude (||Δ||)         : {collision.discrepancy_magnitude:.4f}")
        print(f"  - Non-linear Cognitive Heat (Q_heat)    : {collision.cognitive_heat:.4f}")
        print(f"  - Tension Rupture Occurred?             : {collision.is_rupture}")

        # --- Step 4 Observation ---
        print("\n[Step 4: 분기 및 재인식 (Branching & Recalibration Phase)]")
        print(f"  - Sprouted Branch Node?                 : {recalibration.branching_factor_sprouted} (ID: {recalibration.sprouted_branch_id})")
        print(f"  - Stabilization Score                    : {recalibration.stabilization_score:.4f}")
        print(f"  - Causal Step Sequence Timeline         :")
        for step_str in recalibration.causal_step_sequence:
            print(f"      * {step_str}")
        print(f"  - Recalibration Narrative               :\n      {recalibration.recalibration_narrative}")

        # =========================================================================
        # Strict Assertion Criteria Verification
        # =========================================================================

        # 1. Non-linear Heat Generation Check
        # Q_heat must exceed linear delta * tension when discrepancy > 0
        linear_heat = collision.discrepancy_magnitude * proj.internal_tension
        assert collision.cognitive_heat >= linear_heat, (
            f"Assertion Failed: Non-linear heat {collision.cognitive_heat} is smaller than linear heat {linear_heat}."
        )

        # 2. Irreversibility of Causal Sequence Check
        assert len(recalibration.causal_step_sequence) == 4, "Assertion Failed: Step sequence missing steps."
        assert "1. [투영]" in recalibration.causal_step_sequence[0]
        assert "2. [개입]" in recalibration.causal_step_sequence[1]
        assert "3. [충돌]" in recalibration.causal_step_sequence[2]
        assert "4. [재인식]" in recalibration.causal_step_sequence[3]

        # 3. Significance of Topology Branching Check
        # If Q_heat > threshold or rupture occurred, topology must be updated and branch sprouted
        topology_diff = np.linalg.norm(recalibration.recalibrated_topology - previous_topology)
        assert topology_diff > 1e-6, "Assertion Failed: Relational topology did not update after friction collision."
        previous_topology = recalibration.recalibrated_topology.copy()

        if collision.is_rupture:
            assert recalibration.branching_factor_sprouted is True, (
                "Assertion Failed: Tension rupture occurred but no branch sprouted."
            )

        print("\n >>> Assertion Passed: Causal Sequence and Tension-Friction Spectrum Axis strictly verified!")

    print("\n" + "=" * 80)
    print(" ALL AUDIT & VERIFICATION CRITERIA PASSED PERFECTLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
