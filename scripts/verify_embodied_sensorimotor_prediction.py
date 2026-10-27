"""
scripts/verify_embodied_sensorimotor_prediction.py
===================================================
Verification & Demonstration Script for Embodied Sensorimotor Prediction & Recalibration Loop
(체화된 감각-운동 예측 및 재인식 루프 실증 검증 스크립트)

Demonstrates:
1. Multimodal inputs (Random Image Matrix, Mathematical Equation, Unstructured Text).
2. 4-Stage Biological Sensorimotor Loop:
   Projection ($P_{exp}$) -> Sensorimotor Intervention -> Collision & Cognitive Heat ($Q_{heat}$) -> Relational Recalibration.
3. Sprouting of Branching Factors and Topological Self-Rewiring.
"""

import numpy as np
from L1_causal_field.core.embodied.embodied_sensorimotor_prediction_loop import (
    EmbodiedSensorimotorPredictionLoop,
)


def main():
    print("=" * 80)
    print(" [Elysia] Embodied Sensorimotor Prediction Loop Verification")
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

    for name, input_data, hint in test_cases:
        print(f"\n--- Testing Input: [{name}] ---")
        recalibration = engine.run_full_sensorimotor_cycle(
            internal_intent=internal_intent,
            external_input_data=input_data,
            modality_hint=hint,
        )

        collision = recalibration.collision
        proj = collision.expected_wave

        print(f"1. Projection Wave Norm      : {np.linalg.norm(proj.projection_vector):.4f}")
        print(f"2. Real Feedback Vector Norm : {np.linalg.norm(collision.real_feedback_vector):.4f}")
        print(f"3. Discrepancy Magnitude (||Δ||): {collision.discrepancy_magnitude:.4f}")
        print(f"4. Cognitive Heat (Q_heat)   : {collision.cognitive_heat:.4f}")
        print(f"5. Tension Rupture Occurred? : {collision.is_rupture}")
        print(f"6. Sprouted Branch Node?     : {recalibration.branching_factor_sprouted} (ID: {recalibration.sprouted_branch_id})")
        print(f"7. Recalibration Narrative   :\n   {recalibration.recalibration_narrative}")

    print("\n" + "=" * 80)
    print(" Verification Completed Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
