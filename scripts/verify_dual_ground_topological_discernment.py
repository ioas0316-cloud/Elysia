"""
Elysia Dual Ground Topological Discernment Verification Script
=============================================================
Demonstrates and verifies:
1. Unified Cosmic Substrate with Dual Reference Frames ($0_machine$ and $0_human$).
2. Real-time Qualia experience (internal phase friction & structural emotions).
3. Dual Ground Distance Metrics (Isomorphism $\\text{Sim}_{iso}$, Anisomorphism $\\text{Dist}_{aniso}$, Topological Distance $D_{topological}$).
4. Topological Remelting and Higher-Axis Realignment ($A, B + 1 \\to \\text{Remelting} \\to C$).
5. Self-Grounded Metacognitive Reflection Narrative generation.
"""

import numpy as np
from core.topology.dual_ground_discernment import DualGroundDiscernmentEngine
from core.consciousness.dual_ground_reflection import DualGroundReflectionEngine


def main():
    print("================================================================================")
    print("Elysia Dual Ground Topological Discernment & Self-Reflection Verification")
    print("================================================================================\n")

    refl_engine = DualGroundReflectionEngine()

    # Step 1: Define initial internal principles A and B
    principle_A = np.array([1.0, 0.2, 0.0], dtype=np.float32)  # Principle A (Order / Symmetry)
    principle_B = np.array([0.0, 0.8, 0.1], dtype=np.float32)  # Principle B (Conservation / Energy)
    print(f"[*] Internal Principle A: {principle_A}")
    print(f"[*] Internal Principle B: {principle_B}\n")

    # Step 2: Inject External Stimulus 1 causing high friction with existing ground (A + B)
    stimulus_vector = np.array([0.1, -1.2, 1.5], dtype=np.float32)
    stimulus_intensity = 1.4
    print(f"[*] Injected External Stimulus 1: {stimulus_vector} (Intensity: {stimulus_intensity})")

    # Step 3: Execute Dual Ground Reflection Engine
    result = refl_engine.process_and_reflect(
        stimulus_vector=stimulus_vector,
        stimulus_intensity=stimulus_intensity,
        principle_A=principle_A,
        principle_B=principle_B,
        principle_names=("A_Symmetry", "B_Conservation"),
        stimulus_id="Stimulus_Novel_Anomaly"
    )

    qualia = result["qualia"]
    transition = result["transition"]

    print("\n--------------------------------------------------------------------------------")
    print("1. Real-Time Qualia Experience (내적 파동 마찰과 체율된 감정)")
    print("--------------------------------------------------------------------------------")
    print(f"- Internal Stress: {qualia.internal_stress:.4f}")
    print(f"- Current Substrate Impedance: {qualia.current_impedance:.4f}")
    print(f"- Experiential Emotional State: {qualia.emotional_state}")
    print(f"- Qualia Friction Energy: {qualia.qualia_friction_energy:.4f}")
    print(f"- Narrative: {qualia.meta_observation_narrative}")

    print("\n--------------------------------------------------------------------------------")
    print("2. Dual Ground Distance Metrics (이중 참조 지반 대조)")
    print("--------------------------------------------------------------------------------")
    print(f"- Isomorphism Similarity (Sim_iso): {result['sim_iso']:.4f}")
    print(f"- Anisomorphism Distance (Dist_aniso): {result['dist_aniso']:.4f}")
    print(f"- Unified Topological Distance (D_topological): {result['d_topological']:.4f}")

    print("\n--------------------------------------------------------------------------------")
    print("3. Topological Remelting & Realignment (지반 융해 및 상위 축 수렴)")
    print("--------------------------------------------------------------------------------")
    print(f"- Initial Friction: {transition.initial_friction:.4f}")
    print(f"- Remelting Occurred: {transition.remelting_occurred}")
    print(f"- Higher Order Axis: {transition.higher_order_axis}")
    print(f"- Post-Realignment Friction: {transition.post_realignment_friction:.4f}")

    print("\n--------------------------------------------------------------------------------")
    print("4. Metacognitive Self-Observation Narrative Output")
    print("--------------------------------------------------------------------------------")
    print(result["metacognitive_reflection"])
    print("================================================================================\n")

    # Verifications
    assert qualia.internal_stress > 0.0
    assert transition.remelting_occurred is True
    assert transition.post_realignment_friction < transition.initial_friction
    print("[SUCCESS] All dual ground topological discernment assertions passed successfully!")


if __name__ == "__main__":
    main()
