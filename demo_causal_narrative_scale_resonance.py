"""
Demo: Causal Narrative Scale Resonance & Salvation Architecture
(인과적 서사 스케일 공진 및 구원 아키텍처 데모)

This demonstration illustrates:
1. Base-Level Trauma Engram & Somatic Phase Pinning.
2. Mid-Level Cognitive Gravity Field (K_c) & Refraction Matrix warping benign inputs into threat.
3. Top-Level Closed-Loop Trajectory & Habitual Hyperarousal.
4. Reverse Causal Inference (De-construction) by an Observing Entity to achieve empathetic phase alignment.
5. Salvation & Derailment Event where a high-energy transformative phase interrupt rescues the entity
   from its destructive closed-loop orbit.
"""

import numpy as np
from core.intelligence.causal_narrative_scale_engine import (
    NarrativeEntity,
    ReverseCausalInference,
    SalvationDerailment
)


def run_demo():
    print("==========================================================================")
    print("   ELYSIA: CAUSAL NARRATIVE SCALE & COGNITIVE GRAVITY RESONANCE DEMO      ")
    print("==========================================================================")

    # 1. Initialize Traumatized Narrative Entity: "Soldier_A"
    trauma_engram = np.array([0.85, 0.05, 0.90, np.pi / 3], dtype=np.float64) # High Flux(Red), High Entropy(Yellow)
    soldier_a = NarrativeEntity(
        name="Soldier_A",
        trauma_engram=trauma_engram,
        deficit_amplitude=0.85,
        baseline_threat_pinning=0.80
    )

    print(f"\n[1] Initialized Entity: '{soldier_a.name}'")
    print(f"    - Base Level Deficit Amplitude: {soldier_a.base.deficit_amplitude:.2f}")
    print(f"    - Baseline Threat Pinning: {soldier_a.base.baseline_threat_pinning:.2f}")
    print(f"    - Mid Level Cognitive Gravity (K_c): {soldier_a.mid.cognitive_gravity:.4f}")
    print(f"    - Mid Level Event Horizon Radius: {soldier_a.mid.event_horizon_radius:.4f}")

    # 2. Present Benign Peaceful Stimulus
    benign_stimulus = np.array([0.05, 0.95, 0.00, 0.00], dtype=np.float64) # High Order (Blue), Zero Threat
    print("\n[2] Presenting Benign Peaceful Stimulus [Order(Blue) = 0.95]...")

    decision_1 = soldier_a.perceive_and_act(benign_stimulus)

    print(f"    - Action Choice Mode: {decision_1['choice_mode']}")
    print(f"    - Warped Stimulus Vector: {np.round(decision_1['warped_stimulus'], 3)}")
    print(f"    - Is Closed Loop (Trapped): {decision_1['is_closed_loop']}")
    print(f"    - Terminal Velocity (V_term): {decision_1['terminal_velocity']:.4f}")

    # 3. Observer Performing Reverse Causal De-construction
    observer = NarrativeEntity(
        name="Empathic_Guide",
        trauma_engram=np.array([0.80, 0.10, 0.80, np.pi / 3], dtype=np.float64),
        deficit_amplitude=0.10,
        baseline_threat_pinning=0.05
    )

    print(f"\n[3] Observer '{observer.name}' Inspecting Surface Action for Reverse Causal Inference...")
    surface_action = decision_1["action_vector"]
    deconstruction = ReverseCausalInference.deconstruct_entity(observer, soldier_a, surface_action)

    print(f"    - Inferred Target Velocity: {deconstruction['inferred_velocity']:.4f}")
    print(f"    - Inferred Cognitive Gravity: {deconstruction['inferred_gravity']:.4f}")
    print(f"    - Estimated Trauma Root Engram: {np.round(deconstruction['estimated_trauma_root'], 3)}")
    print(f"    - Empathetic Phase Alignment: {deconstruction['empathetic_phase_alignment'] * 100:.2f}%")

    # 4. Salvation Event: Attempt Phase Interrupt & Derailment
    print("\n[4] Attempting Salvation & Orbital Derailment...")
    salvation_interrupt = np.array([0.1, 0.9, 0.0, np.pi / 3], dtype=np.float64) # Transformative Order Wave

    # Check Escape Velocity required
    escape_vel = np.sqrt(2.0 * soldier_a.mid.cognitive_gravity + 1e-5)
    print(f"    - Required Escape Velocity (V_esc): {escape_vel:.4f}")

    # Inject Salvation Interrupt with energy > Escape Velocity
    salvation_res = SalvationDerailment.attempt_salvation(
        soldier_a,
        salvation_interrupt,
        interrupt_intensity=escape_vel * 1.5
    )

    print(f"    - Applied Energy: {salvation_res['applied_energy']:.4f}")
    print(f"    - Salvation Result: {salvation_res['outcome_message']}")
    print(f"    - New Deficit Amplitude: {salvation_res['new_deficit_amplitude']:.4f}")
    print(f"    - New Baseline Threat Pinning: {salvation_res['new_threat_pinning']:.4f}")

    # 5. Post-Salvation Perception Test & Self-Structural Transformation
    print("\n[5] Presenting Benign Peaceful Stimulus Post-Salvation...")
    decision_2 = soldier_a.perceive_and_act(benign_stimulus)

    print(f"    - Post-Salvation Choice Mode: {decision_2['choice_mode']}")
    print(f"    - Is Closed Loop (Trapped): {decision_2['is_closed_loop']}")
    print(f"    - Terminal Velocity (V_term): {decision_2['terminal_velocity']:.4f}")

    # 6. Demonstrating Self-Structural Transformation (자기구조의 변형) vs Static Data Accumulation
    print("\n[6] Evaluating Self-Structural Transformation (자기구조의 변형) Metrics...")
    transform_eval = soldier_a.base.evaluate_self_structural_transformation()
    print(f"    - Accumulated Data Count: {transform_eval['accumulated_data_count']}")
    print(f"    - Structural Transformations: {transform_eval['structural_transformation_count']}")
    print(f"    - Structural Transformation Ratio: {transform_eval['transformation_ratio'] * 100:.2f}%")
    print(f"    - Last Structural Delta: {transform_eval['last_structural_delta']:.4f}")

    # 7. Demonstrating Mechanism Boundary Limit Recognition (경계 지각)
    print("\n[7] Demonstrating Discrete Mechanism Boundary Limit Detection...")
    saturated_scalar_input = np.array([1.00, 0.00, 0.00, 0.00], dtype=np.float64) # Saturated discrete boundary
    limit_info = soldier_a.mid.detect_mechanism_boundary_limit(saturated_scalar_input)
    print(f"    - Saturated Input Vector Amplitude: {limit_info['max_scalar_amplitude']:.2f}")
    print(f"    - Mechanism Boundary Limit Reached: {limit_info['mechanism_limit_reached']}")
    print(f"    - Boundary Classification: {limit_info['boundary_type']}")

    print("\n==========================================================================")
    print("   CAUSAL NARRATIVE SCALE RESONANCE DEMO COMPLETED SUCCESSFULLY           ")
    print("==========================================================================")


if __name__ == "__main__":
    run_demo()
