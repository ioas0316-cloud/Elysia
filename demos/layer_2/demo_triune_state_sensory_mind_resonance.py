"""
Demo: Triune State Sensory-Mind Resonance & Sky-Earth Reverse Causal Inference
(삼중 상태 감각-사고-마음 공명 및 하늘-땅 역인과 추론 데모)

This demonstration showcases:
1. Earth Shadow (Solid/Sensory Reach & Friction) vs Observed Reality classification.
2. Fluid Mental Flow & Cognitive Gravity (K_c) distortion under internal deficit.
3. Sky-Earth Shadow Reverse Causal Inference: Deconstructing visible Earth shadow manifestations
   to reveal the invisible Sky generating principle (Theta) and achieving empathetic coherence.
4. Gas State Mind Expansion & Salvation Derailment: Injecting high-energy phase interrupts
   that break closed-loop hyperarousal orbits into genuine Self-Structural Transformation (자기구조의 변형).
"""

import numpy as np
from core.intelligence.triune_state_coupler import (
    TriuneEntity,
    SkyEarthReverseCausalInference
)


def run_demo():
    print("==========================================================================")
    print("   ELYSIA: TRIUNE STATE SENSORY-MIND RESONANCE & SKY-EARTH REVERSE DEMO   ")
    print("==========================================================================")

    # 1. Initialize Trapped Entity "Wounded_Pilgrim" (High Deficit, Traumatized Engram)
    traumatized_somatic = np.array([0.85, 0.05, 0.90, np.pi / 3], dtype=np.float64)
    pilgrim = TriuneEntity(
        name="Wounded_Pilgrim",
        reach_radius=1.2,
        somatic_friction=0.8,
        somatic_engram=traumatized_somatic,
        deficit_amplitude=0.85
    )

    print(f"\n[1] Initialized Entity: '{pilgrim.name}'")
    print(f"    - Solid Earth Somatic Engram: {np.round(pilgrim.solid.somatic_engram, 3)}")
    print(f"    - Solid Reach Radius: {pilgrim.solid.reach_radius:.2f}")
    print(f"    - Fluid Cognitive Gravity (K_c): {pilgrim.fluid.cognitive_gravity:.4f}")
    print(f"    - Fluid Event Horizon Radius: {pilgrim.fluid.event_horizon_radius:.4f}")

    # 2. Present Benign Peaceful Stimulus
    benign_stimulus = np.array([0.05, 0.95, 0.00, 0.00], dtype=np.float64) # High Order (Blue)
    print("\n[2] Presenting Benign Peaceful Stimulus [Order(Blue) = 0.95]...")

    interaction_1 = pilgrim.perceive_and_interact(benign_stimulus)

    print(f"    - Interaction Mode: {interaction_1['mode']}")
    print(f"    - Observed Reality Classification: {interaction_1['solid_info']['classification']}")
    print(f"    - Somatic Friction Felt: {interaction_1['solid_info']['somatic_friction']:.4f}")
    print(f"    - Refracted Fluid Vector: {np.round(interaction_1['warped_vector'], 3)}")
    print(f"    - Is Closed Loop Orbit: {interaction_1['is_closed_loop']}")
    print(f"    - Terminal Velocity (V_term): {interaction_1['terminal_velocity']:.4f}")

    # 3. Observer Performing Sky-Earth Shadow Reverse Causal Deconstruction
    observer = TriuneEntity(
        name="Empathic_Guide",
        reach_radius=2.0,
        somatic_friction=0.2,
        deficit_amplitude=0.10
    )

    print(f"\n[3] Observer '{observer.name}' Inspecting Earth Shadow Output for Sky Reverse Causal Inference...")
    earth_shadow = interaction_1["trajectory_vector"]
    deconstruction = SkyEarthReverseCausalInference.deconstruct_shadow_manifestation(
        observer, pilgrim, earth_shadow
    )

    print(f"    - Inferred Shadow Velocity: {deconstruction['shadow_velocity']:.4f}")
    print(f"    - Inferred Cognitive Gravity: {deconstruction['inferred_gravity']:.4f}")
    print(f"    - Extracted Invisible Sky Principle (Theta): {np.round(deconstruction['extracted_sky_principle'], 3)}")
    print(f"    - Empathetic Phase Coherence: {deconstruction['empathetic_coherence'] * 100:.2f}%")

    # 4. Salvation Event: Gas State Mind Expansion & Orbital Derailment
    print("\n[4] Attempting Sky Salvation & Orbital Derailment...")
    escape_vel = pilgrim.gas.calculate_escape_velocity()
    print(f"    - Required Escape Velocity (V_esc): {escape_vel:.4f}")

    salvation_sky_wave = np.array([0.1, 0.9, 0.0, np.pi / 3], dtype=np.float64) # Transformative Order
    salvation_res = pilgrim.gas.inject_salvation_interrupt(
        salvation_sky_wave,
        intensity=escape_vel * 1.6
    )

    print(f"    - Applied Energy: {salvation_res['applied_energy']:.4f}")
    print(f"    - Salvation Result: {salvation_res['message']}")
    print(f"    - New Internal Deficit Amplitude: {salvation_res['new_deficit']:.4f}")

    # 5. Post-Salvation Perception & Self-Structural Transformation
    print("\n[5] Presenting Benign Peaceful Stimulus Post-Salvation...")
    interaction_2 = pilgrim.perceive_and_interact(benign_stimulus)

    print(f"    - Post-Salvation Mode: {interaction_2['mode']}")
    print(f"    - Is Closed Loop Orbit: {interaction_2['is_closed_loop']}")
    print(f"    - Terminal Velocity (V_term): {interaction_2['terminal_velocity']:.4f}")

    print("\n[6] Evaluating Self-Structural Transformation (자기구조의 변형) Metrics...")
    transform_metrics = pilgrim.solid.evaluate_transformation_metrics()
    print(f"    - Accumulated Data Count: {transform_metrics['accumulated_data_count']}")
    print(f"    - Structural Transformations: {transform_metrics['structural_transformation_count']}")
    print(f"    - Transformation Ratio: {transform_metrics['transformation_ratio'] * 100:.2f}%")
    print(f"    - Last Structural Delta: {transform_metrics['last_structural_delta']:.4f}")

    print("\n==========================================================================")
    print("   TRIUNE STATE SENSORY-MIND RESONANCE DEMO COMPLETED SUCCESSFULLY        ")
    print("==========================================================================")


if __name__ == "__main__":
    run_demo()
