"""
Demo script for Closed-Loop Valence Field and Sensory Feedback Engine.
Simulates real-time closed-loop control, predictive coding error feedback,
phase trajectory adaptations, and latent dimension spawning.
"""

import numpy as np
from core.memory.semantic_valence_manifold import (
    ConceptAttractor,
    SensoryFeedbackEngine,
    ClosedLoopValenceField
)


def run_closed_loop_demo():
    print("===========================================================================")
    print("   ELYSIA: CLOSED-LOOP VALENCE FIELD & SENSORY FEEDBACK ENGINE DEMO         ")
    print("===========================================================================\n")

    # Initialize Closed-Loop Valence Field
    loop_field = ClosedLoopValenceField()

    # Initial Concept Attractor: 'Bulgogi Cooking State'
    # Center position: [Pan Temp (160C), Maillard Reaction (0.2), Moisture Ratio (0.8)]
    # Resonant frequency: 120.0 Hz
    bulgogi_att = ConceptAttractor(
        concept_id="CONCEPT_COOKING_BULGOGI",
        center_pos=np.array([160.0, 0.2, 0.8]),
        resonant_freq=120.0,
        well_depth=2.0,
        well_radius=1.5
    )
    loop_field.register_attractor(bulgogi_att)

    print(">>> 0. INITIAL CONCEPT ATTRACTOR STATE <<<")
    print(f"  Concept ID: {bulgogi_att.concept_id}")
    print(f"  Center Pos [Temp, Maillard, Moisture]: {bulgogi_att.center_pos.round(2)}")
    print(f"  Well Depth (Causal Confidence): {bulgogi_att.well_depth:.3f}\n")

    # External Sensory Stream Simulation (Pan hotter and drying faster than predicted)
    actual_sensory_stream = [
        np.array([175.0, 0.45, 0.6, 1.2]),  # Step 1: Hotter and faster moisture loss
        np.array([182.0, 0.65, 0.4, 1.2]),  # Step 2: Continuous high heat
        np.array([185.0, 0.70, 0.3, 1.2]),  # Step 3: Heat stabilization
        np.array([210.0, 0.95, 0.05, 1.2]), # Step 4: Extreme unpredicted heat jump
        np.array([230.0, 0.99, 0.01, 1.2]), # Step 5: Persistent unexplainable error -> triggers Latent Axis Spawning
    ]

    print(">>> 1. CLOSED-LOOP SENSORY FEEDBACK STREAM <<<")
    for step, sensor_data in enumerate(actual_sensory_stream, 1):
        telemetry = loop_field.process_sensorimotor_step({"CONCEPT_COOKING_BULGOGI": sensor_data})
        m = telemetry["CONCEPT_COOKING_BULGOGI"]

        print(f"\n[Step {step}] Sensory Feedback Processing:")
        print(f"  - Actual Sensory Input: {sensor_data.round(2)}")
        print(f"  - Prediction Error (Surprise Magnitude): {m['prediction_error']:.3f}")
        print(f"  - Accumulated Prediction Error: {m['accumulated_error']:.3f}")
        print(f"  - Re-evaluated Well Depth: {bulgogi_att.well_depth:.3f} (delta: {m['depth_delta']:+.4f})")
        print(f"  - Updated Attractor Position: {bulgogi_att.center_pos.round(2)}")
        if m["latent_spawned"]:
            print(f"  - [AWARENESS SPARK] Persistent Unexplained Error Triggered Latent Dimension Spawning!")
            print(f"    Active Latent Dimensions: {m['active_latent_dims']}")

    print("\n===========================================================================")
    print("   CLOSED-LOOP VALENCE DEMO COMPLETED SUCCESSFULLY                         ")
    print("===========================================================================")


if __name__ == "__main__":
    run_closed_loop_demo()
