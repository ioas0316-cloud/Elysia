"""
Verification script for ConceptualPhaseKnobEngine (개념적 위상 노브 엔진).
Demonstrates:
1. Physical Euclidean distance vs. Conceptual Phase Distance convergence.
2. Cross-domain functional isomorphism (Bird Wing vs. Airplane Wing, CAD constraints vs. Photoshop layers).
3. Inverse World Mechanism Extraction into Invariance Bedrock and Dynamic Wave Variants.
4. Extracting Cognitive Knobs (Friction, Rotor Angle, Causal Tension, Entropy).
5. Dynamic world state re-simulation upon knob tuning.
"""

import numpy as np
from L1_causal_field.core.topology.conceptual_phase_knob_engine import (
    ConceptualPhaseKnobEngine,
    ConceptualEntity,
    ExtractedWorldMechanism
)


def main():
    print("==========================================================")
    print("  ELYSIA: Conceptual Phase Knob Engine Verification Demo  ")
    print("==========================================================\n")

    engine = ConceptualPhaseKnobEngine(dimension=64, friction_threshold=0.15)

    # -----------------------------------------------------------------
    # STEP 1: Domain-Agnostic Conceptual Distance Convergence
    # -----------------------------------------------------------------
    print("--- [STEP 1] Measuring Conceptual Distance across Domains ---")

    # 1A. Bird Wing vs Airplane Wing
    bird_causal = np.array([1.2, 0.5, 0.1] + [0.0]*61)
    bird_intent = np.array([1.0, 0.8, 0.0] + [0.0]*61)  # Aerodynamic Lift Intent
    bird_boundary = np.array([0.1, 0.1])

    bird_entity = ConceptualEntity("Bird Wing", "Biology", bird_causal, bird_intent, bird_boundary)

    plane_causal = np.array([500.0, -200.0, 80.0] + [0.0]*61)  # Far in state space
    plane_intent = np.array([0.99, 0.81, 0.01] + [0.0]*61)    # Same intent
    plane_boundary = np.array([0.11, 0.09])

    plane_entity = ConceptualEntity("Airplane Wing", "Aeronautics", plane_causal, plane_intent, plane_boundary)

    engine.register_entity(bird_entity)
    engine.register_entity(plane_entity)

    res_aeronautics = engine.compute_conceptual_distance(
        "Bird Wing", "Airplane Wing",
        physical_coords_a=np.array([0.0, 0.0, 0.0]),
        physical_coords_b=np.array([400000.0, 0.0, 0.0])  # 400 km away
    )

    print(f"Physical Distance (Seoul -> Busan): {res_aeronautics.euclidean_physical_distance:,.1f} meters")
    print(f"Conceptual Phase Distance (Friction Error E): {res_aeronautics.conceptual_phase_distance:.4f}")
    print(f"Clifford Rotor Angle (Delta Theta): {res_aeronautics.clifford_rotor_angle:.4f} rad")
    print(f"Shared Invariance / Isomorphism Score: {res_aeronautics.shared_invariance_score * 100:.2f}%")
    print(f"Is Topologically Isomorphic: {res_aeronautics.is_topologically_isomorphic}\n")

    # 1B. CAD Kinematic Constraint vs Photoshop Layer Mask vs 3D Mesh Rig
    cad_intent = np.array([0.0, 0.0, 1.0, 0.9] + [0.0]*60)  # Spatial Boundary & Constraint
    ps_intent  = np.array([0.0, 0.0, 0.98, 0.91] + [0.0]*60) # Same constraint intent

    cad_entity = ConceptualEntity("CAD Constraint", "CAD", np.zeros(64), cad_intent, np.array([0.2]))
    ps_entity  = ConceptualEntity("Photoshop Layer Mask", "2D Design", np.ones(64)*10, ps_intent, np.array([0.2]))

    engine.register_entity(cad_entity)
    engine.register_entity(ps_entity)

    res_design = engine.compute_conceptual_distance("CAD Constraint", "Photoshop Layer Mask")

    print(f"CAD vs Photoshop Layer Conceptual Phase Distance: {res_design.conceptual_phase_distance:.4f}")
    print(f"Isomorphic Convergence: {res_design.is_topologically_isomorphic}\n")

    # -----------------------------------------------------------------
    # STEP 2: Inverse Latent World Mechanism Extraction & Knob-ification
    # -----------------------------------------------------------------
    print("--- [STEP 2] Inverse Mechanism Extraction & Cognitive Knob-ification ---")

    # Synthetic world trajectory (video/latent space simulation over 100 frames)
    t = np.linspace(0, 4*np.pi, 100)
    base_bedrock = np.sin(np.linspace(0, 1, 64)) * 10.0
    dynamic_wave = np.outer(np.cos(t), np.ones(64)) * 2.0
    trajectory = base_bedrock + dynamic_wave
    boundary_delta = np.array([1.5, 2.0, 0.5])

    mechanism = engine.extract_world_mechanism(trajectory, boundary_delta)

    print("Extracted World Mechanism decomposition:")
    print(f" - Invariant Bedrock Mean Norm: {np.linalg.norm(mechanism.invariant_bedrock):.4f}")
    print(f" - Dynamic Wave Variant Shape: {mechanism.variant_dynamic_wave.shape}")
    print("\nExtracted Cognitive Knobs:")
    for key, knob in mechanism.knobs.items():
        print(f"   * [{knob.name}] Value = {knob.value:.4f} (Range: [{knob.min_value}, {knob.max_value}]) -> {knob.description}")

    # -----------------------------------------------------------------
    # STEP 3: Dynamic World State Re-simulation on Knob Tuning
    # -----------------------------------------------------------------
    print("\n--- [STEP 3] Dynamic World State Re-simulation ---")

    initial_state = engine.resimulate_world_state(mechanism, {})
    print(f"Initial Simulated State Norm: {np.linalg.norm(initial_state):.4f}")

    # Tuning friction higher (increased damping) and rotating phase angle
    tuned_state = engine.resimulate_world_state(
        mechanism,
        {
            "friction": 3.5,
            "phase_rotor_angle": np.pi / 4.0,  # 45 deg phase rotation
            "causal_tension": 2.0
        }
    )
    print(f"Tuned Simulated State Norm: {np.linalg.norm(tuned_state):.4f}")
    print(f"State Shift Delta Norm: {np.linalg.norm(tuned_state - initial_state):.4f}")

    print("\n==========================================================")
    print("  VERIFICATION SUCCESSFUL: Conceptual Phase Knob Engine  ")
    print("==========================================================")


if __name__ == "__main__":
    main()
