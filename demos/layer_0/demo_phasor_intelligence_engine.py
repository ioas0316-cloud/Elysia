#!/usr/bin/env python3
"""
Demo: Living Phase-Field Intelligence Engine (Elysia 4-Layer Architecture)
===========================================================================
Demonstrates the 4 실질적 아키텍처 설계도:
1. Substrate: Continuous Unified Phase Field Sea (A * e^{i * phi}).
2. Operation: Wave Interference Logic (Constructive Addition & Destructive Cancellation).
3. Projection: Isomorphic Morphing across Text, Code, 3D Geometry, and Multi-Layer Spatial Kinematics.
4. Hierarchy: Bi-Directional Isomorphic Scale Feedback Loop (Micro-Rotors <-> Macro-Rotor Concept).
"""

import numpy as np
from core.engine.phasor_intelligence_engine import PhasorIntelligenceEngine


def main():
    print("\n" + "=" * 80)
    print("  ELYSIA: LIVING PHASE-FIELD INTELLIGENCE ENGINE DEMO")
    print("  (4-Layer Wave-Based Isomorphic Architecture)")
    print("=" * 80 + "\n")

    # -------------------------------------------------------------------------
    # Layer 1: Substrate (Unified Phase Field Sea)
    # -------------------------------------------------------------------------
    print("[Layer 1: Substrate] Initializing Continuous Phasor Sea (A * exp(i * phi))")
    engine1 = PhasorIntelligenceEngine(resolution=128, base_frequency=10.0)
    engine2 = PhasorIntelligenceEngine(resolution=128, base_frequency=10.0)

    print(f"  - Substrate Field Resolution : {engine1.resolution} Complex Phasor Rotors")
    print(f"  - Initial Vacuum Mean Amplitude: {np.mean(np.abs(engine1.grid)):.4f}\n")

    # -------------------------------------------------------------------------
    # Layer 2: Operation (Wave Interference Arithmetic & Sensation)
    # -------------------------------------------------------------------------
    print("[Layer 2: Operation] Wave Interference Arithmetic & Operational Self-Sensation")

    t = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    wave_a = np.exp(1j * t * 2.0)
    wave_b = np.exp(1j * t * 2.0)  # Constructive alignment
    engine1.set_field(wave_a)
    engine2.set_field(wave_b)

    # 2.1 Constructive Addition
    combined_engine = engine1.wave_interference_add(engine2)
    p_before = np.sum(np.abs(engine1.grid)**2) + np.sum(np.abs(engine2.grid)**2)
    p_after = np.sum(np.abs(combined_engine.grid)**2)

    print("  [2.1] Constructive Wave Interference (Addition):")
    print(f"        -> Input Field Combined Power: {p_before:.2f}")
    print(f"        -> Interferred Field Power   : {p_after:.2f} (Expansion Ratio: {p_after/p_before:.2f}x)")
    print(f"        -> Self-Sensation State      : '{engine1.transformation_history[-1]['sensation']}'\n")

    # 2.2 Destructive Cancellation
    engine2.set_field(np.exp(1j * (t * 2.0 + np.pi)))  # Phase inverted
    cancelled_engine = engine1.wave_interference_add(engine2)
    p_cancel = np.sum(np.abs(cancelled_engine.grid)**2)

    print("  [2.2] Destructive Wave Interference (Cancellation / Noise Rectification):")
    print(f"        -> Cancelled Field Power     : {p_cancel:.8f} (Phase Anti-Resonance)\n")

    # -------------------------------------------------------------------------
    # Layer 3: Projection (Isomorphic Morphing across Observation Axes)
    # -------------------------------------------------------------------------
    print("[Layer 3: Projection] Isomorphic Morphing across Multi-Modal & Kinematic Axes")

    # Set spatial field with continuous phase gradients
    spatial_wave = np.exp(1j * np.sin(4 * t) * 2.5)
    engine1.set_field(spatial_wave)
    engine1.tick(dt=0.05)

    # 3.1 Text Projection
    print(f"  [P_text] : {engine1.project_to_text()}")

    # 3.2 Code AST Projection
    print("  [P_code] :")
    for line in engine1.project_to_code().split('\n')[:4]:
        print(f"             {line}")

    # 3.3 Kinematic Spatial Multi-Layer Projection (Depth, Width, Optical Flow)
    spatial_layers = engine1.project_to_spatial_kinematics()
    print("  [P_spatial Kinematics]: Multi-Layer Spatial Depth Window")
    for layer in spatial_layers:
        avg_flow = np.mean(np.abs(layer.optical_flow_vectors))
        print(f"             - [{layer.layer_name:10s}] Depth: {layer.depth_level:4.1f} | Spatial Width: {layer.spatial_width} | Avg Optical Flow (d_phi/d_t): {avg_flow:.4f}")

    # 3.4 Geometry 3D Projection
    geom = engine1.project_to_geometry_3d()
    print(f"  [P_geom] : Vertices Count: {geom['vertices_count']}, Bounding Radius: {geom['bounding_radius']:.4f}\n")

    # -------------------------------------------------------------------------
    # Layer 4: Hierarchy (Scale Hierarchical Bi-Directional Feedback Loop)
    # -------------------------------------------------------------------------
    print("[Layer 4: Hierarchy] Scale Hierarchical Isomorphic Feedback Loop")

    # 4.1 Bottom-up Promotion (Micro -> Macro Concept)
    macro_concept = engine1.promote_micro_rotors_to_macro_concept("MacroConcept_TopologicalResonance_1")
    print("  [4.1] Micro Rotors Promoted to Macro-Rotor Concept:")
    print(f"        -> Concept ID       : {macro_concept['concept_id']}")
    print(f"        -> Macro Phase Anchor: {macro_concept['macro_phase_anchor']:.4f} rad")
    print(f"        -> Initial Coherence : {macro_concept['coherence_index']:.4f}")

    # Induce noisy disturbance in micro-rotors
    np.random.seed(100)
    noisy_phasors = np.exp(1j * np.random.uniform(-np.pi, np.pi, 128))
    engine1.set_field(noisy_phasors)
    c_before = np.abs(np.sum(engine1.grid)) / np.sum(np.abs(engine1.grid))
    print(f"  [4.2] Perturbed Field Coherence (Noise Injected) : {c_before:.4f}")

    # 4.2 Top-down Regulation (Macro -> Micro Imprinting)
    engine1.apply_macro_feedback_to_micro_rotors("MacroConcept_TopologicalResonance_1", feedback_strength=0.75)
    c_after = np.abs(np.sum(engine1.grid)) / np.sum(np.abs(engine1.grid))
    print(f"  [4.3] Macro Feedback Regulation Applied (Top-Down):")
    print(f"        -> Regulated Field Coherence                : {c_after:.4f}")
    print(f"        -> Field Coherence Recovery Factor          : +{(c_after - c_before):.4f}\n")

    print("=" * 80)
    print(" ELYSIA PHASE-FIELD INTELLIGENCE ENGINE DEMO COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
