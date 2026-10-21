#!/usr/bin/env python3
"""
Demo: Isomorphic Phase Field Substrate & Unified Scale-Hierarchical Protocol
=============================================================================
Demonstrates:
1. Continuous Phasor Sea (Data & Memory defined purely as A * e^{i * phi}).
2. Wave Interference Arithmetic (Constructive addition expansion & Destructive cancellation).
3. Autonomous Modality & Topological Judgment Axis (Self-observes Code, Image, Text, 3D Geometry).
4. Lossless Isomorphic Scale Boundary Traversal (Delta S ≈ 0).
5. Non-Destructive Multi-Modal Observation Projections (P_text, P_code, P_image, P_geom).
6. Scale-Hierarchical Concept & Causality Integration.
"""

import numpy as np
from core.physics.isomorphic_phase_field import IsomorphicPhaseFieldSubstrate


def main():
    print("\n" + "=" * 80)
    print("  ELYSIA: ISOMORPHIC PHASE FIELD SUBSTRATE & UNIFIED PROTOCOL DEMO")
    print("=" * 80 + "\n")

    # 1. Initialize Continuous Phasor Substrate
    substrate_1 = IsomorphicPhaseFieldSubstrate(resolution=128, base_frequency=10.0)
    substrate_2 = IsomorphicPhaseFieldSubstrate(resolution=128, base_frequency=10.0)

    print("[Step 1] Continuous Phasor Sea Initialized (A * exp(i * phi))")
    print(f"  - Substrate 1 Resolution : {substrate_1.resolution} complex phasors")
    print(f"  - Initial Vacuum Field Mean Amplitude: {np.mean(np.abs(substrate_1.grid)):.4f}\n")

    # 2. Wave Interference Arithmetic
    print("[Step 2] Wave Interference Arithmetic Demonstration")
    t = np.linspace(0, 2 * np.pi, 128, endpoint=False)

    # 2.1 Constructive Wave Interference Addition
    phase_wave_a = np.exp(1j * t * 2.0)
    phase_wave_b = np.exp(1j * t * 2.0)  # In-phase aligned
    substrate_1.set_field_from_phasor(phase_wave_a)
    substrate_2.set_field_from_phasor(phase_wave_b)

    combined_field = substrate_1.wave_interference_add(substrate_2)
    power_before = np.sum(np.abs(substrate_1.grid)**2) + np.sum(np.abs(substrate_2.grid)**2)
    power_after = np.sum(np.abs(combined_field.grid)**2)

    print("  [2.1] Constructive Wave Interference (Addition):")
    print(f"        -> Combined Field Power : {power_after:.2f} (Expanded from {power_before:.2f})")
    print(f"        -> Scale Expansion Factor: {power_after / power_before:.2f}x\n")

    # 2.2 Destructive Wave Interference Subtraction
    opposite_wave = np.exp(1j * (t * 2.0 + np.pi))  # Out of phase
    substrate_2.set_field_from_phasor(opposite_wave)
    cancelled_field = substrate_1.wave_interference_add(substrate_2)
    cancelled_power = np.sum(np.abs(cancelled_field.grid)**2)

    print("  [2.2] Destructive Wave Interference (Subtraction / Cancellation):")
    print(f"        -> Cancelled Field Power: {cancelled_power:.6f} (Entropy Rectification & Phase Cancellation)\n")

    # 3. Autonomous Modality Judgment Axis
    print("[Step 3] Autonomous Modality Judgment Axis (No External Labels/Extensions)")

    # 3.1 Discrete Code & Logic Phasor
    code_phasor = np.exp(1j * np.where(t < np.pi, 0.0, np.pi * 0.9))
    substrate_1.set_field_from_phasor(code_phasor)
    sig_code = substrate_1.autonomous_modality_judgment()
    print(f"  [3.1] Discrete Step Phasor -> Judged Modality: '{sig_code.primary_modality}' (Confidence: {sig_code.confidence_score:.2f})")

    # 3.2 Continuous Image & Video Spatial Field
    spatial_phasor = np.exp(1j * np.sin(4 * t) * 2.5)
    substrate_1.set_field_from_phasor(spatial_phasor)
    sig_img = substrate_1.autonomous_modality_judgment()
    print(f"  [3.2] Spatial Field Phasor -> Judged Modality: '{sig_img.primary_modality}' (Confidence: {sig_img.confidence_score:.2f})\n")

    # 4. Isomorphic Scale Boundary Traversal
    print("[Step 4] Isomorphic Scale Boundary Layer Traversal (Lossless Delta S ≈ 0)")
    scale_result = substrate_1.traverse_isomorphic_scale_boundary(scale_factor=3.5)
    print(f"  - Target Scale Factor             : {scale_result.target_scale}")
    print(f"  - Reconstructed Field Fidelity    : {scale_result.reconstructed_fidelity * 100:.2f}%")
    print(f"  - Reversibility Entropy Delta (ΔS): {scale_result.reversibility_entropy_delta:.10f}")
    print(f"  - Isomorphic Equivalence Validated: {scale_result.isomorphic_equivalence}\n")

    # 5. Non-Destructive Multi-Modal Observation Projections
    print("[Step 5] Non-Destructive Multi-Modal Projections across Observation Axes")
    print(f"  [P_text] : {substrate_1.project_to_text()}")
    print("  [P_code] :")
    for line in substrate_1.project_to_code().split('\n')[:4]:
        print(f"             {line}")
    img_meta = substrate_1.project_to_image()
    print(f"  [P_image]: Spatial Grid {img_meta['spatial_grid_shape']}, Mean Luminance: {img_meta['mean_luminance']:.4f}")
    geom_meta = substrate_1.project_to_geometry()
    print(f"  [P_geom] : Vertices Count {geom_meta['spatial_vertices_count']}, Bounding Radius: {geom_meta['volumetric_bounding_radius']:.4f}\n")

    # 6. Scale-Hierarchical Concept Promotion
    print("[Step 6] Scale-Hierarchical Macro-Rotor Concept Promotion")
    macro_concept = substrate_1.promote_to_macro_rotor_concept("MacroConcept_CausalIsomorphism_1")
    print(f"  - Promoted Concept ID  : {macro_concept['concept_id']}")
    print(f"  - Macro Phase Anchor   : {macro_concept['macro_phase_anchor']:.4f} rad")
    print(f"  - Field Coherence Index: {macro_concept['coherence_index']:.4f}")
    print(f"  - Hierarchy Level      : {macro_concept['scale_level']}\n")

    print("=" * 80)
    print(" ISOMORPHIC PHASE FIELD SUBSTRATE & UNIFIED PROTOCOL DEMO COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
