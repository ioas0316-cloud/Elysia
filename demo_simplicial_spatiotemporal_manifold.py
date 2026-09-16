"""
Demo: Simplicial Complex Spatiotemporal Manifold
==================================================

This demonstration showcases the 0D-4D Simplicial Complex Spatiotemporal Manifold:
- 0D Atom Nodes (Hangul Phonemes, Sound Formants, CIELAB Colors, Physical Textures)
- 1D Causal Line Trajectories
- 2D Syllable Context Planes (Implicit SDF Boundary Domain)
- 3D Integrated Multi-Sensory Manifold (GEMM Projection & Phase-Lock Coherence)
- 4D Spacetime Phase Velocity Dial (Continuous Phase Shift vs. Structural Phase Transition)
- Persistent Homology Regularization (Betti topology conservation against phase collapse)
"""

import math
import numpy as np
import torch

from synaptic_architecture.simplicial_spatiotemporal_manifold import SimplicialManifoldPipeline


def run_demo():
    print("==========================================================================")
    print("  Elysia Continuous Causal Intelligence: Simplicial Spatiotemporal Manifold")
    print("==========================================================================")

    pipeline = SimplicialManifoldPipeline(d_model=64)

    # -------------------------------------------------------------------------
    # 1. Register 0D Atom Nodes & 1D Line Trajectories
    # -------------------------------------------------------------------------
    print("\n[Step 1] Registering 0D Atom Nodes (Hangul Phonemes & Multi-Sensory Signatures)...")

    # Hangul '감' (GAM): 초성 ㄱ (G), 중성 ㅏ (A), 종성 ㅁ (M)
    cho_g = pipeline.add_atom_node("ㄱ", "text_symbol", np.array([1.0, 0.5, 0.0] + [0.1] * 61), (0.9, 0.1, 0.0))
    jung_a = pipeline.add_atom_node("ㅏ", "text_symbol", np.array([0.0, 1.0, 0.5] + [0.1] * 61), (0.1, 0.9, 0.0))
    jong_m = pipeline.add_atom_node("ㅁ", "text_symbol", np.array([0.5, 0.0, 1.0] + [0.1] * 61), (0.1, 0.1, 0.8))

    # Sound Formants (Acoustic)
    snd_formant = pipeline.add_atom_node("Sound_F1_F2", "acoustic_sound", np.array([0.8, 0.7, 0.2] + [0.05] * 61))

    # CIELAB Color Space (Visual)
    vis_color = pipeline.add_atom_node("CIELAB_Red", "visual_color", np.array([0.95, 0.2, 0.1] + [0.05] * 61))

    # Physical Texture
    tex_rough = pipeline.add_atom_node("Texture_Granular", "physical_texture", np.array([0.3, 0.4, 0.9] + [0.05] * 61))

    print(f"  - 0D Nodes Registered: {list(pipeline.nodes_0d.keys())}")

    # -------------------------------------------------------------------------
    # 2. Form 2D Syllable Plane (Implicit SDF Boundary Domain)
    # -------------------------------------------------------------------------
    print("\n[Step 2] Constructing 2D Context Field (Hangul Syllable Plane '감')...")
    field_gam = pipeline.create_hangul_syllable_plane("GAM", "ㄱ", "ㅏ", "ㅁ")

    print(f"  - 2D Plane ID: {field_gam.field_id}")
    print(f"  - SDF Domain Radius: {field_gam.sdf_radius:.4f}")

    sample_pt_inside = field_gam.sdf_center
    sample_pt_outside = field_gam.sdf_center + 5.0

    print(f"  - SDF Check (Center Point): Inside? {field_gam.is_inside_sdf_boundary(sample_pt_inside)} (SDF = {field_gam.compute_sdf(sample_pt_inside).item():.4f})")
    print(f"  - SDF Check (Distorted Point): Inside? {field_gam.is_inside_sdf_boundary(sample_pt_outside)} (SDF = {field_gam.compute_sdf(sample_pt_outside).item():.4f})")

    # -------------------------------------------------------------------------
    # 3. 3D Multi-Sensory Manifold Phase-Locking & Persistent Homology
    # -------------------------------------------------------------------------
    print("\n[Step 3] Synthesizing 3D Multi-Sensory Manifold (Phase-Lock GEMM & Persistent Homology)...")
    integrated_manifold, coherence, homology_loss = pipeline.process_cross_modal_manifold(
        text_nodes=["ㄱ", "ㅏ", "ㅁ"],
        sound_nodes=["Sound_F1_F2"],
        visual_nodes=["CIELAB_Red"],
        texture_nodes=["Texture_Granular"]
    )

    print(f"  - Integrated 3D Manifold Shape: {integrated_manifold.shape}")
    print(f"  - Cross-Modal Coherence Matrix Shape: {coherence.shape}")
    print("  - Coherence Values (Mean Inner Product Phase-Locking):")
    for r in range(coherence.shape[0]):
        row_str = " ".join([f"{coherence[r, c].item():.3f}" for c in range(coherence.shape[1])])
        print(f"    [{row_str}]")
    print(f"  - Persistent Homology Regularization Loss (Beta_0 Topology Invariant): {homology_loss.item():.6f}")

    # -------------------------------------------------------------------------
    # 4. 4D Spacetime Phase Velocity & Transition Dial Simulation
    # -------------------------------------------------------------------------
    print("\n[Step 4] Operating 4D Spacetime Phase Velocity Dial...")

    state = integrated_manifold

    # Case A: Slow Dial Rotation (|v| < 5.0) -> Continuous Phase Shift
    print("  (A) Slow Dial Rotation (dial_delta = 0.2 rad):")
    shifted_slow, info_slow = pipeline.rotate_spacetime_dial(state, dial_rotation_delta=0.2, dt=0.1)
    print(f"      Transition Type: {info_slow['transition_type']}")
    print(f"      Phase Velocity:  {info_slow['phase_velocity']:.2f} rad/s")
    print(f"      Energy Shift:    {info_slow['energy_momentum']:.4f}")

    # Case B: Rapid Dial Rotation (|v| >= 5.0) -> Structural Phase Transition
    print("\n  (B) Rapid Dial Rotation (dial_delta = 1.5 rad):")
    shifted_fast, info_fast = pipeline.rotate_spacetime_dial(state, dial_rotation_delta=1.5, dt=0.1)
    print(f"      Transition Type: {info_fast['transition_type']}")
    print(f"      Phase Velocity:  {info_fast['phase_velocity']:.2f} rad/s")
    print(f"      Energy Shift:    {info_fast['energy_momentum']:.4f}")

    print("\n==========================================================================")
    print("  Simplicial Complex Spatiotemporal Manifold Demo Completed Successfully!")
    print("==========================================================================")


if __name__ == "__main__":
    run_demo()
