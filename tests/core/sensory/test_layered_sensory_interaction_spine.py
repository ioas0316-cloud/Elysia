from __future__ import annotations

from pathlib import Path

from core.sensory.layered_sensory_interaction_spine import (
    DEFAULT_LAYERED_SENSORY_DASHBOARD_PATH,
    EmbodiedPygameSensoryWorld,
    LayeredSensoryInteractionSpine,
    LiveScreenSensoryBridge,
    REAL_EXTERNAL_MP4_PATH,
    compute_human_likeness_audit,
    render_layered_sensory_dashboard,
)


def test_layered_sensory_hierarchy_embodied_and_real_video() -> None:
    """
    Validates ELYSIA's pure non-regression layered sensory spine across BOTH:
      1) Route B (`EmbodiedPygameSensoryWorld`): 4 visually identical red entities +
         unseen green transfer triangles reorganized strictly via bodily contact
         & re-observation into distinct inter-scale `ConnectivityBeam` topologies.
      2) Route A (`LiveScreenSensoryBridge`): real camera MP4 frame decomposition
         into `L_micro` / `L_meso` / `L_macro` coherence inside the same `CausalField`.
    """
    spine = LayeredSensoryInteractionSpine()

    # Route B: Headless real pygame rendering & contact exploration
    world = EmbodiedPygameSensoryWorld(headless=True)
    pygame_snapshot, inferred = world.run_embodied_exploration(spine, max_steps_per_entity=16)

    # Route A: Real external MP4 camera stream through the same 3-scale architecture
    video_summary = LiveScreenSensoryBridge.observe_real_video(
        spine, video_path=REAL_EXTERNAL_MP4_PATH, max_frames=90
    )

    audit = compute_human_likeness_audit(spine, inferred, video_summary)

    print("\n=== [ELYSIA Layered Sensory-Scale Interaction Verification] ===")
    print(
        f"Pairwise Cosine Similarity (Vision Only -> Re-Observed Topology): "
        f"{audit.pre_interaction_mean_similarity:.3f} -> {audit.post_interaction_mean_similarity:.3f} "
        f"(gain=+{audit.structural_differentiation_gain:.3f})"
    )
    for eid in ("ent_1", "ent_2", "ent_3", "ent_4", "ent_unseen_a", "ent_unseen_b"):
        c = spine.columns[eid]
        print(
            f"  {eid:14s} -> {inferred[eid]:26s} | "
            f"micro<->meso={c.beam_micro_meso:.3f} | meso<->macro={c.beam_meso_macro:.3f} | "
            f"triad(R/B/Y)=({c.chromatic_triad[0]:.2f}/{c.chromatic_triad[1]:.2f}/{c.chromatic_triad[2]:.2f}) | "
            f"emergent_boundary={c.boundary_voxel_id is not None}"
        )
    print(
        f"Real External Video ({REAL_EXTERNAL_MP4_PATH.name}): "
        f"frames={video_summary['frames_processed']} | L_micro={video_summary['mean_micro_activity']:.3f} | "
        f"L_meso={video_summary['mean_meso_activity']:.3f} | L_macro={video_summary['mean_macro_persistence']:.3f} | "
        f"Beam Coherence={audit.real_video_beam_coherence:.3f}"
    )

    dash_path = render_layered_sensory_dashboard(
        spine=spine,
        audit=audit,
        inferred_affordances=inferred,
        pygame_snapshot=pygame_snapshot,
        video_summary=video_summary,
        output_path=Path(DEFAULT_LAYERED_SENSORY_DASHBOARD_PATH),
    )
    assert dash_path.exists() and dash_path.stat().st_size > 10_000

    # --- Structural & Human-Likeness Assertions ---
    assert audit.pre_interaction_mean_similarity > 0.90
    assert audit.post_interaction_mean_similarity < 0.65
    assert audit.structural_differentiation_gain > 0.30

    assert audit.wall_resistance_locked is True
    assert audit.crate_flux_coupled is True
    assert audit.illusion_beam_torn is True
    assert audit.gate_emergent_boundary_spawned is True
    assert audit.unseen_transfer_correct is True
    assert audit.real_video_beam_coherence > 0.50

    assert "ent_4_L_boundary_transition" in spine.causal_field.voxels
    assert spine.columns["ent_3"].beam_micro_meso < 0.25
    assert spine.columns["ent_1"].beam_micro_meso > 0.75
