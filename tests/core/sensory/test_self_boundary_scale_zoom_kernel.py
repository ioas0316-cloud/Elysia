from __future__ import annotations

from pathlib import Path

from core.sensory.self_boundary_scale_zoom_kernel import (
    DEFAULT_ZOOM_DASHBOARD_PATH,
    REAL_EXTERNAL_MP4_PATH,
    SelfBoundaryScaleZoomKernel,
    render_self_boundary_zoom_dashboard,
    run_code_execution_zoom_scenario,
    run_real_video_adaptive_zoom_scenario,
    run_tool_avatar_boundary_scenario,
)


def test_unified_self_boundary_scale_zoom_across_three_domains() -> None:
    """
    Runs ONE persistent `SelfBoundaryScaleZoomKernel` (zero per-domain branches,
    zero MSE regression) continuously across three disparate worlds:
      1. Embodied Tool Control & Cutscene Severance
      2. Real AST + SafeASTSandbox Python Runtime Fault Isolation & Healing
      3. Real External Camera MP4 Video Adaptive Resolution Filtering
    """
    kernel = SelfBoundaryScaleZoomKernel()

    d1 = run_tool_avatar_boundary_scenario(kernel)
    d2 = run_code_execution_zoom_scenario(kernel)
    d3 = run_real_video_adaptive_zoom_scenario(
        kernel, video_path=REAL_EXTERNAL_MP4_PATH, max_frames=75
    )

    print("\n=== [ELYSIA Unified Self-Boundary Scale Zoom Log] ===")
    d1_snaps = d1["snapshots"]
    active_end = d1_snaps[9]
    severed_end = d1_snaps[-1]
    print(
        f"[Domain 1: Tool/Avatar] Active self={active_end.extended_self_ids} "
        f"ext={active_end.external_world_ids} | "
        f"Severed self={severed_end.extended_self_ids} "
        f"zoom={active_end.zoom_scale:.2f} -> {severed_end.zoom_scale:.2f} ({severed_end.zoom_band})"
    )

    d2_snaps = d2["snapshots"]
    min_bug_zoom = min(s.zoom_scale for s in d2_snaps[3:7])
    end_heal_zoom = d2_snaps[-1].zoom_scale
    print(
        f"[Domain 2: AST Code Fault] Fault isolated -> {d2['isolated_function']}() "
        f"({int(d2['isolated_ast_nodes'])}/{d2['total_ast_nodes']} AST nodes) | "
        f"zoom: Macro={d2_snaps[2].zoom_scale:.2f} -> Micro={min_bug_zoom:.2f} -> Healed={end_heal_zoom:.2f}"
    )

    print(
        f"[Domain 3: Real MP4 Video] Frames={d3['frames_processed']} | "
        f"Mean Compute Saved={d3['mean_compute_saved'] * 100:.1f}% | "
        f"CausalField Voxels={len(kernel.causal_field.voxels)} Beams={len(kernel.causal_field.beams)}"
    )

    dash_path = render_self_boundary_zoom_dashboard(
        kernel=kernel,
        d1=d1,
        d2=d2,
        d3=d3,
        output_path=Path(DEFAULT_ZOOM_DASHBOARD_PATH),
    )
    assert dash_path.exists() and dash_path.stat().st_size > 10_000

    # --- Domain 1 Assertions ---
    assert "avatar_tool" in active_end.extended_self_ids
    assert "wall_env" in active_end.external_world_ids
    assert "npc_other" in active_end.external_world_ids
    assert "avatar_tool" not in severed_end.extended_self_ids
    assert any(s.boundary_event == "BOUNDARY_RETRACTED_FROM_TOOL" for s in d1_snaps[10:])
    assert severed_end.zoom_scale > 1.35

    # --- Domain 2 Assertions ---
    assert d2["isolated_function"] == "compute_ratio"
    assert min_bug_zoom < 0.45
    assert end_heal_zoom > 0.90

    # --- Domain 3 Assertions ---
    assert d3["frames_processed"] == 75
    assert d3["mean_compute_saved"] > 0.25

    # --- Persistent CausalField Coexistence ---
    assert kernel.self_core_id in kernel.causal_field.voxels
    for expected_vid in ("avatar_tool", "wall_env", "npc_other", "code_sandbox", "external_camera_stream"):
        assert expected_vid in kernel.causal_field.voxels
