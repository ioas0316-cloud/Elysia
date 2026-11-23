"""
test_spacetime_causal_game_video_engine.py

Blind spacetime causal discovery, pixel-only closed-loop visual-motor control,
and bottom-up real video autonomous agency inference tests (Zero VLM/LLM).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.sensory.spacetime_causal_game_video_engine import (
    ContinuousCausalArena,
    SpacetimeCausalDiscoveryEngine,
)

VIDEO_PATH = Path(r"c:\Elysia\demos\real_world_assets\real_external_nature_video.mp4")


def test_blind_spacetime_entity_discovery_and_visual_motor_control() -> None:
    arena = ContinuousCausalArena(width=240, height=180)
    engine = SpacetimeCausalDiscoveryEngine()

    game_rep = engine.discover_and_control_arena(arena, control_steps=22)

    print(
        f"\n[Arena] self_err={game_rep.self_localization_error_px:.2f}px | "
        f"ext_err={game_rep.external_agent_localization_error_px:.2f}px | "
        f"gain={game_rep.learned_motor_gain_yx} | "
        f"dist {game_rep.control_initial_dist_px:.1f} -> {game_rep.control_final_dist_px:.1f} "
        f"(-{game_rep.control_distance_reduction_ratio * 100:.1f}%)"
    )

    assert game_rep.self_localization_error_px <= 18.0
    assert game_rep.external_agent_localization_error_px <= 22.0
    assert game_rep.learned_motor_gain_yx[0] > 0.5
    assert game_rep.learned_motor_gain_yx[1] > 0.5
    assert game_rep.control_distance_reduction_ratio >= 0.40

    roles = {d.entity_role for d in game_rep.phase1_discoveries}
    assert {
        "SELF_BODY",
        "EXTERNAL_AUTONOMOUS_AGENT",
        "ILLUMINATION_SHADOW",
        "STATIC_ENVIRONMENT",
    }.issubset(roles)

    for d in game_rep.phase1_discoveries:
        assert d.voxel_id in engine.causal_field.voxels
        v = engine.causal_field.voxels[d.voxel_id]
        assert v.position.shape == (6,)
        assert v.tensor.shape == (6,)
    assert len(engine.causal_field.beams) >= 3


def test_real_external_video_bottom_up_agency_inference() -> None:
    if not VIDEO_PATH.exists():
        pytest.skip("Missing real external nature video")

    engine = SpacetimeCausalDiscoveryEngine()
    video_rep = engine.discover_autonomous_agent_in_real_video(
        VIDEO_PATH, max_frames=10, frame_stride=6, max_width=240
    )

    print(
        f"\n[Video] frames={video_rep.frames_processed} | "
        f"traj_len={len(video_rep.discovered_agent_trajectory_yx)} | "
        f"inferred_agency={video_rep.inferred_autonomous_agency_score:.3f}"
    )

    assert video_rep.frames_processed >= 8
    assert len(video_rep.discovered_agent_trajectory_yx) >= 6
    assert video_rep.inferred_autonomous_agency_score >= 0.45
    assert video_rep.voxel_id in engine.causal_field.voxels
