r"""
Verification suite for ELYSIA Unified Fractal Scale-Rotor Hierarchy:
Tests top-down fractal differentiation (`Scale 4 -> 3 -> 2 -> 1 -> 0 Jamo`) inside
ONE unified graph, bottom-up cross-modal integration (`vision + audio + utterance +
action -> Scale 4 scene`), cross-modal spreading resonance with zero silos, Scale-0
Jamo-grounded novel Korean assembly, and single-file `scale_rotor_universe.json` persistence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.embodiment import (
    AgencyMode,
    ScaleRotorEmbodiedMind,
    ScaleRotorHierarchy,
    ScreenRetinaSensor,
    UniversalMotorActuator,
)


def test_fractal_differentiation_from_scene_to_hangul_jamo_in_one_graph():
    """Differentiates multimodal episodes from Scale 4 down to Scale 0 Hangul jamo inside ONE graph."""
    uni = ScaleRotorHierarchy()
    spec = [0.45, 0.72, 0.35, 0.50]

    u_node = uni.differentiate_text("한글이 조용히 흐른다.", spectrum=spec, situation="방 안", cause="글")
    v_node = uni.differentiate_visual([0.4] * 48, motion_flux=0.05, contrast_std=0.20, warm_cool_ratio=1.15, spectrum=spec)
    a_node = uni.differentiate_audio({"rms_energy": 0.12, "zero_cross_rate": 0.08, "duration_s": 1.2}, spectrum=spec)
    scene = uni.integrate_lived_scene("ep_1", v_node, a_node, u_node, "observe_attend", spec)

    scales = uni.scale_summary()
    for sc in (0, 1, 2, 3, 4):
        assert scales[sc] >= 1

    assert scene.scale == 4
    assert "syl::한" in uni.nodes
    assert uni.nodes["syl::한"].children == ["jamo::ㅎ", "jamo::ㅏ", "jamo::ㄴ"]
    assert "unit::한글" in uni.nodes
    assert "syl::한" in uni.nodes["unit::한글"].children
    assert u_node.node_id in scene.children
    assert v_node.node_id in scene.children
    assert a_node.node_id in scene.children


def test_cross_modal_integration_and_spreading_resonance_without_silos():
    """Visual cues spread 2 hops across shared Scale 4 parents to co-integrated language & actions."""
    uni = ScaleRotorHierarchy()

    warm_spec = [0.75, 0.65, 0.30, 0.85]
    v_warm = uni.differentiate_visual([0.9, 0.4, 0.2] * 16, motion_flux=0.04, contrast_std=0.22, warm_cool_ratio=1.80, spectrum=warm_spec)
    u_warm = uni.differentiate_text("노을이 따뜻하다.", spectrum=warm_spec)
    uni.integrate_lived_scene("warm_ep", v_warm, None, u_warm, "scroll_explore", warm_spec)

    cool_spec = [0.20, 0.30, 0.85, -0.60]
    v_cool = uni.differentiate_visual([0.1, 0.2, 0.8] * 16, motion_flux=0.40, contrast_std=0.10, warm_cool_ratio=0.40, spectrum=cool_spec)
    u_cool = uni.differentiate_text("폭풍이 차갑다.", spectrum=cool_spec)
    uni.integrate_lived_scene("cool_ep", v_cool, None, u_cool, "click_salient", cool_spec)

    # Query purely with visual cue `v_warm` (no text cue!): must resonate toward warm utterance units & scroll_explore
    unit_hits = uni.resonate([v_warm.node_id], target_spectrum=warm_spec, target_scale=2, node_prefix="unit::", top_k=6)
    hit_surfaces = [nd.surface for nd, _sc in unit_hits]
    assert "노을" in hit_surfaces or "따뜻하" in hit_surfaces

    act_ranked = uni.score_actions_unified([v_warm.node_id], target_spectrum=warm_spec, candidate_actions=["click_salient", "scroll_explore"])
    assert act_ranked[0][0] == "scroll_explore"
    assert act_ranked[0][1] > act_ranked[1][1]


def test_unified_hierarchy_composes_novel_korean_using_scale0_jamo_nodes():
    """Composes non-verbatim Korean from unified graph using Scale-0 jamo coda checks for particle harmony."""
    uni = ScaleRotorHierarchy()
    spec = [0.40, 0.70, 0.30, 0.60]
    sentences = [
        "마음이 고요해진다.",
        "바다가 넓게 열린다.",
        "바람이 조용히 불어온다.",
        "소리가 멀리 퍼진다.",
        "마음은 하늘을 바라본다.",
    ]
    nodes = [uni.differentiate_text(s, spectrum=spec) for s in sentences]

    assert uni._stem_has_coda_from_scale0(uni.nodes["unit::마음"]) is True
    assert uni._stem_has_coda_from_scale0(uni.nodes["unit::바다"]) is False

    composed, meta = uni.compose_utterance_from_hierarchy(
        cue_node_ids=[nodes[0].node_id, nodes[1].node_id],
        user_text="마음이 조용하네",
        target_spectrum=spec,
    )
    assert composed
    assert composed not in sentences
    assert meta["grounded_in_real_world"] is True
    assert meta["novel_vs_corpus"] is True
    assert len(meta["used_nodes"]) >= 2


def test_single_file_persistence_roundtrip_and_embodied_mind_integration(tmp_path: Path):
    """Unified hierarchy persists to `scale_rotor_universe.json` and reloads identically inside EmbodiedMind."""
    state_dir = tmp_path / "uni_embodied"
    retina = ScreenRetinaSensor(frame_provider=lambda: np.full((32, 32, 3), 140, dtype=np.uint8))
    actuator = UniversalMotorActuator(dry_run=True)

    mind1 = ScaleRotorEmbodiedMind(state_dir)
    tick_res = mind1.perceive_and_act_tick(
        frame_snap=retina.capture(),
        user_idle_s=1.0,
        agency_mode=AgencyMode.OBSERVE_AND_CHAT,
        idle_threshold_s=15.0,
        actuator=actuator,
    )
    chat_res = mind1.converse_with_user("바다가 잔잔하네")

    assert tick_res["universe_nodes"] > 0
    assert chat_res["universe_nodes"] > tick_res["universe_nodes"]
    assert all(chat_res["universe_scales"][sc] >= 1 for sc in (0, 1, 2, 3, 4))

    uni_path = state_dir / "scale_rotor_universe.json"
    assert uni_path.is_file()

    mind2 = ScaleRotorEmbodiedMind(state_dir)
    assert len(mind2.universe.nodes) == len(mind1.universe.nodes)
    assert len(mind2.universe.links) == len(mind1.universe.links)
    assert mind2.universe.scale_summary() == mind1.universe.scale_summary()
