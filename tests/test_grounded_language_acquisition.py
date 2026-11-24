r"""
Verification suite for ELYSIA Grounded Language Acquisition (Child-Like Naming):
Tests pre-linguistic discernment & objectification (`obj::*`) before words,
cross-situational symbol grounding (`unit::* --names--> obj::*`) rejecting
one-off/noisy non-grounded symbols, grounded world-object speech generation,
and persistent grounding through live screen observation + Korean user dialogue.
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


def test_prelinguistic_objectification_creates_unnamed_world_objects_before_words():
    """Discovers invariant `obj::*` phenomena without any words and reports naming deficit."""
    uni = ScaleRotorHierarchy()
    cool_still_spec = [0.05, 0.90, 0.50, 0.20]
    vis_node = uni.differentiate_visual(
        [0.10, 0.20, 0.85] * 16,
        motion_flux=0.01,
        contrast_std=0.10,
        warm_cool_ratio=0.35,
        spectrum=cool_still_spec,
    )
    objs = uni.objectify_world_phenomena(
        visual_node=vis_node,
        audio_node=None,
        action_kind="observe_attend",
        felt_spectrum=cool_still_spec,
        visual_change=0.0,
    )
    obj_ids = [o.node_id for o in objs]
    assert "obj::vis_still_cool" in obj_ids
    assert uni.count_grounded_names() == 0

    utt, meta = uni.express_from_grounded_objects(obj_ids)
    assert utt == ""
    assert meta["naming_deficit"] is True
    assert "obj::vis_still_cool" in meta["unnamed_objects"]


def test_cross_situational_invariance_acquires_true_naming_links_and_rejects_ungrounded_noise():
    """
    Words co-occurring consistently with a discerned world object acquire `names` links;
    one-off tokens ('일리아스') and non-specific fillers across unrelated objects ('그냥') are rejected.
    """
    uni = ScaleRotorHierarchy()
    cool_spec = [0.10, 0.85, 0.45, 0.55]

    # 3 lived scenes where still+cool visual phenomenon co-occurs with "조용" / "고요"
    for i, text in enumerate(
        (
            "그냥 조용하고 차분하다.",
            "일리아스 조용하다.",
            "화면이 조용하고 고요하다.",
        )
    ):
        v = uni.differentiate_visual([0.1, 0.2, 0.8] * 16, motion_flux=0.02, contrast_std=0.10, warm_cool_ratio=0.40, spectrum=cool_spec)
        u = uni.differentiate_text(text, spectrum=cool_spec)
        uni.integrate_lived_scene(f"cool_{i}", v, None, u, "observe_attend", cool_spec, visual_change=0.01)

    # Now expose filler word "그냥" across unrelated distinct objects so specificity drops < 0.45
    other_scenes = [
        ("warm_still", [0.85, 0.40, 0.15] * 16, 0.01, 1.80, "scroll_explore", 0.0, [0.4, 0.6, 0.7, 0.6]),
        ("warm_move", [0.90, 0.30, 0.10] * 16, 0.45, 1.90, "click_salient", 0.35, [0.8, 0.5, 0.2, 0.5]),
        ("cool_move", [0.10, 0.30, 0.80] * 16, 0.40, 0.45, "open_web_search", 0.30, [0.7, 0.5, 0.3, -0.4]),
    ]
    for sid, vec, mflux, wcr, act, vchg, sp in other_scenes:
        v = uni.differentiate_visual(vec, motion_flux=mflux, contrast_std=0.12, warm_cool_ratio=wcr, spectrum=sp)
        u = uni.differentiate_text(f"그냥 {act} 느낌이다.", spectrum=sp)
        uni.integrate_lived_scene(sid, v, None, u, act, sp, visual_change=vchg)

    cool_names = [nd.surface for nd, _w in uni.get_grounded_names_for_object("obj::vis_still_cool", min_weight=0.45)]
    assert any("조용" in n for n in cool_names)
    assert "일리아스" not in cool_names
    assert "그냥" not in cool_names


def test_grounded_expression_speaks_from_active_world_objects_not_empty_symbols():
    """
    Active objects invoke their grounded acquired names rather than ungrounded symbols
    like '일리아스' even if ungrounded utterances sit inside the hierarchy.
    """
    uni = ScaleRotorHierarchy()
    cool_spec = [0.10, 0.80, 0.40, 0.55]
    warm_spec = [0.60, 0.75, 0.35, 0.60]

    for i in range(2):
        vc = uni.differentiate_visual([0.1, 0.2, 0.8] * 16, 0.02, 0.10, 0.40, cool_spec)
        uc = uni.differentiate_text("차분하고 조용하다.", cool_spec)
        uni.integrate_lived_scene(f"c_{i}", vc, None, uc, "observe_attend", cool_spec)

        vw = uni.differentiate_visual([0.9, 0.4, 0.2] * 16, 0.02, 0.10, 1.80, warm_spec)
        uw = uni.differentiate_text("온기가 따뜻하다.", warm_spec)
        uni.integrate_lived_scene(f"w_{i}", vw, None, uw, "observe_attend", warm_spec)

    # Inject an ungrounded one-off literary symbol that has NO object link
    uni.differentiate_text("일리아스 서사시다.", [0.3, 0.5, 0.4, 0.2])

    utt, meta = uni.express_from_grounded_objects(["obj::vis_still_cool"], target_spectrum=cool_spec)
    assert utt
    assert meta["grounded_by_objectification"] is True
    assert "일리아스" not in utt
    assert any(tok in utt for tok in ("조용", "차분"))


def test_embodied_mind_acquires_names_through_live_screen_plus_user_dialogue_and_persists(tmp_path: Path):
    """Screen perception + co-occurring user dialogue acquires persistent object-word links."""
    state_dir = tmp_path / "grounded_mind"
    cool_screen = np.tile(np.array([30, 50, 180], dtype=np.uint8), (32, 32, 1))
    retina = ScreenRetinaSensor(frame_provider=lambda: cool_screen)
    actuator = UniversalMotorActuator(dry_run=True)

    mind1 = ScaleRotorEmbodiedMind(state_dir)
    # 1. Tick before any words -> detects unnamed objects & asks honestly
    snap = retina.capture()
    t1 = mind1.perceive_and_act_tick(snap, 0.0, AgencyMode.OBSERVE_AND_CHAT, 15.0, actuator)
    assert "obj::vis_still_cool" in t1["active_objects"]

    # 2. User talks naturally about the same cool still screen across 2 encounters
    mind1.converse_with_user("지금 화면이 조용하고 차분하네", frame_snap=retina.capture())
    r2 = mind1.converse_with_user("맞아 화면이 정말 조용하고 차분하다", frame_snap=retina.capture())

    assert r2["grounded_names_count"] >= 1
    assert "obj::vis_still_cool" in r2["active_objects"]

    # 3. Reload cleanly from disk and verify grounded names persist & speak
    mind2 = ScaleRotorEmbodiedMind(state_dir)
    assert mind2.universe.count_grounded_names() == r2["grounded_names_count"]
    utt, meta = mind2.universe.express_from_grounded_objects(["obj::vis_still_cool"])
    assert utt
    assert meta["grounded_by_objectification"] is True
