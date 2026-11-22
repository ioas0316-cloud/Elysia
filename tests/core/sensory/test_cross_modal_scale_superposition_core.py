r"""
test_cross_modal_scale_superposition_core.py

Zero-mock empirical verification of ELYSIA's foundational
Cross-Modal Scale Superposition, Orchestration & Distillation Core:
  1. Unifies genuine MP4 optical physics (`OcularScaleOrgan`), `CLIP ViT-B/32`,
     and real human speech prosody (`.wav`) onto ONE shared 8-axis manifold in `CausalField`.
  2. Proves measured coral-red petal & yellow center hues + radial blooming expansion
     with high constructive overlap (zero hardcoded fake labels).
  3. Proves top-down Self-Reobservation tears the `ConnectivityBeam` automatically when
     speech contradicts dynamic/expansion invariants, without keyword blacklists.
  4. Proves `ElysiaLiveOrganism` halts ungrounded hallucinations via `EpistemicVoid` and
     self-corrects any contradictory vocal draft.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np

from core.sensory.cross_modal_scale_superposition_core import (
    STRUCTURAL_SCALE_AXES,
    CrossModalScaleSuperpositionCore,
)
from core.sensory.elysia_live_organism_runtime import (
    REAL_MP4_VIDEO_PATH,
    REAL_SPEECH_WAV_PATH,
    ElysiaLiveOrganism,
)


def _load_real_mp4_frames(sample_count: int = 20) -> List[np.ndarray]:
    assert REAL_MP4_VIDEO_PATH.exists(), f"Missing MP4: {REAL_MP4_VIDEO_PATH}"
    cap = cv2.VideoCapture(str(REAL_MP4_VIDEO_PATH))
    assert cap.isOpened()
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = set(np.linspace(0, max(0, total - 1), sample_count).astype(int).tolist())
    frames: List[np.ndarray] = []
    i = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        if i in idxs:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    return frames


def test_real_multimodal_superposition_and_distillation() -> None:
    core = CrossModalScaleSuperpositionCore()
    frames = _load_real_mp4_frames(20)

    oc_proj = core.absorb_ocular_video(frames, node_id="real_mp4_flower")
    clip_proj = core.absorb_clip_video_cognitive_structure(frames, node_id="real_mp4_flower")
    assert REAL_SPEECH_WAV_PATH.exists(), f"Missing WAV: {REAL_SPEECH_WAV_PATH}"
    aud_proj = core.absorb_audio_prosodic_structure(
        REAL_SPEECH_WAV_PATH, node_id="real_audio_speech"
    )

    flower_node = core.orchestrate_and_distill("real_mp4_flower", [oc_proj, clip_proj])
    audio_node = core.orchestrate_and_distill("real_audio_speech", [aud_proj])

    print("\n[Ocular vector]", np.round(oc_proj.structural_vector, 3))
    print("[CLIP   vector]", np.round(clip_proj.structural_vector, 3))
    print("[Audio  vector]", np.round(aud_proj.structural_vector, 3))
    print(
        f"[Distilled overlap={flower_node.constructive_overlap_score:.4f}, mass={flower_node.semantic_mass:.3f}]",
        flower_node.factual_summary,
    )

    ax = {name: i for i, name in enumerate(STRUCTURAL_SCALE_AXES)}
    # Real petal opponent color MUST be coral-red (warm_red_chroma > 0) + yellow core (yellow_blue_chroma > 0)
    assert flower_node.distilled_vector[ax["warm_red_chroma"]] > 0.25
    assert flower_node.distilled_vector[ax["yellow_blue_chroma"]] > 0.35
    assert flower_node.distilled_vector[ax["radial_expansion"]] > 0.25
    assert flower_node.distilled_vector[ax["dynamic_flux"]] > 0.25
    assert flower_node.constructive_overlap_score > 0.50

    # CLIP must rank blooming red flower higher than motionless rock & kimchi stew
    spec = flower_node.clip_concept_spectrum
    assert spec["blooming_red_flower"] > spec["motionless_grey_rock"]
    assert spec["blooming_red_flower"] > spec["korean_kimchi_stew"]

    # CausalField topology checks
    assert "real_mp4_flower::ocular" in core.causal_field.voxels
    assert "real_mp4_flower::clip_vision" in core.causal_field.voxels
    assert "real_mp4_flower::distilled" in core.causal_field.voxels
    assert "real_audio_speech::audio_prosody" in core.causal_field.voxels
    assert "real_audio_speech::distilled" in core.causal_field.voxels
    assert audio_node.semantic_mass > 0.5
    assert len(core.causal_field.beams) >= 3


def test_self_reobservation_tears_beam_on_contradiction_without_keywords() -> None:
    core = CrossModalScaleSuperpositionCore()
    frames = _load_real_mp4_frames(16)
    oc_proj = core.absorb_ocular_video(frames, node_id="real_mp4_flower")
    clip_proj = core.absorb_clip_video_cognitive_structure(frames, node_id="real_mp4_flower")
    core.orchestrate_and_distill("real_mp4_flower", [oc_proj, clip_proj])

    contradictions = [
        "영상을 보니 꽃이 가만히 있다는 느낌이 들었어요.",
        "꽃이 아무 변화 없이 가만히 정지해 있고 움직이지 않는다.",
    ]
    truthful_statements = [
        "배경은 가만히 고정되어 있고, 중앙의 붉은 꽃잎은 바깥으로 활짝 피어나며 움직였습니다.",
        "붉은빛 꽃잎이 중심에서 바깥으로 활짝 피어나며 노란 중심을 드러내고, 채도가 점점 선명해졌습니다.",
    ]

    bad_verdicts = [core.reobserve_own_utterance(s, "real_mp4_flower") for s in contradictions]
    good_verdicts = [
        core.reobserve_own_utterance(s, "real_mp4_flower") for s in truthful_statements
    ]

    for v in bad_verdicts:
        print(
            f"\n[Contradiction] tension={v.max_beam_tension:.3f}, torn={v.torn_axes}, align={v.superposition_alignment:.3f}"
        )
        assert v.beam_torn is True
        assert v.max_beam_tension >= 0.55

    for v in good_verdicts:
        print(
            f"\n[Truthful] tension={v.max_beam_tension:.3f}, torn={v.torn_axes}, align={v.superposition_alignment:.3f}"
        )
        assert v.beam_torn is False
        assert v.max_beam_tension < 0.55

    max_good_tension = max(v.max_beam_tension for v in good_verdicts)
    min_bad_tension = min(v.max_beam_tension for v in bad_verdicts)
    assert min_bad_tension > max_good_tension


def test_live_organism_epistemic_void_and_grounded_self_reobservation(tmp_path: Path) -> None:
    state_file = tmp_path / "live_state.json"
    org = ElysiaLiveOrganism(state_path=state_file, load_existing_memory=False)

    vid_mem, dash_path = org.observe_real_mp4_video_stream()
    assert "붉은빛" in vid_mem.ocular_causal_summary
    assert "노란 꽃잎" not in vid_mem.ocular_causal_summary
    assert vid_mem.constructive_overlap_score > 0.50
    assert Path(dash_path).exists() and Path(dash_path).stat().st_size > 15000

    # 1. Ungrounded domain -> Epistemic Void gate halts before Qwen hallucination
    ep_void = org.converse_live("김치찌개 맛있게 끓이는 방법 알려줘")
    print(f"\n[Void Turn] {ep_void.generation_latency_sec:.3f}s -> {ep_void.final_spoken_response}")
    assert ep_void.epistemic_void is True
    assert ep_void.generation_latency_sec < 1.0
    assert "반죽" not in ep_void.final_spoken_response

    # 2. Grounded MP4 video question -> Self-reobservation prevents contradiction and evasive non-answers
    ep_vid = org.converse_live("방금 본 실제 영상에서 꽃이 어떻게 변했어?")
    print(
        f"\n[Video Turn] ({ep_vid.generation_latency_sec:.2f}s, corrected={ep_vid.self_corrected}, align={ep_vid.reobservation_alignment:.3f}, tension={ep_vid.max_beam_tension:.3f})"
    )
    print("  raw  :", ep_vid.raw_llm_response)
    print("  final:", ep_vid.final_spoken_response)
    assert ep_vid.epistemic_void is False
    assert ep_vid.causal_guard_passed is True
    assert ep_vid.reobservation_alignment >= 0.42
    assert ep_vid.max_beam_tension < 0.55

    # 3. Grounded archive / boundary question
    ep_phil = org.converse_live("경계가 왜 단순한 벽이 아니라 나를 비추는 거울인지 네 생각은 어때?")
    print(
        f"\n[Philosophy Turn] ({ep_phil.generation_latency_sec:.2f}s, mode={ep_phil.sovereign_intent_mode}, support={ep_phil.epistemic_support:.3f})"
    )
    print("  final:", ep_phil.final_spoken_response)
    assert ep_phil.epistemic_void is False
    assert ep_phil.sovereign_intent_mode == "causal_philosophy"
    assert ep_phil.causal_guard_passed is True
    assert len(ep_phil.final_spoken_response.strip()) >= 15

    # 4. Cold-reload persistence check
    saved = org.save_organism_state()
    assert saved.exists()
    org2 = ElysiaLiveOrganism(state_path=state_file, load_existing_memory=True)
    assert org2.awakening_count == 2
    assert len(org2.watched_videos) == 1
    assert len(org2.dialogue_history) == 3
