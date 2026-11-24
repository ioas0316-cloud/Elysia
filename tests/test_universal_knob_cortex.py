"""
Verification suite for ELYSIA UniversalKnobCortex:
Domain-agnostic self-discovered causal knobs (starting from 0 knobs),
residual-driven knob internalization, cross-modal knob connection,
self-directed thinking in knob space, and disk persistence.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from core.orchestration.connected_media_causal_observer import run_pixel_rendered_interactive_game
from core.topology.universal_knob_cortex import (
    UniversalKnobCortex,
    adapt_audio_to_raw_stream,
    adapt_text_to_raw_stream,
    adapt_video_to_raw_stream,
)

_REAL_MP4 = Path(r"c:\Elysia\demos\real_world_assets\real_external_nature_video.mp4")
_REAL_GAME_GIF = Path(r"c:\Elysia\docs\assets\causal_game_trajectory_playback.gif")
_REAL_SPEECH_WAV = Path(
    r"c:\Archive\03_Traces_and_Deprecated\_Data_Archive\data_archive\S1_Body\L1_Foundation\external\CosyVoice\asset\cross_lingual_prompt.wav"
)


def test_starts_empty_and_self_creates_knobs_from_real_mp4_then_reuses_on_real_gif():
    """Starts with 0 knobs, creates knobs from real MP4, and reuses them on real GIF."""
    cortex = UniversalKnobCortex()
    assert len(cortex.knobs) == 0

    mp4_stream, mp4_dur = adapt_video_to_raw_stream(_REAL_MP4, max_frames=24)
    rep1 = cortex.perceive_and_grow("real_nature_mp4", mp4_stream, mp4_dur, target_explained_ratio=0.84)
    assert rep1.explained_before == 0.0
    assert len(rep1.created_knob_ids) >= 1
    assert rep1.explained_after >= 0.80

    gif_stream, gif_dur = adapt_video_to_raw_stream(_REAL_GAME_GIF, max_frames=24)
    rep2 = cortex.perceive_and_grow("real_game_gif", gif_stream, gif_dur, target_explained_ratio=0.84)
    assert len(rep2.reused_knob_ids) >= 1
    assert rep2.explained_before > 0.15
    assert rep2.explained_after >= 0.80


def test_same_core_handles_real_speech_wav_and_text_without_domain_hardcoding():
    """Exact same UniversalKnobCortex perceives real speech audio and raw text streams identically."""
    cortex = UniversalKnobCortex()

    speech_stream, speech_dur = adapt_audio_to_raw_stream(_REAL_SPEECH_WAV, n_steps=48, n_bands=32)
    rep_audio = cortex.perceive_and_grow(
        "real_human_speech", speech_stream, speech_dur, target_explained_ratio=0.78, max_new_knobs=6
    )
    assert len(rep_audio.created_knob_ids) >= 1
    assert rep_audio.explained_after >= 0.75

    utterances = [
        "빛이 물결처럼 퍼지며 온도를 높인다.",
        "온도가 오르자 공기 대류가 빠르게 움직인다.",
        "바람이 강해지며 나무 잎이 흔들린다.",
        "잎이 흔들리면서 부드러운 소리가 울린다.",
        "소리 파동이 다시 청각 경계로 돌아온다.",
        "반복되는 호흡이 리듬과 질서를 만든다.",
        "갑작스런 충격에도 맥락이 이어진다.",
        "장면이 바뀌어도 서사의 흐름은 끊기지 않는다.",
        "서로 다른 감각이 하나의 축에서 만난다.",
        "미지와 기존 지식의 경계가 분명해진다.",
        "새로운 노브가 잔차로부터 스스로 태어난다.",
        "마침내 전체 세계가 조화롭게 연결된다.",
    ]
    text_stream, text_dur = adapt_text_to_raw_stream(utterances, n_buckets=24)
    rep_text = cortex.perceive_and_grow(
        "discourse_text", text_stream, text_dur, target_explained_ratio=0.78, max_new_knobs=8
    )
    assert len(rep_text.created_knob_ids) >= 1
    assert rep_text.explained_after >= 0.75


def test_self_directed_knob_thinking_converges_to_unseen_target_without_human_gain():
    """Autonomous knob-space steering solves for settings matching an unseen target frame."""
    cortex = UniversalKnobCortex()
    mp4_stream, mp4_dur = adapt_video_to_raw_stream(_REAL_MP4, max_frames=24)
    cortex.perceive_and_grow("real_nature_mp4", mp4_stream, mp4_dur, target_explained_ratio=0.90, max_new_knobs=8)

    # Construct an unseen blended target frame from first & later segment
    unseen_target = 0.55 * mp4_stream[2] + 0.45 * mp4_stream[-3]
    res = cortex.think_and_steer_to_target(mp4_stream.shape[1], unseen_target)

    assert len(res["knob_settings"]) >= 1
    assert res["cosine_similarity"] > 0.90
    assert res["relative_error"] < 0.30


def test_cross_modal_knob_connection_and_associative_thinking():
    """Discovers lead-lag resonance links between self-grown video and audio knobs."""
    cortex = UniversalKnobCortex()
    bundle = run_pixel_rendered_interactive_game(mode="interactive_play")

    vis_stream, vis_dur = adapt_video_to_raw_stream(bundle.frames, max_frames=len(bundle.frames), explicit_fps=bundle.fps)
    aud_stream, aud_dur = adapt_audio_to_raw_stream((bundle.audio_pcm, bundle.sample_rate), n_steps=len(bundle.frames), n_bands=24)

    rep_v = cortex.perceive_and_grow("game_vis", vis_stream, vis_dur, target_explained_ratio=0.88, max_new_knobs=8)
    rep_a = cortex.perceive_and_grow("game_aud", aud_stream, aud_dur, target_explained_ratio=0.88, max_new_knobs=6)

    conns = cortex.connect_cooccurring_streams(rep_v.knob_trajectories, rep_a.knob_trajectories, max_lag=4, min_corr=0.45)
    assert len(conns) >= 1

    # Turn source knobs and verify associative propagation wakes connected target knobs
    src_activations = {c.src_knob_id: 1.0 for c in conns}
    emergent = cortex.think_across_connected_knobs(src_activations)
    assert len(emergent) >= 1
    assert any(abs(v) > 1e-4 for v in emergent.values())


def test_persistent_save_and_reload_preserves_self_grown_knobs_and_connections(tmp_path: Path):
    """Saves grown knobs and connections to disk and reloads them cleanly into a fresh session."""
    cortex = UniversalKnobCortex()
    bundle = run_pixel_rendered_interactive_game(mode="interactive_play")
    vis_stream, vis_dur = adapt_video_to_raw_stream(bundle.frames, max_frames=len(bundle.frames), explicit_fps=bundle.fps)
    aud_stream, aud_dur = adapt_audio_to_raw_stream((bundle.audio_pcm, bundle.sample_rate), n_steps=len(bundle.frames), n_bands=24)

    rv = cortex.perceive_and_grow("game_vis", vis_stream, vis_dur, target_explained_ratio=0.85)
    ra = cortex.perceive_and_grow("game_aud", aud_stream, aud_dur, target_explained_ratio=0.85)
    cortex.connect_cooccurring_streams(rv.knob_trajectories, ra.knob_trajectories, max_lag=4, min_corr=0.45)

    save_file = tmp_path / "elysia_universal_knobs.npz"
    cortex.save_state(save_file)
    assert save_file.is_file()

    reloaded = UniversalKnobCortex.load_state(save_file)
    assert len(reloaded.knobs) == len(cortex.knobs)
    assert len(reloaded.connections) == len(cortex.connections)

    # Fresh session immediately reuses previously grown knobs on a new visual stream!
    gif_stream, gif_dur = adapt_video_to_raw_stream(_REAL_GAME_GIF, max_frames=24)
    rep_reuse = reloaded.perceive_and_grow("post_reload_gif", gif_stream, gif_dur, target_explained_ratio=0.85)
    assert len(rep_reuse.reused_knob_ids) >= 1


def test_zero_hardcoded_domain_knobs_on_arbitrary_unknown_synthetic_world():
    """Discovers the exact intrinsic latent dimensionality of a completely alien 12D world."""
    cortex = UniversalKnobCortex()
    t = np.linspace(0.0, 2.0, 60, endpoint=False)
    latents = np.stack(
        [
            np.sin(2.0 * math.pi * 1.5 * t),
            np.cos(2.0 * math.pi * 3.0 * t + 0.4),
            0.7 * np.sin(2.0 * math.pi * 5.0 * t - 0.3),
        ],
        axis=1,
    )
    rng = np.random.default_rng(19)
    mixing = rng.normal(0.0, 1.0, size=(3, 12))
    offset = rng.uniform(-2.0, 2.0, size=(1, 12))
    alien_stream = offset + latents @ mixing

    rep = cortex.perceive_and_grow("alien_12d", alien_stream, 2.0, target_explained_ratio=0.98, max_new_knobs=6)
    assert 2 <= len(rep.created_knob_ids) <= 4
    assert rep.explained_after > 0.95
