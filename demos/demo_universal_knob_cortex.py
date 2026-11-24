r"""
ELYSIA Demo: UniversalKnobCortex — Zero Hardcoded Knobs, Self-Growing & Connected Thinking
===========================================================================================

Starts with ZERO knobs (`{}`), perceives unknown synthetic spaces, real MP4 video,
real Game GIF (reusing internalized knobs from MP4!), real human speech WAV,
and interactive pixel gameplay (auto-linking video <-> audio knobs), steers
autonomously in knob space toward unseen targets, and persists grown knobs to disk.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import tempfile

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import elysia_loader

elysia_loader.install()

from core.orchestration.connected_media_causal_observer import run_pixel_rendered_interactive_game
from core.topology.universal_knob_cortex import (
    UniversalKnobCortex,
    adapt_audio_to_raw_stream,
    adapt_text_to_raw_stream,
    adapt_video_to_raw_stream,
)


def _section(title: str) -> None:
    print("\n" + "=" * 92)
    print(f"  {title}")
    print("=" * 92)


def run_demo() -> None:
    print("=" * 92)
    print("  ELYSIA — UNIVERSAL SELF-GROWING KNOB CORTEX (0 KNOBS -> SELF-GROWN & CONNECTED)")
    print("=" * 92)

    cortex = UniversalKnobCortex()
    print(f"  초기 내적화 노브 수: {len(cortex.knobs)}개 (사전 정의된 색·음높이·움직임 규칙 0개)")

    # 1. Alien 12D latent world (no prior domain knowledge)
    _section("1. 정체 불명 12차원 미지 세계 유입 -> 내재 노브 자가 발견")
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
    alien_stream = rng.uniform(-2.0, 2.0, size=(1, 12)) + latents @ rng.normal(0.0, 1.0, size=(3, 12))
    rep_alien = cortex.perceive_and_grow("alien_12d", alien_stream, 2.0, target_explained_ratio=0.98, max_new_knobs=6)
    print(
        f"  [미지 12D 세계] 설명률 {rep_alien.explained_before*100:.1f}% -> {rep_alien.explained_after*100:.1f}%, "
        f"자생 노브={len(rep_alien.created_knob_ids)}개, 소요={rep_alien.elapsed_ms:.2f}ms"
    )

    # 2. Real External MP4 -> Real Game GIF (Reuse + Residual Growth!)
    _section("2. 실제 자연 MP4 자생 노브 생성 -> 게임 궤적 GIF 기존 노브 재사용 + 미설명 잔차 확장")
    mp4_path = Path(r"c:\Elysia\demos\real_world_assets\real_external_nature_video.mp4")
    gif_path = Path(r"c:\Elysia\docs\assets\causal_game_trajectory_playback.gif")
    speech_path = Path(
        r"c:\Archive\03_Traces_and_Deprecated\_Data_Archive\data_archive\S1_Body\L1_Foundation\external\CosyVoice\asset\cross_lingual_prompt.wav"
    )

    mp4_s, mp4_d = adapt_video_to_raw_stream(mp4_path, max_frames=24)
    rep_mp4 = cortex.perceive_and_grow("real_nature_mp4", mp4_s, mp4_d, target_explained_ratio=0.84)
    print(
        f"  [실제 자연 MP4] 사전 설명률={rep_mp4.explained_before*100:.1f}% -> 사후={rep_mp4.explained_after*100:.1f}%, "
        f"재사용={len(rep_mp4.reused_knob_ids)}개, 신규 자생={len(rep_mp4.created_knob_ids)}개 ({rep_mp4.elapsed_ms:.1f}ms)"
    )

    gif_s, gif_d = adapt_video_to_raw_stream(gif_path, max_frames=24)
    rep_gif = cortex.perceive_and_grow("real_game_gif", gif_s, gif_d, target_explained_ratio=0.84)
    print(
        f"  [실제 게임 GIF] 기존 MP4 노브로 즉시 설명={rep_gif.explained_before*100:.1f}% -> 잔차 확장 후={rep_gif.explained_after*100:.1f}%, "
        f"재사용={len(rep_gif.reused_knob_ids)}개, 신규 추가={len(rep_gif.created_knob_ids)}개 ({rep_gif.elapsed_ms:.1f}ms)"
    )

    # 3. Same core on Real Human Speech WAV & Vocabulary-Free Text
    _section("3. 동일한 단일 코어로 실제 사람 음성 WAV & 단어사전 없는 텍스트 스트림 내적화")
    sp_s, sp_d = adapt_audio_to_raw_stream(speech_path, n_steps=48, n_bands=32)
    rep_sp = cortex.perceive_and_grow("real_human_speech", sp_s, sp_d, target_explained_ratio=0.78, max_new_knobs=6)
    print(
        f"  [실제 사람 음성] 설명률={rep_sp.explained_before*100:.1f}% -> {rep_sp.explained_after*100:.1f}%, "
        f"자생 노브={len(rep_sp.created_knob_ids)}개 ({rep_sp.elapsed_ms:.1f}ms)"
    )

    utterances = [
        "빛이 물결처럼 퍼지며 온도를 높인다.",
        "온도가 오르자 공기 대류가 빠르게 움직인다.",
        "바람이 강해지며 나무 잎이 흔들린다.",
        "잎이 흔들리면서 부드러운 소리가 울린다.",
        "소리 파동이 다시 청각 경계로 돌아온다.",
        "반복되는 호흡이 리듬과 질서를 만든다.",
        "갑작스런 충격에도 맥락이 이어진다.",
        "장면이 바뀌어도 서사의 흐름은 끊기지 않는다.",
    ]
    tx_s, tx_d = adapt_text_to_raw_stream(utterances, n_buckets=24)
    rep_tx = cortex.perceive_and_grow("discourse_text", tx_s, tx_d, target_explained_ratio=0.78, max_new_knobs=6)
    print(
        f"  [언어 흐름 스트림] 설명률={rep_tx.explained_before*100:.1f}% -> {rep_tx.explained_after*100:.1f}%, "
        f"자생 노브={len(rep_tx.created_knob_ids)}개 ({rep_tx.elapsed_ms:.1f}ms)"
    )

    # 4. Connected Audiovisual Gameplay -> Cross-Modal Knob Resonance & Associative Thinking
    _section("4. 상호작용 게임(영상↔오디오) 자생 노브 교차 연결 · 노브 공간 자기조향 · 영속 저장")
    bundle = run_pixel_rendered_interactive_game(mode="interactive_play")
    gv_s, gv_d = adapt_video_to_raw_stream(bundle.frames, max_frames=len(bundle.frames), explicit_fps=bundle.fps)
    ga_s, ga_d = adapt_audio_to_raw_stream((bundle.audio_pcm, bundle.sample_rate), n_steps=len(bundle.frames), n_bands=24)
    rep_gv = cortex.perceive_and_grow("game_vis", gv_s, gv_d, target_explained_ratio=0.88, max_new_knobs=8)
    rep_ga = cortex.perceive_and_grow("game_aud", ga_s, ga_d, target_explained_ratio=0.88, max_new_knobs=6)
    conns = cortex.connect_cooccurring_streams(rep_gv.knob_trajectories, rep_ga.knob_trajectories, max_lag=4, min_corr=0.45)
    print(
        f"  [게임 교차 노브 연결] 영상↔오디오 공명 연결={len(conns)}개 발견 "
        f"(최대 상관={max((abs(c.resonance_corr) for c in conns), default=0.0):.3f})"
    )
    if conns:
        top_c = max(conns, key=lambda c: abs(c.resonance_corr))
        print(
            f"      대표 연결: {top_c.src_knob_id} --(시차 {top_c.best_lag_steps}틱, r={top_c.resonance_corr:+.3f})--> {top_c.dst_knob_id}"
        )
        assoc = cortex.think_across_connected_knobs({top_c.src_knob_id: 1.0})
        print(f"      연상 사고 전파: {top_c.src_knob_id}=1.0 회전 -> 연결 노브 {len(assoc)}개 연쇄 활성")

    # 5. Autonomous Knob Steering to Unseen Target + Disk Persistence
    unseen_target = 0.55 * mp4_s[2] + 0.45 * mp4_s[-3]
    steer = cortex.think_and_steer_to_target(mp4_s.shape[1], unseen_target)
    print(
        f"  [노브 공간 자기 조향] 한 번도 보지 못한 혼합 목표 프레임 수렴: "
        f"코사인 유사도={steer['cosine_similarity']:.4f}, 상대 오차={steer['relative_error']:.4f}, "
        f"사용 노브={len(steer['knob_settings'])}개"
    )

    with tempfile.TemporaryDirectory(prefix="elysia_ukc_") as tmp_dir:
        save_p = Path(tmp_dir) / "universal_knobs.npz"
        cortex.save_state(save_p)
        reloaded = UniversalKnobCortex.load_state(save_p)
        print(
            f"  [디스크 영속 저장/복원] 파일 크기={save_p.stat().st_size:,d} bytes, "
            f"복원 노브={len(reloaded.knobs)}개, 교차 연결={len(reloaded.connections)}개"
        )

    print("\n" + "=" * 92)
    print("  ELYSIA UNIVERSAL KNOB CORTEX DEMO COMPLETED CLEANLY (EXIT 0)")
    print("=" * 92)


if __name__ == "__main__":
    run_demo()
