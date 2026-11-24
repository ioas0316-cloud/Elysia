r"""
ELYSIA v3.2 Demo: Generative O(K) Scale-Rotor Connected Media Observer & Self-Improver
======================================================================================

Demonstrates TRUE generative causal discernment:
  1. Fast vectorized Fourier Phase-Shift & FFT Wiener-Khinchin observation of real
     disk MP4, real game GIF, real human speech WAV, and continuous symphony WAV.
  2. Closed-loop autonomous recreation & self-improvement (`Cycle 0 -> 1 -> 2`):
     Elysia recreates actual RGB video frames & PCM audio strictly from her
     Scale-Rotor Spectrum, re-observes her own recreation, harvests residuals,
     and strictly drives down visual & acoustic error while keeping high compression.
  3. Generative novel world creation: drives learned Causal Transfer Operators `H`
     with a never-seen control rhythm to synthesize a brand-new pixel video + PCM
     soundtrack in milliseconds and self-verifies causality.
  4. Vocabulary-free volitional vs reactive narrative discernment (tested on unknown
     alien vocabulary) + visual cut-bridging continuity + 20-organ self-audit.
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import elysia_loader

elysia_loader.install()

from core.orchestration.connected_media_causal_observer import (
    ConnectedMediaCausalObserver,
    build_audiovisual_narrative_scene,
    run_pixel_rendered_interactive_game,
)
from core.orchestration.elysia_unified_living_cycle import ElysiaUnifiedLivingCycle
from core.orchestration.ice_water_gas_rotor_runtime import IceWaterGasRotorRuntime
from core.physics.causal_field import CausalField


def _section(title: str) -> None:
    print("\n" + "=" * 92)
    print(f"  {title}")
    print("=" * 92)


def run_demo() -> None:
    print("=" * 92)
    print("  ELYSIA v3.2 — GENERATIVE O(K) SCALE-ROTOR RECREATION, SELF-IMPROVEMENT & CREATION")
    print("=" * 92)

    mp4_path = Path(r"c:\Elysia\demos\real_world_assets\real_external_nature_video.mp4")
    gif_path = Path(r"c:\Elysia\docs\assets\causal_game_trajectory_playback.gif")
    speech_path = Path(
        r"c:\Archive\03_Traces_and_Deprecated\_Data_Archive\data_archive\S1_Body\L1_Foundation\external\CosyVoice\asset\cross_lingual_prompt.wav"
    )
    symphony_path = Path(r"c:\Elysia\data\elysia_drama_symphony.wav")

    with tempfile.TemporaryDirectory(prefix="elysia_v32_media_") as tmp_dir:
        slab_path = Path(tmp_dir) / "elysia_v32.ice"
        with IceWaterGasRotorRuntime(slab_path=slab_path) as mem_rt:
            cycle = ElysiaUnifiedLivingCycle(memory_runtime=mem_rt)
            observer = ConnectedMediaCausalObserver(living_cycle=cycle)

            # 1. Fast O(K) real disk media perception
            _section("1. 실제 디스크 매체 O(K) 푸리에 위상속도 & FFT 음성 관측")
            rep_mp4 = observer.observe_episode("real_nature_mp4", video_source=mp4_path, max_frames=30)
            rep_gif = observer.observe_episode("real_game_gif", video_source=gif_path, max_frames=39)
            rep_speech = observer.observe_episode("real_human_speech", audio_source=speech_path)
            rep_music = observer.observe_episode("real_drama_symphony", audio_source=symphony_path)

            print(f"  [MP4]      {rep_mp4.summary_ko} ({rep_mp4.elapsed_ms:.1f}ms)")
            print(f"  [GIF]      {rep_gif.summary_ko} ({rep_gif.elapsed_ms:.1f}ms)")
            print(f"  [사람음성] {rep_speech.summary_ko} ({rep_speech.elapsed_ms:.1f}ms)")
            print(f"  [교향곡]   {rep_music.summary_ko} ({rep_music.elapsed_ms:.1f}ms)")

            # 2. Closed-loop autonomous recreation & self-improvement (Game + Real MP4/Speech!)
            _section("2. 닫힌 자기재창조 → 자기재관측 → 잔차수확 자기개선 (Cycle 0 → 1 → 2)")
            game_bundle = run_pixel_rendered_interactive_game(mode="interactive_play")
            imp_game = observer.learn_recreate_and_self_improve(
                episode_id="interactive_game_rotor",
                video_source=game_bundle.frames,
                video_fps=game_bundle.fps,
                audio_source=(game_bundle.audio_pcm, game_bundle.sample_rate),
                max_frames=len(game_bundle.frames),
                max_cycles=3,
            )
            print(f"  [게임 자기개선] {imp_game.summary_ko}")
            for c in imp_game.cycles:
                print(
                    f"      Cycle {c.cycle_index}: 시각MSE={c.visual_mse:.5f}, 운동흐름상관={c.motion_flow_correlation:+.3f}, "
                    f"압축={c.visual_compression_ratio:6.1f}x, 매듭={c.visual_knot_count} | "
                    f"음향MSE={c.audio_mse:.5f}, 포락선상관={c.audio_envelope_correlation:+.3f} ({c.elapsed_ms:.1f}ms)"
                )

            imp_real = observer.learn_recreate_and_self_improve(
                episode_id="real_mp4_and_human_speech",
                video_source=mp4_path,
                audio_source=speech_path,
                max_frames=24,
                max_cycles=3,
            )
            print(f"  [실제 MP4+사람음성 자기개선] {imp_real.summary_ko}")

            # 3. Self-agency, desync falsification & NOVEL WORLD CREATION from learned operators
            _section("3. 자기 행위주체성 분별 & 학습된 인과 전달 연산자(H)로 한 번도 본 적 없는 새 세계 창조")
            rep_game = observer.observe_bundle(game_bundle)
            rep_video = observer.observe_bundle(run_pixel_rendered_interactive_game(mode="passive_video_playback"))
            rep_desync = observer.observe_bundle(run_pixel_rendered_interactive_game(mode="desynchronized_audio"))
            print(f"  [상호작용 게임] 행위주체={rep_game.agency_mode.value}, 인과={rep_game.causality_mode.value} ({rep_game.elapsed_ms:.1f}ms)")
            print(f"  [녹화 영상재생] 행위주체={rep_video.agency_mode.value}, 인과={rep_video.causality_mode.value} ({rep_video.elapsed_ms:.1f}ms)")
            av_desync = [lk for lk in rep_desync.causal_links if lk.relation_kind == "audiovisual_physical_synchrony"]
            print(f"  [오디오 47% 이동 반사실 대조] 시청각 동기 링크={'유지' if av_desync else '붕괴(미검출)'}")

            novel_ctrl = np.zeros(40, dtype=np.float64)
            novel_ctrl[[3, 4, 13, 14, 24, 25, 33, 34]] = 1.0
            created = observer.create_novel_world_from_principles(rep_game, novel_ctrl)
            print(f"  [신규 세계 재창조] {created.summary_ko}")

            # 4. Vocabulary-free unknown-language causality + Cut-bridging + 20-organ audit
            _section("4. 단어 사전 없는 미지 언어 서사 분별 · 시각 컷 교량 · 20기관 자기성찰")
            rep_vol = observer.observe_bundle(build_audiovisual_narrative_scene("volitional", use_unknown_language=True))
            rep_rea = observer.observe_bundle(build_audiovisual_narrative_scene("reactive", use_unknown_language=True))
            print(f"  [미지언어 의지적 서사] {rep_vol.summary_ko}")
            for br in rep_vol.continuity_bridges:
                print(f"      -> [컷 교량] {br.summary_ko}")
            print(f"  [미지언어 반응적 서사] {rep_rea.summary_ko}")

            field = CausalField(dimensions=3)
            sync = observer.sync_to_causal_field(causal_field=field)
            field.step(dt=0.1)
            audit = cycle.introspect_own_architecture()
            print(
                f"  [4D CausalField & 20기관 감사] 복셀={len(field.voxels)}개, 빔={sync['beams_linked']}개, "
                f"Ice 크리스탈={mem_rt.ice.num_crystals}개, SHA-256={cycle.verify_autobiographical_integrity()}, "
                f"검증 기관={audit.total_organ_files}개 ({audit.total_organ_classes} 클래스, {audit.total_organ_lines} 라인)"
            )

    print("\n" + "=" * 92)
    print("  ELYSIA v3.2 GENERATIVE SCALE-ROTOR DEMO COMPLETED CLEANLY (EXIT 0)")
    print("=" * 92)


if __name__ == "__main__":
    run_demo()
