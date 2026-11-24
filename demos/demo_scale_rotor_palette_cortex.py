r"""
ELYSIA Demo: Micro–Meso–Macro Scale-Rotor Palette Cortex vs Opaque FFT Codec
===========================================================================

Demonstrates the paradigm shift from chopping frames into flat arrays / opaque
FFT grids to representing causal reality as continuous inspectable Palette &
Trajectory Rotor Knobs across Micro (RGB ratio), Meso (spatial motion), and
Macro (persistent multi-media curriculum & cross-world axis redistribution).
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import elysia_loader

elysia_loader.install()

from core.orchestration.connected_media_causal_observer import (
    encode_audio_to_scale_rotor,
    encode_video_to_scale_rotor,
)
from core.topology.scale_rotor_palette_cortex import (
    MacroScaleRotorCortex,
    RotorWave1D,
    _load_rgb_tensor,
)


def _section(title: str) -> None:
    print("\n" + "=" * 92)
    print(f"  {title}")
    print("=" * 92)


def run_demo() -> None:
    print("=" * 92)
    print("  ELYSIA — MICRO-MESO-MACRO SCALE-ROTOR PALETTE CORTEX (BEFORE vs AFTER)")
    print("=" * 92)

    curriculum = [
        ("visual", "real_nature_mp4", Path(r"c:\Elysia\demos\real_world_assets\real_external_nature_video.mp4")),
        ("visual", "game_traj_gif", Path(r"c:\Elysia\docs\assets\causal_game_trajectory_playback.gif")),
        ("visual", "zoom_flight_gif", Path(r"c:\Elysia\docs\assets\elysia_3d_perspective_zoom_flight.gif")),
        ("visual", "cinema_gif", Path(r"c:\Elysia\docs\assets\spacetime_structural_cinema.gif")),
        (
            "acoustic",
            "real_human_speech",
            Path(
                r"c:\Archive\03_Traces_and_Deprecated\_Data_Archive\data_archive\S1_Body\L1_Foundation\external\CosyVoice\asset\cross_lingual_prompt.wav"
            ),
        ),
        ("acoustic", "drama_symphony", Path(r"c:\Elysia\data\elysia_drama_symphony.wav")),
    ]

    # 1. Micro RGB Ratio Knob Wave Convergence
    _section("1. 미시(Micro) 색채 비율 노브 파형 수렴: (15,10,15) -> (10,15,20) -> (10,25,20)")
    dur = 1.0
    t_pts = np.linspace(0.0, dur, 6, endpoint=False)
    r_seq = np.array([15.0, 10.0, 10.0, 15.0, 20.0, 15.0])
    g_seq = np.array([10.0, 15.0, 25.0, 20.0, 15.0, 10.0])
    b_seq = np.array([15.0, 20.0, 20.0, 15.0, 10.0, 15.0])
    rw = RotorWave1D.fit(r_seq, dur, max_modes=3)
    gw = RotorWave1D.fit(g_seq, dur, max_modes=3)
    bw = RotorWave1D.fit(b_seq, dur, max_modes=3)
    r_rec, g_rec, b_rec = rw.evaluate(t_pts), gw.evaluate(t_pts), bw.evaluate(t_pts)
    mae = float(np.mean(np.abs(r_rec - r_seq) + np.abs(g_rec - g_seq) + np.abs(b_rec - b_seq)) / 3.0)
    print(f"  원본 6틱 RGB 파형 -> 3모드 RotorWave1D 재수렴 (MAE={mae:.4f}):")
    for k in range(6):
        print(
            f"    t={t_pts[k]:.2f}s: 원본=({r_seq[k]:4.1f},{g_seq[k]:4.1f},{b_seq[k]:4.1f}) -> "
            f"노브수렴=({r_rec[k]:4.1f},{g_rec[k]:4.1f},{b_rec[k]:4.1f})"
        )

    # 2. Persistent Continuous Ingestion of 6 Real Workspace Media Files
    _section("2. 단일 영속 피질(MacroScaleRotorCortex)에 실물 미디어 6종 연속 유입·축적")
    cortex = MacroScaleRotorCortex()
    cur_res = cortex.absorb_workspace_media_curriculum(curriculum)
    print(
        f"  연속 유입 완료: 시각 세계={cur_res['total_visual_worlds']}종, 음향 세계={cur_res['total_acoustic_worlds']}종 "
        f"(총 소요={cur_res['total_elapsed_ms']:.1f}ms)"
    )
    for wid, vf in cortex.visual_worlds.items():
        print(
            f"    [시각] {wid:18s}: 레이어={len(vf.layers)}층, 파라미터={vf.parameter_count:4d}개, "
            f"압축비={vf.compression_ratio:7.1f}x, 추출={vf.extract_ms:.1f}ms"
        )
    for aid, af in cortex.acoustic_worlds.items():
        print(
            f"    [음향] {aid:18s}: 파라미터={af.parameter_count:4d}개, "
            f"압축비={af.compression_ratio:7.1f}x, 휴지비={af.pause_fraction:.2f}, 추출={af.extract_ms:.1f}ms"
        )

    # 3. Independent Knob Redistribution & Cross-World 3-Way Axis Direction
    _section("3. 독립 노브 재분배(온기/템포) & 이종 3세계 교차 축 재분배 연출")
    vf_nature = cortex.visual_worlds["real_nature_mp4"]
    t_grid = np.arange(36, dtype=np.float64) / 12.0
    t0 = time.perf_counter()
    fr_base = vf_nature.render_frames(t_grid, height=64, width=96)
    fr_warm = vf_nature.render_frames(t_grid, height=64, width=96, rgb_gain=(1.35, 1.0, 0.70))
    knob_ms = (time.perf_counter() - t0) * 1000.0
    print(
        f"  [노브 조작 'real_nature_mp4'] R/B 비 변화: 기본={np.mean(fr_base[...,0])/(np.mean(fr_base[...,2])+1):.3f} -> "
        f"따뜻한 온기 노브={np.mean(fr_warm[...,0])/(np.mean(fr_warm[...,2])+1):.3f} (2회 렌더링={knob_ms:.1f}ms)"
    )

    directed = cortex.direct_cross_world_composition(
        composition_id="nature_palette+zoom_motion+human_speech",
        palette_world_id="real_nature_mp4",
        motion_world_id="zoom_flight_gif",
        cadence_world_id="real_human_speech",
        n_frames=36,
        fps=12.0,
        height=64,
        width=96,
    )
    print(f"  [교차 연출] {directed.summary_ko}")

    # 4. Empirical Before (Opaque FFT Codec) vs After (Palette-Rotor Cortex) Comparison
    _section("4. Before (불투명 FFT 계수 격자) vs After (미시-중시-거시 팔레트 로터) 실측 비교")
    for wid, path in [("real_nature_mp4", curriculum[0][2]), ("game_traj_gif", curriculum[1][2])]:
        raw_frames, fps, _ = _load_rgb_tensor(path, max_frames=36, resize_wh=(96, 64))
        times = np.arange(len(raw_frames), dtype=np.float64) / fps

        # Before: Opaque 2D-FFT Spatiotemporal Grid
        tb0 = time.perf_counter()
        fft_spec = encode_video_to_scale_rotor(raw_frames, fps=fps, kh_max=14, kw_max=14, k_temporal=10, max_knots=3)
        rec_fft = fft_spec.recreate_frames_at(times)
        fft_ms = (time.perf_counter() - tb0) * 1000.0
        fft_mse = float(np.mean((raw_frames.astype(np.float64) / 255.0 - rec_fft.astype(np.float64) / 255.0) ** 2))

        # After: Inspectable Palette-Rotor Field
        pal_field = cortex.visual_worlds[wid]
        ta0 = time.perf_counter()
        rec_pal = pal_field.render_frames(times, height=64, width=96)
        pal_ms = (time.perf_counter() - ta0) * 1000.0
        pal_mse = float(np.mean((raw_frames.astype(np.float64) / 255.0 - rec_pal.astype(np.float64) / 255.0) ** 2))

        print(f"  [{wid}]")
        print(
            f"    Before (FFT 격자)   : 파라미터={fft_spec.parameter_count:6d}개, 압축={fft_spec.compression_ratio:6.1f}x, "
            f"인코드+렌더={fft_ms:5.1f}ms, 픽셀MSE={fft_mse:.5f} | 노브 직접조작=불가(불투명 복소격자)"
        )
        print(
            f"    After  (팔레트 로터): 파라미터={pal_field.parameter_count:6d}개, 압축={pal_field.compression_ratio:6.1f}x, "
            f"렌더={pal_ms:5.1f}ms, 픽셀MSE={pal_mse:.5f} | 노브 직접조작=가능(RGB·위치·템포·이종교차)"
        )

    # Also compare acoustic FFT bin storage vs AcousticCadenceRotorField
    from scipy.io import wavfile

    sr, data = wavfile.read(str(curriculum[4][2]))
    pcm = np.asarray(data, dtype=np.float64)
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    pcm = pcm / (np.max(np.abs(pcm)) + 1e-12)
    tb0 = time.perf_counter()
    aud_fft = encode_audio_to_scale_rotor(pcm, sr, k_bins=16)
    _ = aud_fft.recreate_pcm()
    aud_fft_ms = (time.perf_counter() - tb0) * 1000.0

    af_speech = cortex.acoustic_worlds["real_human_speech"]
    ta0 = time.perf_counter()
    _ = af_speech.synthesize_pcm(sample_rate=sr)
    aud_pal_ms = (time.perf_counter() - ta0) * 1000.0
    print("  [real_human_speech (13.75s 실제 사람 음성)]")
    print(
        f"    Before (STFT Top-K) : 파라미터={aud_fft.parameter_count:6d}개, 압축={aud_fft.compression_ratio:6.1f}x, "
        f"인코드+합성={aud_fft_ms:5.1f}ms | 발성/속도/피치 노브 분리=불가"
    )
    print(
        f"    After  (음향 로터)  : 파라미터={af_speech.parameter_count:6d}개, 압축={af_speech.compression_ratio:6.1f}x, "
        f"합성={aud_pal_ms:5.1f}ms | 에너지·피치(F0)·유성음 노브=독립 조작 가능"
    )

    print("\n" + "=" * 92)
    print("  ELYSIA SCALE-ROTOR PALETTE CORTEX DEMO COMPLETED CLEANLY (EXIT 0)")
    print("=" * 92)


if __name__ == "__main__":
    run_demo()
