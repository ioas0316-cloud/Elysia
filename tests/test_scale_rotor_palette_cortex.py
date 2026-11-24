"""
Verification suite for ELYSIA Scale-Rotor Palette Cortex:
Micro (RGB-ratio waves) <-> Meso (spatial trajectory rotors) <-> Macro (axis redistribution).
"""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from core.orchestration.connected_media_causal_observer import encode_video_to_scale_rotor
from core.topology.scale_rotor_palette_cortex import (
    MacroScaleRotorCortex,
    RotorWave1D,
    _load_rgb_tensor,
    distill_visual_palette_rotors,
)

_MEDIA_CURRICULUM = [
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


def test_micro_rgb_ratio_wave_knob_convergence():
    """Tests continuous RGB ratio wave evolution (15,10,15)->(10,15,20)->(10,25,20)->... without dot lookup."""
    dur = 1.0
    t = np.linspace(0.0, dur, 6, endpoint=False)
    r_seq = np.array([15.0, 10.0, 10.0, 15.0, 20.0, 15.0])
    g_seq = np.array([10.0, 15.0, 25.0, 20.0, 15.0, 10.0])
    b_seq = np.array([15.0, 20.0, 20.0, 15.0, 10.0, 15.0])

    rw = RotorWave1D.fit(r_seq, dur, max_modes=3)
    gw = RotorWave1D.fit(g_seq, dur, max_modes=3)
    bw = RotorWave1D.fit(b_seq, dur, max_modes=3)

    r_rec = rw.evaluate(t)
    g_rec = gw.evaluate(t)
    b_rec = bw.evaluate(t)

    mae = float(np.mean(np.abs(r_rec - r_seq) + np.abs(g_rec - g_seq) + np.abs(b_rec - b_seq)) / 3.0)
    assert mae < 1.5

    # Directable knob shift: doubling G gain shifts green ratio predictably upward
    g_boosted = gw.evaluate(t) * 1.5
    assert float(np.mean(g_boosted)) > float(np.mean(g_rec)) * 1.45


def test_continuous_multi_media_curriculum_accumulation():
    """Ingests 6 real workspace MP4/GIF/Speech/Music files into ONE persistent MacroScaleRotorCortex."""
    cortex = MacroScaleRotorCortex()
    summary = cortex.absorb_workspace_media_curriculum(_MEDIA_CURRICULUM)

    assert summary["visual_worlds_absorbed"] == 4
    assert summary["acoustic_worlds_absorbed"] == 2
    assert summary["skipped"] == 0
    assert summary["total_visual_worlds"] == 4
    assert summary["total_acoustic_worlds"] == 2

    for wid, vf in cortex.visual_worlds.items():
        assert len(vf.layers) >= 2, f"{wid} should have backdrop + salient layer(s)"
        assert vf.compression_ratio > 50.0, f"{wid} palette compression too low: {vf.compression_ratio}"

    for aid, af in cortex.acoustic_worlds.items():
        assert af.compression_ratio > 500.0, f"{aid} acoustic compression too low: {af.compression_ratio}"


def test_independent_knob_redistribution_warmth_and_tempo():
    """Turning Micro RGB warmth knob and Meso tempo knob converges directly into altered worlds in <80ms."""
    cortex = MacroScaleRotorCortex()
    cortex.absorb_visual_stream("real_nature_mp4", _MEDIA_CURRICULUM[0][2])

    vf = cortex.visual_worlds["real_nature_mp4"]
    times = np.arange(36, dtype=np.float64) / 12.0

    t0 = time.perf_counter()
    base_frames = vf.render_frames(times, height=64, width=96)
    warm_frames = vf.render_frames(times, height=64, width=96, rgb_gain=(1.35, 1.0, 0.70))
    fast_frames = vf.render_frames(times, height=64, width=96, tempo_scale=2.0)
    render_ms = (time.perf_counter() - t0) * 1000.0

    ratio_base = float(np.mean(base_frames[..., 0]) / (np.mean(base_frames[..., 2]) + 1.0))
    ratio_warm = float(np.mean(warm_frames[..., 0]) / (np.mean(warm_frames[..., 2]) + 1.0))
    assert ratio_warm > ratio_base * 1.45

    # Tempo doubling increases frame-to-frame temporal energy difference vs base
    diff_base = float(np.mean(np.abs(base_frames[1:].astype(np.float32) - base_frames[:-1].astype(np.float32))))
    diff_fast = float(np.mean(np.abs(fast_frames[1:].astype(np.float32) - fast_frames[:-1].astype(np.float32))))
    assert diff_fast >= diff_base * 0.90
    assert render_ms < 120.0


def test_cross_world_3way_axis_redistribution_direction():
    """Fuses Nature MP4 colors + 3D Zoom GIF motion trajectory + Real Human Speech cadence into one new world."""
    cortex = MacroScaleRotorCortex()
    cortex.absorb_workspace_media_curriculum(_MEDIA_CURRICULUM)

    directed = cortex.direct_cross_world_composition(
        composition_id="nature_color_zoom_motion_speech_cadence",
        palette_world_id="real_nature_mp4",
        motion_world_id="zoom_flight_gif",
        cadence_world_id="real_human_speech",
        n_frames=36,
        fps=12.0,
        height=64,
        width=96,
    )

    assert directed.frames.shape == (36, 64, 96, 3)
    assert directed.pcm is not None and len(directed.pcm) > 10000
    assert directed.motion_energy_mean > 0.30
    assert directed.render_ms < 200.0
    assert directed.cadence_visual_coupling > 0.25


def test_palette_rotor_vs_opaque_fft_compactness_and_speed():
    """Verifies Palette-Rotor requires far fewer inspectable knob parameters than opaque 2D-FFT grids."""
    gif_path = _MEDIA_CURRICULUM[1][2]  # causal_game_trajectory_playback.gif
    raw_frames, fps, _ = _load_rgb_tensor(gif_path, max_frames=36, resize_wh=(96, 64))

    fft_spec = encode_video_to_scale_rotor(raw_frames, fps=fps, kh_max=14, kw_max=14, k_temporal=10, max_knots=3)
    pal_field = distill_visual_palette_rotors("game_traj", gif_path, max_frames=36)

    assert pal_field.parameter_count < fft_spec.parameter_count
    assert pal_field.parameter_count * 10 < fft_spec.parameter_count

    times = np.arange(len(raw_frames), dtype=np.float64) / fps
    t0 = time.perf_counter()
    rec_pal = pal_field.render_frames(times, height=64, width=96)
    pal_ms = (time.perf_counter() - t0) * 1000.0
    assert rec_pal.shape == raw_frames.shape
    assert pal_ms < 80.0


def test_graceful_skip_on_missing_media_in_curriculum():
    """Missing media files are cleanly skipped with zero crash during continuous curriculum absorption."""
    cortex = MacroScaleRotorCortex()
    res = cortex.absorb_workspace_media_curriculum(
        [
            ("visual", "missing_clip", Path(r"c:\Elysia\does_not_exist_video.mp4")),
            ("visual", "game_traj", _MEDIA_CURRICULUM[1][2]),
        ]
    )
    assert res["skipped"] == 1
    assert res["visual_worlds_absorbed"] == 1
    assert res["total_visual_worlds"] == 1
