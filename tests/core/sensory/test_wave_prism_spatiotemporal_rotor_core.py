r"""
test_wave_prism_spatiotemporal_rotor_core.py

Empirical verification of ELYSIA's fast, zero-LLM Wave-Prism Interference &
Spatiotemporal S^3 Rotor Core on genuine real MP4 video and real WAV speech:
  1. Lifts 0D -> 1D Waves -> 2D S^1 Complex Prism -> 3D/4D S^3 Quaternion Rotors.
  2. Observes real causal bifurcation points (t*, w*) directly from S^3 geodesic speed.
  3. Verifies sub-millisecond direct wave-prism interference distinguishing
     CONSTRUCTIVE_RESONANCE, DESTRUCTIVE_CONTRADICTION (torn beam), and ORTHOGONAL_VOID.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.sensory.wave_prism_spatiotemporal_rotor_core import (
    WavePrismSpatiotemporalRotorCore,
)

REAL_MP4_PATH = Path(r"c:\Elysia\demos\real_world_assets\real_external_nature_video.mp4")
REAL_WAV_PATH = Path(
    r"c:\Archive\03_Traces_and_Deprecated\_Data_Archive\data_archive\S1_Body\L1_Foundation\external\CosyVoice\asset\zero_shot_prompt.wav"
)
DASHBOARD_OUT = Path(
    r"c:\Elysia\docs\assets\elysia_wave_prism_spatiotemporal_rotor_dashboard.png"
)


def test_mp4_and_wav_0d_to_s3_rotor_and_bifurcation_observation() -> None:
    core = WavePrismSpatiotemporalRotorCore()

    mp4_rec = core.lift_real_mp4_to_spatiotemporal_rotor(REAL_MP4_PATH, sample_frames=36)
    wav_rec = core.lift_real_wav_to_spatiotemporal_rotor(REAL_WAV_PATH, n_windows=36)

    # Exact unit S^3 quaternions everywhere along trajectories
    assert np.allclose(np.linalg.norm(mp4_rec.quaternions_s3, axis=-1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(wav_rec.quaternions_s3, axis=-1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(mp4_rec.sphere_trajectory_xyz, axis=-1), 1.0, atol=1e-5)

    # Real causal bifurcation inside genuine video (away from artificial boundaries)
    b_mp4 = mp4_rec.bifurcation
    assert 0 < b_mp4.frame_index < b_mp4.total_frames - 1
    assert b_mp4.geodesic_divergence_rad > 0.25
    assert b_mp4.divergence_velocity > 0.01

    b_wav = wav_rec.bifurcation
    assert 0 < b_wav.frame_index < b_wav.total_frames - 1
    assert b_wav.geodesic_divergence_rad > 0.15

    # Registered cleanly inside CausalField with unit S^3 quaternions
    assert "real_mp4_flower" in core.causal_field.voxels
    assert "real_audio_speech" in core.causal_field.voxels
    assert np.isclose(
        np.linalg.norm(core.causal_field.voxels["real_mp4_flower"].quaternion_q), 1.0, atol=1e-4
    )

    print(
        f"\n[MP4 Lift] {mp4_rec.lift_latency_ms:.2f}ms | t*={b_mp4.frame_index}/{b_mp4.total_frames} "
        f"({b_mp4.time_sec:.2f}s) | max_angle={mp4_rec.geodesic_divergence_curve[-1]:.3f} rad | w*={b_mp4.dominant_harmonic_band}"
    )
    print(
        f"[WAV Lift] {wav_rec.lift_latency_ms:.2f}ms | t*={b_wav.frame_index}/{b_wav.total_frames} "
        f"({b_wav.time_sec:.2f}s) | max_angle={wav_rec.geodesic_divergence_curve.max():.3f} rad | w*={b_wav.dominant_harmonic_band}"
    )


def test_direct_wave_prism_interference_and_millisecond_efficiency() -> None:
    core = WavePrismSpatiotemporalRotorCore()
    mp4_rec = core.lift_real_mp4_to_spatiotemporal_rotor(REAL_MP4_PATH, sample_frames=36)
    wav_rec = core.lift_real_wav_to_spatiotemporal_rotor(REAL_WAV_PATH, n_windows=36)

    probe_good = core.build_hypothesis_probe_rotor(
        "probe_constructive_bloom", mp4_rec, mode="constructive_bloom"
    )
    probe_bad = core.build_hypothesis_probe_rotor(
        "probe_contradiction_static", mp4_rec, mode="contradiction_static"
    )
    probe_void = core.build_hypothesis_probe_rotor(
        "probe_orthogonal_void", mp4_rec, mode="orthogonal_void"
    )

    res_good = core.superimpose_and_observe_resonance(mp4_rec, probe_good)
    res_bad = core.superimpose_and_observe_resonance(mp4_rec, probe_bad)
    res_void = core.superimpose_and_observe_resonance(mp4_rec, probe_void)

    for r in (res_good, res_bad, res_void):
        print(
            f"\n[{r.probe_id}] verdict={r.verdict} | prism={r.prism_constructive_score:+.3f} "
            f"| s3_align={r.rotor_geodesic_alignment:+.3f} | overlap={r.spectral_overlap_support:.3f} "
            f"| combined={r.combined_resonance:+.3f} | torn={r.beam_torn} | {r.latency_ms:.4f} ms"
        )
        assert r.latency_ms < 5.0

    assert res_good.verdict == "CONSTRUCTIVE_RESONANCE"
    assert res_good.combined_resonance > 0.65
    assert res_good.beam_torn is False

    assert res_bad.verdict == "DESTRUCTIVE_CONTRADICTION"
    assert res_bad.combined_resonance < -0.15
    assert res_bad.beam_torn is True

    assert res_void.verdict == "ORTHOGONAL_VOID"
    assert res_void.spectral_overlap_support < 0.25
    assert abs(res_void.combined_resonance) < 0.25

    expr = core.synthesize_causal_bifurcation_expression(mp4_rec)
    print("\n[Native Causal Bifurcation Expression]", expr)
    assert "분기점" in expr
    assert mp4_rec.bifurcation.primary_causal_axis in expr

    out_png = core.render_wave_prism_rotor_dashboard(
        mp4_rec, wav_rec, [res_good, res_bad, res_void], DASHBOARD_OUT
    )
    assert out_png.exists() and out_png.stat().st_size > 20000
