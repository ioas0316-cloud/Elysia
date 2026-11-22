r"""
test_spectral_scale_rotor_engine.py

Zero-LLM verification of ELYSIA's `(Amplitude, Period, Common Principle Axis)`
Spectral Rotor architecture:
  1. Exact Parseval energy conservation across `split_macro_to_micro` and vector phasor
     addition in `compose_micro_to_macro`.
  2. Real `.mp4` flower video & `.wav` human speech lifted directly onto `(4x4)` spectral
     arrays preserving Macro static background vs Meso/Micro blooming dynamics.
  3. Direct Korean sentence lifting & `< 2ms` phase interference discrimination across
     mixed-scale truth, static contradiction, color/collapse contradiction, and ungrounded
     void — with ZERO PyTorch, ZERO Transformers, and ZERO LLM inference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.sensory.spectral_scale_rotor_engine import (
    COMMON_PRINCIPLE_AXES,
    SpectralScaleRotorEngine,
)

REAL_MP4_PATH = Path(r"c:\Elysia\demos\real_world_assets\real_external_nature_video.mp4")
REAL_WAV_PATH = Path(
    r"c:\Archive\03_Traces_and_Deprecated\_Data_Archive\data_archive\S1_Body\L1_Foundation\external\CosyVoice\asset\zero_shot_prompt.wav"
)
SPECTRAL_DASHBOARD_OUT = Path(
    r"c:\Elysia\docs\assets\elysia_spectral_scale_rotor_dashboard.png"
)


def test_macro_micro_split_compose_energy_and_axis_invariance() -> None:
    eng = SpectralScaleRotorEngine()

    macro_amps = np.array([0.8, 0.6, 0.5, 0.4], dtype=np.float32)
    macro_phs = np.array([0.0, 0.0, 0.0, float(np.pi)], dtype=np.float32)
    micro = eng.split_macro_to_micro("macro_split", macro_amps, macro_phs)

    # Exact Parseval energy conservation along each Common Principle Axis
    ax_e = micro.axis_energies()
    for i, ax in enumerate(COMMON_PRINCIPLE_AXES):
        assert np.isclose(ax_e[ax], float(macro_amps[i] ** 2), atol=1e-5)

    # Opposing micro rotors along the same Axis x Period cancel destructively
    anti_phs = (macro_phs + np.pi).astype(np.float32)
    micro_anti = eng.split_macro_to_micro("macro_split_anti", macro_amps, anti_phs)
    composed_cancel = eng.compose_micro_to_macro("cancel_result", [micro, micro_anti])
    assert composed_cancel.total_energy() < 1e-6

    # Co-phase micro rotors reinforce constructively (2x amplitude -> 4x energy)
    composed_boost = eng.compose_micro_to_macro("boost_result", [micro, micro])
    assert np.isclose(composed_boost.total_energy(), 4.0 * micro.total_energy(), atol=1e-4)

    sum_amp, _ = eng.collapse_to_macro_axis_summary(micro)
    assert np.allclose(sum_amp, macro_amps, atol=1e-5)


def test_real_mp4_and_wav_spectral_array_lifting() -> None:
    eng = SpectralScaleRotorEngine()
    mp4_spec = eng.lift_mp4_to_spectral_rotor_array(REAL_MP4_PATH, sample_frames=36)
    wav_spec = eng.lift_wav_to_spectral_rotor_array(REAL_WAV_PATH, n_windows=36)

    assert np.isclose(mp4_spec.total_energy(), 1.0, atol=1e-4)
    assert np.isclose(wav_spec.total_energy(), 1.0, atol=1e-4)

    # Along flux_vs_anchor: Macro T=1.0 holds negative polarity (static BG ~ pi),
    # while Meso/Micro T<=0.5 hold positive polarity (dynamic blooming ~ 0)!
    signed_mp4 = mp4_spec.amplitudes * np.cos(mp4_spec.phases)
    print("\n[MP4 (Axis x Period) Signed Spectral Array]\n", np.round(signed_mp4, 3))
    print("[WAV (Axis x Period) Signed Spectral Array]\n", np.round(wav_spec.amplitudes * np.cos(wav_spec.phases), 3))

    assert signed_mp4[0, 0] < -0.10  # Macro background anchor (phase pi)
    assert float(np.sum(signed_mp4[0, 1:])) > 0.15  # Meso/micro dynamic flux (phase 0)
    assert float(np.sum(signed_mp4[1, 1:])) > 0.15  # Outward bloom expansion (phase 0)
    assert float(np.sum(signed_mp4[2, 1:])) > 0.15  # Coral-red petals (phase 0)
    assert float(np.sum(signed_mp4[3, 1:])) > 0.10  # Yellow core (phase 0)

    assert "real_mp4_flower" in eng.causal_field.voxels
    assert "real_audio_speech" in eng.causal_field.voxels


def test_direct_korean_language_to_wave_prism_superposition_zero_llm() -> None:
    eng = SpectralScaleRotorEngine()
    mp4_spec = eng.lift_mp4_to_spectral_rotor_array(REAL_MP4_PATH, sample_frames=36)
    wav_spec = eng.lift_wav_to_spectral_rotor_array(REAL_WAV_PATH, n_windows=36)

    cases = [
        (
            "Truth: BG anchor + red bloom",
            "배경은 가만히 고정되어 있고, 중앙의 붉은 꽃잎은 바깥으로 활짝 피어났다.",
            "CONSTRUCTIVE_RESONANCE",
            False,
        ),
        (
            "False: static motionless flower",
            "꽃이 아무 변화 없이 가만히 정지해 있다.",
            "DESTRUCTIVE_CONTRADICTION",
            True,
        ),
        (
            "False: cool blue wilted bud",
            "파란 차가운 꽃잎이 시들어 오므라들었다.",
            "DESTRUCTIVE_CONTRADICTION",
            True,
        ),
        (
            "Void: ungrounded kimchi stew",
            "김치찌개 맛있게 끓이는 방법 알려줘",
            "ORTHOGONAL_VOID",
            False,
        ),
    ]

    recorded_verdicts = []
    for short_label, text, expected_verdict, expected_torn in cases:
        probe = eng.lift_korean_sentence_to_spectral_rotor_array(text, f"probe_{short_label}")
        verdict = eng.superimpose_spectral_arrays(mp4_spec, probe)
        recorded_verdicts.append((short_label, verdict))

        print(
            f"\n[{short_label}] verdict={verdict.verdict} | score={verdict.resonance_score:+.3f} "
            f"| support={verdict.overlap_support:.3f} | torn={verdict.torn_coordinates} "
            f"| time={verdict.latency_ms:.4f} ms"
        )
        assert verdict.latency_ms < 2.0
        assert verdict.verdict == expected_verdict
        assert verdict.beam_torn is expected_torn

        if expected_verdict == "CONSTRUCTIVE_RESONANCE":
            assert verdict.resonance_score > 0.65
        elif expected_verdict == "DESTRUCTIVE_CONTRADICTION":
            assert verdict.resonance_score < -0.25
        else:
            assert probe.total_energy() == 0.0

    decoded = eng.decode_spectral_array_to_korean(mp4_spec)
    print("\n[Zero-LLM Direct Spectral Readout]", decoded)
    assert "정적 배경" in decoded
    assert "붉은빛" in decoded
    assert "활짝 피어나" in decoded

    out_png = eng.render_spectral_scale_dashboard(
        mp4_spec, wav_spec, recorded_verdicts, SPECTRAL_DASHBOARD_OUT
    )
    assert out_png.exists() and out_png.stat().st_size > 20000
