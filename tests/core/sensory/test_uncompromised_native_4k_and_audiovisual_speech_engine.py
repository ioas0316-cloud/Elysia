r"""
test_uncompromised_native_4k_and_audiovisual_speech_engine.py

Verifies:
1. Uncompromised Native 4K (`2160 x 3840 x 3` = 8,294,400 pixels) Original Array
   Decomposition, Exact Recomposition (`PSNR > 60 dB`), Sameness (Skeleton Invariance
   `> 0.99`) vs Difference (Macro Chromatic Shift), and 1:1 Native 4K Optical Pan
   without pre-downsampling or synthetic toy `sin/cos` rocks.
2. Real Human Speech WAV (`zero_shot_prompt.wav`, 24,000 Hz, 83,520 samples) +
   Lip Articulation + Facial Expression + Subtitle Co-Spectrum Language Acquisition
   and novel subtitle audio-visual re-creation.
"""

from __future__ import annotations

from pathlib import Path

from core.sensory.uncompromised_native_4k_and_audiovisual_speech_engine import (
    UncompromisedNative4KAndAudioVisualEngine,
)


def test_uncompromised_native_4k_original_array_sameness_and_difference() -> None:
    engine = UncompromisedNative4KAndAudioVisualEngine()
    rep, paths = engine.run_native_4k_original_array_verification()

    # 1. Exact native 4K array dimensions (NO pre-downsampling!)
    assert rep.native_shape_hwc == (2160, 3840, 3)
    assert rep.total_rgb_elements == 2160 * 3840 * 3

    # 2. Exact Ground-Truth Recomposition across all 8.29M pixels
    assert rep.recomposition_max_abs_error_f32 < 1e-5
    assert rep.recomposition_psnr_db >= 90.0
    assert rep.skeleton_gradient_sameness_cosine > 0.9999

    # 3. Discerning Sameness (invariant granite/pine/river skeleton) vs Difference (macro chromatic shift)
    assert rep.sunset_skeleton_sameness_cosine > 0.985
    assert rep.sunset_chromatic_difference_norm > 0.05
    assert rep.moonlight_skeleton_sameness_cosine > 0.985
    assert rep.moonlight_chromatic_difference_norm > 0.08

    # 4. Full 4K array tamper localization SNR
    assert rep.native_4k_tamper_detection_snr > 50.0

    assert Path(paths["output_4k_inspection_path"]).exists()
    assert Path(paths["output_4k_crop_video_path"]).exists()


def test_real_human_speech_lip_expression_subtitle_language_learning() -> None:
    engine = UncompromisedNative4KAndAudioVisualEngine()
    rep, paths = engine.run_real_audiovisual_drama_language_learning()

    # 1. Untruncated real human speech WAV processed (24,000 Hz, 83,520 samples)
    assert rep.real_wav_sample_rate == 24000
    assert rep.real_wav_total_samples == 83520
    assert rep.real_wav_duration_sec > 3.4

    # 2. Discovered cross-modal correlations between Voice Spectrum, Lip Aperture, Facial Expression, and Subtitle
    assert rep.lip_aperture_to_f1_vowel_corr > 0.95
    assert rep.lip_closure_to_bilabial_consonant_corr > 0.95
    assert rep.brow_cheek_expression_to_pitch_prosody_corr > 0.90

    # 3. Novel subtitle re-creation lip-voice synchronization
    assert rep.recreated_lip_voice_sync_correlation > 0.98
    assert Path(paths["output_av_dashboard_path"]).exists()
    assert Path(paths["output_av_video_gif_path"]).exists()
