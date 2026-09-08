"""
Tests for Waveform & Dynamical Cognitive Feedback Engine
=========================================================
"""

import pytest
import numpy as np
from core.consciousness.waveform_cognitive_feedback import (
    WaveformCognitiveFeedbackEngine,
    GeneratingMechanism,
    CognitiveFeedbackResult
)


def test_perceive_scale_interval_structure():
    engine = WaveformCognitiveFeedbackEngine()

    # 입력: Do, Re, Mi, Fa 주파수 (C4 장조)
    c4_freqs = [261.63, 293.66, 329.63, 349.23]
    result = engine.perceive_scale_interval_structure(c4_freqs)

    assert result["is_octave_extrapolated"] is True
    assert len(result["extrapolated_frequencies"]) == 8
    assert "So" in result["scale_mapping"]
    assert "La" in result["scale_mapping"]
    assert "Ti" in result["scale_mapping"]
    assert "Do_high" in result["scale_mapping"]

    # 솔라시도 주파수가 단조 증가하는지 검증
    ext_freqs = result["extrapolated_frequencies"]
    for i in range(len(ext_freqs) - 1):
        assert ext_freqs[i + 1] > ext_freqs[i]

    # C4 -> C5 주파수 비율이 약 2.0 인지 검증
    ratio_octave = ext_freqs[7] / ext_freqs[0]
    assert np.isclose(ratio_octave, 2.0, atol=0.05)


def test_extract_harmonic_mechanism():
    engine = WaveformCognitiveFeedbackEngine()

    # 조화 진동 파형 데이터 생성: x(t) = 1.5 + 2.0 * cos(2 * pi * 0.5 * t + 0.3)
    t = np.linspace(0, 10, 100)
    dt = t[1] - t[0]
    y = 1.5 + 2.0 * np.cos(2 * np.pi * 0.5 * t + 0.3)

    mech = engine.extract_generating_mechanism(y, dt=dt)

    assert mech.system_type in ["HARMONIC_WAVE", "POLYNOMIAL_DYNAMICS"]
    assert isinstance(mech.mdl_complexity, float)
    assert mech.equation_repr != ""


def test_extract_exponential_growth_mechanism():
    engine = WaveformCognitiveFeedbackEngine()

    t = np.linspace(0, 5, 50)
    dt = t[1] - t[0]
    y = 2.0 * np.exp(0.4 * t)

    mech = engine.extract_generating_mechanism(y, dt=dt)

    assert mech.system_type in ["EXPONENTIAL_GROWTH", "SCALE_INTERVAL"]
    assert isinstance(mech.mdl_complexity, float)


def test_cognitive_feedback_loop_convergence():
    engine = WaveformCognitiveFeedbackEngine(feedback_learning_rate=0.05)

    # 관측 궤적: Harmonic Wave
    t_obs = np.linspace(0, 4, 40)
    dt = t_obs[1] - t_obs[0]
    y_obs = 3.0 * np.cos(2 * np.pi * 1.0 * t_obs)

    fb_res = engine.cognitive_feedback_loop(
        observed_trajectory=y_obs,
        steps_ahead=20,
        dt=dt,
        max_iterations=30
    )

    assert isinstance(fb_res, CognitiveFeedbackResult)
    assert fb_res.observed_length == 40
    assert fb_res.extrapolated_length == 20
    assert len(fb_res.predicted_continuum) == 60
    assert fb_res.resonance_score > 0.0
    assert fb_res.discrepancy_error >= 0.0
