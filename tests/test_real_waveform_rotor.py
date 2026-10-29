import pytest
import numpy as np
from L1_causal_field.synaptic_architecture.phase_rotor_transition_engine import (
    RealWaveformGenerator,
    RealWaveformRotorVerifier,
    PhaseRotorTransitionEngine,
)


def test_real_waveform_generator_shapes_and_normalization():
    generator = RealWaveformGenerator(feature_dim=64)
    t = 0.25

    wave_harmonic = generator.generate_harmonic_resonance(t)
    wave_noise = generator.generate_friction_noise_shock(t)
    wave_kinematic = generator.generate_kinematic_trajectory(t)

    assert wave_harmonic.shape == (64,)
    assert wave_noise.shape == (64,)
    assert wave_kinematic.shape == (64,)

    assert pytest.approx(np.linalg.norm(wave_harmonic), abs=1e-4) == 1.0
    assert pytest.approx(np.linalg.norm(wave_noise), abs=1e-4) == 1.0
    assert pytest.approx(np.linalg.norm(wave_kinematic), abs=1e-4) == 1.0


test_real_waveform_generator_shapes_and_normalization()


def test_mock_stream_buffer_loading():
    generator = RealWaveformGenerator(feature_dim=64)
    raw_data = np.random.randn(64).astype(np.float32).tobytes()

    wave = generator.load_from_mock_stream_buffer(raw_data)
    assert wave.shape == (64,)
    assert pytest.approx(np.linalg.norm(wave), abs=1e-4) == 1.0


def test_waveform_judgment_categorical_separation():
    verifier = RealWaveformRotorVerifier(feature_dim=64, data_scale_gb=0.3)
    res = verifier.verify_waveform_judgment(samples_per_category=15)

    assert "category_summary" in res
    assert "separation_margin" in res
    assert res["separation_margin"] > 0.05
    assert res["judgment_accurate"] is True

    summary = res["category_summary"]
    # Harmonic resonance and kinematic trajectory should have higher coherence than friction noise
    assert summary["Harmonic_Resonance"]["avg_coherence"] > summary["Friction_Noise_Shock"]["avg_coherence"]
    assert summary["Friction_Noise_Shock"]["avg_q_heat"] > summary["Harmonic_Resonance"]["avg_q_heat"]


def test_wave_weaving_reconstruction_fidelity():
    verifier = RealWaveformRotorVerifier(feature_dim=64, data_scale_gb=0.3)
    res = verifier.verify_wave_weaving_reconstruction(steps=20)

    assert "avg_cosine_similarity" in res
    assert "avg_reconstruction_fidelity" in res
    assert res["avg_cosine_similarity"] > 0.90
    assert res["avg_reconstruction_fidelity"] > 0.90
    assert res["reconstruction_valid"] is True


def test_multi_scale_integration_benchmark():
    verifier = RealWaveformRotorVerifier(feature_dim=64, data_scale_gb=0.3)
    scale_report = verifier.run_full_scale_integration_benchmark(scales=[0.3, 10.0, 30.0, 100.0])

    assert len(scale_report) == 4
    for row in scale_report:
        assert row["separation_margin"] > 0.05
        assert row["reconstruction_fidelity"] > 0.90
        assert row["vram_gb"] <= 2.50
