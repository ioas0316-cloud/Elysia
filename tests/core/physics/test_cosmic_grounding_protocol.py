"""
Unit Tests for Cosmic Grounding Protocol & Unified Spectrum Engine
===================================================================
Tests key dynamics:
1. Kuramoto Phase Locking & Observation Collapse (R -> 1.0)
2. Holographic Field Manifestation (Interference Pattern Generation)
3. Synesthetic Spectrum Axis Transformation (Acoustic <-> Visual <-> Spectrum)
4. Discriminative Sensory Observation & Dynamic Variable Impedance Tuning
"""

import pytest
import torch
from core.physics.cosmic_grounding_protocol import (
    UnifiedCosmicSpectrumEngine,
    SpectrumFieldState,
    CosmicGroundingMetrics,
    HolographicFieldRenderer,
    SynestheticSpectrumMapper,
    DiscriminativeSensoryObservationLens,
)


def test_kuramoto_phase_locking_observation_collapse():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = UnifiedCosmicSpectrumEngine(
        num_micro_nodes=500,
        spectrum_dim=64,
        coupling_strength_K=10.0,
        dt=0.03,
        device=device
    )

    init_R, _ = engine.compute_kuramoto_order()
    assert init_R.item() < 0.9, "Initial micro nodes should be unsynchronized"

    # Step phase locking towards macro phase pi
    final_R = engine.step_phase_locking(target_macro_phase=torch.pi, steps=30)
    assert final_R.item() > 0.85, f"Expected Kuramoto order parameter R > 0.85, got {final_R.item()}"
    assert engine.sensory_lens.impedance.item() < 0.2, "Variable impedance should drop upon high order parameter alignment"


def test_holographic_field_manifestation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    renderer = HolographicFieldRenderer(resolution=(32, 32), device=device)

    freqs = torch.tensor([5.0, 10.0, 15.0], device=device)
    amps = torch.tensor([1.0, 0.8, 0.5], device=device)
    phases = torch.tensor([0.0, torch.pi / 4, torch.pi / 2], device=device)
    chroma = torch.ones((3, 3), device=device) / 3.0

    spectrum = SpectrumFieldState(
        frequencies=freqs,
        amplitudes=amps,
        phases=phases,
        chromatic_vectors=chroma
    )

    manifestation = renderer.manifest_field(spectrum=spectrum, variable_impedance=0.1)
    assert manifestation.shape == (32, 32), f"Expected shape (32, 32), got {manifestation.shape}"
    assert torch.max(manifestation) <= 1.0 + 1e-5, "Manifestation field should be normalized"
    assert torch.min(manifestation) >= 0.0, "Manifestation intensity field must be non-negative"


def test_synesthetic_spectrum_transformation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mapper = SynestheticSpectrumMapper(num_spectrum_modes=64, device=device)

    # Generate synthetic acoustic PCM signal
    t = torch.linspace(0, 1, 1000, device=device)
    pcm = torch.sin(2 * torch.pi * 440.0 * t) + 0.5 * torch.sin(2 * torch.pi * 880.0 * t)

    # Acoustic -> Spectrum Field State
    spectrum = mapper.acoustic_to_spectrum(pcm, sample_rate=44100.0)
    assert spectrum.frequencies.shape[0] == 64
    assert spectrum.amplitudes.shape[0] == 64

    # Spectrum -> Acoustic Waveform Synthesis
    resynthesized_pcm = mapper.spectrum_to_acoustic(spectrum, num_samples=500)
    assert resynthesized_pcm.shape[0] == 500
    assert torch.std(resynthesized_pcm) > 0.0, "Resynthesized PCM wave should have non-zero variance"


def test_discriminative_sensory_observation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lens = DiscriminativeSensoryObservationLens(spectrum_dim=64, device=device)

    # Simulated discrete raw byte stream
    raw_stream = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1] * 10, device=device)

    spectrum = lens.observe_stream(raw_stream)
    assert spectrum.frequencies.shape[0] == 64
    assert spectrum.chromatic_vectors.shape == (64, 3)

    lens.tune_impedance(order_parameter_R=0.95)
    assert abs(lens.impedance.item() - 0.05) < 1e-3, "Impedance should adjust to 1.0 - R"


def test_full_cosmic_grounding_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = UnifiedCosmicSpectrumEngine(
        num_micro_nodes=300,
        spectrum_dim=64,
        coupling_strength_K=12.0,
        dt=0.02,
        device=device
    )

    raw_stream = torch.randint(0, 2, (100,), device=device)
    visual_field, acoustic_wave, metrics = engine(
        raw_data_stream=raw_stream,
        target_macro_phase=torch.pi / 2.0,
        steps=20
    )

    assert visual_field.shape == (64, 64)
    assert acoustic_wave.shape[0] == 500
    assert isinstance(metrics, CosmicGroundingMetrics)
    assert metrics.observation_collapse_efficiency > 80.0, "Efficiency should exceed 80%"
    assert metrics.holographic_coherence > 0.0
