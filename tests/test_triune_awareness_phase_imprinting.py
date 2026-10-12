"""
Tests for Triune Awareness Lens and Phase Imprinting Engine
"""

import pytest
import numpy as np

from core.lens.triune_awareness_lens import TriuneAwarenessLens, TriuneSpectrum
from core.consciousness.phase_imprinting_engine import PhaseImprintingEngine, SemanticDefinitionNode


def test_triune_awareness_lens_perception():
    lens = TriuneAwarenessLens(base_resistance=1.5, intrinsic_phase=0.5, intrinsic_cycle=1.0)

    signal = np.array([1.0, 2.0, 1.5, 0.5])
    spectrum = lens.perceive(signal)

    assert isinstance(spectrum, TriuneSpectrum)
    assert spectrum.amplitude > 0.0
    assert 0.0 <= spectrum.phase <= 2 * np.pi
    assert 0.0 <= spectrum.resonance <= 1.0
    assert 0.0 <= spectrum.interference <= 1.0
    assert pytest.approx(spectrum.resonance + spectrum.interference, abs=1e-5) == 1.0


def test_differential_awareness():
    lens = TriuneAwarenessLens()

    # Send identical signals twice to observe high resonance (sameness)
    sig1 = np.array([1.0, 0.0, -1.0, 0.0])
    spectrum1 = lens.perceive(sig1)

    spectrum2 = lens.perceive(sig1, reference_waveform=sig1)

    # Sameness should produce high resonance
    assert spectrum2.resonance > 0.7
    assert spectrum2.interference < 0.3

    # Send opposing/different signal to observe higher interference
    sig_opposite = np.array([-1.0, 0.0, 1.0, 0.0])
    spectrum3 = lens.perceive(sig_opposite, reference_waveform=sig1)
    assert spectrum3.interference > spectrum2.interference


def test_hierarchical_definition_grounding():
    lens = TriuneAwarenessLens()
    engine = PhaseImprintingEngine(lens=lens, recalibration_rate=0.2)

    # Ground a semantic definition: "Harmonic_State" at (1.0, 1.5, np.pi/2)
    engine.register_semantic_definition(
        name="Harmonic_State",
        target_spectrum=(1.0, 1.5, np.pi / 2),
        causal_meaning="System operating in balanced resonance",
    )

    engine.register_semantic_definition(
        name="High_Tension_Error",
        target_spectrum=(1.0, 5.0, np.pi),
        causal_meaning="Anomalous resistance detected",
    )

    mock_spectrum = TriuneSpectrum(
        cycle=1.0,
        amplitude=1.4,
        phase=np.pi / 2 + 0.1,
        resonance=0.9,
        interference=0.1,
    )

    best_node, affinity = engine.Ground_hierarchical_definition(mock_spectrum)
    assert best_node is not None
    assert best_node.name == "Harmonic_State"
    assert affinity > 0.5


def test_phase_imprinting_recalibration():
    lens = TriuneAwarenessLens(base_resistance=1.0, intrinsic_phase=0.0)
    engine = PhaseImprintingEngine(lens=lens, recalibration_rate=0.2)

    initial_resistance = lens.resistance_matrix[0, 0]
    initial_phase = lens.intrinsic_phase

    data_stream = np.array([2.0, 3.0])
    spectrum_before = lens.perceive(data_stream)

    # Perform action resulting in higher outcome tension (3.5) and feedback phase (np.pi)
    record = engine.imprint_phase_experience(
        spectrum=spectrum_before,
        outcome_tension=3.5,
        causal_feedback_phase=np.pi,
    )

    # Verify recalibration of lens properties
    assert lens.resistance_matrix[0, 0] > initial_resistance
    assert lens.intrinsic_phase != initial_phase
    assert len(engine.imprint_history) == 1

    # Post-imprinting perception should reflect the altered lens state
    spectrum_after = lens.perceive(data_stream)
    assert spectrum_after.amplitude != spectrum_before.amplitude
