import pytest
import numpy as np
from core.intelligence.phasor_trinity_attention import PhasorTrinityAttentionField


def test_initialization():
    field = PhasorTrinityAttentionField(num_scales=4, base_dimension=32)
    assert field.num_scales == 4
    assert field.base_dimension == 32
    assert field.query_phasors.shape == (4, 32)
    assert field.query_phasors.dtype == np.complex64


def test_set_query_state():
    field = PhasorTrinityAttentionField(num_scales=3, base_dimension=16)
    target_amps = np.ones((3, 16)) * 2.0
    target_phases = np.full((3, 16), np.pi / 4)

    field.set_query_state(amplitudes=target_amps, phases=target_phases)

    np.testing.assert_allclose(np.abs(field.query_phasors), target_amps, rtol=1e-5)
    np.testing.assert_allclose(np.angle(field.query_phasors), target_phases, rtol=1e-5)


def test_phase_locking_constructive_vs_destructive():
    field = PhasorTrinityAttentionField(
        num_scales=2, base_dimension=8, coupling_strength=1.0
    )

    # Set Query phase to zero
    field.set_query_state(
        amplitudes=np.ones((2, 8)), phases=np.zeros((2, 8))
    )

    # Constructive Key (phase = 0) vs Destructive Key (phase = pi)
    key_constructive = np.exp(1j * np.zeros((2, 8)))  # delta_phi = 0
    key_destructive = np.exp(1j * np.full((2, 8), np.pi))  # delta_phi = pi

    value = np.ones((2, 8), dtype=np.complex64)

    # Combined batch input of N=2 entities
    keys = np.stack([key_constructive, key_destructive], axis=0)  # (2, 2, 8)
    values = np.stack([value, value], axis=0)  # (2, 2, 8)

    emergent_value, coherence, weights = field.compute_trinity_resonance(
        keys, values
    )

    # Normalized resonance weight for entity 0 (constructive) should be greater than entity 1 (destructive)
    assert np.all(weights[0] > weights[1])
    assert coherence.shape == (2,)


def test_scale_hierarchy_feedback_evolution():
    field = PhasorTrinityAttentionField(num_scales=4, base_dimension=16)
    initial_phase = np.angle(field.query_phasors).copy()

    key = np.exp(1j * np.zeros((4, 16)))
    value = np.exp(1j * np.full((4, 16), np.pi / 2))

    # Step through resonance feedback loop
    for _ in range(5):
        field.compute_trinity_resonance(key, value, time_delta=0.1)

    # Phase history should record evolution
    assert len(field.phase_history) == 5

    # Query phases should have evolved away from initial state
    current_phase = np.angle(field.query_phasors)
    assert not np.allclose(initial_phase, current_phase)


def test_phase_spectrum():
    field = PhasorTrinityAttentionField(num_scales=3, base_dimension=8)
    spectrum = field.get_phase_spectrum()

    assert "amplitudes" in spectrum
    assert "phases" in spectrum
    assert "frequencies" in spectrum
    assert len(spectrum["frequencies"]) == 3
