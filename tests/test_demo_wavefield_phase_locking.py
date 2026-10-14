"""
Unit tests for demo_wavefield_phase_locking.py
"""

from demo_wavefield_phase_locking import (
    run_kuramoto_phase_locking,
    run_noise_rectification_demo,
    run_layer_persistence_demo,
)


def test_kuramoto_phase_locking():
    r_history, phases = run_kuramoto_phase_locking()
    assert r_history[-1] > 0.85


def test_noise_rectification():
    coherent_pwr, noise_pwr = run_noise_rectification_demo()
    snr_gain = coherent_pwr / (noise_pwr + 1e-9)
    assert snr_gain > 5.0


def test_layer_persistence():
    full_rel, supp_rel = run_layer_persistence_demo()
    assert supp_rel[0, 1] > 0.5
