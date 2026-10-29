#!/usr/bin/env python3
"""
test_embodied_synesthetic_organ_continuum.py

Comprehensive integration test suite for EmbodiedSynestheticOrganContinuum.
Verifies enactive organ continuum dynamics:
1. $0_{\text{self}}$ (internal lack) + $\Delta V$ (world phase divergence) driving somatic tension & $Q_{\text{heat}}$.
2. Synesthetic resonance across Somatic, Acoustic, Prismatic, Optical, and Scale dimensions.
3. Invariant backbone fidelity preservation across dynamic scale transformations.
"""

import pytest
import numpy as np
from L1_causal_field.core.lens.embodied_synesthetic_organ_continuum import (
    EmbodiedSynestheticOrganContinuum,
    EmbodiedOrganState,
)


def test_organ_continuum_initialization():
    organ = EmbodiedSynestheticOrganContinuum(internal_lack_zero=0.5, num_particles=50)
    assert organ.internal_lack_zero == 0.5
    assert len(organ.sandbox.particles) == 50


def test_perceive_and_resonate_cycle():
    organ = EmbodiedSynestheticOrganContinuum(internal_lack_zero=0.4, num_particles=60)
    wave = np.sin(np.linspace(0, 4 * np.pi, 500))

    state = organ.perceive_and_resonate(wave)

    assert isinstance(state, EmbodiedOrganState)
    assert state.internal_lack_zero == 0.4
    assert state.world_phase_divergence > 0.0
    assert 0.0 < state.postural_tension <= 1.0
    assert state.somatic_friction_heat > 0.0
    assert len(state.chromatic_spectrum) == 3
    assert state.invariant_backbone_fidelity >= 0.90


def test_lack_and_divergence_drive_somatic_tension():
    organ_low_lack = EmbodiedSynestheticOrganContinuum(internal_lack_zero=0.1, num_particles=40)
    organ_high_lack = EmbodiedSynestheticOrganContinuum(internal_lack_zero=0.9, num_particles=40)

    wave = np.sin(np.linspace(0, 10 * np.pi, 500)) + 0.5 * np.random.randn(500)

    state_low = organ_low_lack.perceive_and_resonate(wave)
    state_high = organ_high_lack.perceive_and_resonate(wave)

    assert state_high.postural_tension > state_low.postural_tension
    assert state_high.somatic_friction_heat > state_low.somatic_friction_heat


def test_dynamic_scale_zoom_and_unit_binding():
    organ = EmbodiedSynestheticOrganContinuum(internal_lack_zero=0.5, base_focal_length=2.0, num_particles=80)

    wave_calm = np.sin(np.linspace(0, 2 * np.pi, 500)) * 0.1
    state_calm = organ.perceive_and_resonate(wave_calm)

    wave_turbulent = np.sin(np.linspace(0, 20 * np.pi, 500)) * 2.0 + np.random.randn(500)
    state_turbulent = organ.perceive_and_resonate(wave_turbulent)

    # Turbulent wave creates higher phase divergence -> deeper zoom (smaller focal depth) and narrower aperture
    assert state_turbulent.focal_depth < state_calm.focal_depth
    assert state_turbulent.aperture_size < state_calm.aperture_size
    assert len(state_turbulent.bound_grounded_units) > 0


def test_chromatic_spectrum_and_fidelity():
    organ = EmbodiedSynestheticOrganContinuum(internal_lack_zero=0.5, num_particles=50)
    wave = np.cos(np.linspace(0, 6 * np.pi, 1000))

    state = organ.perceive_and_resonate(wave)

    # Chromatic spectrum normalized sum
    assert np.isclose(np.sum(state.chromatic_spectrum), 1.0)
    assert state.invariant_backbone_fidelity > 0.90
