#!/usr/bin/env python3
"""
test_scale_hierarchy_continuity.py

Unit and integration tests for Elysia Scale Hierarchy Continuity & Ecosystem Coupling
(L0 Substrate -> L1 Constraint & Prismatic Gate -> L2 Surface Tension Attractor -> L3 Macro Deck).
"""

import sys
from pathlib import Path
import pytest
import numpy as np

_current_file = Path(__file__).resolve() if '__file__' in globals() else Path.cwd()
_root = _current_file.parents[2] if '__file__' in globals() else Path.cwd()
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import elysia_loader
elysia_loader.install()

from core.layer_1_constraint.sensory_prismatic_gate import SensoryPrismaticGate, ContinuousWaveformLens
from core.layer_2_resonant.surface_tension_attractor import SurfaceTensionAttractor, PhaseRelaxationEngine
from core.layer_3_macro.macro_causal_deck import MacroCausalDeck, SpatiotemporalScrubber
from core.orchestration.scale_hierarchy_orchestrator import ScaleHierarchyOrchestrator


def test_layer1_prismatic_gate_continuous_refraction():
    gate = SensoryPrismaticGate(lens_density=500)
    wave = np.sin(np.linspace(0, 4 * np.pi, 500)) + 0.1 * np.random.randn(500)
    out = gate.refract_external_aperture(wave)

    assert "focused_field" in out
    assert "chromatic_spectrum" in out
    assert "phase_coherence" in out
    assert "mental_imagery_field" in out

    assert len(out["chromatic_spectrum"]) == 3
    assert np.isclose(np.sum(out["chromatic_spectrum"]), 1.0)
    assert 0.0 <= out["phase_coherence"] <= 1.0
    assert out["mental_imagery_field"].shape == (64, 64)


def test_layer2_surface_tension_attractor_energy_minimization():
    attractor = SurfaceTensionAttractor(tension_coeff=0.2)
    noisy_imagery = np.random.randn(32, 32) * 2.0
    out = attractor.converge_imagery_attractor(noisy_imagery)

    assert out["relaxed_energy"] < out["initial_energy"]
    assert out["convergence_ratio"] < 1.0
    assert out["steps_taken"] > 0
    assert out["relaxed_imagery_field"].shape == (32, 32)


def test_layer3_macro_deck_and_scrubber():
    deck = MacroCausalDeck(macro_dim=32, memory_capacity=10)
    relaxed_imagery = np.random.randn(16, 16)
    spectrum = np.array([0.4, 0.4, 0.2])

    res1 = deck.process_macro_cycle(relaxed_imagery, spectrum, phase_coherence=0.8, timestamp=0.1)
    res2 = deck.process_macro_cycle(relaxed_imagery, spectrum, phase_coherence=0.8, timestamp=0.2)

    assert res2["macro_tension"] >= 0.0
    assert 0.0 < res2["sensitivity_dial"] <= 1.0
    assert deck.scrubber.total_recorded == 2

    scrubbed = deck.scrubber.scrub_trajectory(target_timestamp=0.1)
    assert scrubbed is not None
    assert len(scrubbed) == 32


def test_full_scale_hierarchy_orchestrator_continuity():
    orchestrator = ScaleHierarchyOrchestrator()
    sample_wave = np.sin(np.linspace(0, 6 * np.pi, 1000))

    # Cycle 1: Low friction
    res1 = orchestrator.process_embodied_cycle(sample_wave, world_friction=0.0, dt=0.1)
    assert res1["continuity_verified"] is True
    dial_before = res1["feedback_sensitivity_dial"]

    # Cycle 2: High friction shock
    shock_wave = sample_wave + 3.0 * np.random.randn(1000)
    res2 = orchestrator.process_embodied_cycle(shock_wave, world_friction=2.0, dt=0.1)

    assert res2["L2_relaxed_energy"] < res2["L2_initial_energy"]
    assert res2["L3_macro_tension"] > res1["L3_macro_tension"]
    # Top-down sensitivity dial should decrease to protect system from shock overload
    assert res2["feedback_sensitivity_dial"] < dial_before
    assert res2["accumulated_scar"] > 0.0
