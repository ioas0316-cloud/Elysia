"""
Tests for Prismatic Phase Transition Engine & Spacetime Deck
============================================================
Validates the 4-stage phase transition pipeline:
1. Substrate propagation (micro numerical wavefield)
2. Prismatic refraction into chromatic signatures & structural constraints
3. Macro Spacetime Deck & local ROI phase scrubbing ("Eye of Agamotto")
4. Meta-Observer invariant core integrity & adaptive feedback
"""

import numpy as np
import pytest

from core.physics.prismatic_phase_transition_engine import (
    PrismaticPhaseTransitionEngine,
    SubstrateLayer,
    PrismaticTransitionConstraint,
    MacroSpacetimeDeck,
    MetaObserverInvariantCore
)


def test_substrate_layer_propagation():
    """Test Stage 1: Substrate propagation and numerical injection."""
    substrate = SubstrateLayer(grid_shape=(8, 8))
    data = np.ones((8, 8), dtype=np.float64) * 2.0
    state1 = substrate.inject_input(data)

    assert state1.grid_shape == (8, 8)
    assert np.allclose(state1.raw_signal, 2.0)
    assert state1.potential_energy > 0.0

    state2 = substrate.propagate(dt=0.1)
    assert state2.time_step == pytest.approx(0.1)


def test_prismatic_refraction_and_chromatic_signatures():
    """Test Stage 2: Prismatic refraction from substrate to chromatic spectrum."""
    substrate = SubstrateLayer(grid_shape=(8, 8))
    substrate.inject_input(np.random.rand(8, 8))
    sub_state = substrate.propagate(dt=0.1)

    constraint = PrismaticTransitionConstraint(spectrum_depth=3)
    prism_state = constraint.refract_substrate(sub_state, constraint_stiffness=1.5)

    assert prism_state.chromatic_field.shape == (8, 8, 3)
    assert len(prism_state.nodes) > 0
    assert 0.0 <= prism_state.phase_coherence <= 1.0
    assert prism_state.constraint_tension >= 0.0


def test_macro_spacetime_deck_local_roi_scrubbing():
    """Test Stage 3: Global rotation and local ROI O(1) continuous scrubbing."""
    deck = MacroSpacetimeDeck(grid_shape=(8, 8))
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:5, 2:5] = True
    deck.set_roi(mask)

    initial_phases = deck.waveform_tensor.phases.copy()

    # Scrub local ROI by +1.5 radians
    deck.scrub_roi_phase(1.5)

    # ROI cells should have changed phase
    assert not np.allclose(deck.waveform_tensor.phases[mask], initial_phases[mask])

    # Non-ROI cells must remain untouched (phase invariant)
    assert np.allclose(deck.waveform_tensor.phases[~mask], initial_phases[~mask])


def test_meta_observer_invariant_core_feedback():
    """Test Stage 4: Meta-Observer coherence tracking and adaptive feedback."""
    observer = MetaObserverInvariantCore(invariant_target_coherence=0.8)
    engine = PrismaticPhaseTransitionEngine(grid_shape=(8, 8))

    state = engine.step(dt=0.1)
    feedback = state.meta_feedback

    assert isinstance(feedback.invariant_coherence, float)
    assert feedback.invariant_intact is True
    assert "constraint_stiffness" in feedback.feedback_adjustment


def test_end_to_end_phase_transition_engine():
    """Test full multi-step execution and adaptive resilience under external noise."""
    engine = PrismaticPhaseTransitionEngine(grid_shape=(8, 8))

    # Set local ROI for scrubbing
    roi = np.zeros((8, 8), dtype=bool)
    roi[:4, :4] = True
    engine.deck.set_roi(roi)

    states = []
    for step in range(10):
        noisy_input = np.random.randn(8, 8) if step % 3 == 0 else None
        scrub_delta = 0.5 if step == 5 else 0.0
        st = engine.step(numerical_input=noisy_input, dt=0.1, roi_scrub_delta=scrub_delta)
        states.append(st)

    assert len(states) == 10
    assert states[-1].step_count == 10
    assert states[-1].meta_feedback.invariant_intact is True
