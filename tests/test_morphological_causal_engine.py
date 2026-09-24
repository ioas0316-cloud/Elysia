"""
Unit tests for Bidirectional Phase Negotiation, Crystallized DNA Anchor,
and Morphological Plasticity Engine.
"""

import math
import pytest
from core.embodied.phase_negotiation import BidirectionalPhaseNegotiator, SensoryWaveStream
from core.evolution.dna_anchor import CrystallizedDNAAnchor, MorphologicalGenome
from core.embodied.morphological_engine import MorphologicalPlasticityEngine, EnvironmentPressure


def test_bidirectional_phase_negotiation_convergence():
    negotiator = BidirectionalPhaseNegotiator(learning_rate=0.2, coupling_strength=0.9)
    external_wave = SensoryWaveStream(frequency=2.0, phase=0.0, amplitude=1.5)

    # Perform negotiation steps with advancing external wave phase
    dt = 0.1
    for _ in range(50):
        external_wave.phase = (external_wave.phase + external_wave.frequency * dt) % (2.0 * math.pi)
        res = negotiator.step_negotiation(external_wave, time_delta=dt)

    # Check phase error reduction and resonance score
    assert abs(negotiator.q_err) < 0.4
    assert negotiator.resonance_score > 0.8
    assert abs(negotiator.internal_wave.frequency - external_wave.frequency) < 0.2


def test_crystallized_dna_anchor():
    dna_store = CrystallizedDNAAnchor()

    # Check primal ancestral anchors loaded
    assert "DNA_STREAMLINED_AQUATIC" in dna_store.crystallized_anchors
    assert "DNA_AERODYNAMIC_WING" in dna_store.crystallized_anchors
    assert "DNA_ARTICULATED_HAND" in dna_store.crystallized_anchors
    assert "DNA_LOAD_BEARING_LEGS" in dna_store.crystallized_anchors

    # Query closest anchor for wing-like feature vector
    wing_like_features = [0.2, 0.9, 0.2, 0.4, 0.85, 0.1, 0.3, 0.7]
    closest = dna_store.query_closest_anchor(wing_like_features)
    assert closest.anchor_id == "DNA_AERODYNAMIC_WING"

    # Test crystallization under high stability score
    custom_genome = MorphologicalGenome(
        anchor_id="CUSTOM_EVOLVED",
        name="Custom Evolved Form",
        drag_coefficient=0.1,
        lift_coefficient=0.8,
        grasp_articulation=0.3,
        structural_rigidity=0.5,
        resonance_frequency=3.0,
        feature_vector=[0.1] * 8
    )
    success = dna_store.crystallize_pattern("CUSTOM_EVOLVED", "Custom Form", custom_genome, stability_score=0.92)
    assert success is True
    assert "CUSTOM_EVOLVED" in dna_store.crystallized_anchors


def test_morphological_plasticity_fluid_adaptation():
    engine = MorphologicalPlasticityEngine()

    # Apply heavy fluid pressure (high drag)
    fluid_env = EnvironmentPressure(fluid_density=2.5, current_velocity=3.0)

    for _ in range(30):
        history = engine.adapt_morphology(fluid_env, time_delta=0.1, morph_rate=0.2)

    # Check that drag coefficient converged down towards streamlined aquatic template (0.08)
    assert engine.current_genome.drag_coefficient < 0.25
    assert history["target_anchor"] == "Streamlined Hydrodynamic Form"


def test_morphological_plasticity_aerodynamic_lift_adaptation():
    engine = MorphologicalPlasticityEngine()

    # Apply high airflow velocity (requires lift)
    air_env = EnvironmentPressure(air_flow_velocity=5.0)

    for _ in range(30):
        history = engine.adapt_morphology(air_env, time_delta=0.1, morph_rate=0.2)

    # Check that lift coefficient converged up towards aerodynamic wing template (0.92)
    assert engine.current_genome.lift_coefficient > 0.70
    assert history["target_anchor"] == "Aerodynamic Wing Structure"


def test_morphological_plasticity_resource_scarcity_hand_adaptation():
    engine = MorphologicalPlasticityEngine()

    # Apply severe resource scarcity (hunger requiring tool manipulation)
    hungry_env = EnvironmentPressure(resource_scarcity=4.0)

    for _ in range(30):
        history = engine.adapt_morphology(hungry_env, time_delta=0.1, morph_rate=0.2)

    # Check that grasp articulation converged up towards articulated hand template (0.95)
    assert engine.current_genome.grasp_articulation > 0.75
    assert history["target_anchor"] == "Articulated Manipulator Interface"
