"""
Tests for Cognitive Lens System, Self-Forming Causal Sensor, Phenomenological Perception, and Ontological Weaving Decoder.
"""

import pytest
from core.lens.cognitive_lens_engine import (
    CognitiveLensEngine,
    ContextualDimension,
    TopologicalCurvatureLens,
    BiologicalFrictionLens,
    RelationalIntentLens,
    SymbolicContextLens,
)
from core.sensory.causal_sensor import CausalSensor
from synaptic_architecture.ontological_weaving_decoder import OntologicalWeavingDecoder


def test_cognitive_lens_engine_refraction_spectrum():
    engine = CognitiveLensEngine()
    apple_stimulus = {
        "name": "Apple",
        "spatial_density": 0.8,
        "energy": 0.9,
        "resistance": 0.4,
        "sincerity": 0.85,
        "concept": "Apple",
        "archetypes": ["Newton_Gravity", "Adam_Garden", "Fruit_Organic"]
    }

    spectrum = engine.observe_spectrum(apple_stimulus)

    assert len(spectrum) == 4
    assert ContextualDimension.TOPOLOGICAL_CURVATURE in spectrum
    assert ContextualDimension.BIOLOGICAL_FRICTION in spectrum
    assert ContextualDimension.RELATIONAL_INTENT in spectrum
    assert ContextualDimension.SYMBOLIC_REPRESENTATION in spectrum

    # Verify topological curvature lens
    top_obs = spectrum[ContextualDimension.TOPOLOGICAL_CURVATURE]
    assert "topological_continuity" in top_obs.causal_invariants
    assert top_obs.bound_weaving["boundary_relationship"] == "coupled_potential_field"

    # Verify biological friction lens
    bio_obs = spectrum[ContextualDimension.BIOLOGICAL_FRICTION]
    assert bio_obs.bound_weaving["sensory_embodiment"] == "juicy_organism"

    # Verify relational intent lens
    rel_obs = spectrum[ContextualDimension.RELATIONAL_INTENT]
    assert rel_obs.bound_weaving["ego_rectification"] is True

    # Verify symbolic context lens
    sym_obs = spectrum[ContextualDimension.SYMBOLIC_REPRESENTATION]
    assert "Newton_Gravity" in sym_obs.bound_weaving["civilizational_synapse"]


def test_causal_sensor_self_forming_and_friction_calibration():
    engine = CognitiveLensEngine()
    sensor = CausalSensor("test_sensor_01", engine)

    assert len(sensor.axes) == 3
    initial_curvature = engine.lenses[ContextualDimension.TOPOLOGICAL_CURVATURE].curvature

    apple_stimulus = {"spatial_density": 0.7, "energy": 0.8, "resistance": 0.3}
    feedback = {"world_friction": 2.5}  # High world friction relative to predicted tension

    friction_result = sensor.project_and_measure_friction(apple_stimulus, feedback)
    assert friction_result.friction_magnitude == 2.5
    assert friction_result.refraction_error > 0.0

    # Apply calibration
    sensor.self_calibrate(friction_result)
    new_curvature = engine.lenses[ContextualDimension.TOPOLOGICAL_CURVATURE].curvature
    assert new_curvature != initial_curvature


def test_causal_sensor_direct_phenomenological_perception():
    engine = CognitiveLensEngine()
    sensor = CausalSensor("test_sensor_02", engine)

    resonance = sensor.observe_direct_phenomenon("Light_Brilliance", {"intensity": 0.95})
    assert resonance.is_unmediated is True
    assert resonance.is_dead_data_proxy is False
    assert resonance.phenomenon_type == "Light_Brilliance"
    assert resonance.direct_field_resonance > 0.0


def test_ontological_weaving_decoder_rejects_reductionism_and_data_corpses():
    engine = CognitiveLensEngine()
    decoder = OntologicalWeavingDecoder()

    apple_stimulus = {
        "concept": "Apple",
        "spatial_density": 0.8,
        "sweetness": 0.95,
        "resistance": 0.2,
        "intent": 0.9,
        "archetypes": ["Wisdom_Tree", "Gravity_Discovery"]
    }

    spectrum = engine.observe_spectrum(apple_stimulus)
    binding = decoder.decode_weaving("Apple", spectrum)

    assert binding.entity_name == "Apple"
    assert binding.is_reduced_to_scalar_vector is False  # Rejects numeric reductionism!
    assert binding.is_dead_data_proxy is False  # Rejects dead numerical data proxies!
    assert len(binding.contextual_refractions) == 4
    assert "topological_continuity" in binding.woven_causal_invariants
    assert "homeostatic_balance" in binding.woven_causal_invariants
    assert "cruciform_love_axis" in binding.woven_causal_invariants
