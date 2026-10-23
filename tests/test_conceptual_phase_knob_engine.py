"""
Unit tests for ConceptualPhaseKnobEngine.
"""

import math
import numpy as np
import pytest
from L1_causal_field.core.topology.conceptual_phase_knob_engine import (
    ConceptualPhaseKnobEngine,
    ConceptNode
)


def test_concept_registration():
    engine = ConceptualPhaseKnobEngine(dimension=16)
    raw_vec = [1.0] * 16
    node = engine.register_concept("test_concept", raw_vec)

    assert node.name == "test_concept"
    assert len(node.feature_vector) == 16
    assert np.isclose(np.linalg.norm(node.feature_vector), 1.0)
    # Check invariant vs variant split
    assert np.all(node.invariance_core[8:] == 0.0)
    assert np.all(node.variant_wave[:8] == 0.0)


def test_conceptual_distance_calculation():
    engine = ConceptualPhaseKnobEngine(dimension=16)
    v1 = [1.0] * 8 + [0.0] * 8
    v2 = [0.9] * 8 + [0.1] * 8
    v_opposite = [-1.0] * 8 + [0.0] * 8

    engine.register_concept("c1", v1)
    engine.register_concept("c2", v2)
    engine.register_concept("c_opp", v_opposite)

    delta_theta_12, friction_12 = engine.compute_conceptual_distance("c1", "c2")
    assert friction_12 < 0.1
    assert delta_theta_12 >= 0.0

    delta_theta_opp, friction_opp = engine.compute_conceptual_distance("c1", "c_opp")
    assert friction_opp < 0.01  # abs(dot_product) is close to 1.0


def test_knob_creation_and_adjustment():
    engine = ConceptualPhaseKnobEngine(dimension=16)
    engine.register_concept("gravity", [0.5] * 16)

    knob_key = engine.create_knob_from_concept("gravity", initial_value=1.0)
    assert knob_key == "knob_gravity"
    assert engine.active_knobs[knob_key] == 1.0

    result = engine.adjust_knob(knob_key, value=3.0)
    assert result["status"] == "rearranged"
    assert engine.active_knobs[knob_key] == 3.0
    assert result["system_phase_tension"] == pytest.approx(0.3)


def test_unregistered_concept_error():
    engine = ConceptualPhaseKnobEngine(dimension=16)
    engine.register_concept("c1", [1.0] * 16)

    with pytest.raises(ValueError, match="Registered concepts required"):
        engine.compute_conceptual_distance("c1", "unregistered")


def test_unregistered_knob_error():
    engine = ConceptualPhaseKnobEngine(dimension=16)
    with pytest.raises(KeyError):
        engine.adjust_knob("non_existent_knob", 2.0)
