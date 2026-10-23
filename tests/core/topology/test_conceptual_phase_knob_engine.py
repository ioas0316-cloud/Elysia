"""
Unit tests for ConceptualPhaseKnobEngine (개념적 위상 노브 엔진).
"""

import pytest
import numpy as np
from L1_causal_field.core.topology.conceptual_phase_knob_engine import (
    ConceptualPhaseKnobEngine,
    ConceptualEntity,
    ConceptualDistanceResult,
    ExtractedWorldMechanism,
    CognitiveKnob
)


def test_conceptual_phase_distance_isomorphism():
    engine = ConceptualPhaseKnobEngine(dimension=64, friction_threshold=0.15)

    # Bird wing: aerodynamic lift intent
    bird_causal = np.random.randn(64)
    bird_intent = np.zeros(64)
    bird_intent[0] = 1.0  # Lift & drag reduction dimension
    bird_intent[1] = 0.5
    bird_boundary = np.array([0.1, 0.2])

    bird_entity = ConceptualEntity(
        name="bird_wing",
        domain="biology",
        causal_vector=bird_causal,
        intent_lens=bird_intent,
        boundary_condition=bird_boundary
    )

    # Airplane wing: same aerodynamic intent, completely different domain/vector
    airplane_causal = np.random.randn(64) + 100.0  # Far in Euclidean space
    airplane_intent = np.zeros(64)
    airplane_intent[0] = 0.98
    airplane_intent[1] = 0.52
    airplane_boundary = np.array([0.12, 0.19])

    airplane_entity = ConceptualEntity(
        name="airplane_wing",
        domain="aeronautics",
        causal_vector=airplane_causal,
        intent_lens=airplane_intent,
        boundary_condition=airplane_boundary
    )

    engine.register_entity(bird_entity)
    engine.register_entity(airplane_entity)

    res = engine.compute_conceptual_distance(
        "bird_wing",
        "airplane_wing",
        physical_coords_a=np.array([0.0, 0.0, 0.0]),
        physical_coords_b=np.array([400000.0, 0.0, 0.0])  # 400km physical distance
    )

    # Physical distance is huge (400,000)
    assert res.euclidean_physical_distance == pytest.approx(400000.0)

    # Conceptual distance (friction error) is close to 0 due to functional isomorphism
    assert res.conceptual_phase_distance < 0.1
    assert res.shared_invariance_score > 0.9
    assert res.is_topologically_isomorphic is True
    assert res.clifford_rotor_angle < 0.1


def test_inverse_world_mechanism_extraction_and_knobs():
    engine = ConceptualPhaseKnobEngine(dimension=16)

    # Generate synthetic world state trajectory (100 time steps, 16 state dims)
    t = np.linspace(0, 10, 100)
    base_state = np.ones(16) * 5.0
    wave_variant = np.sin(t)[:, None] * np.ones((100, 16))
    noise = np.random.normal(0, 0.1, (100, 16))

    trajectory = base_state + wave_variant + noise
    boundary_constraints = np.array([0.5, 0.5, 1.0])

    mechanism = engine.extract_world_mechanism(trajectory, boundary_constraints)

    assert isinstance(mechanism, ExtractedWorldMechanism)
    assert len(mechanism.invariant_bedrock) == 16
    assert mechanism.variant_dynamic_wave.shape == (100, 16)

    # Check extracted Cognitive Knobs
    assert "friction" in mechanism.knobs
    assert "rotor_angle" in mechanism.knobs
    assert "causal_tension" in mechanism.knobs
    assert "entropy" in mechanism.knobs

    friction_knob = mechanism.knobs["friction"]
    assert 0.001 <= friction_knob.value <= 5.0


def test_world_state_resimulation_on_knob_tuning():
    engine = ConceptualPhaseKnobEngine(dimension=16)

    trajectory = np.random.randn(50, 16)
    boundary = np.ones(4)

    mechanism = engine.extract_world_mechanism(trajectory, boundary)

    initial_sim_state = engine.resimulate_world_state(mechanism, {})

    # Adjust friction knob higher (increased damping)
    damped_sim_state = engine.resimulate_world_state(
        mechanism,
        {"friction": 4.0, "causal_tension": 2.0}
    )

    # Ensure state changed dynamically
    assert not np.allclose(initial_sim_state, damped_sim_state)
    assert len(damped_sim_state) == 16
