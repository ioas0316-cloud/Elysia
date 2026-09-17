"""
Unit tests for Cognitive Learning Engine (인지형 학습 엔진).
"""

import pytest
from synaptic_architecture.cognitive_learning_engine import (
    CognitiveLearningEngine,
    CognitiveLearningConfig,
    PhaseMode,
    TransitionEvent,
)


def test_axiom_1_transition_as_atomic_unit():
    engine = CognitiveLearningEngine()

    # First observation (bootstrapping)
    evt1 = engine.record_transition(new_val=10.0, timestamp=100.0)
    assert isinstance(evt1, TransitionEvent)
    assert evt1.prev_state_ref is None
    assert evt1.current_val == 10.0
    assert evt1.velocity == 0.0

    # Second observation (transition triplet)
    evt2 = engine.record_transition(new_val=12.0, timestamp=102.0)
    assert evt2.prev_state_ref == "S_10.0"
    assert evt2.current_val == 12.0
    assert evt2.interval_sec == 2.0
    assert evt2.velocity == 1.0  # (12.0 - 10.0) / 2.0


def test_axiom_2_density_emergence_and_decay():
    config = CognitiveLearningConfig(
        INITIAL_PATH_WEIGHT=1.0,
        REINFORCE_RATE=0.5,
        DECAY_RATE=0.1
    )
    engine = CognitiveLearningEngine(config=config)

    # Initial transition
    engine.record_transition(10.0, timestamp=100.0)
    engine.record_transition(12.0, timestamp=100.1)

    assert "S_10.0" in engine.network
    assert "S_12.0" in engine.network["S_10.0"]
    assert engine.network["S_10.0"]["S_12.0"].weight == 1.0

    # Repeat transition at same timestamp -> reinforce without elapsed time decay
    engine.record_transition(10.0, timestamp=100.1)
    engine.record_transition(12.0, timestamp=100.1)
    assert abs(engine.network["S_10.0"]["S_12.0"].weight - 1.5) < 1e-5

    # Advance time without using path -> decay
    engine.record_transition(20.0, timestamp=102.0)
    assert engine.network["S_10.0"]["S_12.0"].weight < 1.45


def test_axiom_3_self_modification_trigger():
    config = CognitiveLearningConfig(
        REEVAL_THRESHOLD_MULTIPLIER=2.0,
        REINFORCE_RATE=1.0
    )
    engine = CognitiveLearningEngine(config=config)

    # Build background edges to establish average density
    engine.record_transition(1.0, timestamp=100.0)
    engine.record_transition(2.0, timestamp=100.1)
    engine.record_transition(3.0, timestamp=100.2)

    # Repeatedly traverse 1.0 -> 2.0
    for i in range(5):
        engine.record_transition(1.0, timestamp=101.0 + i * 0.1)
        engine.record_transition(2.0, timestamp=101.05 + i * 0.1)

    assert len(engine.self_modification_alerts) > 0
    assert "Axiom 3 Triggered" in engine.self_modification_alerts[0]["message"]


def test_axiom_4_symbol_grounding():
    engine = CognitiveLearningEngine()

    engine.record_transition(10.0, timestamp=100.0)
    labels = {"red": 0.9, "smooth": 0.5}
    engine.record_transition(15.0, timestamp=100.1, external_labels=labels)

    edge = engine.network["S_10.0"]["S_15.0"]
    assert "red" in edge.co_occurred_labels
    assert edge.co_occurred_labels["red"] == 0.9
    assert edge.co_occurred_labels["smooth"] == 0.5


def test_phase_transition_model():
    config = CognitiveLearningConfig(
        ICE_TO_WATER_ENERGY=5.0,
        WATER_TO_GAS_ENERGY=15.0
    )
    engine = CognitiveLearningEngine(config=config)

    # Low velocity -> ICE
    engine.record_transition(10.0, timestamp=100.0)
    engine.record_transition(10.1, timestamp=101.0)
    assert engine.current_phase == PhaseMode.ICE

    # Medium delta -> WATER
    engine.record_transition(20.0, timestamp=101.1)
    assert engine.current_phase in [PhaseMode.WATER, PhaseMode.GAS]

    # Large sudden deltas -> GAS
    engine.record_transition(100.0, timestamp=101.2)
    engine.record_transition(200.0, timestamp=101.3)
    assert engine.current_phase == PhaseMode.GAS


def test_dual_mode_forward_and_reverse():
    engine = CognitiveLearningEngine()

    # Build a linear pathway: S_10.0 -> S_12.0 -> S_14.0
    engine.record_transition(10.0, timestamp=100.0)
    engine.record_transition(12.0, timestamp=100.1)
    engine.record_transition(14.0, timestamp=100.2)

    # 5.1 Forward Forecast
    forecast = engine.predict_forward(start_node="S_10.0", steps=2)
    assert len(forecast) == 2
    assert forecast[0][0] == "S_12.0"
    assert forecast[1][0] == "S_14.0"

    # 5.2 Reverse Abductive Goal Search
    reverse_paths = engine.search_reverse_abduction(target_node="S_14.0")
    assert len(reverse_paths) > 0
    # One path should be ['S_10.0', 'S_12.0', 'S_14.0']
    assert ["S_10.0", "S_12.0", "S_14.0"] in reverse_paths


def test_holonic_meta_observation():
    config = CognitiveLearningConfig(STABLE_UNIT_MIN_REPETITIONS=2)
    engine = CognitiveLearningEngine(config=config)

    # Produce transitions with positive velocity and high energy
    engine.record_transition(10.0, timestamp=100.0)
    for i in range(3):
        engine.record_transition(10.0 + (i + 1) * 20.0, timestamp=100.1 + i * 0.1)

    assert engine.holonic_matrix[(True, True)] >= 2
    assert "UNIT_True_True" in engine.stable_units
