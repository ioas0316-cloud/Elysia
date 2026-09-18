"""
Unit tests for Cognitive Learning Engine (인지형 학습 엔진).
"""

import pytest
from synaptic_architecture.cognitive_learning_engine import (
    CognitiveLearningEngine,
    CognitiveLearningConfig,
    PhaseMode,
    TransitionEvent,
    InputDispatcher,
    InputType,
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
    reverse_res = engine.search_reverse_abduction(target_node="S_14.0")
    retrace = reverse_res["historical_retrace_paths"]
    assert len(retrace) > 0
    assert ["S_10.0", "S_12.0", "S_14.0"] in retrace


def test_input_dispatcher_and_multi_modal_encoding():
    engine = CognitiveLearningEngine()

    # Scalar input
    evt1, t1 = engine.dispatch_and_record(15.5, timestamp=100.0)
    assert t1 == InputType.SCALAR
    assert engine.current_state_node == "S_15.5"

    # Categorical input
    evt2, t2 = engine.dispatch_and_record("STATE_OVERHEAT", timestamp=101.0)
    assert t2 == InputType.CATEGORICAL
    assert engine.current_state_node == "SYM_STATE_OVERHEAT"

    # Vector input
    evt3, t3 = engine.dispatch_and_record([1.0, 2.0, 3.0], timestamp=102.0)
    assert t3 == InputType.VECTOR
    assert engine.current_state_node == "VEC_[1.0, 2.0, 3.0]"

    # Second vector input (verifies VectorEncoder L2 norm velocity computation)
    evt4, t4 = engine.dispatch_and_record([1.0, 5.0, 3.0], timestamp=104.0)
    assert t4 == InputType.VECTOR
    assert engine.current_state_node == "VEC_[1.0, 5.0, 3.0]"
    assert abs(evt4.velocity - 1.5) < 1e-3  # sqrt((5-2)^2) / 2.0 = 3.0 / 2.0 = 1.5
    assert "S_15.5" in engine.network or "SYM_STATE_OVERHEAT" in engine.network


def test_volume_encoding_and_element_velocities():
    engine = CognitiveLearningEngine()

    # First volume frame (Axiom 1.1)
    vol1 = {"temp": 20.0, "pressure": 100.0, "mode": "STABLE"}
    evt1, t1 = engine.dispatch_and_record(vol1, timestamp=100.0)
    assert t1 == InputType.VOLUME
    assert "VOL_{" in engine.current_state_node
    assert evt1.element_velocities == {"temp": 0.0, "pressure": 0.0, "mode": 0.0}

    # Second volume frame after 2 seconds
    vol2 = {"temp": 24.0, "pressure": 100.0, "mode": "OVERHEAT"}
    evt2, t2 = engine.dispatch_and_record(vol2, timestamp=102.0)
    assert t2 == InputType.VOLUME
    assert evt2.element_velocities["temp"] == 2.0  # (24 - 20) / 2
    assert evt2.element_velocities["pressure"] == 0.0  # (100 - 100) / 2
    assert evt2.element_velocities["mode"] == 0.5  # categorical shift 1.0 / 2.0


def test_holonic_relative_selection_pressure():
    config = CognitiveLearningConfig(STABLE_UNIT_MIN_REPETITIONS=2, STABLE_UNIT_RELATIVE_RATIO=0.4)
    engine = CognitiveLearningEngine(config=config)

    # Produce 80% (True, True) transitions and 20% (False, False)
    engine.dispatch_and_record(10.0, timestamp=100.0)
    # 8 High delta transitions -> (True, True)
    for i in range(8):
        engine.dispatch_and_record(10.0 + (i + 1) * 30.0, timestamp=100.1 + i * 0.1)

    # 2 Low/negative delta transitions -> (False, False)
    for i in range(2):
        engine.dispatch_and_record(10.0 - (i + 1) * 0.01, timestamp=101.0 + i * 0.1)

    # (True, True) ratio is 8/10 = 80% >= 40% -> Stable
    assert "UNIT_True_True" in engine.stable_units
    # (False, False) count is 2 >= 2, but ratio is 2/10 = 20% < 40% -> Filtered out by selection pressure
    assert "UNIT_False_False" not in engine.stable_units


def test_composite_node_multiplicative_binding():
    engine = CognitiveLearningEngine()

    engine.dispatch_and_record(10.0, timestamp=100.0)
    engine.dispatch_and_record(12.0, timestamp=100.1, external_labels={"smooth": 2.0})

    cid = "COMP_S_10.0->S_12.0_LABEL_smooth"
    assert cid in engine.composite_nodes

    cnode = engine.composite_nodes[cid]
    edge_weight = engine.network["S_10.0"]["S_12.0"].weight
    expected_binding = edge_weight * 2.0 * 1.0  # density_a * density_b * phase_alignment
    assert abs(cnode.binding_strength - expected_binding) < 1e-5

    # Multiplicative test: if density_a or density_b or phase_alignment is zero, binding strength is zero
    cnode_zero = engine.create_or_update_composite_node(
        source_a="A", source_b="B", density_a=5.0, density_b=0.0, phase_alignment=1.0
    )
    assert cnode_zero.binding_strength == 0.0


def test_convergence_vs_divergence_branching():
    config = CognitiveLearningConfig(STABLE_UNIT_MIN_REPETITIONS=2, STABLE_UNIT_RELATIVE_RATIO=0.3)
    engine = CognitiveLearningEngine(config=config)

    # Generate repeated (True, True) -> Convergence (is_exclusive = False)
    engine.dispatch_and_record(10.0, timestamp=100.0)
    for i in range(5):
        engine.dispatch_and_record(10.0 + (i + 1) * 20.0, timestamp=100.1 + i * 0.1)

    comp_id = "COMP_COND_VEL_True_COND_ENG_True"
    assert comp_id in engine.composite_nodes
    assert engine.composite_nodes[comp_id].is_exclusive is False

    # Generate repeated (True, False) -> Divergence (is_exclusive = True)
    # Drain energy below ICE_TO_WATER_ENERGY/2 while keeping small positive velocity
    engine.accumulated_energy = 0.0
    engine.current_state_node = "S_10.0"
    engine.last_scalar_value = 10.0
    for i in range(5):
        engine.dispatch_and_record(10.0 + (i + 1) * 0.01, timestamp=102.0 + i * 0.1)

    excl_id = "EXCL_COND_VEL_True_COND_ENG_False"
    assert excl_id in engine.composite_nodes
    assert engine.composite_nodes[excl_id].is_exclusive is True


def test_axiom_5_temporal_phase_alignment_and_cache():
    config = CognitiveLearningConfig(PHASE_ALIGNMENT_THRESHOLD=0.5)
    engine = CognitiveLearningEngine(config=config)

    # Rhythmic observations with identical interval (0.1s)
    engine.dispatch_and_record(10.0, timestamp=100.0)
    engine.dispatch_and_record(12.0, timestamp=100.1)
    engine.dispatch_and_record(14.0, timestamp=100.2)

    assert len(engine.transient_cache) > 0
    tb_unit = list(engine.transient_cache.values())[0]
    assert tb_unit.phase_alignment > 0.9  # Nearly 1.0 since interval diff is 0.0s


def test_memory_tier_promotions():
    config = CognitiveLearningConfig(
        PROMOTION_CACHE_TO_RAM_COUNT=2,
        PROMOTION_RAM_TO_SSD_WEIGHT=1.8,
        REINFORCE_RATE=1.0
    )
    engine = CognitiveLearningEngine(config=config)

    # Repeated rhythmic transitions: S_10.0 -> S_12.0 -> S_14.0 -> S_16.0 ...
    engine.dispatch_and_record(10.0, timestamp=100.0)
    engine.dispatch_and_record(12.0, timestamp=100.1)
    engine.dispatch_and_record(14.0, timestamp=100.2)
    engine.dispatch_and_record(12.0, timestamp=100.3)
    engine.dispatch_and_record(14.0, timestamp=100.4)

    # Verify promotion to Persistent SSD
    assert len(engine.persistent_seeds) > 0
    path_seeds = [k for k in engine.persistent_seeds.keys() if "SEED_PATH_" in k]
    assert len(path_seeds) > 0


def test_novel_recombined_pathway_synthesis():
    engine = CognitiveLearningEngine()

    # Route A: S_10.0 -> S_12.0 with grounding label "lift"
    engine.dispatch_and_record(10.0, timestamp=100.0)
    engine.dispatch_and_record(12.0, timestamp=100.1, external_labels={"lift": 2.0})

    # Route B: S_50.0 -> S_55.0 with grounding label "lift" (no direct edge to S_12.0)
    engine.dispatch_and_record(50.0, timestamp=101.0)
    engine.dispatch_and_record(55.0, timestamp=101.1, external_labels={"lift": 2.0})

    reverse_res = engine.search_reverse_abduction(target_node="S_12.0")
    novel = reverse_res["novel_recombined_pathways"]

    assert len(novel) > 0
    # S_50.0 -> S_55.0 should be discovered as a novel recombined pathway bridging via "lift"
    found_bridge = any("lift" in n["shared_grounded_principles"] for n in novel)
    assert found_bridge

    # Roadmap Step 4: Combinatorial design synthesis for unseen combinations
    hypotheses = reverse_res["combinatorial_design_hypotheses"]
    assert len(hypotheses) > 0
    assert any("UNSEEN COMBINATION" in h["hypothetical_edge"] for h in hypotheses)


def test_axiom_4_label_un_grounding_decay():
    config = CognitiveLearningConfig(GROUNDING_DECAY=0.1)
    engine = CognitiveLearningEngine(config=config)

    engine.dispatch_and_record(10.0, timestamp=100.0)
    engine.dispatch_and_record(12.0, timestamp=100.1, external_labels={"fragile": 0.5})

    edge = engine.network["S_10.0"]["S_12.0"]
    assert "fragile" in edge.co_occurred_labels

    # Advance time by 6 seconds without reinforcement -> 0.5 - 0.1 * 6 = -0.1 <= 0 -> removed (un-grounded)
    engine.dispatch_and_record(14.0, timestamp=106.1)

    assert "fragile" not in edge.co_occurred_labels


def test_axiom_3_dispatcher_rule_reevaluation_alert():
    config = CognitiveLearningConfig(DISPATCHER_UNKNOWN_LIMIT=2)
    engine = CognitiveLearningEngine(config=config)

    class CustomObject:
        pass

    engine.dispatch_and_record(CustomObject(), timestamp=100.0)
    assert not any(a.get("type") == "DISPATCHER_RULE_REEVALUATION" for a in engine.self_modification_alerts)

    engine.dispatch_and_record(CustomObject(), timestamp=101.0)
    alerts = [a for a in engine.self_modification_alerts if a.get("type") == "DISPATCHER_RULE_REEVALUATION"]
    assert len(alerts) == 1
    assert "Dispatcher classification rules require self-review" in alerts[0]["message"]


def test_dual_source_bootstrap_discrepancy_and_realignment():
    config = CognitiveLearningConfig(DEMOTION_DISCREPANCY_LIMIT=2)
    engine = CognitiveLearningEngine(config=config)

    # Section 7.1: Load imported seed prior
    engine.load_seed_priors({"rule_temperature": {"expected": 20.0}})
    assert "SEED_PRIOR_rule_temperature" in engine.persistent_seeds

    # Experience observations contradicting seed expectations (actual: 30.0 vs expected: 20.0)
    evt1, obs1 = engine.process_dual_source_observation(
        raw_data=30.0, seed_prior_id="rule_temperature", expected_next_val=20.0, timestamp=100.0
    )
    assert obs1 is not None
    assert obs1.activation_count == 1

    # Second consistent discrepancy -> triggers seed re-alignment alert
    evt2, obs2 = engine.process_dual_source_observation(
        raw_data=30.0, seed_prior_id="rule_temperature", expected_next_val=20.0, timestamp=101.0
    )
    assert obs2 is not None
    assert obs2.activation_count == 2

    realign_alerts = [a for a in engine.self_modification_alerts if a.get("type") == "SEED_REALIGNMENT_TRIGGERED"]
    assert len(realign_alerts) == 1
    assert realign_alerts[0]["seed_id"] == "rule_temperature"
    assert engine.persistent_seeds["SEED_PRIOR_rule_temperature"]["realigned_value"] == 30.0
