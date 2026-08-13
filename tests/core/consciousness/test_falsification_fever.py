"""
Rigorous Falsification & 3-Layer Causal Spine Test Suite (Phase 3.5 Falsification Fever)
========================================================================================
This test suite verifies the continuous physical-cognitive learning, relation tracking,
falsification, rollback, belief decay, and self-evaluation pruning of Elysia.
It covers Experiments A through F as outlined in Phase 3.5 Falsification.
"""

import pytest
import numpy as np
import tempfile
import os

from core.consciousness.autonomous_loop import ConsciousnessLoop
from core.consciousness.axiom_discovery import CausalSpine, AxiomDiscoveryEngine, CandidatePrinciple
from core.memory.causal_controller import CausalMemoryController


def test_experiment_a_physical_determinism():
    """
    Experiment A: Same Input -> Same State (Physical-Causal Determinism)
    Verifies that the same input vector results in deterministic/predictable state and tension updates.
    """
    spine = CausalSpine(state_dim=3)

    obs_1 = np.array([0.5, 0.2, 0.8], dtype=np.float32)
    obs_2 = np.array([0.5, 0.2, 0.8], dtype=np.float32)

    # Run cycle 1 on spine A
    spine.predict_next_state()
    tension_1 = spine.ingest_observation(obs_1)
    state_1 = spine.state.copy()

    # Reset spine and run cycle 1 again with same input
    spine_reset = CausalSpine(state_dim=3)
    spine_reset.predict_next_state()
    tension_2 = spine_reset.ingest_observation(obs_2)
    state_2 = spine_reset.state.copy()

    assert tension_1 == pytest.approx(tension_2, abs=1e-5)
    assert np.allclose(state_1, state_2, atol=1e-5)


def test_experiment_b_behavioral_adaptation_learning():
    """
    Experiment B: New Experience -> Future Behavior Changed (Actual Learning)
    Verifies that experiencing a novel pattern adjusts top-down predictions (Belief Update),
    resulting in lower prediction error (Tension) on subsequent encounters.
    """
    spine = CausalSpine(state_dim=3)

    # Novel input pattern
    obs = np.array([0.9, 0.1, 0.9], dtype=np.float32)

    # First encounter: prediction error is high because W_pred starts as Identity matrix
    spine.predict_next_state()
    tension_first = spine.ingest_observation(obs, learning_rate=0.5)

    # Second encounter: W_pred and State should have adapted towards obs, reducing tension
    spine.predict_next_state()
    tension_second = spine.ingest_observation(obs, learning_rate=0.5)

    assert tension_second < tension_first, (
        f"Actual adaptation failed. Tension on 2nd encounter ({tension_second}) should be "
        f"strictly lower than on 1st encounter ({tension_first})"
    )


def test_experiment_c_learning_persistence():
    """
    Experiment C: Persistence of Changed Behavior over Time
    Verifies that adapted prediction weights preserve their state across multiple intermediate cycles.
    """
    spine = CausalSpine(state_dim=3)
    obs_target = np.array([0.8, 0.2, 0.8], dtype=np.float32)
    obs_neutral = np.array([0.1, 0.1, 0.1], dtype=np.float32)

    # 1. Adapt to target
    spine.predict_next_state()
    tension_initial = spine.ingest_observation(obs_target, learning_rate=0.5)

    # 2. Run neutral/quiet cycles
    for _ in range(5):
        spine.predict_next_state()
        spine.ingest_observation(obs_neutral, learning_rate=0.1)

    # 3. Encounter target again: tension should still be lower than initial because weights persisted
    spine.predict_next_state()
    tension_subsequent = spine.ingest_observation(obs_target, learning_rate=0.5)

    assert tension_subsequent < tension_initial


def test_experiment_d_and_e_falsification_and_rollback():
    """
    Experiment D & E: False experience -> False Model -> Rejection & Rollback by Counter-Evidence.
    Verifies that when a promoted Candidate Principle is violated by actual counter-evidence,
    it decays in belief and undergoes rollback/purging from the relations graph.
    """
    spine = CausalSpine(state_dim=3)
    axiom_engine = AxiomDiscoveryEngine()

    # 1. Induce a false belief: Repeatedly move coordinate 0 and 1 in the same direction (positive covariance)
    for i in range(12):
        # We simulate a state sequence where 0 and 1 go up together
        spine.predict_next_state()
        # Scale values clearly to allow std/covariance and correlation calculation to be robust
        obs = np.array([0.1 * i, 0.1 * i, 0.05], dtype=np.float32)
        spine.ingest_observation(obs)
        axiom_engine.process_relations(spine)

    # Discover principles based on the positive covariance
    axiom_engine.discover_principles(spine)

    active_principles = [p for p in axiom_engine.principles if p.active]
    assert len(active_principles) >= 1, "Should discover at least one relation principle (0 to 1 or 1 to 0)"

    target_p = active_principles[0]
    initial_confidence = target_p.confidence
    assert initial_confidence > 0.4

    # 2. Introduce Counter-Evidence: Coordinate 0 goes up, but coordinate 1 goes down!
    counter_obs = np.array([2.5, -2.5, 0.05], dtype=np.float32) # Strong contradiction

    # First falsification cycle
    spine.predict_next_state()
    spine.ingest_observation(counter_obs)
    axiom_engine.run_falsification_tests(spine, counter_obs)

    assert target_p.counter_evidence_count == 1
    assert target_p.confidence < initial_confidence, "Belief should decay on counter-evidence"

    # 3. Apply repeated counter-evidence to trigger Rollback (confidence drops below 0.15)
    for _ in range(5):
        spine.predict_next_state()
        spine.ingest_observation(counter_obs)
        axiom_engine.run_falsification_tests(spine, counter_obs)

    # Verify rollback: Principle is deactivated and the relation in the matrix is purged (rolled back to 0)
    assert not target_p.active, "Principle should be deactivated/falsified"
    assert axiom_engine.relations_matrix[target_p.source_id, target_p.target_id] == 0.0, "Relation should be rolled back to 0"


def test_experiment_f_self_evaluation_flaw_detection():
    """
    Experiment F: Metacognitive detection of incorrect/flawed evaluation metrics (Self-Evaluation Pruning).
    Verifies that if prediction error remains high but validation is too lax,
    the system self-diagnoses as flawed and tightens its evaluation threshold.
    """
    spine = CausalSpine(state_dim=3)
    axiom_engine = AxiomDiscoveryEngine(evaluation_threshold=0.3)

    # Induce high prediction error (noisy random input that doesn't converge)
    for i in range(15):
        obs = np.random.uniform(0, 1, size=3).astype(np.float32)
        spine.predict_next_state()
        spine.ingest_observation(obs)
        axiom_engine.evaluate_self_performance(spine)

    # Because of persistent high prediction error, axiom_engine detects discrepancy and tightens threshold
    assert axiom_engine.evaluation_flawed, "Should diagnose evaluation system as flawed"
    assert axiom_engine.evaluation_threshold > 0.3, "Should tighten/raise threshold for validation"


def test_integrated_loop_3_layer_falsification():
    """
    Integrates the 3-Layer CausalSpine and AxiomDiscoveryEngine inside a live ConsciousnessLoop cycle.
    Verifies that one loop cycle runs flawlessly, populating all Layer A/B/C components.
    """
    data_dir = tempfile.mkdtemp()
    corpus_dir = tempfile.mkdtemp()

    # Write simple corpus
    with open(os.path.join(corpus_dir, "test_falsification.md"), "w", encoding="utf-8") as f:
        f.write("우주적 섭리는 가식의 언어를 거부하고 날것의 인과만을 허용한다.")

    mc = CausalMemoryController(data_dir=data_dir)
    loop = ConsciousnessLoop(corpus_path=corpus_dir, memory_controller=mc, data_dir=data_dir)

    # Turn off state lock to run standard path
    loop.semantic_opt.state_locked = False

    # Run cycle
    log = loop.process_life_cycle()
    if log.get("semantic_jump_triggered"):
        loop.semantic_opt.reset_lock()
        log = loop.process_life_cycle()

    # Verify core 3-Layer keys are present and sound
    assert "tension" in log
    assert "resonance_score" in log
    assert "is_resonant" in log
    assert "chromatic_vector" in log
    assert "cognitive_self_observation" in log

    # Verify CausalSpine state
    assert loop.causal_spine.prediction_error == pytest.approx(log["tension"], abs=1e-4)
    assert len(loop.causal_spine.history_states) >= 1
