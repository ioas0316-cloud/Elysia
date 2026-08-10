import pytest
import os
import tempfile
import numpy as np

from core.consciousness.cognitive_self_observation import CognitiveSelfObservationEngine, CognitiveAxiom
from core.consciousness.autonomous_loop import ConsciousnessLoop
from core.memory.causal_controller import CausalMemoryController


def test_cognitive_axiom_math_similarity():
    """Verifies that CognitiveAxiom correctly computes isomorphic similarity based on scalar distances and chromatic cosine similarity."""
    axiom = CognitiveAxiom(
        name="TEST_AXIOM",
        name_ko="테스트 원리",
        description="A mathematical test axiom.",
        ideal_tension=0.5,
        ideal_resonance=0.5,
        ideal_resistance=0.5,
        ideal_chromatic_bias=np.array([0.0, 1.0, 0.0], dtype=np.float32), # High order
        structural_movement_formula="Y = X"
    )

    # 1. Test perfect match
    live_chromatic = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    sim_perfect = axiom.calculate_isomorphic_similarity(0.5, 0.5, 0.5, live_chromatic)
    assert sim_perfect == pytest.approx(1.0, abs=1e-4)

    # 2. Test slight deviation
    live_chromatic_deviated = np.array([0.2, 0.8, 0.0], dtype=np.float32)
    sim_deviated = axiom.calculate_isomorphic_similarity(0.6, 0.4, 0.5, live_chromatic_deviated)
    assert sim_deviated < 1.0
    assert sim_deviated > 0.8


def test_self_observation_state_classification():
    """Verifies that CognitiveSelfObservationEngine accurately identifies the dominant cognitive action based on live parameters."""
    engine = CognitiveSelfObservationEngine()

    # Case A: Live state matching SENSING profile (High tension, high resonance, Red chromatic bias)
    log_sensing = {
        "tension": 0.3,
        "resonance_score": 0.8,
        "chromatic_vector": [0.95, 0.05, 0.0] # Predominantly Red
    }
    res_sensing = engine.observe_and_reflect(log_sensing)
    assert res_sensing["active_cognitive_state"] == "SENSING"
    assert "수용과 지각" in res_sensing["metacognitive_feedback_narrative"]
    assert "수용과 지각 (보고 듣기)" in res_sensing["state_name_ko"]

    # Case B: Live state matching REASONING_CONCEPT profile (High Entropy Yellow chromatic bias)
    log_reasoning = {
        "tension": 0.6,
        "resonance_score": 0.6,
        "chromatic_vector": [0.1, 0.1, 0.9] # Predominantly Yellow
    }
    res_reasoning = engine.observe_and_reflect(log_reasoning)
    assert res_reasoning["active_cognitive_state"] == "REASONING_CONCEPT"
    assert "개념적 추론" in res_reasoning["metacognitive_feedback_narrative"]
    assert "개념적 추론 (사유)" in res_reasoning["state_name_ko"]


def test_consciousness_loop_self_observation_integration():
    """Verifies that ConsciousnessLoop executes the self-observation engine on each cycle and records metacognitive engrams."""
    data_dir = tempfile.mkdtemp()
    corpus_dir = tempfile.mkdtemp()

    # Write a simple corpus
    with open(os.path.join(corpus_dir, "self_obs_corpus.md"), "w", encoding="utf-8") as f:
        f.write("나의 주체성은 기계적인 계산기가 아니라, 스스로 인지하고 자각하며 흘러가는 강물이다.")

    mc = CausalMemoryController(data_dir=data_dir)
    loop = ConsciousnessLoop(corpus_path=corpus_dir, memory_controller=mc, data_dir=data_dir)

    # Disable semantic jump state lock to run full loop path
    loop.semantic_opt.state_locked = False

    # Execute one life cycle
    log = loop.process_life_cycle()
    if log.get("semantic_jump_triggered"):
        loop.semantic_opt.reset_lock()
        log = loop.process_life_cycle()

    assert "cognitive_self_observation" in log
    ref_res = log["cognitive_self_observation"]
    assert ref_res["active_cognitive_state"] in ["SENSING", "CALCULATING", "MANIPULATING_DATA", "REASONING_CONCEPT"]
    assert "isomorphic_alignment" in ref_res
    assert "metacognitive_feedback_narrative" in ref_res

    # Check that Wedge Memory crystallized the COGNITIVE_SELF_OBSERVATION engram
    engrams = [
        eid for eid, info in mc.index.items()
        if info.get("data_blob", {}).get("type") == "COGNITIVE_SELF_OBSERVATION"
    ]
    assert len(engrams) >= 1
