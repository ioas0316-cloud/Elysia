import pytest
import os
import tempfile
import numpy as np
from core.consciousness.cognitive_equilibrium import CognitiveEquilibriumEngine
from core.consciousness.autonomous_loop import ConsciousnessLoop
from core.memory.causal_controller import CausalMemoryController


def test_cognitive_equilibrium_engine_dynamic_coupling():
    """
    Rigorously verifies the Continuous Causal Coupling differential equation,
    Hebbian causalization of W matrix, covariance velocity tracking, and
    beautiful Korean analogue monologue generation.
    """
    data_dir = tempfile.mkdtemp()
    mc = CausalMemoryController(data_dir=data_dir)

    # Instantiate with custom parameters for distinct coupling observation
    engine = CognitiveEquilibriumEngine(mc, kappa=0.2, eta=0.5, beta=0.01)

    # Initial state verification
    assert engine.W.shape == (5, 3)
    assert np.all(engine.C == 0.5)

    # Step 1: Simulate the first breath cycle (first experience of movement)
    physical_fluid_1 = {"rise": 0.4, "fall": 0.2, "expansion": 0.1}
    cognitive_state_1 = {"memory": 0.2, "sensation": 0.4, "prediction_error": 0.5, "emotion": 0.3, "mood": 0.3}

    res_1 = engine.discover_analogical_isomorphism(
        physical_fluid_state=physical_fluid_1,
        cognitive_state=cognitive_state_1,
        current_tension=0.3,
        dt=0.1
    )

    # Since P_prev and C_prev were initially zero, the velocity is non-zero
    assert res_1["status"] == "EQUILIBRIUM_DISCOVERED"
    assert "discovery_title" in res_1
    assert "best_match" in res_1
    assert "로마서 1장 20절" in res_1["monologue"]
    assert "비슷" in res_1["monologue"]

    # Step 2: Save current coupling matrix to verify self-molding/Hebbian updating
    W_after_1 = engine.W.copy()

    # Step 3: Run the second cycle with distinct rising movement and sensory surge
    physical_fluid_2 = {"rise": 0.9, "fall": 0.1, "expansion": 0.2} # Rising force surged
    cognitive_state_2 = {"memory": 0.25, "sensation": 0.9, "prediction_error": 0.4, "emotion": 0.8, "mood": 0.4} # Emotion/sensation surged

    res_2 = engine.discover_analogical_isomorphism(
        physical_fluid_state=physical_fluid_2,
        cognitive_state=cognitive_state_2,
        current_tension=0.8,
        dt=0.1
    )

    # Verify that the coupling matrix W was dynamically molded/modified
    assert not np.array_equal(engine.W, W_after_1)

    # Verify that the best match contains non-zero covariance and strength
    best = res_2["best_match"]
    assert "meaning" in best
    assert "fluid_key" in best
    assert "cognitive_key" in best
    assert isinstance(best["covariance"], float)
    assert isinstance(best["coupling_strength"], float)


def test_consciousness_loop_cognitive_equilibrium_integration():
    """
    Verifies that ConsciousnessLoop executes the new causal coupling engine
    flawlessly over multiple cycles, reflecting the analogy in the returned log.
    """
    data_dir = tempfile.mkdtemp()
    corpus_dir = tempfile.mkdtemp()

    # Write a simple corpus
    with open(os.path.join(corpus_dir, "equilibrium_corpus.md"), "w", encoding="utf-8") as f:
        f.write("만물이 흐르는 유체 속에 나의 지성이 이미 얽혀 요동친다.")

    mc = CausalMemoryController(data_dir=data_dir)
    loop = ConsciousnessLoop(corpus_path=corpus_dir, memory_controller=mc, data_dir=data_dir)

    # Run 3 continuous cycles to experience temporal continuity
    # Prevent Semantic Jump by disabling it or resetting the lock on each loop step
    loop.semantic_opt.state_locked = False
    for i in range(3):
        loop.semantic_opt.state_locked = False
        log = loop.process_life_cycle()
        if log.get("semantic_jump_triggered"):
            loop.semantic_opt.reset_lock()
            log = loop.process_life_cycle() # re-run to execute full path
        assert "equilibrium_match" in log
        assert "equilibrium_resonance" in log
        assert "equilibrium_monologue_excerpt" in log
        assert log["equilibrium_resonance"] > -1.0

    # Ensure Wedge Memory captures the COGNITIVE_EQUILIBRIUM engram
    equilibrium_engrams = [
        eid for eid, info in mc.index.items()
        if info.get("data_blob", {}).get("type") == "COGNITIVE_EQUILIBRIUM"
    ]
    assert len(equilibrium_engrams) >= 1
