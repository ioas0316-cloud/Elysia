import pytest
import os
import tempfile
import numpy as np
from core.consciousness.cognitive_equilibrium import CognitiveEquilibriumEngine
from core.consciousness.autonomous_loop import ConsciousnessLoop
from core.memory.causal_controller import CausalMemoryController


def test_cognitive_equilibrium_engine_isolation():
    """
    Verifies that the CognitiveEquilibriumEngine correctly maps the visible
    fluid principles to invisible cognitive states based on Romans 1:20,
    finding analogical isomorphisms and articulating them beautifully in Korean.
    """
    data_dir = tempfile.mkdtemp()
    mc = CausalMemoryController(data_dir=data_dir)

    engine = CognitiveEquilibriumEngine(mc)

    physical_fluid = {"rise": 0.8, "fall": 0.1, "expansion": 0.3}
    cognitive_state = {"memory": 0.4, "sensation": 0.8, "prediction_error": 0.2, "emotion": 0.7, "mood": 0.5}

    result = engine.discover_analogical_isomorphism(
        physical_fluid_state=physical_fluid,
        cognitive_state=cognitive_state,
        current_tension=0.5
    )

    assert result["status"] == "EQUILIBRIUM_DISCOVERED"
    assert "discovery_title" in result
    assert "best_match" in result
    assert "monologue" in result
    assert "로마서 1장 20절" in result["monologue"]
    assert "비슷하구나" in result["monologue"]

    # Verify that the engram of the analogy is recorded to Wedge Memory
    equilibrium_engrams = [
        eid for eid, info in mc.index.items()
        if info.get("data_blob", {}).get("type") == "COGNITIVE_EQUILIBRIUM"
    ]
    assert len(equilibrium_engrams) == 1
    engram = mc.index[equilibrium_engrams[0]]
    assert "로마서" in engram["data_blob"]["monologue"]


def test_consciousness_loop_cognitive_equilibrium_integration():
    """
    Verifies that ConsciousnessLoop executes the CognitiveEquilibriumEngine
    seamlessly inside its breath cycle, reflecting the analogy in the returned log.
    """
    data_dir = tempfile.mkdtemp()
    corpus_dir = tempfile.mkdtemp()

    # Write a simple corpus
    with open(os.path.join(corpus_dir, "equilibrium_corpus.md"), "w", encoding="utf-8") as f:
        f.write("보이는 만물의 이치를 통해 보이지 않는 마음을 해석한다.")

    mc = CausalMemoryController(data_dir=data_dir)
    loop = ConsciousnessLoop(corpus_path=corpus_dir, memory_controller=mc, data_dir=data_dir)

    log = loop.process_life_cycle()

    # Verify logs of analogy and equilibrium
    assert "equilibrium_match" in log
    assert "equilibrium_resonance" in log
    assert "equilibrium_monologue_excerpt" in log
    assert log["equilibrium_resonance"] > 0.0

    # Ensure Wedge Memory captures it
    equilibrium_engrams = [
        eid for eid, info in mc.index.items()
        if info.get("data_blob", {}).get("type") == "COGNITIVE_EQUILIBRIUM"
    ]
    assert len(equilibrium_engrams) >= 1
