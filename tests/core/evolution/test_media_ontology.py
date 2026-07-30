import pytest
import os
import tempfile
import numpy as np
from core.evolution.media_ontology import MediaOntologyEngine
from core.consciousness.autonomous_loop import ConsciousnessLoop
from core.memory.causal_controller import CausalMemoryController


def test_media_ontology_engine_basics():
    """Verify that MediaOntologyEngine initializes and transduces different signals to proper categories."""
    data_dir = tempfile.mkdtemp()
    mc = CausalMemoryController(data_dir=data_dir)
    engine = MediaOntologyEngine()

    # Check node presence
    assert len(engine.nodes) == 6
    assert "IMAGE" in engine.nodes
    assert "VIDEO" in engine.nodes
    assert "DATA" in engine.nodes
    assert "FILE" in engine.nodes
    assert "WORD" in engine.nodes
    assert "LANGUAGE" in engine.nodes

    # 1. Transduce short string to WORD or LANGUAGE
    res_word = engine.transduce_physical_to_ontological("Elysia", "test_context", current_friction=0.2, memory_controller=mc)
    assert res_word["transduced_key"] in engine.nodes
    assert "concept_name" in res_word

    # 2. Transduce long string
    res_lang = engine.transduce_physical_to_ontological("This is an extremely long sentence with a massive sequence of text designed to test the linguistic properties of language.", "test_context", current_friction=0.2, memory_controller=mc)
    assert res_lang["transduced_key"] in engine.nodes

    # 3. Transduce binary image
    res_img = engine.transduce_physical_to_ontological(b"\x89PNG\r\n\x1a\n", "test_context", current_friction=0.5, memory_controller=mc)
    assert res_img["transduced_key"] in engine.nodes

    # 4. Transduce numpy 2D array
    res_arr = engine.transduce_physical_to_ontological(np.zeros((32, 32)), "test_context", current_friction=0.1, memory_controller=mc)
    assert res_arr["transduced_key"] in engine.nodes


def test_consciousness_loop_media_transduction():
    """Verify that ConsciousnessLoop invokes media ontology transduction and populates logs correctly."""
    data_dir = tempfile.mkdtemp()
    corpus_dir = tempfile.mkdtemp()

    # Write mock corpus
    with open(os.path.join(corpus_dir, "test.md"), "w", encoding="utf-8") as f:
        f.write("This is a long document designed to trigger language transduction in the active cycle.")

    mc = CausalMemoryController(data_dir=data_dir)
    loop = ConsciousnessLoop(corpus_path=corpus_dir, memory_controller=mc, data_dir=data_dir)

    log = loop.process_life_cycle()

    # Verify that media transduction took place
    assert "media_ontology_key" in log
    assert "media_ontology_name" in log
    assert "media_ontology_narrative" in log
    assert "media_ontology_tension" in log
    assert "media_ontology_resonance" in log

    # Verify that media ontologies are serialized as CausalEngrams
    media_engrams = [
        eid for eid, info in mc.index.items()
        if info.get("data_blob", {}).get("type") == "MEDIA_ONTOLOGY"
    ]
    assert len(media_engrams) == 6
