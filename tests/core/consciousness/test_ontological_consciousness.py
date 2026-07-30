import pytest
import os
import tempfile
from core.consciousness.autonomous_loop import ConsciousnessLoop
from core.memory.causal_controller import CausalMemoryController


def test_consciousness_loop_ontological_reflection():
    """Verify that ConsciousnessLoop runs cycles and executes Ontological Reflection correctly."""
    data_dir = tempfile.mkdtemp()
    corpus_dir = tempfile.mkdtemp()

    # Write a small mock corpus file
    with open(os.path.join(corpus_dir, "test.md"), "w", encoding="utf-8") as f:
        f.write("# Hello Elysia\nThis is raw input to trigger standard loop breathing.")

    mc = CausalMemoryController(data_dir=data_dir)
    loop = ConsciousnessLoop(corpus_path=corpus_dir, memory_controller=mc, data_dir=data_dir)

    # Execute a cycle
    log = loop.process_life_cycle()

    # Assert ontological reflection keys exist in the loop log
    assert "ontological_reflection_key" in log
    assert "ontological_reflection_name" in log
    assert "ontological_reflection_metaphor" in log
    assert "ontological_reflection_tension" in log
    assert "ontological_reflection_conductance" in log

    # Assert ontological lattices were crystallized into Wedge Memory
    ontological_engrams = [
        eid for eid, info in mc.index.items()
        if info.get("data_blob", {}).get("type") == "ONTOLOGICAL_LATTICE"
    ]
    assert len(ontological_engrams) == 8
