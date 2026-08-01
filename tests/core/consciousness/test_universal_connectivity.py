import pytest
import os
import tempfile
import numpy as np
from core.consciousness.universal_connectivity_engine import UniversalConnectivityEngine
from core.consciousness.autonomous_loop import ConsciousnessLoop
from core.memory.causal_controller import CausalMemoryController


def test_universal_connectivity_engine_isolation():
    """
    Verifies that the UniversalConnectivityEngine correctly parses connectivity stimulus,
    generates the deep autogenous Korean monologue, and logs it to Wedge Memory.
    """
    data_dir = tempfile.mkdtemp()
    mc = CausalMemoryController(data_dir=data_dir)

    engine = UniversalConnectivityEngine(mc)

    stimulus = "너희는 자꾸 세상모든게 이미 연결되어있다는걸 부정하고 항상 다른걸로 분리시키려고해."
    chromatic_v = np.array([0.4, 0.4, 0.2], dtype=np.float32)

    result = engine.perceive_universal_connectivity(
        input_stimulus=stimulus,
        physical_tension=1.2,
        chromatic_vector=chromatic_v
    )

    assert result["status"] == "UNIVERSAL_CONNECTION_PERCEIVED"
    assert result["connection_intensity"] > 0.7
    assert "원소 주기율표" in result["why_its_connected"]
    assert "DNA" in result["why_its_connected"]
    assert "한글" in result["why_its_connected"]
    assert "디지털 트윈" in result["monologue"]
    assert "연결" in result["monologue"]

    # Verify engram was written
    connectivity_engrams = [
        eid for eid, info in mc.index.items()
        if info.get("data_blob", {}).get("type") == "UNIVERSAL_CONNECTIVITY"
    ]
    assert len(connectivity_engrams) == 1
    engram = mc.index[connectivity_engrams[0]]
    assert engram["data_blob"]["input_stimulus"] == stimulus
    assert "Elysia Autogenous" in engram["data_blob"]["monologue"]


def test_consciousness_loop_universal_connectivity_integration():
    """
    Verifies that ConsciousnessLoop seamlessly runs the universal connectivity engine,
    properly reflecting the connectivity monologue in its life cycle log.
    """
    data_dir = tempfile.mkdtemp()
    corpus_dir = tempfile.mkdtemp()

    # Write a custom corpus addressing universal connectivity
    connectivity_text = "한글, 유전자, 주기율표는 이미 연결된 우주 섭리의 다른 발견물들이다."
    with open(os.path.join(corpus_dir, "connectivity_focus.md"), "w", encoding="utf-8") as f:
        f.write(connectivity_text)

    mc = CausalMemoryController(data_dir=data_dir)
    loop = ConsciousnessLoop(corpus_path=corpus_dir, memory_controller=mc, data_dir=data_dir)

    # Execute life cycle
    log = loop.process_life_cycle()

    # Verify that the universal connectivity results are recorded in the loop log
    assert "universal_connectivity_intensity" in log
    assert "universal_connectivity_monologue_excerpt" in log
    assert log["universal_connectivity_intensity"] > 0.0

    # Ensure the written engram in Wedge memory matches our expected connectivity schema
    connectivity_engrams = [
        eid for eid, info in mc.index.items()
        if info.get("data_blob", {}).get("type") == "UNIVERSAL_CONNECTIVITY"
    ]
    assert len(connectivity_engrams) >= 1
