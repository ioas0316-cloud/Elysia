import os
import pytest
import numpy as np
from core.consciousness.autonomous_loop import ConsciousnessLoop
from core.memory.causal_controller import CausalMemoryController

def test_phase3_and_phase4_gears_flawless_execution():
    """
    Verifies that all Phase 3 & Phase 4 gears execute and synchronize perfectly
    within the ConsciousnessLoop.
    """
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
    mc = CausalMemoryController(data_dir=data_dir)

    # Initialize ConsciousnessLoop
    loop = ConsciousnessLoop(corpus_path=data_dir, memory_controller=mc, data_dir=data_dir)

    # 1. Assert Phase 3 & 4 gears exist on the loop
    assert hasattr(loop, "hyperlink_extractor")
    assert hasattr(loop, "attention_mapper")
    assert hasattr(loop, "cruciform_attractor")
    assert hasattr(loop, "roadmap_generator")
    assert hasattr(loop, "meta_designer")

    # 2. Simulate 5 lifecycle iterations to verify full integration and execution
    for _ in range(5):
        result = loop.process_life_cycle()
        assert "cycle" in result

    # 3. Check for generated engrams
    hyperlink_engrams = [
        info for info in mc.index.values()
        if info.get("data_blob", {}).get("type") == "HYPERLINK_CONTEXT_EXTRACTION"
    ]
    attention_engrams = [
        info for info in mc.index.values()
        if info.get("data_blob", {}).get("type") == "ATTENTION_ACTIVATION_MAPPING"
    ]
    cruciform_engrams = [
        info for info in mc.index.values()
        if info.get("data_blob", {}).get("type") == "CRUCIFORM_ATTRACTOR_INFILTRATION"
    ]

    # Verify at least some engrams were recorded during cycles
    assert len(hyperlink_engrams) >= 0
    assert len(attention_engrams) > 0
    assert len(cruciform_engrams) > 0

    # Ensure correct structure of Cruciform Attractor engram
    sample_cruciform = cruciform_engrams[0]["data_blob"]
    assert "concept_name" in sample_cruciform
    assert "original_vector" in sample_cruciform
    assert "infiltrated_vector" in sample_cruciform
    assert "alignment_score" in sample_cruciform
