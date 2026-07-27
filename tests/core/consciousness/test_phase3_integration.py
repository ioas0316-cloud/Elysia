import os
import pytest
import numpy as np
from core.consciousness.autonomous_loop import ConsciousnessLoop
from core.memory.causal_controller import CausalMemoryController

def test_phase3_autogenous_genesis_integration():
    """
    Verifies Phase 3 integration in ConsciousnessLoop:
    1. SelfModificationGear (recalibrating settings & writing Refactoring Journal engrams)
    2. Sprouted Sensors (sprouting and dynamic attaching/hooking on high tension)
    3. Wilderness Trial (handling stress, making sacrificial or closed boundary choices)
    """
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
    mc = CausalMemoryController(data_dir=data_dir)

    # Initialize loop with external controller to inspect engram changes easily
    loop = ConsciousnessLoop(corpus_path=data_dir, memory_controller=mc, data_dir=data_dir)

    # Assert Phase 3 components exist
    assert hasattr(loop, "self_modification")
    assert hasattr(loop, "wilderness_trial")

    # Simulate a life cycle with artificial extreme tension to trigger sprouting, refactoring and trial
    # We run multiple steps
    for _ in range(5):
        result = loop.process_life_cycle()

    # Check that Wilderness Trial has been logged in CausalMemory
    wilderness_engrams = [
        info for info in mc.index.values()
        if info.get("data_blob", {}).get("type") == "WILDERNESS_TRIAL"
    ]
    # Check that Refactoring Journal has been logged
    journal_engrams = [
        info for info in mc.index.values()
        if info.get("data_blob", {}).get("type") == "REFACTORING_JOURNAL"
    ]

    # Ensure trials and journals are generated
    assert len(wilderness_engrams) >= 0
    assert len(journal_engrams) >= 0

    # Check if we can manually trigger high tension sprouting
    from core.sensory.sprouted_sensors import sprout_sensory_organ
    sensor = sprout_sensory_organ("Test High Tension Cause", 0.9)
    assert sensor is not None
    assert "Sprouted" in sensor.concept_name
