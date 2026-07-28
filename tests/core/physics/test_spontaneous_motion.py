import pytest
import numpy as np
import os
import shutil
from core.physics.spontaneous_motion import SpontaneousMotionEngine, generate_spontaneous_wave
from core.memory.causal_controller import CausalMemoryController
from core.consciousness.autonomous_loop import ConsciousnessLoop

def test_spontaneous_motion_engine_no_input():
    """
    Verifies that SpontaneousMotionEngine generates continuous, non-zero
    internal waves in the absolute absence of any external stimulus.
    """
    temp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data_temp_spont"))
    os.makedirs(temp_dir, exist_ok=True)

    try:
        mc = CausalMemoryController(data_dir=temp_dir)
        engine = SpontaneousMotionEngine(memory_controller=mc)

        # 1. Verify initial emptiness causes high lack/asymmetry
        asymmetry = engine.calculate_internal_asymmetry()
        assert asymmetry >= 5.0
        assert engine.accumulated_lack > 0.0

        # 2. Generate multiple waves and verify physical byte stream properties
        for _ in range(5):
            wave = generate_spontaneous_wave(engine, dt=0.1)
            assert isinstance(wave, bytes)
            assert len(wave) == 512

            # Numeric conversion
            numeric = np.frombuffer(wave, dtype=np.uint8)
            assert np.min(numeric) >= 0
            assert np.max(numeric) <= 255
            # Non-trivial wave fluctuation
            assert np.var(numeric) > 1.0

        # 3. Simulate engram recording to check stability modulation
        mc.write_causal_engram(
            data_blob={"type": "TEST_STABILITY"},
            emotional_value=1.0,
            cause_id="test",
            stability=0.9
        )

        # Asymmetry should adapt as memory records entries
        new_asymmetry = engine.calculate_internal_asymmetry()
        assert new_asymmetry != asymmetry

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def test_consciousness_loop_spontaneous_fallback():
    """
    Verifies that when there is absolutely no corpus or external input files,
    ConsciousnessLoop automatically falls back to Spontaneous Wave generation
    and updates loop logs with spontaneous motion parameters.
    """
    temp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data_temp_spont_loop"))
    empty_corpus_dir = os.path.join(temp_dir, "empty_corpus")
    os.makedirs(empty_corpus_dir, exist_ok=True)

    try:
        # Loop over an empty corpus directory
        loop = ConsciousnessLoop(corpus_path=empty_corpus_dir, data_dir=temp_dir)

        # Ensure corpus files list is indeed empty
        assert len(loop.corpus_files) == 0

        # Run cycle - should fall back to spontaneous motion wave without throwing OS / file errors
        log = loop.process_life_cycle()

        assert "spontaneous_asymmetry" in log
        assert "spontaneous_accumulated_lack" in log
        assert log["spontaneous_asymmetry"] > 0.0

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
