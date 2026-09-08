"""
Unit Tests for Omni-Modal Causal Cognition Engine
"""

import numpy as np
import pytest
from core.memory.omni_modal_cognition_engine import OmniModalCognitionEngine


def test_omni_modal_cognition_engine():
    engine = OmniModalCognitionEngine(
        omni_dim=64, num_attractors=128, friction_threshold=0.20, random_seed=42
    )

    stream = {
        "visual": np.random.randn(128).astype(np.float32),
        "auditory": np.random.randn(64).astype(np.float32),
        "textual": np.random.randn(64).astype(np.float32),
        "intent": np.random.randn(16).astype(np.float32),
    }

    res1 = engine.process_omni_stream(stream)
    assert "predicted_phase" in res1
    assert "mode" in res1
    assert "friction" in res1

    # Check scar consolidation pulls repeated stream into System 1 O(1) intuition
    if "System_2" in res1["mode"]:
        frictions = [res1["friction"]]
        for _ in range(10):
            res_next = engine.process_omni_stream(stream)
            frictions.append(res_next["friction"])
        assert frictions[-1] <= frictions[0]
