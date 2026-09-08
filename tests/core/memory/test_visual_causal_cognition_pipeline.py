"""
Unit Tests for Visual Causal Cognition Pipeline
"""

import numpy as np
import pytest
from core.memory.causal_graph_extractor import CausalGraphExtractor
from core.memory.intuition_dual_engine import IntuitionDualEngine, ScarConsolidator
from core.memory.visual_causal_cognition_pipeline import VisualCausalCognitionPipeline


def test_causal_graph_extractor():
    extractor = CausalGraphExtractor(spatial_dim=(64, 64), max_nodes=4)
    f0 = np.zeros((64, 64, 3), dtype=np.float32)
    f1 = np.zeros((64, 64, 3), dtype=np.float32)

    # Add motion block in frame 1
    f1[10:30, 10:30, :] = 1.0

    res = extractor.forward(f0, f1)

    assert "node_masks" in res
    assert "node_centers" in res
    assert "causal_edges" in res
    assert "pivot_points" in res
    assert "causal_graph" in res
    assert "causal_vector" in res

    assert res["node_centers"].shape == (4, 2)
    assert res["causal_edges"].shape == (4, 4)
    assert len(res["causal_graph"]["V"]) == 4


def test_intuition_dual_engine():
    engine = IntuitionDualEngine(dim=16, num_attractors=32, friction_threshold=0.20, random_seed=42)
    x = np.random.randn(16).astype(np.float32)

    res1 = engine.forward(x)
    assert "predicted_state" in res1
    assert "mode" in res1
    assert "friction" in res1

    # Check that after System 2 runs and consolidates Scar, friction decreases on re-exposure
    if "System_2" in res1["mode"]:
        # Re-run multiple times to observe Scar Consolidation pulling stimulus into System 1
        frictions = [res1["friction"]]
        for _ in range(10):
            res_next = engine.forward(x)
            frictions.append(res_next["friction"])
        assert frictions[-1] <= frictions[0]


def test_visual_causal_cognition_pipeline_end_to_end():
    pipeline = VisualCausalCognitionPipeline(
        spatial_dim=(64, 64),
        max_nodes=4,
        attr_dim=32,
        num_attractors=64,
        friction_threshold=0.25,
        random_seed=100,
    )

    f0 = np.random.rand(64, 64, 3).astype(np.float32)
    f1 = np.random.rand(64, 64, 3).astype(np.float32)

    output = pipeline.process_frames(f0, f1)

    assert "predicted_state" in output
    assert "mode" in output
    assert "friction" in output
    assert "causal_graph" in output
    assert "node_masks" in output
    assert "node_centers" in output
    assert "causal_edges" in output

    assert output["predicted_state"].shape == (32,)
    assert output["causal_graph"]["V"][0]["type"] == "V_bg"
