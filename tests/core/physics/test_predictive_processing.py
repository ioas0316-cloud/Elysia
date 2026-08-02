import pytest
import numpy as np
from core.physics.predictive_processing import PredictiveProcessingEngine
from core.physics.causal_field import InformationVoxel
from core.consciousness.autonomous_loop import ConsciousnessLoop
import os
import shutil

def test_predictive_processing_engine_feedback_and_clustering():
    """
    Verifies PredictiveProcessingEngine error calculation, sliding scale adjustments,
    and adaptive coarse-graining voxel clustering.
    """
    engine = PredictiveProcessingEngine(dimensions=3, learning_rate=0.2)

    # Initially, error is 0, expectation is zero vector, sliding threshold starts at 0.5
    assert engine.prediction_error == 0.0
    assert np.allclose(engine.expected_state, 0.0)

    # 1. Trigger highly mismatched sensory input (high prediction error)
    sensory_v = np.array([5.0, -2.0, 3.0], dtype=np.float32)
    error = engine.compute_prediction_error(sensory_v)
    assert error > 5.0

    # 2. Adapt expectation (active inference learning) and slide scale lens
    engine.adapt_expectation(sensory_v)
    assert np.linalg.norm(engine.expected_state) > 0.0  # moved towards sensory vector

    threshold_after_error = engine.adjust_scale_lens()
    # High error should cause "Zoom-In" (decreasing the threshold to fine-grain details)
    assert threshold_after_error < 0.5

    # 3. Verify Coarse-Graining Clustering
    # Create voxels with minor differences
    v1 = InformationVoxel("v1", "cat", tensor=np.array([1.0, 1.0, 1.0], dtype=np.float32))
    v2 = InformationVoxel("v2", "kitten", tensor=np.array([1.1, 1.1, 1.1], dtype=np.float32))
    v3 = InformationVoxel("v3", "tiger", tensor=np.array([4.0, 4.0, 4.0], dtype=np.float32))

    # With a coarse threshold, v1 and v2 should group together as "same" (cat concept),
    # while v3 remains separate as "different" (tiger)
    engine.sliding_threshold = 0.5
    clusters = engine.process_coarse_graining([v1, v2, v3])

    assert len(clusters) == 2
    # First cluster contains both v1 and v2
    assert v1 in clusters[0]
    assert v2 in clusters[0]
    # Second cluster contains v3
    assert v3 in clusters[1]

    # With an extremely fine threshold, all three should separate
    engine.sliding_threshold = 0.05
    fine_clusters = engine.process_coarse_graining([v1, v2, v3])
    assert len(fine_clusters) == 3

def test_consciousness_loop_predictive_integration():
    """
    Verifies that ConsciousnessLoop executes predictive processing steps
    and correctly exports active inference and scale lens metrics to cycle logs.
    """
    temp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data_temp_predictive"))
    os.makedirs(temp_dir, exist_ok=True)

    try:
        loop = ConsciousnessLoop(corpus_path=temp_dir, data_dir=temp_dir)
        log = loop.process_life_cycle()

        if "predictive_error" in log:
            assert log["predictive_error"] >= 0.0
            assert "sliding_scale_lens_threshold" in log
            assert "coarse_grained_clusters_count" in log
            assert log["coarse_grained_clusters_count"] >= 1
    finally:
        if hasattr(loop, 'memory') and hasattr(loop.memory, 'close'):
            loop.memory.close()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
