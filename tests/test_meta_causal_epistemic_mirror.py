"""
Unit and Integration Test Suite for Meta-Causal Epistemic Mirror Engine
=======================================================================
Verifies:
1. MetaCausalTrajectoryTensor: Recording steps, calculating struggle/spark metrics, and convergence tracking.
2. SynestheticTranslationEngine: Cognitive friction calculation and synesthetic bridge generation.
3. IsomorphicMirrorLayer: Synchronizing human cognitive phase dynamics with Elysia CausalField.
4. MetaCausalEpistemicMirror: Multi-cycle epistemic reflection and anchoring integration.
"""

import pytest
import numpy as np
from core.consciousness.meta_causal_epistemic_mirror import (
    MetaCausalTrajectoryTensor,
    SynestheticTranslationEngine,
    IsomorphicMirrorLayer,
    MetaCausalEpistemicMirror
)


def test_meta_causal_trajectory_tensor():
    dimension = 32
    tensor = MetaCausalTrajectoryTensor(dimension=dimension, history_capacity=50)

    # Record 10 mock cognitive steps
    for i in range(10):
        phase_state = np.sin(np.linspace(0, np.pi, dimension) + i * 0.1)
        friction = 0.8 - i * 0.07
        resonance = 0.2 + i * 0.07
        err_vec = np.ones(dimension) * (0.8 - i * 0.07)

        tensor.record_step(phase_state, friction, resonance, err_vec)

    metrics = tensor.compute_trajectory_metrics()
    assert "total_friction_struggle" in metrics
    assert "intuitive_spark_count" in metrics
    assert "topological_invariant_norm" in metrics
    assert metrics["topological_invariant_norm"] > 0.0
    assert len(tensor.phase_trajectory_history) == 10


def test_synesthetic_translation_engine():
    dimension = 32
    engine = SynestheticTranslationEngine(dimension=dimension)

    source_phase = np.zeros(dimension)
    target_phase = np.ones(dimension) * (np.pi / 2.0)  # Divergence ~ pi/2

    friction = engine.compute_cognitive_friction(source_phase, target_phase)
    assert friction["total_cognitive_friction"] > 0.0
    assert len(friction["disconnect_indices"]) > 0

    bridge = engine.generate_synesthetic_bridge(
        source_phase=source_phase,
        target_phase=target_phase,
        friction_analysis=friction,
        context_label="Protein Folding Observation"
    )

    assert "metaphor_type" in bridge
    assert "bridge_steering_vector" in bridge
    assert len(bridge["bridge_steering_vector"]) == dimension


def test_isomorphic_mirror_layer_cycle():
    dimension = 32
    layer = IsomorphicMirrorLayer(dimension=dimension)

    # Single cycle test
    sensory_input = np.random.uniform(0, 2 * np.pi, dimension)
    result = layer.reflect_and_synchronize(external_human_input_signal=sensory_input)

    assert "order_parameter_R" in result
    assert "cognitive_friction" in result
    assert "trajectory_metrics" in result
    assert "isomorphism_score" in result
    assert result["isomorphism_score"] >= 0.0
    assert len(layer.causal_field.voxels) > 0


def test_meta_causal_epistemic_mirror_full_pipeline():
    dimension = 32
    mirror = MetaCausalEpistemicMirror(dimension=dimension)

    # Run 10 reflection cycles
    for cycle in range(10):
        input_sig = np.random.normal(0, 0.5, dimension)
        res = mirror.process_epistemic_reflection(sensory_input=input_sig, concept_label="Causal Gravity")
        assert res["order_parameter_R"] is not None

    status = mirror.get_epistemic_status()
    assert status["reflection_counter"] == 10
    assert status["dimension"] == dimension
    assert status["isomorphism_score"] >= 0.0


if __name__ == "__main__":
    pytest.main([__file__])
