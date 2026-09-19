import pytest
import numpy as np
from synaptic_architecture.active_probing_cognition_engine import ActiveProbingCognitionEngine


def test_active_probing_boundary_dynamics():
    engine = ActiveProbingCognitionEngine(dim=3, viscosity=0.1)

    # 1. Action vector u
    u_vector = np.array([1.0, 0.0, 0.0])

    # 2. External stress tensor
    external_stress = np.array([
        [2.0, 0.1, 0.0],
        [0.1, 1.0, 0.0],
        [0.0, 0.0, 0.5]
    ])

    # Test c = 1.0 (Self -> zero topological friction)
    res_self = engine.apply_active_probing(u_vector, external_stress, control_c=1.0)
    assert np.allclose(res_self["topological_friction"], 0.0)

    # Test c = 0.0 (World -> maximum topological friction)
    res_world = engine.apply_active_probing(u_vector, external_stress, control_c=0.0)
    assert not np.allclose(res_world["topological_friction"], 0.0)
    assert res_world["g_metric"].shape == (3, 3)


def test_symbol_deconstruction_and_alignment():
    engine = ActiveProbingCognitionEngine(dim=3)

    # Deconstruct a custom symbol
    entry = engine.deconstruct_symbol(
        label="점탄성",
        action_profile={
            "delta_F_over_delta_x": 0.6,
            "delta_F_over_delta_v": 0.8,
            "control_c": 0.05
        },
        scale="physical"
    )

    assert entry["label"] == "점탄성"
    assert entry["stiffness"] == 0.6
    assert entry["viscosity"] == 0.8
    assert "점탄성" in engine.concept_registry


def test_unknown_concept_reverse_tracing():
    engine = ActiveProbingCognitionEngine(dim=3)

    unknown_stress = np.array([
        [1.5, 0.2, 0.0],
        [0.2, 0.8, 0.0],
        [0.0, 0.0, 0.2]
    ])

    trace_res = engine.trace_unknown_concept(unknown_stress)
    assert "connected_root" in trace_res
    assert trace_res["connected_root"] in engine.concept_registry
    assert trace_res["mismatch_norm"] >= 0.0


def test_four_stage_cognitive_operations():
    engine = ActiveProbingCognitionEngine(dim=3)

    u = np.array([0.5, 0.5, 0.0])
    ext_stress = np.eye(3) * 1.2

    # 1. Cognition
    cog_res = engine.cognition(u, ext_stress, control_c=0.1)
    assert "topological_friction" in cog_res

    # 2. Thought
    thought_stress = engine.thought(u)
    assert thought_stress.shape == (3, 3)

    # 3. Judgment
    judgment_res = engine.judgment(predicted_stress=thought_stress, actual_stress=ext_stress)
    assert "predictive_delta" in judgment_res
    assert "is_valid" in judgment_res

    # 4. Discrimination
    disc_res = engine.discrimination("단단함", "점성")
    assert disc_res["is_distinct"] is True
    assert disc_res["boundary_distance"] > 0.0
