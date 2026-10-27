"""
tests/core/embodied/test_embodied_sensorimotor_loop.py
======================================================
Unit tests for EmbodiedSensorimotorPredictionLoop (체화된 감각-운동 예측 및 재인식 루프)
"""

import numpy as np
import pytest
from L1_causal_field.core.embodied.embodied_sensorimotor_prediction_loop import (
    EmbodiedSensorimotorPredictionLoop,
    ProjectionWave,
    SensorimotorCollisionResult,
    RelationalRecalibrationOutput,
)


def test_projection_and_transjection():
    engine = EmbodiedSensorimotorPredictionLoop(vector_dim=8)
    intent = np.array([1.0, 0.5, -0.5, 0.2, 0.0, 0.0, 0.0, 0.0])

    projection = engine.project_causal_expectation(internal_intent=intent)

    assert isinstance(projection, ProjectionWave)
    assert len(projection.projection_vector) == 8
    assert projection.internal_tension == 1.0
    assert len(projection.intended_action_vector) == 8
    assert "내면 장력" in projection.causal_hypothesis


def test_sensorimotor_intervention_and_collision():
    engine = EmbodiedSensorimotorPredictionLoop(vector_dim=8)
    intent = np.ones(8)
    projection = engine.project_causal_expectation(internal_intent=intent)

    # 1. 2D Image array input
    img_data = np.random.randn(16, 16)
    collision_img = engine.execute_sensorimotor_intervention(
        projection=projection,
        external_input_data=img_data,
        modality_hint="image",
    )

    assert isinstance(collision_img, SensorimotorCollisionResult)
    assert len(collision_img.real_feedback_vector) == 8
    assert collision_img.cognitive_heat >= 0.0
    assert collision_img.organ_discernment is not None

    # 2. Mathematical equation/text string input
    formula_text = "E = mc^2"
    collision_formula = engine.execute_sensorimotor_intervention(
        projection=projection,
        external_input_data=formula_text,
        modality_hint="text_formula",
    )

    assert isinstance(collision_formula, SensorimotorCollisionResult)
    assert len(collision_formula.discrepancy_vector) == 8


def test_recalibration_and_branching():
    engine = EmbodiedSensorimotorPredictionLoop(vector_dim=8, heat_threshold_rupture=0.5, branching_threshold=0.3)
    intent = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Big discrepancy input to trigger cognitive heat rupture and branching
    unmatched_data = np.ones((32, 32)) * 10.0
    result = engine.run_full_sensorimotor_cycle(
        internal_intent=intent,
        external_input_data=unmatched_data,
        modality_hint="image",
    )

    assert isinstance(result, RelationalRecalibrationOutput)
    assert result.branching_factor_sprouted is True
    assert result.sprouted_branch_id is not None
    assert "분기 노드" in result.recalibration_narrative
    assert result.recalibrated_topology.shape == (8, 8)


def test_stabilization_on_low_discrepancy():
    engine = EmbodiedSensorimotorPredictionLoop(vector_dim=8)
    intent = np.zeros(8)

    # 투영 예측과 일치하는 외부 자극 피드백을 전달하여 오차 Δ < 0.4 달성
    # projection_vector for zero intent with eye topology will be zero
    zero_data = np.zeros(8)
    # Get projection
    proj = engine.project_causal_expectation(internal_intent=intent)
    # Execute cycle
    collision = engine.execute_sensorimotor_intervention(
        projection=proj,
        external_input_data=zero_data,
        modality_hint="data",
    )
    # Manually set zero discrepancy for perfect matching test
    collision.discrepancy_magnitude = 0.1
    collision.cognitive_heat = 0.1
    result = engine.recalibrate_relational_topology(collision)

    assert result.stabilization_score > 0.0
    assert result.branching_factor_sprouted is False
    assert "바위처럼 단단하게 고착" in result.recalibration_narrative
