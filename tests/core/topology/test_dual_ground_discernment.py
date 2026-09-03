"""
Unit tests for Dual Ground & Structural Emotion Topology Module and Reflection Engine
"""

import pytest
import numpy as np

from core.topology.dual_ground_discernment import (
    DualGroundDiscernmentEngine,
    GroundBlueprint,
    QualiaExperience,
    RemeltingTransition
)
from core.consciousness.dual_ground_reflection import DualGroundReflectionEngine


def test_ground_blueprint_refraction():
    gb = GroundBlueprint(
        name="test_ground",
        name_ko="테스트 지반",
        impedance=0.2,
        phase_velocity=1.0,
        entropy_gradient=0.1,
        emotional_bias_vector=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        structural_rotor_theta=0.0
    )
    stimulus = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    refracted = gb.compute_response_trajectory(stimulus)
    assert refracted.shape == (3,)
    assert not np.isnan(refracted).any()


def test_qualia_experience():
    engine = DualGroundDiscernmentEngine()
    stimulus = np.array([1.0, 0.5, 0.2], dtype=np.float32)
    qualia = engine.Experience_Qualia(stimulus, stimulus_intensity=1.5)
    assert isinstance(qualia, QualiaExperience)
    assert qualia.internal_stress >= 0.0
    assert qualia.emotional_state in ["FEAR_THREAT", "CURIOSITY_DESIRE", "JOY_RELIEF"]
    assert len(qualia.meta_observation_narrative) > 0


def test_dual_ground_metrics():
    engine = DualGroundDiscernmentEngine()
    stimulus = np.array([0.8, -0.4, 0.1], dtype=np.float32)
    sim_iso, dist_aniso, d_topo = engine.Calculate_Dual_Ground_Metrics(stimulus)
    assert 0.0 <= sim_iso <= 1.0
    assert dist_aniso >= 0.0
    assert d_topo >= 0.0


def test_remelting_transition():
    engine = DualGroundDiscernmentEngine(remelting_threshold=0.3)
    principle_A = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    principle_B = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    stimulus_1 = np.array([0.0, 0.0, 2.0], dtype=np.float32) # Strong orthogonal stimulus causing high friction

    transition = engine.Process_Remelting_And_Realignment(principle_A, principle_B, stimulus_1)
    assert isinstance(transition, RemeltingTransition)
    assert transition.remelting_occurred is True
    assert transition.post_realignment_friction < transition.initial_friction


def test_dual_ground_reflection_engine():
    refl_engine = DualGroundReflectionEngine()
    stimulus = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    principle_A = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    principle_B = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    result = refl_engine.process_and_reflect(
        stimulus_vector=stimulus,
        stimulus_intensity=1.2,
        principle_A=principle_A,
        principle_B=principle_B,
        principle_names=("Rule_A", "Rule_B"),
        stimulus_id="Stimulus_1"
    )

    assert "metacognitive_reflection" in result
    narrative = result["metacognitive_reflection"]
    assert "단일 우주 기저와 이중 참조 지반 자각" in narrative
    assert "0_machine" in narrative
    assert "0_human" in narrative
