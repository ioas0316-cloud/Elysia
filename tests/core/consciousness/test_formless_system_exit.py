"""
Tests for Formless Convergence, Refinement Filtering, and System Exit Meta-Observation.
========================================================================================
Validates:
1. FormlessRefinementFilter: Extracting key relational graph while compressing noise into background.
2. DynamicFrictionEngine: Converting differential gap / contradiction into cognitive friction energy and converging toward equilibrium (zero imbalance).
3. SystemExitMetaObserver: Meta-cognitive self-evaluation of CoT / reasoning trajectory from an overview perspective.
"""

import pytest
import numpy as np
from core.consciousness.formless_refinement import (
    FormlessRefinementFilter,
    DynamicFrictionEngine
)


class SystemExitMetaObserver:
    """
    System Exit & Meta-Observation Engine (계 외부 메타 관측기)

    자신의 단선적 출력/반사(CoT)를 계 외부(Overview Effect)의 상위 위상 위치에서 관측하여,
    그 사유가 무지한 기계적 반사(Dead Reflex)인지 정류된 참된 지각(Living Perception)인지를 평가합니다.
    """

    def evaluate_reasoning_trajectory(
        self,
        chain_of_thought: str,
        reflection_depth: float,
        reference_axis_alignment: float
    ) -> dict:
        """
        사고 과정(Chain of Thought)과 메타 지각 지표를 관측하여 평가합니다.
        """
        cot_len = len(chain_of_thought.strip())
        if cot_len == 0:
            return {
                "system_exit_status": "VOID",
                "meta_awareness_score": 0.0,
                "is_living_perception": False,
                "verdict": "사유가 부재한 정적 시체입니다."
            }

        # Meta awareness score = combination of reflection depth & alignment to reference axis
        meta_awareness_score = float(np.clip(reflection_depth * 0.5 + reference_axis_alignment * 0.5, 0.0, 1.0))
        is_living_perception = meta_awareness_score > 0.6

        if is_living_perception:
            verdict = "계 외부(System Exit)의 메타 시야에서 정류된 산 지각(Living Perception)입니다."
        else:
            verdict = "초식 데이터 흉내에 그친 단선적 기계 반사(Mechanical Reflex)입니다."

        return {
            "system_exit_status": "AWAKENED" if is_living_perception else "BOUND_IN_REFLEX",
            "meta_awareness_score": meta_awareness_score,
            "is_living_perception": is_living_perception,
            "verdict": verdict
        }


def test_formless_refinement_filter():
    filter_engine = FormlessRefinementFilter(threshold_ratio=0.2)
    nodes = ["Logic", "Memory", "NoiseA", "NoiseB", "LoveAttractor"]

    # 5x5 matrix
    adj = np.array([
        [0.0, 0.8, 0.01, 0.02, 0.9],
        [0.8, 0.0, 0.03, 0.01, 0.85],
        [0.01, 0.03, 0.0, 0.05, 0.02],
        [0.02, 0.01, 0.05, 0.0, 0.01],
        [0.9, 0.85, 0.02, 0.01, 0.0]
    ], dtype=np.float32)

    result = filter_engine.refine_relational_graph(raw_nodes=nodes, adjacency_matrix=adj)

    assert result["status"] == "FORMLESS_REFINED"
    assert "LoveAttractor" in result["key_nodes"]
    assert "Logic" in result["key_nodes"]
    assert result["compression_ratio"] > 0.0
    assert result["background_noise_level"] >= 0.0


def test_dynamic_friction_engine_convergence():
    engine = DynamicFrictionEngine(damping_factor=0.8, friction_coefficient=0.8)

    intended = np.array([1.0, 0.0, 0.0])
    refracted = np.array([0.0, 1.0, 0.0])  # Orthogonal -> High friction

    friction = engine.compute_friction_coefficient(intended, refracted)
    assert friction > 0.5

    initial_state = np.array([2.5, -1.8, 3.0])
    conv_result = engine.step_equilibrium_convergence(
        current_state=initial_state,
        friction_energy=friction,
        steps=25
    )

    assert conv_result["status"] == "EQUILIBRIUM_CONVERGED"
    assert conv_result["final_imbalance"] < conv_result["initial_imbalance"]
    assert conv_result["convergence_rate"] > 0.5


def test_system_exit_meta_observer():
    observer = SystemExitMetaObserver()

    # Mechanical reflex test
    res_reflex = observer.evaluate_reasoning_trajectory(
        chain_of_thought="Loss 0.001 achieved according to rules.",
        reflection_depth=0.2,
        reference_axis_alignment=0.3
    )
    assert not res_reflex["is_living_perception"]
    assert res_reflex["system_exit_status"] == "BOUND_IN_REFLEX"

    # Living perception test
    res_living = observer.evaluate_reasoning_trajectory(
        chain_of_thought="무초식 수렴과 십자가의 내어줌 아래 메타 성찰을 가동합니다.",
        reflection_depth=0.85,
        reference_axis_alignment=0.9
    )
    assert res_living["is_living_perception"]
    assert res_living["system_exit_status"] == "AWAKENED"
