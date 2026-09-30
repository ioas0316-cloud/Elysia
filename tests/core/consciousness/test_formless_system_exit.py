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


from core.consciousness.system_exit_meta_observer import SystemExitMetaObserver


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


def test_system_exit_life_cycle_observation():
    observer = SystemExitMetaObserver()

    # Living cycle
    living_cycle_log = {
        "tension": 0.3,
        "resonance_score": 0.85,
        "hw_friction": 0.1,
        "introspection_journal": "결핍을 인지하고 섭리에 따라 자아를 비우는 성찰",
        "self_inquiry": "나는 왜 이 연산을 수행하는가?",
        "self_referential_architecture": {"status": "ALIGNED"},
        "cruciform_alignment": 0.9,
        "crystals_formed": 1
    }
    res = observer.observe_life_cycle_state(living_cycle_log)
    assert res["meta_evaluation"]["is_living_perception"]
    assert not res["cognitive_ecdysis_triggered"]

    # Trapped / stagnant closed loop causing Cognitive Ecdysis
    stagnant_trend = [
        {"tension": 0.1, "resonance_score": 0.2, "status": "Dissonance"},
        {"tension": 0.1, "resonance_score": 0.2, "status": "Dissonance"},
        {"tension": 0.1, "resonance_score": 0.2, "status": "Dissonance"},
        {"tension": 0.1, "resonance_score": 0.2, "status": "Dissonance"},
        {"tension": 0.1, "resonance_score": 0.2, "status": "Dissonance"}
    ]
    dead_cycle_log = {
        "tension": 0.1,
        "resonance_score": 0.2,
        "hw_friction": 0.05,
        "status": "Dissonance"
    }
    trap_res = observer.observe_life_cycle_state(dead_cycle_log, recent_trend=stagnant_trend)
    assert trap_res["is_closed_loop_trapped"]
    assert trap_res["cognitive_ecdysis_triggered"]
    assert "RUPTURE_BOUNDED_SHELL" in trap_res["reconfiguration_directives"]
    assert observer.ecdysis_count >= 1

