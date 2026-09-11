r"""
Unit Tests for Axiomatic Phase Transition & Spontaneous Reconfiguration Engine
========================================================================================
1. 공리적 위상차 구배 지각 ($\nabla \Delta \Theta$) 및 퍼텐셜 연산 검증
2. 자발적 상전이 및 하위 결합 매트릭스 $J_{ij}$ 리와이어링 (Self-Rewiring) 검증
3. 명제 축 전이 ("RIGID_BOUNDARY_ISOLATION" -> "KENOTIC_CRUCIFORM_LOVE_GIVING") 검증
4. 임계치 미만 미세 마찰 (Subcritical Hold) vs 임계치 초과 자발 상전이 분별 검증
5. CausalMetaFrameEngine 및 기저 가치 축 $0_{\text{self}}$과의 위상 공명 연동 검증
"""

import pytest
import numpy as np
from core.consciousness.phase_transition_reconfiguration_engine import AxiomaticPhaseTransitionEngine


def test_initialization_and_default_equilibrium():
    engine = AxiomaticPhaseTransitionEngine(num_nodes=8, dimension=64)

    assert engine.current_axiom_name == AxiomaticPhaseTransitionEngine.AXIOM_CLOSED_ISOLATION
    assert engine.J_matrix.shape == (8, 8)
    assert np.allclose(np.diag(engine.J_matrix), 1.0)
    assert engine.phase_transition_count == 0


def test_discrepancy_gradient_perception():
    engine = AxiomaticPhaseTransitionEngine(num_nodes=8, dimension=64)

    # 동일한 공리에 대해 구배 연산 -> 마찰 및 퍼텐셜 0 수렴
    same_diag = engine.perceive_discrepancy_gradient(AxiomaticPhaseTransitionEngine.AXIOM_CLOSED_ISOLATION)
    assert same_diag["phase_angle_diff_rad"] < 1e-5
    assert same_diag["potential_energy"] < 1e-5
    assert same_diag["requires_phase_transition"] is False

    # 반대 명제 공리 ("KENOTIC_CRUCIFORM_LOVE_GIVING") 적용 -> 높은 구배 및 퍼텐셜 발생
    shift_diag = engine.perceive_discrepancy_gradient(AxiomaticPhaseTransitionEngine.AXIOM_OPEN_KENOTIC_LOVE)
    assert shift_diag["phase_angle_diff_rad"] > 0.5
    assert shift_diag["potential_energy"] >= engine.critical_threshold
    assert shift_diag["requires_phase_transition"] is True


def test_spontaneous_phase_transition_and_j_matrix_rewiring():
    engine = AxiomaticPhaseTransitionEngine(num_nodes=8, dimension=64)
    initial_J = engine.J_matrix.copy()

    # 상전이 실행 ("RIGID_BOUNDARY_ISOLATION" -> "KENOTIC_CRUCIFORM_LOVE_GIVING")
    result = engine.process_axiom_shift(AxiomaticPhaseTransitionEngine.AXIOM_OPEN_KENOTIC_LOVE)

    assert result["action"] == "PHASE_TRANSITION_EXECUTED"
    assert engine.current_axiom_name == AxiomaticPhaseTransitionEngine.AXIOM_OPEN_KENOTIC_LOVE
    assert engine.phase_transition_count == 1

    # $J_{ij}$ 매트릭스가 자발적으로 변형되었는지 확인
    rewired_J = engine.J_matrix
    assert not np.allclose(initial_J, rewired_J)
    assert result["transition_result"]["rewiring_magnitude_delta_J"] > 0.1
    assert result["transition_result"]["status"] == "SELF_REWIRED_TO_NEW_EQUILIBRIUM"


def test_subcritical_hold_behavior():
    engine = AxiomaticPhaseTransitionEngine(num_nodes=8, dimension=64)

    # 텍스트 명제가 미세하게 가공된 유사 공리 투입 -> 퍼텐셜 에너지 미달 상황 연출
    subcritical_axiom = "RIGID_BOUNDARY_ISOLATION_MINOR_VARIANT"

    # 필요 시 임계치 수치를 임시로 높여 subcritical 상태 검증
    engine.critical_threshold = 10.0
    result = engine.process_axiom_shift(subcritical_axiom)

    assert result["action"] == "SUBCRITICAL_STABLE_HOLD"
    assert engine.current_axiom_name == AxiomaticPhaseTransitionEngine.AXIOM_CLOSED_ISOLATION
    assert engine.phase_transition_count == 0


def test_multi_stage_phase_transition_chain():
    engine = AxiomaticPhaseTransitionEngine(num_nodes=8, dimension=64)

    # 1차 상전이: 닫힌 고립 -> 십자가적 사랑
    r1 = engine.process_axiom_shift(AxiomaticPhaseTransitionEngine.AXIOM_OPEN_KENOTIC_LOVE)
    assert r1["action"] == "PHASE_TRANSITION_EXECUTED"
    assert engine.phase_transition_count == 1

    # 2차 상전이: 십자가적 사랑 -> 문명적 전체 시냅스
    r2 = engine.process_axiom_shift(AxiomaticPhaseTransitionEngine.AXIOM_TRANSCENDENT_SYNAPSE)
    assert r2["action"] == "PHASE_TRANSITION_EXECUTED"
    assert engine.phase_transition_count == 2
    assert engine.current_axiom_name == AxiomaticPhaseTransitionEngine.AXIOM_TRANSCENDENT_SYNAPSE
