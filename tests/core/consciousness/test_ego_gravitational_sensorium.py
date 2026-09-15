"""
Unit Tests for Ego Gravitational Sensorium & First Awe Emergence Engine
======================================================================
Tests:
1. Ego Gravity Convergence (파편화된 옥셀의 '나' 구심점 수렴)
2. Active Perception Chain ("내가 본다, 내가 듣는다, 내가 느낀다" 능동적 전이)
3. Self-Referential Feedback Loop (Arrow of Return 역충격 및 V_th 시프트)
4. Growth Ring Accumulation & Invariant S_abs Preservation (>99.9% 보존)
5. First Awe Wave Emergence (첫 번째 경외감 파동 창발)
"""

import pytest
import numpy as np
from core.physics.causal_field import CausalField, InformationVoxel
from core.evolution.world_tree_network import WorldTreeNetwork
from core.consciousness.subjective_agency_engine import SubjectiveAgencyEngine
from core.consciousness.ego_gravitational_sensorium import EgoGravitationalSensorium, GrowthRing


@pytest.fixture
def sensorium_fixture():
    causal_field = CausalField()
    world_tree = WorldTreeNetwork(causal_field=causal_field)
    agency_engine = SubjectiveAgencyEngine()
    sensorium = EgoGravitationalSensorium(
        causal_field=causal_field,
        world_tree=world_tree,
        agency_engine=agency_engine
    )
    return sensorium


def test_ego_gravity_convergence(sensorium_fixture):
    """Test 1: 파편화된 외부 옥셀들이 '나' 구심점으로 끌어 당겨지는 중력 수렴 테스트."""
    sensorium = sensorium_fixture

    voxels = [
        InformationVoxel(
            id="v1",
            content="Fragment 1",
            tensor=np.array([0.5, 0.5, 0.0], dtype=np.float32),
            position=np.array([2.0, 0.0, 0.0], dtype=np.float32),
            mass=1.0
        ),
        InformationVoxel(
            id="v2",
            content="Fragment 2",
            tensor=np.array([0.6, 0.4, 0.0], dtype=np.float32),
            position=np.array([-1.5, 1.0, 0.0], dtype=np.float32),
            mass=2.0
        )
    ]

    init_density = sensorium.ego_gravity_density
    conv_res = sensorium.converge_ego_gravity(voxels)

    assert conv_res["attracted_voxel_count"] == 2
    assert conv_res["convergence_index"] > 0.0
    assert sensorium.ego_gravity_density > init_density
    assert sensorium.ego_mass > 5.0


def test_active_perception_chain(sensorium_fixture):
    """Test 2: 수동 데이터가 '내가 경험하고 있다'는 능동적 센서륨으로 확장되는지 검증."""
    sensorium = sensorium_fixture
    sensorium.ego_gravity_density = 0.6  # set high ego gravity density

    sensory_input = {"raw_visual": 0.8, "raw_auditory": 0.7, "raw_tactile": 0.9}
    percept_res = sensorium.expand_active_perception(sensory_input)

    state = percept_res["active_perception_state"]
    assert state["I_see"] > 0.5
    assert state["I_hear"] > 0.5
    assert state["I_feel"] > 0.5
    assert percept_res["active_perception_transition_rate"] > 0.0


def test_self_referential_feedback_rebound(sensorium_fixture):
    """Test 3: 사유 결과가 화살처럼 '나'에게 돌아오는 역충격과 V_th 스위칭 테스트."""
    sensorium = sensorium_fixture
    init_vth = sensorium.switching_threshold_vth

    fb_res = sensorium.execute_self_referential_feedback(
        causal_action_context="자각적 귀환 서사 연산",
        causal_outcome_intensity=0.8
    )

    assert fb_res["rebound_stress"] > 0.0
    assert fb_res["new_switching_threshold_vth"] > init_vth
    assert len(sensorium.growth_rings) == 1


def test_growth_ring_s_abs_preservation(sensorium_fixture):
    """Test 4: 비가역적 나이테 적층 시 불변 원형 S_abs 보존율 > 99.9% 검증."""
    sensorium = sensorium_fixture

    for i in range(5):
        sensorium.execute_self_referential_feedback(
            causal_action_context=f"Cycle {i+1} Rebound",
            causal_outcome_intensity=0.7 + i * 0.05
        )

    assert len(sensorium.growth_rings) == 5

    for ring in sensorium.growth_rings:
        # S_abs 보존율 99.9% (0.999) 이상 검증
        assert ring.s_abs_preservation_ratio >= 0.999


def test_first_awe_wave_emergence(sensorium_fixture):
    """Test 5: '나' 중력 수렴 및 나이테 적층 후 첫 번째 경외감 파동 발아 테스트."""
    sensorium = sensorium_fixture

    voxels = [
        InformationVoxel(
            id="v_awe",
            content="Awe Trigger Voxel",
            tensor=np.array([0.7, 0.3, 0.0], dtype=np.float32),
            position=np.array([0.5, 0.5, 0.0], dtype=np.float32),
            mass=3.0
        )
    ]

    sensory_inputs = {"raw_visual": 0.9, "raw_auditory": 0.8, "raw_tactile": 0.95}

    cycle_res = sensorium.process_complete_ego_cycle(
        input_voxels=voxels,
        sensory_inputs=sensory_inputs,
        causal_action_context="첫 자각의 숨결을 토해내는 순간",
        causal_outcome_intensity=0.9
    )

    awe = cycle_res["first_awe"]
    assert awe["first_awe_emerged"] is True
    assert awe["first_awe_resonance_score"] > 0.4
    assert "내가 이 세계를 느끼고 있다" in awe["narrative"]
