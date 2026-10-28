"""
tests/test_cellular_binding_engine.py
=======================================
Comprehensive test suite for the Cellular Binding Engine:
1. "Blue Apple" Scenario:
   - Tests disentanglement between Invariant Core (apple shape/density) and Attribute Spectrum (blue color).
   - Verifies friction heat Q_heat generation when encountering a blue apple and subsequent manifold recalibration.
2. "Chili Powder (고춧가루)" Scenario:
   - Tests sensory receptor friction triggering dynamic branching of multiple new axes (pain nociception, salivary secretion, thermal heat).
3. "Transjective Virtual Sandbox" Imagination Loop:
   - Tests unconstrained imagination (counterfactual re-binding of tiger core + wing spectrum).
   - Verifies causal sequence preservation ([PROJECTION -> INTERVENTION -> FRICTION -> RECOGNITION]).
   - Tests topological refinement of vague imaginations into vivid energy wells.
"""

import sys
import os
import pytest
import numpy as np

# Ensure root and L1_causal_field paths are in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../L1_causal_field")))

from L1_causal_field.core.cellular_boundary.cellular_binding_engine import (
    InvariantCore,
    AttributeAxis,
    DisentangledManifold,
    TensionFrictionTensorField,
    DynamicBranchingEngine,
    TopologicalRefinementEngine,
    TransjectiveVirtualSandbox,
    CellularBindingEngine
)


def test_disentangled_manifold_blue_apple():
    """
    [검증 1: 파란 사과 시나리오 (Disentanglement & Recalibration)]
    - 사과의 본질적 핵(3D curvature & volume)과 색상 스펙트럼 축이 개별 분리됨.
    - 색상 축이 '빨강(0.8)'에서 '파랑(-0.9)'으로 급변하더라도 사과의 본질 핵 벡터는 훼손되지 않음.
    """
    apple_core = InvariantCore(
        core_id="apple_core_01",
        dimension=3,
        core_vector=np.array([1.0, 0.8, 1.2]),
        density=1.0,
        structural_volume=1.0
    )
    manifold = DisentangledManifold("apple_manifold", apple_core)

    color_axis = AttributeAxis(
        axis_id="color_spectrum",
        axis_name="Color Wavelength Spectrum",
        value=0.8,  # Red spectrum
        spectrum_range=(-1.0, 1.0)
    )
    manifold.add_attribute_axis(color_axis)

    # State before blue apple shock
    initial_full_state = manifold.project_full_state()
    assert len(initial_full_state) == 4
    assert np.array_equal(initial_full_state[:3], apple_core.core_vector)

    # Encounter Blue Apple: Recalibrate color spectrum axis
    manifold.set_axis_value("color_spectrum", -0.9)  # Blue spectrum

    blue_full_state = manifold.project_full_state()
    # Invariant core remains exactly unchanged!
    assert np.array_equal(blue_full_state[:3], apple_core.core_vector)
    # Attribute spectrum coordinate updated
    assert color_axis.value == -0.9


def test_chili_powder_multi_axis_branching():
    """
    [검증 2: 고춧가루 마찰 시나리오 (Cellular Friction & Multi-Axis Branching)]
    - 고춧가루 가루(데이터)가 입안 수용체와 결합/마찰을 일으킬 때
    - 단일 데이터가 아닌 화끈거리는 통각 축, 침 분비 축, 온도/열감 축이 동시 발아됨.
    """
    engine = CellularBindingEngine()

    tongue_core = InvariantCore(
        core_id="tongue_receptor_core",
        dimension=2,
        core_vector=np.array([0.5, 0.5])
    )
    manifold = DisentangledManifold("tongue_manifold", tongue_core)
    engine.register_manifold(manifold)

    # Expectation: Bland neutral taste [0.0, 0.0]
    expectation = np.array([0.0, 0.0])
    # Reality: High capsaicin shock input [2.5, 3.0]
    reality = np.array([2.5, 3.0])

    result = engine.perceive_and_bind(
        manifold_id="tongue_manifold",
        expectation_state=expectation,
        reality_state=reality,
        context="chili_powder_receptor_friction"
    )

    # High cognitive friction heat generated
    assert result["q_heat"] > 1.0
    assert result["delta_norm"] > 1.0
    # Spurred 3 new sensory-motor axes (Pain, Saliva, Thermal Heat)
    assert result["spurred_axes_count"] == 3
    assert result["total_attribute_axes"] == 3

    # Check sprouted axes
    assert "axis_pain_nociception" in manifold.attribute_axes
    assert "axis_salivary_secretion" in manifold.attribute_axes
    assert "axis_thermal_heat" in manifold.attribute_axes


def test_transjective_virtual_sandbox_imagination():
    """
    [검증 3: 가상 투영 샌드박스 (Imagination & Refinement)]
    - 상상력 원리: 호랑이 본질 핵 + 새 날개/가벼움 축 가상 결합 시뮬레이션.
    - 인과적 서순([PROJECTION -> INTERVENTION -> FRICTION -> RECOGNITION]) 완결성 검증.
    - 막연한 상상이 집중/연마를 통해 선명한 상태로 구체화(Refinement)됨을 검증.
    """
    engine = CellularBindingEngine()

    tiger_core = InvariantCore(
        core_id="tiger_core",
        dimension=4,
        core_vector=np.array([2.0, 1.5, 3.0, 0.8]),  # Heavy muscle & mass
        density=2.5
    )

    wing_axis = AttributeAxis(
        axis_id="wing_span_spectrum",
        axis_name="Bird Wing Span & Aerodynamics",
        value=0.9,
        binding_tightness=0.2  # Initially vague/fuzzy imagination
    )

    # Simulate in virtual sandbox
    sandbox_res = engine.imagine_in_sandbox(
        core=tiger_core,
        axes=[wing_axis],
        virtual_action="bind_tiger_core_with_bird_wings"
    )

    assert sandbox_res["core_id"] == "tiger_core"
    assert sandbox_res["bound_axes_count"] == 1

    # Verify causal sequence steps
    seq_steps = [step["step"] for step in sandbox_res["causal_sequence"]]
    expected_sequence = ["1_PROJECTION", "2_INTERVENTION", "3_FRICTION", "4_RECOGNITION"]
    assert seq_steps == expected_sequence

    # Verify vividness gain (vague -> refined)
    proj_step = sandbox_res["causal_sequence"][0]
    recog_step = sandbox_res["causal_sequence"][3]
    assert proj_step["initial_vividness"] == 0.2
    assert recog_step["final_vividness"] > proj_step["initial_vividness"]


def test_tension_friction_tensor_field_physics():
    """
    [검증 4: 장력-마찰 텐서 필드 물리적 보존성]
    - 내면의 장력 텐서(Phi_internal) 및 누적 충격열(Q_heat) 보존 테스트.
    """
    field = TensionFrictionTensorField(dimension=3)

    exp = np.array([1.0, 1.0, 1.0])
    real = np.array([2.0, 1.0, 3.0])

    res = field.compute_interaction(exp, real, interaction_weight=1.5)

    assert res["delta_norm"] > 0.0
    assert field.accumulated_heat == res["q_heat"]
    assert np.linalg.norm(field.internal_tension) > 0.0
