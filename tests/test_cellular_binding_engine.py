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
4. "Dream & Relaxation State" Zero-Friction Topological Drift Mode:
   - Verifies external friction resistance W_real = 0 and Q_heat = 0.
   - Verifies zero-friction drift without structural rupture or errors.
5. "Multi-Scale Coupled Variational System" Framework:
   - Tests local Lagrangians (RBC, WBC, Eye, Brain) and coupling tensor Gamma_ij.
   - Verifies S_total = sum(L_i) + sum(Gamma_ij * <phi_i, phi_j>).
6. "Incongruity / Uncanniness Alarm (위화감 감지)":
   - Tests multi-axis conflict (auditory positive vs visual tremor/anxiety vs context tension).
   - Verifies Q_heat friction spike generation and dynamic spurring of "axis_incongruity_intuition".
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
    CellularBindingNode,
    DynamicBranchingEngine,
    TopologicalRefinementEngine,
    TransjectiveVirtualSandbox,
    DreamStateEngine,
    CoupledMultiScaleVariationalSystem,
    IncongruityDetector,
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


def test_dream_state_zero_friction_topological_drift():
    """
    [검증 4: 꿈 & 이완 표류 모드 (Dream State Zero-Friction Drift)]
    - 외부 마찰 저항 W_real = 0, Q_heat = 0
    - 세포 결합 노드 및 속성 축들이 저항 없이 자유 무중력 표류.
    - 비합리적 변화가 발생해도 structural_rupture = False, error_rate = 0.0 보장.
    """
    engine = CellularBindingEngine()

    core = InvariantCore("dreamer_core", 3, np.array([1.0, 1.0, 1.0]))
    manifold = DisentangledManifold("dream_manifold", core)

    flying_axis = AttributeAxis("flying_spectrum", "Levitation Axis", value=0.5, spectrum_range=(-10.0, 10.0))
    face_axis = AttributeAxis("father_face_spectrum", "Father Face Shift", value=0.0, spectrum_range=(-5.0, 5.0))
    manifold.add_attribute_axis(flying_axis)
    manifold.add_attribute_axis(face_axis)

    engine.register_manifold(manifold)

    nodes = [CellularBindingNode(f"node_{i}", position=np.array([0.0, 0.0, 0.0])) for i in range(3)]

    dream_res = engine.drift_in_dream_state("dream_manifold", steps=5, coupling_nodes=nodes)

    assert dream_res["dream_mode"] is True
    assert dream_res["external_friction"] == 0.0
    assert dream_res["q_heat"] == 0.0
    assert dream_res["structural_rupture"] is False
    assert dream_res["error_rate"] == 0.0
    assert len(dream_res["drift_history"]) == 5


def test_multi_scale_coupled_variational_system():
    """
    [검증 5: 다중 스케일 결합 변분 원리 (Multi-Scale Coupled Variational System)]
    - 각 국소 최소 작용 L_i (RBC, WBC, Eye, Brain) 계산
    - 세포 결합 커플링 텐서 Gamma_ij 계산
    - 전역 작용 S_total = sum(L_i) + sum(Gamma_ij * <phi_i, phi_j>) 검증
    """
    system = CoupledMultiScaleVariationalSystem()

    module_states = {
        "rbc_drag": np.array([1.2, 0.5, 0.8]),
        "wbc_search": np.array([0.4, 1.5, 0.2]),
        "eye_compression": np.array([2.0, 0.1, 1.1]),
        "brain_free_energy": np.array([0.1, 0.2, 0.3])
    }

    res = system.compute_total_action(module_states)

    assert "s_total" in res
    assert "local_actions" in res
    assert len(res["local_actions"]) == 4
    assert res["coupling_energy"] > 0.0
    assert res["s_total"] == res["sum_local_actions"] + res["coupling_energy"]


def test_incongruity_uncanniness_detection_and_branching():
    """
    [검증 6: 위화감 (Incongruity / Uncanniness Alarm) 및 동적 축 발아]
    - 청각(말: "괜찮아", +1.0) vs 시각(표정 떨림/불안, -0.9) vs 맥락 장력(-0.8) 충돌
    - Q_heat 스파이크 발생 및 위화감 직관 축("axis_incongruity_intuition") 자가 발아.
    """
    engine = CellularBindingEngine()

    core = InvariantCore("social_perception_core", 2, np.array([1.0, 1.0]))
    manifold = DisentangledManifold("social_manifold", core)
    engine.register_manifold(manifold)

    auditory_val = 1.0   # "I'm fine" (Positive words)
    visual_val = -0.9    # Micro-expression tremor (Anxiety)
    context_val = -0.8   # Heavy atmosphere

    inc_res = engine.detect_and_handle_incongruity(
        manifold_id="social_manifold",
        auditory_val=auditory_val,
        visual_val=visual_val,
        context_val=context_val
    )

    assert inc_res["incongruity_detected"] is True
    assert inc_res["q_heat_spike"] > 1.0
    assert inc_res["status"] == "UNCAUGHT_DECEPTIVE_INCONGRUITY_ALARM"
    assert "spurred_axes" in inc_res
    assert "axis_incongruity_intuition" in manifold.attribute_axes
    assert "axis_deception_sensitivity" in manifold.attribute_axes


def test_tension_friction_tensor_field_physics():
    """
    [검증 7: 장력-마찰 텐서 필드 물리적 보존성]
    - 내면의 장력 텐서(Phi_internal) 및 누적 충격열(Q_heat) 보존 테스트.
    """
    field = TensionFrictionTensorField(dimension=3)

    exp = np.array([1.0, 1.0, 1.0])
    real = np.array([2.0, 1.0, 3.0])

    res = field.compute_interaction(exp, real, interaction_weight=1.5)

    assert res["delta_norm"] > 0.0
    assert field.accumulated_heat == res["q_heat"]
    assert np.linalg.norm(field.internal_tension) > 0.0
