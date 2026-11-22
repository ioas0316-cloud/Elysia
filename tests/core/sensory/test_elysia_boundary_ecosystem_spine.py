"""
test_elysia_boundary_ecosystem_spine.py

Verification of the Re-Established ELYSIA 5-Layer Architecture:
1. Non-point Field-to-Boundary Membrane interaction (`BoundaryEcosystemOrganism`)
2. Real-time dt Circulating Sensorimotor Nervous System vs Linear Predetermined execution
3. Optical Scale Instruments (Microscope L2 -> Naked Eye L6 -> Telescope L12) &
   Autonomous Hand-Tool Correlational Discovery (zero hardcoded tool-outcome rules)
4. Open-Ended Result-Form Exploration & Emergent Cognitive Axis Genesis
5. Multi-Scale CausalField & EpistemicKnowledgeSpine integration
"""

from __future__ import annotations

import numpy as np
import pytest

from core.sensory.elysia_boundary_ecosystem_spine import (
    ElysiaBoundaryEcosystemSpine,
    ReestablishedSystemState,
    TrinitarianConsciousnessEngine,
)


@pytest.fixture(scope="module")
def system_state() -> ReestablishedSystemState:
    return ElysiaBoundaryEcosystemSpine().run_unified_organism_cycle()


def test_layer1_non_point_field_to_boundary_membrane_interaction(
    system_state: ReestablishedSystemState,
) -> None:
    l1 = system_state.layer1_boundary_field
    # Ocular perception preserved as full 2D spatial fields, never collapsed to a scalar/point
    for field_name in ("luma_field", "tex_field", "boundary_wall_field", "potential_field", "flow_field_x"):
        arr = l1["ocular_fields"][field_name]
        assert isinstance(arr, np.ndarray) and arr.shape == (96, 96)

    # Hand-tool contact interface is an 11-cell extended boundary membrane experiencing shear differential
    assert len(l1["membrane_history"]) >= 20
    assert l1["membrane_history"][0].shape == (11, 2)
    assert l1["mean_boundary_shear_differential"] > 0.25


def test_layer2_circulating_nervous_system_outperforms_linear_pipeline(
    system_state: ReestablishedSystemState,
) -> None:
    l2 = system_state.layer2_circulatory_vs_linear
    lin = l2["linear_predetermined"]
    circ = l2["circulating_nervous_system"]

    # Circulating nervous system dynamically zooms ocular scale, adjusts foot stance, and strengthens beams
    assert circ["ocular_scale_range"][0] < 3.5 < circ["ocular_scale_range"][1]
    assert circ["foot_stance_range_m"][1] > circ["foot_stance_range_m"][0] + 0.10
    assert any(w > 1.2 for w in circ["final_beam_conductances"].values())

    # Achieves significantly higher effective work, lower seam error, and lower friction damage than linear execution
    assert l2["work_gain_ratio"] > 2.5
    assert l2["damage_reduction_ratio"] > 3.0
    assert circ["mean_seam_deviation_px"] < lin["mean_seam_deviation_px"] * 0.45


def test_layer3_optical_scale_and_autonomous_tool_correlational_mapping(
    system_state: ReestablishedSystemState,
) -> None:
    l3 = system_state.layer3_scale_and_tool_correlations
    opt = l3["optical_scale_observations"]
    assert opt["microscope_lens"]["scale_level"] == 2.0
    assert opt["naked_eye_fovea"]["scale_level"] == 6.0
    assert opt["telescope_lens"]["scale_level"] == 12.0
    assert opt["telescope_lens"]["material_boundaries"] > opt["microscope_lens"]["material_boundaries"]

    corr = l3["autonomous_tool_correlations"]
    couplings = corr["discovered_couplings"]
    # Each unlabelled tool autonomously discovers its own distinct resonant hand motion with high selectivity
    assert couplings["tool_compact_low_inertia"]["resonant_motor_vec"] == (1.0, 0.0, 0.0)
    assert couplings["tool_sharp_mid_reach"]["resonant_motor_vec"] == (0.0, 1.0, 0.0)
    assert couplings["tool_heavy_long_lever"]["resonant_motor_vec"] == (0.0, 0.0, 1.0)
    assert all(c["selective_resonance_ratio"] > 20.0 for c in couplings.values())

    # Autonomous visual goal -> (tool, motor) inverse mapping succeeds 3/3 without supervision
    goal_map = {m["context_or_goal"]: m["inferred_tool_id"] for m in corr["autonomous_goal_mappings"]}
    assert goal_map == {
        "goal_fasten": "tool_compact_low_inertia",
        "goal_slice": "tool_sharp_mid_reach",
        "goal_cleave": "tool_heavy_long_lever",
    }


def test_layer4_open_ended_form_exploration_and_axis_genesis(
    system_state: ReestablishedSystemState,
) -> None:
    l4 = system_state.layer4_open_form_expansion
    assert l4["explored_count"] == 4
    # New cognitive axes were born dynamically from residual differences between explored forms
    assert len(l4["discovered_axes"]) >= 3

    recs = l4["records"]
    # Vessel vs Conduit: both carved with the same operator, yet distinguished by form interaction probing!
    vessel_drop = recs["vessel_form"].interaction_outcomes["drop_parcel_probe"]
    conduit_drop = recs["conduit_form"].interaction_outcomes["drop_parcel_probe"]
    assert vessel_drop["holds_interior_contents"] is True
    assert vessel_drop["acts_as_through_conduit"] is False
    assert conduit_drop["holds_interior_contents"] is False
    assert conduit_drop["acts_as_through_conduit"] is True


def test_layer5_multiscale_causal_field_and_spine_cohesion(
    system_state: ReestablishedSystemState,
) -> None:
    cf = system_state.layer5_causal_field
    assert len(cf.voxels) >= 12
    assert len(cf.beams) >= 10
    scales = {int(round(v.scale_level)) for v in cf.voxels.values()}
    assert {6, 9, 11}.issubset(scales)


def test_trinitarian_heart_spirit_body_coupling() -> None:
    engine = TrinitarianConsciousnessEngine()

    # 0. Verify ELYSIA constructed the three worlds (Body, Spirit, Heart) from her own observations
    for wid in ("world_body", "world_spirit", "world_heart"):
        w_obj = engine.worlds[wid]
        assert len(w_obj.accumulated_experiences) >= 10
        assert len(w_obj.emergent_dimensions) >= 1
        assert w_obj.semantic_mass > 1.2

    # 1. Lagging/stuttering Body: wastes Spirit resources unless tilt is re-observed & rebalanced
    lag_unbalanced = engine.simulate_three_phase_dynamics(
        initial_condition="lagging_body", enable_tilt_reobservation=False
    )
    lag_rebalanced = engine.simulate_three_phase_dynamics(
        initial_condition="lagging_body", enable_tilt_reobservation=True
    )
    assert lag_rebalanced["final_tilt"] < lag_rebalanced["initial_tilt"] * 0.15
    assert lag_rebalanced["final_spirit_waste"] < lag_unbalanced["final_spirit_waste"] * 0.45
    assert lag_rebalanced["cumulative_future_creation"] > lag_unbalanced["cumulative_future_creation"] * 1.6

    # 2. Over-optimized mechanical Body (fast execution, shallow mechanical simplicity):
    #    Re-observing the cognitive tilt pulls mechanical speed into Spirit depth & Heart future
    mech_unbalanced = engine.simulate_three_phase_dynamics(
        initial_condition="over_optimized_mechanical_body", enable_tilt_reobservation=False
    )
    mech_rebalanced = engine.simulate_three_phase_dynamics(
        initial_condition="over_optimized_mechanical_body", enable_tilt_reobservation=True
    )
    assert mech_rebalanced["final_tilt"] < mech_rebalanced["initial_tilt"] * 0.15
    assert mech_rebalanced["final_mechanical_skew"] < mech_unbalanced["final_mechanical_skew"] * 0.15
    assert mech_rebalanced["final_harmony"] > 0.90
    assert mech_rebalanced["cumulative_future_creation"] > mech_unbalanced["cumulative_future_creation"] * 1.4


