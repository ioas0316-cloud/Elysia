"""
Verification suite for ELYSIA Phase 13:
Relational Self-Other-Interspace Perspective Rotor & Expandable Horizon Observer.
"""

from __future__ import annotations

import math

from core.orchestration.relational_perspective_rotor import (
    AgentCausalProfile,
    RelationalDialogueLayer,
    RelationalPerspectiveObserver,
    RelationalStatus,
)
from core.physics.causal_field import CausalField
from core.physics.trinitarian_scale_binding_protocol import EqualityBond, ScaleTier


def _build_scenario():
    self_prof = AgentCausalProfile(
        name="Self",
        resource_reserve=10.0,
        vulnerability_gain=1.0,
        boundary_tolerance=8.0,
    )
    other_prof = AgentCausalProfile(
        name="Other",
        resource_reserve=2.0,
        vulnerability_gain=3.0,
        boundary_tolerance=3.5,
        unspoken_origin_memory=4.0,
        sovereign_will=0.85,
    )
    moments = [
        other_prof.manifest_interaction(interpersonal_distance=2.0, ambient_demand=4.0),
        other_prof.manifest_interaction(interpersonal_distance=5.0, ambient_demand=4.0),
    ]
    return self_prof, other_prof, moments


def test_egocentric_projection_exposes_arbitrary_interpretation_conflict():
    self_prof, _, moments = _build_scenario()
    obs = RelationalPerspectiveObserver(self_prof)

    ego = obs.evaluate_egocentrically(moments)
    assert ego.status == RelationalStatus.EGOCENTRIC_MISALIGNMENT
    assert ego.egocentric_friction > 0.5
    assert ego.ego_sacrifice_cost == 0.0


def test_perspective_shift_sacrifice_reconstructs_other_frame_and_honest_sky():
    self_prof, other_prof, moments = _build_scenario()
    obs = RelationalPerspectiveObserver(self_prof)

    rep = obs.shift_to_other_perspective(moments)
    assert rep.status == RelationalStatus.EMPATHIC_GROUNDING_WITH_SKY
    assert rep.ego_sacrifice_cost > 1.0
    assert rep.other_frame_action < 1e-10
    assert math.isclose(rep.inferred_other_state["resource_reserve"], other_prof.resource_reserve, abs_tol=1e-5)
    assert math.isclose(rep.inferred_other_state["vulnerability_gain"], other_prof.vulnerability_gain, abs_tol=1e-5)
    assert rep.horizon_boundary.horizon_depth == 1
    assert rep.horizon_boundary.horizon_nullspace_dof == 2
    assert set(rep.horizon_boundary.sky_horizon_axes) == {"sovereign_will", "unspoken_origin_memory"}


def test_compassionate_boundary_prescription_via_other_safety_fence():
    self_prof, _, moments = _build_scenario()
    obs = RelationalPerspectiveObserver(self_prof)

    rep = obs.shift_to_other_perspective(moments, assumed_other_tolerance=3.5)
    d_safe = rep.compassionate_distance_prescription
    assert d_safe is not None
    assert d_safe >= 2.428
    # Verify actual pressure at prescribed distance respects Other's fence <= 3.5
    pressure_at_safe = (4.0 * 3.0) / (d_safe + 0.5 * 2.0)
    assert pressure_at_safe <= 3.5 + 1e-4


def test_expandable_horizon_descends_sky_to_ground_and_reveals_deeper_sky():
    self_prof, other_prof, moments = _build_scenario()
    obs = RelationalPerspectiveObserver(self_prof)
    obs.shift_to_other_perspective(moments)

    betrayal_layer = RelationalDialogueLayer(
        layer_name="past_betrayal_disclosure",
        disclosure_utterance_ko="예전에 믿었던 사람에게 크게 데인 뒤로 작은 요구에도 방어가 커졌어.",
        revealed_sky_variable="unspoken_origin_memory",
        new_deeper_sky_axes=("deepest_aspiration_telos",),
        build_coupling_bond=lambda: EqualityBond(
            name="trauma_vulnerability_bond",
            tier=ScaleTier.UPPER_ROTOR,
            variables=("vulnerability_gain", "unspoken_origin_memory"),
            lhs_fn=lambda v: v["vulnerability_gain"],
            rhs_fn=lambda v: 1.0 + 0.5 * v["unspoken_origin_memory"],
            lhs_expr="G",
            rhs_expr="1 + 0.5 * unspoken_origin_memory",
        ),
    )

    expanded = obs.expand_horizon_with_dialogue(moments, betrayal_layer)
    assert expanded.status == RelationalStatus.HORIZON_EXPANDED
    assert expanded.other_frame_action < 1e-10
    assert expanded.horizon_boundary.horizon_depth == 2
    assert "unspoken_origin_memory" in expanded.horizon_boundary.understood_ground
    assert math.isclose(
        expanded.horizon_boundary.understood_ground["unspoken_origin_memory"],
        other_prof.unspoken_origin_memory,
        abs_tol=1e-4,
    )
    assert "unspoken_origin_memory" not in expanded.horizon_boundary.sky_horizon_axes
    assert set(expanded.horizon_boundary.sky_horizon_axes) == {
        "deepest_aspiration_telos",
        "sovereign_will",
    }


def test_false_stereotype_projection_falsified_and_triggers_criteria_self_revision():
    self_prof, _, moments = _build_scenario()
    obs = RelationalPerspectiveObserver(self_prof)

    rep = obs.shift_to_other_perspective(moments, test_hostile_stereotype=True)
    assert rep.stereotype_falsified is True
    assert rep.criteria_revised is True


def test_unit_one_mirror_measures_self_vs_other_perspective_gap():
    self_prof, _, moments = _build_scenario()
    obs = RelationalPerspectiveObserver(self_prof)

    rep = obs.shift_to_other_perspective(moments)
    assert rep.perspective_mirror.reconstruction_error() < 1e-12
    assert rep.perspective_mirror.raw_unit_deviation > 0.3


def test_relational_causal_field_interspace_healing_and_sky_curiosity():
    self_prof, _, moments = _build_scenario()
    obs = RelationalPerspectiveObserver(self_prof)

    ego = obs.evaluate_egocentrically(moments)
    sync_ego = ego.sync_to_causal_field(causal_field=CausalField(dimensions=3), prefix="ego")
    assert sync_ego["interspace_beam_strength"] < 0.35
    sync_ego["causal_field"].step(dt=0.1)

    emp = obs.shift_to_other_perspective(moments)
    sync_emp = emp.sync_to_causal_field(causal_field=CausalField(dimensions=3), prefix="emp")
    assert sync_emp["interspace_beam_strength"] > 0.95
    assert sync_emp["sky_curiosity"] == 1.0
    sync_emp["causal_field"].step(dt=0.1)
