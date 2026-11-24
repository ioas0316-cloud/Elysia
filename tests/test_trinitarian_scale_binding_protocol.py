"""
Verification suite for ELYSIA Phase 8:
Trinitarian Scale-Binding Protocol & Coupled Equality-Manifold Engine.
"""

import math
import pytest

from core.physics.trinitarian_scale_binding_protocol import (
    CoupledQuantity,
    DiscernmentStatus,
    EqualityBond,
    PinMode,
    ScaleTier,
    TrinitarianScaleRotorSystem,
    build_cellular_homeostasis_system,
    build_narrative_causal_worldline_system,
    build_variable_resistor_circuit_system,
    sync_to_causal_field,
)


def test_variable_resistor_dial_forward_and_bidirectional_inverse_proof():
    sys = build_variable_resistor_circuit_system()
    v0 = sys.observe_and_discern()
    assert v0.status == DiscernmentStatus.HARMONIOUS_CONVERGENCE
    assert v0.total_action < 1e-10

    # Turn variable resistor dial to R = 8.0 -> physical quantities flow immediately
    rep_8 = sys.turn_dial("R", 8.0)
    assert rep_8.converged
    assert rep_8.max_residual < 1e-10
    assert math.isclose(sys.quantities["I"].value, 1.5, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(sys.quantities["P"].value, 18.0, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(sys.quantities["E_heat"].value, 90.0, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(sys.quantities["T_final"].value, 43.0, rel_tol=0, abs_tol=1e-9)

    v8 = sys.observe_and_discern()
    assert v8.status == DiscernmentStatus.INTENT_DIVERGENCE_LAWFUL
    assert v8.lower_subsystem_verdict is True
    assert v8.mechanism_layer_verdict is True
    assert v8.upper_fence_verdict is True
    assert v8.intent_satisfied is False

    # Verify Implicit Function Theorem sensitivities against analytic calculus at R=8, V=12
    assert math.isclose(v8.sensitivities["I"]["R"], -12.0 / 64.0, rel_tol=0, abs_tol=1e-5)
    assert math.isclose(v8.sensitivities["P"]["R"], -144.0 / 64.0, rel_tol=0, abs_tol=1e-5)

    # Bidirectional counterfactual inverse proof prescribes exact R -> 4.0
    assert "R" in v8.prescribed_adjustments
    assert math.isclose(v8.prescribed_adjustments["R"], 4.0, rel_tol=0, abs_tol=1e-4)

    # Turning dial again to R = 5.0 also stays exact
    rep_5 = sys.turn_dial("R", 5.0)
    assert rep_5.converged and rep_5.max_residual < 1e-10
    assert math.isclose(sys.quantities["I"].value, 2.4, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(sys.quantities["P"].value, 28.8, rel_tol=0, abs_tol=1e-9)

    # Upper Rotor closed-loop regulation brings dial smoothly back to R=4.0, P=36W
    before, after = sys.regulate_to_equilibrium(adjustable_dials=["R"])
    assert before.status == DiscernmentStatus.INTENT_DIVERGENCE_LAWFUL
    assert after.status == DiscernmentStatus.HARMONIOUS_CONVERGENCE
    assert math.isclose(sys.quantities["R"].value, 4.0, rel_tol=0, abs_tol=1e-4)
    assert math.isclose(sys.quantities["P"].value, 36.0, rel_tol=0, abs_tol=1e-3)


def test_circuit_overheat_local_valid_upper_violation_and_regulation():
    sys = build_variable_resistor_circuit_system()

    # Dropping resistance to 1.0 Ohm keeps Layer-1 & Layer-2 equalities exact,
    # yet blows past Upper Rotor's thermal safety fence (T_final = 169 C > 85 C).
    rep = sys.turn_dial("R", 1.0)
    assert rep.converged
    assert rep.max_residual < 1e-10
    assert math.isclose(sys.quantities["I"].value, 12.0, abs_tol=1e-8)
    assert math.isclose(sys.quantities["T_final"].value, 169.0, abs_tol=1e-6)

    v = sys.observe_and_discern()
    assert v.lower_subsystem_verdict is True
    assert v.mechanism_layer_verdict is True
    assert v.upper_fence_verdict is False
    assert v.status == DiscernmentStatus.LOCAL_VALID_UPPER_VIOLATION
    assert any(b["bond"] == "thermal_safety_fence" for b in v.broken_bonds)

    before, after = sys.regulate_to_equilibrium(adjustable_dials=["R"])
    assert before.status == DiscernmentStatus.LOCAL_VALID_UPPER_VIOLATION
    assert after.status == DiscernmentStatus.HARMONIOUS_CONVERGENCE
    assert sys.quantities["T_final"].value <= 85.0 + 1e-5


def test_structural_non_causal_tamper_detection_and_self_healing():
    sys = build_variable_resistor_circuit_system()

    # Direct non-causal cheat injection into current I without changing V or R
    sys.inject_tamper("I", 10.0)
    v = sys.observe_and_discern()
    assert v.status == DiscernmentStatus.STRUCTURAL_NON_CAUSAL_RUPTURE
    assert v.lower_subsystem_verdict is False

    ohm_breaks = [b for b in v.broken_bonds if b["bond"] == "ohm_law"]
    assert len(ohm_breaks) == 1
    assert math.isclose(ohm_breaks[0]["residual"], -28.0, abs_tol=1e-6)
    assert math.isclose(v.prescribed_adjustments["I"], 3.0, abs_tol=1e-6)

    before, after = sys.regulate_to_equilibrium()
    assert before.status == DiscernmentStatus.STRUCTURAL_NON_CAUSAL_RUPTURE
    assert after.status == DiscernmentStatus.HARMONIOUS_CONVERGENCE
    assert math.isclose(sys.quantities["I"].value, 3.0, abs_tol=1e-6)
    assert sys.quantities["I"].pin_mode == PinMode.FREE


def test_cellular_mitosis_digital_cancer_trinitarian_discernment():
    sys = build_cellular_homeostasis_system()
    assert sys.observe_and_discern().status == DiscernmentStatus.HARMONIOUS_CONVERGENCE

    # Cell activates oncogenic proliferation dial r_div = 0.85
    rep = sys.turn_dial("r_div", 0.85)
    assert rep.converged and rep.max_residual < 1e-10

    # Isolated Lower Subsystem approves 100% of mitosis & ATP rules
    lower = sys.observe_lower_perspective()
    assert lower["lower_valid"] is True
    assert lower["max_residual"] < 1e-10

    # Upper Organism Rotor perceives hypoxic & over-capacity rupture
    verdict = sys.observe_and_discern()
    assert verdict.lower_subsystem_verdict is True
    assert verdict.mechanism_layer_verdict is True
    assert verdict.upper_fence_verdict is False
    assert verdict.status == DiscernmentStatus.LOCAL_VALID_UPPER_VIOLATION
    broken_names = {b["bond"] for b in verdict.broken_bonds}
    assert {"tissue_capacity_fence", "vascular_oxygen_fence"}.issubset(broken_names)
    assert any(
        chain[0] == "r_div" and chain[-1] == "O2_demand"
        for chain in verdict.causal_chains
    )

    # Downward regulation restores organism homeostasis cleanly
    _, after = sys.regulate_to_equilibrium(adjustable_dials=["r_div"])
    assert after.status == DiscernmentStatus.HARMONIOUS_CONVERGENCE
    assert after.total_action < 1e-8
    assert sys.quantities["N_next"].value <= 125.0 + 1e-5
    assert sys.quantities["O2_demand"].value <= 140.0 + 1e-5
    assert sys.quantities["N_next"].value >= 102.0 - 1e-5


def test_narrative_causal_worldline_vs_contextless_exploit():
    sys = build_narrative_causal_worldline_system()
    v0 = sys.observe_and_discern()
    assert v0.status == DiscernmentStatus.HARMONIOUS_CONVERGENCE

    # Weak prior history (no training/trust built), yet agent claims +50 mastery leap
    sys.pin_value("prep_past", 0.5, mode=PinMode.FIXED_BOUNDARY, relax=False)
    sys.pin_value("trust_past", 0.5, mode=PinMode.FIXED_BOUNDARY, relax=False)
    sys.turn_dial("effort_cost", 4.0, relax=False)
    sys.turn_dial("claimed_mastery_jump", 50.0, relax=True)

    lower = sys.observe_lower_perspective()
    assert lower["lower_valid"] is True  # Discrete tick check approves effort <= energy!

    verdict = sys.observe_and_discern(adjustable_dials=["claimed_mastery_jump"])
    assert verdict.lower_subsystem_verdict is True
    assert verdict.mechanism_layer_verdict is True
    assert verdict.upper_fence_verdict is False
    assert verdict.status == DiscernmentStatus.LOCAL_VALID_UPPER_VIOLATION
    assert any(b["bond"] == "narrative_coherence_fence" for b in verdict.broken_bonds)
    assert math.isclose(sys.quantities["causal_mastery_gain"].value, 1.0, abs_tol=1e-6)

    # Regulating claimed outcome dial caps mastery jump to authentic narrative trajectory
    sys.regulate_to_equilibrium(adjustable_dials=["claimed_mastery_jump"])
    assert sys.quantities["claimed_mastery_jump"].value <= 1.0 + 1e-5


def test_epistemic_humility_underdetermined_nullspace_detection():
    sys = TrinitarianScaleRotorSystem(name="underdetermined_line")
    sys.add_quantity(CoupledQuantity("x", 2.0, ScaleTier.VALUE, "", "state", PinMode.FREE))
    sys.add_quantity(CoupledQuantity("y", 3.0, ScaleTier.VALUE, "", "state", PinMode.FREE))
    sys.add_quantity(CoupledQuantity("target_sum", 10.0, ScaleTier.VALUE, "", "fixed", PinMode.FIXED_BOUNDARY))
    sys.add_bond(
        EqualityBond(
            name="sum_law",
            tier=ScaleTier.FORMULA,
            variables=("x", "y", "target_sum"),
            lhs_fn=lambda v: v["x"] + v["y"],
            rhs_fn=lambda v: v["target_sum"],
            lhs_expr="x + y",
            rhs_expr="target_sum",
        )
    )

    rep = sys.relax_least_action()
    assert rep.converged
    assert math.isclose(
        sys.quantities["x"].value + sys.quantities["y"].value, 10.0, abs_tol=1e-9
    )
    assert rep.nullspace_dof == 1

    verdict = sys.observe_and_discern()
    assert verdict.nullspace_dof == 1
    assert "nullspace DoF=1" in verdict.four_axis_attribution["boundary_ignorance"]


def test_causal_field_sync_physical_beam_tear_and_heal():
    sys = build_variable_resistor_circuit_system()
    sys.inject_tamper("I", 10.0)
    torn_verdict = sys.observe_and_discern()

    torn_sync = sync_to_causal_field(sys, verdict=torn_verdict, prefix="torn")
    torn_field = torn_sync["causal_field"]
    assert torn_sync["min_beam_strength"] < 0.1
    torn_field.step(dt=0.1)

    # Now heal via Upper Rotor regulation and verify beam restoration
    _, healed_verdict = sys.regulate_to_equilibrium()
    healed_sync = sync_to_causal_field(sys, verdict=healed_verdict, prefix="healed")
    healed_field = healed_sync["causal_field"]
    assert healed_sync["min_beam_strength"] > 0.99
    healed_field.step(dt=0.1)
