"""
Verification suite for ELYSIA Phase 11:
Extended Cognitive Organism (Tools, Vehicles, Internet & Academic Disciplines).
"""

from __future__ import annotations

import math

from core.orchestration.extended_cognitive_organism import (
    ExtendedCognitiveOrganism,
    ExtendedWorldState,
    ExternalKnowledgeNetwork,
)
from core.physics.causal_field import CausalField
from core.physics.trinitarian_scale_binding_protocol import DiscernmentStatus


def test_naked_biological_horizon_honestly_detects_infrared_vacuum():
    world = ExtendedWorldState(source_temp_k=3200.0)
    org = ExtendedCognitiveOrganism()

    # Even with all 3 biological organs healthy, 905.55 nm IR exceeds naked senses
    perc = org.perceive_extended(world, active_organs=("eye", "ear", "hand"))
    assert perc.total_action < 1e-10
    assert perc.vacuum_report.vacuum_nullspace_dof == 1
    assert "emission_wavelength_nm" in perc.vacuum_report.irreducible_vacuum_variables


def test_grafting_spectrometer_collapses_invisible_ir_vacuum():
    world = ExtendedWorldState(source_temp_k=3200.0)
    org = ExtendedCognitiveOrganism()
    org.graft_spectrometer(offset_deg=0.0)

    perc = org.perceive_extended(world, active_organs=("eye", "ear", "hand"))
    assert perc.total_action < 1e-10
    assert perc.vacuum_report.vacuum_nullspace_dof == 0
    assert "emission_wavelength_nm" not in perc.vacuum_report.irreducible_vacuum_variables
    assert math.isclose(
        perc.inferred_world["emission_wavelength_nm"],
        world.emission_wavelength_nm,
        abs_tol=1e-3,
    )


def test_internet_knowledge_network_falsifies_myths_and_internalizes_wien_law():
    world = ExtendedWorldState(source_temp_k=3200.0)
    org = ExtendedCognitiveOrganism()
    org.graft_spectrometer()

    report = org.query_and_verify_knowledge_network(
        world=world,
        network=ExternalKnowledgeNetwork(),
        target_vacuum_variable="emission_wavelength_nm",
        discover_workspace_keyword="hangul",
    )
    assert report.accepted_laws == ["wien_displacement_law"]
    assert set(report.rejected_falsehoods) == {
        "internet_linear_myth",
        "internet_wrong_constant_law",
    }
    assert "wien_displacement_law" in org.internalized_disciplines
    assert len(report.workspace_harvested_tools) >= 1


def test_internalized_academic_law_transfers_zero_shot_without_physical_tool():
    cal_world = ExtendedWorldState(source_temp_k=3200.0)
    org = ExtendedCognitiveOrganism()
    org.graft_spectrometer()
    org.query_and_verify_knowledge_network(cal_world, ExternalKnowledgeNetwork())

    # Detach physical Spectrometer AND blind biological Eye on a brand-new 5800K sun-like world
    org.detach_tool("spectrometer")
    hot_world = ExtendedWorldState(source_temp_k=5800.0)
    perc = org.perceive_extended(hot_world, active_organs=("ear", "hand"))

    assert perc.total_action < 1e-10
    assert math.isclose(perc.inferred_world["source_temp_k"], 5800.0, abs_tol=1e-2)
    assert math.isclose(
        perc.inferred_world["emission_wavelength_nm"],
        hot_world.emission_wavelength_nm,
        abs_tol=1e-2,
    )
    # Only surface pigment remains unobservable; radiation wavelength is solved mentally!
    assert "emission_wavelength_nm" not in perc.vacuum_report.irreducible_vacuum_variables
    assert set(perc.vacuum_report.irreducible_vacuum_variables.keys()) == {
        "eye_hue",
        "source_pigment_hue",
    }


def test_automobile_motor_extension_braking_fence_and_far_horizon_parallax():
    far_world = ExtendedWorldState(source_pos_x=120.0, source_power_w=800.0, source_temp_k=3200.0)
    org = ExtendedCognitiveOrganism()
    org.graft_automobile(friction_mu=0.35, safety_margin_m=15.0)

    drive_rep = org.drive_automobile_with_safety(
        world=far_world,
        desired_speed_ms=35.0,
        drive_duration_s=2.0,
        active_organs=("ear", "hand"),
        disabled_channels=("touch_contact_force",),
        clearance_limit_m=120.0,
    )
    assert drive_rep.verdict_before.status == DiscernmentStatus.LOCAL_VALID_UPPER_VIOLATION
    assert drive_rep.safety_regulated is True
    assert drive_rep.regulated_speed_ms < 22.0
    assert drive_rep.perception_after_drive.body_pos_x + drive_rep.stopping_distance_m + 15.0 <= 120.0 + 1e-3

    # The regulated ~41m car drive breaks static degeneracy and locks far-horizon distance & power
    perc = drive_rep.perception_after_drive
    assert perc.total_action < 1e-10
    assert math.isclose(perc.inferred_world["source_pos_x"], 120.0, abs_tol=1e-3)
    assert math.isclose(perc.inferred_world["source_power_w"], 800.0, abs_tol=1e-2)


def test_broken_spectrometer_isolated_and_self_calibrated_by_embodied_consensus():
    world = ExtendedWorldState(source_temp_k=3200.0)
    org = ExtendedCognitiveOrganism()
    org.graft_spectrometer(offset_deg=0.0)
    org.query_and_verify_knowledge_network(world, ExternalKnowledgeNetwork())

    # Spectrometer drifts by +5.0 deg while body senses + internalized Wien law agree
    fault = org.diagnose_and_calibrate_spectrometer(
        world=world,
        true_spectrometer_offset_deg=5.0,
        active_organs=("eye", "ear", "hand"),
    )
    assert fault.fault_detected is True
    assert fault.conflict_action_before > 1e-4
    assert fault.body_consensus_action < 1e-8
    assert abs(fault.estimated_offset_deg - 5.0) < 1e-4
    assert fault.recalibrated_action_after < 1e-10
    assert abs(org.grafted_tools["spectrometer"]["offset_deg"] - 5.0) < 1e-4


def test_extended_cognitive_organism_causal_field_and_growth_rings():
    world = ExtendedWorldState(source_temp_k=3200.0)
    org = ExtendedCognitiveOrganism()

    naked_perc = org.perceive_extended(world, active_organs=("eye", "ear", "hand"))
    sync_naked = org.sync_extended_to_causal_field(naked_perc, causal_field=CausalField(dimensions=3), prefix="naked")
    sync_naked["causal_field"].step(dt=0.1)

    org.graft_spectrometer()
    org.graft_automobile()
    org.query_and_verify_knowledge_network(world, ExternalKnowledgeNetwork())
    assert len(org.cognitive_growth_rings) >= 3

    ext_perc = org.perceive_extended(world, active_organs=("eye", "ear", "hand"))
    sync_ext = org.sync_extended_to_causal_field(ext_perc, causal_field=CausalField(dimensions=3), prefix="ext")
    sync_ext["causal_field"].step(dt=0.1)

    assert sync_ext["upper_rotor_mass"] > sync_naked["upper_rotor_mass"]
