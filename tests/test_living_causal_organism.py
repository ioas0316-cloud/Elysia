"""
Verification suite for ELYSIA Phase 10:
Living Embodied Multi-Organ Causal Organism & Sensory Vacuum Substitution.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from core.orchestration.living_causal_organism import (
    EmbodiedWorldState,
    LivingCausalOrganism,
    OrganismStatus,
)
from core.physics.causal_field import CausalField


def test_full_organism_living_harmony_and_mirrors():
    world = EmbodiedWorldState()
    org = LivingCausalOrganism()
    perc = org.perceive(world, active_organs=("eye", "ear", "hand"))

    assert perc.status == OrganismStatus.LIVING_HARMONY
    assert perc.total_action < 1e-10
    assert perc.vacuum_report.vacuum_nullspace_dof == 0
    assert math.isclose(perc.inferred_world["source_pos_x"], world.source_pos_x, abs_tol=1e-6)
    assert math.isclose(perc.inferred_world["source_vel_x"], world.source_vel_x, abs_tol=1e-6)
    assert math.isclose(perc.inferred_world["source_freq_hz"], world.source_freq_hz, abs_tol=1e-6)
    assert math.isclose(perc.inferred_world["source_power_w"], world.source_power_w, abs_tol=1e-6)
    assert math.isclose(perc.inferred_world["source_temp_k"], world.source_temp_k, abs_tol=1e-5)
    assert math.isclose(perc.inferred_world["source_pigment_hue"], world.source_pigment_hue, abs_tol=1e-6)
    assert math.isfinite(perc.hangul_wave_coherence)


def test_blind_eye_vacuum_filled_by_ear_hand_and_honest_pigment_void():
    world = EmbodiedWorldState()
    org = LivingCausalOrganism()

    # Eye is completely closed/blind -> Ear + Hand fill all cross-modal channels
    perc = org.perceive(world, active_organs=("ear", "hand"))
    assert perc.status == OrganismStatus.VACUUM_COMPENSATED
    assert perc.total_action < 1e-10
    assert perc.vacuum_report.missing_organs == ["eye"]

    true_imp = world.generate_sensory_impressions(body_pos_x=0.0, body_vel_x=0.0)
    assert math.isclose(perc.inferred_world["source_pos_x"], world.source_pos_x, abs_tol=1e-5)
    assert math.isclose(perc.inferred_world["source_vel_x"], world.source_vel_x, abs_tol=1e-5)
    assert math.isclose(perc.inferred_world["source_temp_k"], world.source_temp_k, abs_tol=1e-3)
    assert math.isclose(perc.reconstructed_organs["eye_apparent_dist"], true_imp["eye_apparent_dist"], abs_tol=1e-5)
    assert math.isclose(perc.reconstructed_organs["eye_apparent_vel"], true_imp["eye_apparent_vel"], abs_tol=1e-5)
    assert math.isclose(perc.reconstructed_organs["eye_color_warmth"], true_imp["eye_color_warmth"], abs_tol=1e-5)

    # Filled vs irreducible honest void (SVD right nullspace cleanly detects pigment)
    filled = set(perc.vacuum_report.vacuum_filled_variables.keys())
    assert {"eye_apparent_dist", "eye_apparent_vel", "eye_color_warmth"}.issubset(filled)
    assert set(perc.vacuum_report.irreducible_vacuum_variables.keys()) == {"eye_hue", "source_pigment_hue"}
    assert perc.vacuum_report.vacuum_nullspace_dof == 1


def test_deaf_ear_vacuum_filled_by_eye_and_hand():
    world = EmbodiedWorldState()
    org = LivingCausalOrganism()

    perc = org.perceive(world, active_organs=("eye", "hand"))
    assert perc.status == OrganismStatus.VACUUM_COMPENSATED
    assert perc.total_action < 1e-10
    assert perc.vacuum_report.vacuum_nullspace_dof == 0

    true_imp = world.generate_sensory_impressions(body_pos_x=0.0, body_vel_x=0.0)
    assert math.isclose(perc.reconstructed_organs["ear_intensity"], true_imp["ear_intensity"], abs_tol=1e-6)
    assert math.isclose(perc.reconstructed_organs["ear_doppler_hz"], true_imp["ear_doppler_hz"], abs_tol=1e-5)

    true_wave = world.synthesize_acoustic_wave(body_pos_x=0.0, body_vel_x=0.0, n_samples=256)
    corr = float(np.corrcoef(perc.reconstructed_acoustic_wave, true_wave)[0, 1])
    assert corr > 0.9999


def test_dual_sensory_loss_resolved_by_embodied_body_motion():
    world = EmbodiedWorldState()
    org = LivingCausalOrganism()

    # Blind Eye AND Numb Touch Contact Force -> distance & power statically entangled
    static_perc = org.perceive(
        world,
        active_organs=("ear", "hand"),
        disabled_channels=("touch_contact_force",),
    )
    assert static_perc.vacuum_report.vacuum_nullspace_dof >= 2

    # Active motor probe: stepping body by delta_x=1.5 creates parallax across time
    moved_perc = org.probe_by_body_motion(
        world,
        delta_x=1.5,
        active_organs=("ear", "hand"),
        disabled_channels=("touch_contact_force",),
    )
    assert moved_perc.total_action < 1e-10
    assert math.isclose(moved_perc.inferred_world["source_pos_x"], world.source_pos_x, abs_tol=1e-5)
    assert math.isclose(moved_perc.inferred_world["source_power_w"], world.source_power_w, abs_tol=1e-4)
    assert math.isclose(moved_perc.inferred_world["source_vel_x"], world.source_vel_x, abs_tol=1e-5)
    assert math.isclose(moved_perc.inferred_world["source_freq_hz"], world.source_freq_hz, abs_tol=1e-5)
    assert math.isclose(moved_perc.inferred_world["source_temp_k"], world.source_temp_k, abs_tol=1e-3)
    assert moved_perc.vacuum_report.vacuum_nullspace_dof == 1
    assert all(
        "hue" in name or "pigment" in name
        for name in moved_perc.vacuum_report.irreducible_vacuum_variables
    )


def test_unknown_distorted_eye_self_calibrated_by_ear_hand_and_body():
    world = EmbodiedWorldState()
    org = LivingCausalOrganism(eye_scale_bias=1.0, eye_offset_bias=0.0)

    before, after = org.self_calibrate_eye_via_body(
        world,
        delta_x=1.0,
        true_eye_scale_bias=1.6,
        true_eye_offset_bias=1.2,
    )
    assert before.status == OrganismStatus.CROSS_ORGAN_CONFLICT
    assert before.total_action > 1e-4
    assert org.last_outlier_organ == "eye"
    assert after.status == OrganismStatus.LIVING_HARMONY
    assert after.total_action < 1e-10
    assert abs(org.eye_scale_bias - 1.6) < 1e-5
    assert abs(org.eye_offset_bias - 1.2) < 1e-5


def test_reconstructed_blind_optical_surface_and_living_mirrors():
    world = EmbodiedWorldState()
    org = LivingCausalOrganism()
    blind_perc = org.perceive(world, active_organs=("ear", "hand"))

    true_thermal_img = world.synthesize_optical_surface(body_pos_x=0.0, size=32, include_pigment=False)
    corr = float(
        np.corrcoef(
            blind_perc.reconstructed_optical_surface.ravel(),
            true_thermal_img.ravel(),
        )[0, 1]
    )
    assert corr > 0.99

    assert blind_perc.visual_mirror.reconstruction_error() < 1e-6
    assert blind_perc.acoustic_mirror.reconstruction_error() < 1e-6
    v_lin = blind_perc.visual_mirror.trace_causal_lineage()
    a_lin = blind_perc.acoustic_mirror.trace_causal_lineage()
    assert v_lin["globe_name"] == "organism_visual_field"
    assert a_lin["globe_name"] == "organism_acoustic_field"


def test_organism_causal_field_vacuum_suction_and_4d_tensors():
    world = EmbodiedWorldState()
    org = LivingCausalOrganism()

    healthy = org.perceive(world, active_organs=("eye", "ear", "hand"))
    sync_h = healthy.sync_to_causal_field(causal_field=CausalField(dimensions=3), prefix="org_h")
    assert sync_h["vacuum_curiosity"] == 0.0
    sync_h["causal_field"].step(dt=0.1)

    blind = org.perceive(world, active_organs=("ear", "hand"))
    sync_b = blind.sync_to_causal_field(causal_field=CausalField(dimensions=3), prefix="org_b")
    assert sync_b["vacuum_curiosity"] > 0.5
    sync_b["causal_field"].step(dt=0.1)
