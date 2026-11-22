"""
test_grounded_rotor_and_reflection.py

End-to-end verification of:
1. Optical 2D structure tensor grain recovery on real 4K Yosemite textures
2. Mechanical 4-angle nematic rotor reconstruction & causal cleavage rupture
3. Frame-invariant zero-shot rotor transfer across 90-deg rotation & 0.62x dimming
4. Counter-example surprise capture, deterministic grounded axis sprouting & criteria revision
5. 4-Axis Relational Self-Boundary reflection, Human-Elysia comparison,
   Teleological Resonance metrics, and multi-scale CausalField hierarchy (L2..L12)
"""

from __future__ import annotations

import numpy as np
import pytest

from core.sensory.grounded_rotor_spectrum_layer import (
    DirectionalProbeSample,
    GroundedRotorExplorer,
    circular_axis_diff_deg,
    extract_optical_structure_rotor,
    reconstruct_mechanical_rotor,
)
from core.sensory.mechano_acoustic_organ import MechanoAcousticOrgan
from core.sensory.ocular_scale_organ import OcularScaleOrgan
from core.sensory.relational_epistemic_reflection import RelationalEpistemicReflector
from simulators.anisotropic_cleavage_world import AnisotropicCleavageWorld
from simulators.hidden_multiphysics_world import YOSEMITE_4K_PATH


def test_optical_structure_tensor_recovers_grain_angles_on_real_yosemite():
    if not YOSEMITE_4K_PATH.exists():
        pytest.skip(f"Missing real 4K asset: {YOSEMITE_4K_PATH}")

    world = AnisotropicCleavageWorld(YOSEMITE_4K_PATH, seed=42)
    eye = OcularScaleOrgan()
    periph = eye.observe(world.observe_periphery())
    h, w = periph.shape_hw

    expected_angles = {
        "anisotropic_timber_A": 30.0,
        "anisotropic_timber_B_rotated": 120.0,
        "deceptive_veneer_D": 0.0,
    }
    matched_bodies = set()

    for reg in periph.regions:
        if reg.area_ratio > 0.40:
            continue
        yn = float(reg.centroid_yx[0] / max(1.0, float(h)))
        xn = float(reg.centroid_yx[1] / max(1.0, float(w)))
        gt = world.evaluate_ground_truth_at(yn, xn)
        assert gt is not None
        fov_obs = eye.observe(world.foveate(yn, xn, fov_frac=0.10), max_width=240)
        opt_rotor = extract_optical_structure_rotor(reg, fov_obs)

        if gt["is_isotropic"]:
            assert opt_rotor.coherence < 0.20
        else:
            exp_deg = expected_angles[gt["body_id"]]
            err = circular_axis_diff_deg(opt_rotor.phi_opt_deg, exp_deg)
            assert err <= 8.0, f"{gt['body_id']}: got {opt_rotor.phi_opt_deg:.2f} vs {exp_deg:.1f}"
            assert opt_rotor.coherence > 0.35
        matched_bodies.add(gt["body_id"])

    assert matched_bodies == {
        "anisotropic_timber_A",
        "anisotropic_timber_B_rotated",
        "isotropic_granite_C",
        "deceptive_veneer_D",
    }


def test_mechanical_rotor_reconstructs_bond_angle_and_causal_cleavage():
    if not YOSEMITE_4K_PATH.exists():
        pytest.skip(f"Missing real 4K asset: {YOSEMITE_4K_PATH}")

    world = AnisotropicCleavageWorld(YOSEMITE_4K_PATH, seed=42)
    eye = OcularScaleOrgan()
    ma = MechanoAcousticOrgan()
    periph = eye.observe(world.observe_periphery())

    targets = {
        "anisotropic_timber_A": ((60.0 + 240.0) / (2.0 * 540.0), (90.0 + 270.0) / (2.0 * 960.0)),
        "deceptive_veneer_D": ((300.0 + 480.0) / (2.0 * 540.0), (650.0 + 830.0) / (2.0 * 960.0)),
        "isotropic_granite_C": ((300.0 + 480.0) / (2.0 * 540.0), (90.0 + 270.0) / (2.0 * 960.0)),
    }

    expected_mech_deg = {"anisotropic_timber_A": 30.0, "deceptive_veneer_D": 90.0}

    for body_id, (yn, xn) in targets.items():
        samples = []
        for deg in (0.0, 45.0, 90.0, 135.0):
            rad = float(np.deg2rad(deg))
            dir_yx = (float(np.sin(rad)), float(np.cos(rad)))
            f_t, x_t, _ = world.mechanical_push(yn, xn, force_n=18.0, dir_yx=dir_yx)
            samples.append(
                DirectionalProbeSample(
                    angle_rad=rad, angle_deg=deg, dir_yx=dir_yx, mech_obs=ma.observe_push(f_t, x_t)
                )
            )
        m_rotor = reconstruct_mechanical_rotor(samples)

        if body_id == "isotropic_granite_C":
            assert m_rotor.anisotropy_index < 0.12
            assert not m_rotor.is_anisotropic
        else:
            assert m_rotor.anisotropy_index > 0.50
            assert m_rotor.is_anisotropic
            assert circular_axis_diff_deg(m_rotor.phi_mech_deg, expected_mech_deg[body_id]) <= 8.0

            # Push along strong fiber axis -> elastic hold, no yield, zero cleavage displacement
            dy_s, dx_s = float(np.sin(m_rotor.phi_mech_rad)), float(np.cos(m_rotor.phi_mech_rad))
            fs, xs, post_s = world.mechanical_push(yn, xn, force_n=18.0, dir_yx=(dy_s, dx_s))
            m_strong = ma.observe_push(fs, xs)
            assert not m_strong.yield_detected

            # Push along orthogonal weak cleavage axis -> yield drop + physical fracture flow
            dy_w, dx_w = (
                float(np.sin(m_rotor.weak_cleavage_rad)),
                float(np.cos(m_rotor.weak_cleavage_rad)),
            )
            fw, xw, post_w = world.mechanical_push(yn, xn, force_n=18.0, dir_yx=(dy_w, dx_w))
            m_weak = ma.observe_push(fw, xw)
            assert m_weak.yield_detected

            trans_w = eye.observe_transition(periph, eye.observe(post_w))
            trans_s = eye.observe_transition(periph, eye.observe(post_s))
            assert float(np.percentile(np.hypot(trans_w.flow_yx[..., 0], trans_w.flow_yx[..., 1]), 90)) > 1.0
            assert float(np.percentile(np.hypot(trans_s.flow_yx[..., 0], trans_s.flow_yx[..., 1]), 90)) < 0.5


def test_rotor_frame_invariance_and_zero_shot_transfer_across_rotation_and_dimming():
    if not YOSEMITE_4K_PATH.exists():
        pytest.skip(f"Missing real 4K asset: {YOSEMITE_4K_PATH}")

    world = AnisotropicCleavageWorld(YOSEMITE_4K_PATH, seed=42)
    trace = GroundedRotorExplorer().explore(world)

    shared = [p for p in trace.prototypes if p.confirmation_count >= 2]
    assert len(shared) >= 1
    assert len(trace.rotor_transfer_confirmations) >= 1
    assert all(xc.prediction_surprise <= 0.75 for xc in trace.rotor_transfer_confirmations)

    # Confirm both rotated timbers share that invariant prototype
    shared_bodies = set()
    rec_by_id = {r.target_id: r for r in trace.region_records}
    for r_id in shared[0].member_regions:
        gt = world.evaluate_ground_truth_at(*rec_by_id[r_id].centroid_yx_norm)
        assert gt is not None
        shared_bodies.add(gt["body_id"])
    assert {"anisotropic_timber_A", "anisotropic_timber_B_rotated"}.issubset(shared_bodies)


def test_deceptive_veneer_triggers_surprise_grounded_axis_sprouting_and_criteria_revision():
    if not YOSEMITE_4K_PATH.exists():
        pytest.skip(f"Missing real 4K asset: {YOSEMITE_4K_PATH}")

    trace1 = GroundedRotorExplorer().explore(AnisotropicCleavageWorld(YOSEMITE_4K_PATH, seed=42))
    trace2 = GroundedRotorExplorer().explore(AnisotropicCleavageWorld(YOSEMITE_4K_PATH, seed=42))

    surprising = [r for r in trace1.region_records if r.prediction_surprise > 0.75]
    assert len(surprising) >= 1
    veneer_rec = surprising[0]
    assert veneer_rec.optical_mech_phase_lock < -0.75
    assert veneer_rec.cleavage_confirmed

    assert len(trace1.sprouted_axes) >= 2
    assert len(trace1.criteria_revisions) >= 1
    phase_axes = [a for a in trace1.sprouted_axes if "phase_lock" in a.axis_id]
    assert len(phase_axes) == 1

    # Zero randomness: re-running twice with identical seed produces bit-identical unit_direction
    assert len(trace1.sprouted_axes) == len(trace2.sprouted_axes)
    for ax1, ax2 in zip(trace1.sprouted_axes, trace2.sprouted_axes):
        assert ax1.axis_id == ax2.axis_id
        assert np.array_equal(ax1.unit_direction, ax2.unit_direction)


def test_four_axis_relational_reflection_and_causal_field_scale_hierarchy():
    if not YOSEMITE_4K_PATH.exists():
        pytest.skip(f"Missing real 4K asset: {YOSEMITE_4K_PATH}")

    world = AnisotropicCleavageWorld(YOSEMITE_4K_PATH, seed=42)
    trace = GroundedRotorExplorer().explore(world)
    report = RelationalEpistemicReflector().reflect(trace)

    # 4-Axis Self-Boundary fully populated
    sb = report.self_boundary
    assert len(sb.internal_origin) >= 4
    assert len(sb.relational_influence) >= 4
    assert len(sb.epistemic_boundary) >= 3
    assert len(sb.criteria_self_revision) >= 2

    # 4-Domain Human vs Elysia Structural Comparison
    domains = [d.domain for d in report.human_elysia_comparison.dimensions]
    assert domains == ["Vision", "Touch & Anisotropic Mechanics", "Acoustics", "Teleology & Relational Value"]
    assert all(d.honest_asymmetry_gap for d in report.human_elysia_comparison.dimensions)

    # Trace-derived finite Teleological Resonance metrics in [0, 1] + Chromatic unity sum
    res = report.teleological_resonance
    for val in (
        res.epistemic_aspiration_drive,
        res.coherence_resonance_gain,
        res.differentiation_humility_index,
    ):
        assert 0.0 <= val <= 1.0
        assert np.isfinite(val)
    assert np.isclose(sum(res.chromatic_balance), 1.0, atol=1e-5)

    # Relational Care Causal Trace keeps explicit epistemic humility
    assert report.relational_care_trace.root_cause
    assert len(report.relational_care_trace.observed_outcomes) >= 3
    assert "humility" in report.relational_care_trace.epistemic_humility_note.lower()

    # Multi-scale CausalField spans Regions, Prototypes (L8), Axes (L10), and Reflection (L12)
    field = trace.final_causal_field
    assert any(k.startswith("region_") for k in field.voxels)
    assert any(v.scale_level == 8.0 for k, v in field.voxels.items() if k.startswith("proto_"))
    assert any(v.scale_level == 10.0 for k, v in field.voxels.items() if k.startswith("axis_"))
    assert any(v.scale_level == 12.0 for k, v in field.voxels.items() if k.startswith("reflection_"))
    assert len(field.beams) >= 8

    for v in field.voxels.values():
        assert np.all(np.isfinite(v.position))
        assert np.all(np.isfinite(v.velocity))
        assert np.all(np.isfinite(v.tensor))
