"""
test_spacetime_structural_cinema.py

End-to-end verification of SpacetimeStructuralCinema & DynamicSpacetimeWorld:
1. Blind moving/rotating 4K disk tracking without geometry peeking
2. Clifford Spacetime Rotor generator extraction (omega, drift) & compression (>50x)
3. Continuous scrub prediction accuracy on UNSEEN interpolated (0.75s) and future (2.0s) frames
4. Exact algebraic Clifford rewind (<1e-12), bifurcation classification & slow-motion cleavage zoom
5. Counterfactual video branches (weak cleavage vs strong hold vs stale t=0 angle at t=2.0)
6. Multi-scale CausalField hierarchy spanning L2..L14 with zero non-finite states
"""

from __future__ import annotations

import numpy as np
import pytest

from core.sensory.grounded_rotor_spectrum_layer import (
    circular_axis_diff_deg,
    extract_optical_structure_rotor,
)
from core.sensory.ocular_scale_organ import OcularScaleOrgan
from core.sensory.spacetime_structural_cinema import SpacetimeStructuralCinema
from simulators.dynamic_spacetime_world import DynamicSpacetimeWorld
from simulators.hidden_multiphysics_world import YOSEMITE_4K_PATH


def test_dynamic_spacetime_world_rotates_and_tracks_real_4k_disks():
    if not YOSEMITE_4K_PATH.exists():
        pytest.skip(f"Missing real 4K asset: {YOSEMITE_4K_PATH}")

    world = DynamicSpacetimeWorld(YOSEMITE_4K_PATH, seed=42)
    eye = OcularScaleOrgan()

    for t in (0.0, 0.5, 1.0, 1.5, 2.0):
        obs = eye.observe(world.observe_frame_at(t))
        h, w = obs.shape_hw
        fg = [r for r in obs.regions if r.area_ratio <= 0.40]
        assert len(fg) == 3, f"Expected 3 distinct disks at t={t}, got {len(fg)}"

        bodies_seen = set()
        for reg in fg:
            yn = float(reg.centroid_yx[0] / float(h))
            xn = float(reg.centroid_yx[1] / float(w))
            gt = world.evaluate_ground_truth_at(t, yn, xn)
            assert gt is not None
            bodies_seen.add(gt["body_id"])
            if not gt["is_isotropic"]:
                fov_obs = eye.observe(world.foveate_at(t, yn, xn, fov_frac=0.09), max_width=240)
                opt = extract_optical_structure_rotor(reg, fov_obs)
                assert circular_axis_diff_deg(opt.phi_opt_deg, gt["phi_opt_deg"]) <= 8.0
                assert opt.coherence > 0.35

        assert bodies_seen == {"rotating_timber_A", "moving_granite_B", "rotating_veneer_C"}


def test_spacetime_cinema_extracts_generators_and_compresses_video_principle():
    if not YOSEMITE_4K_PATH.exists():
        pytest.skip(f"Missing real 4K asset: {YOSEMITE_4K_PATH}")

    world = DynamicSpacetimeWorld(YOSEMITE_4K_PATH, seed=42)
    cinema = SpacetimeStructuralCinema()
    session = cinema.ingest(world, sample_times=(0.0, 0.5, 1.0, 1.5))

    assert len(session.tracks) == 3
    assert session.compression_ratio > 50.0
    assert session.raw_video_bytes > session.structural_tracks_bytes * 50

    timber_tracks = [t for t in session.tracks if t.prototype_label == "anisotropic_phase_locked_timber"]
    veneer_tracks = [t for t in session.tracks if t.prototype_label == "anisotropic_cross_ply_veneer"]
    granite_tracks = [t for t in session.tracks if t.prototype_label == "isotropic_stiff_solid"]
    assert len(timber_tracks) == 1
    assert len(veneer_tracks) == 1
    assert len(granite_tracks) == 1

    tr_a = timber_tracks[0]
    tr_c = veneer_tracks[0]
    assert abs(float(np.rad2deg(tr_a.omega_rad_per_s)) - 30.0) <= 5.0
    assert abs(float(np.rad2deg(tr_c.omega_rad_per_s)) - (-25.0)) <= 5.0
    assert tr_c.optical_mech_phase_lock < -0.75
    assert tr_a.optical_mech_phase_lock > 0.75


def test_continuous_scrub_predicts_unseen_interpolated_and_future_frames():
    if not YOSEMITE_4K_PATH.exists():
        pytest.skip(f"Missing real 4K asset: {YOSEMITE_4K_PATH}")

    world = DynamicSpacetimeWorld(YOSEMITE_4K_PATH, seed=42)
    cinema = SpacetimeStructuralCinema()
    session = cinema.ingest(world, sample_times=(0.0, 0.5, 1.0, 1.5))

    track_map = {t.track_id: t for t in session.tracks}
    assert len(session.scrub_validations) == 6  # 3 tracks x (0.75, 2.0)

    for val in session.scrub_validations:
        tr = track_map[val.track_id]
        assert val.centroid_error_norm < 0.03
        if val.optical_angle_error_deg is not None:
            assert val.optical_angle_error_deg <= 8.0
        gt = world.evaluate_ground_truth_at(val.t_test, *val.observed_pos_yx_norm)
        assert gt is not None
        if tr.is_anisotropic:
            assert circular_axis_diff_deg(val.predicted_phi_mech_deg, gt["phi_mech_deg"]) <= 8.0
            assert val.cleavage_confirmed
            assert val.cleavage_flow_mag >= 1.0


def test_clifford_rewind_exact_invertibility_and_bifurcation_tracing():
    if not YOSEMITE_4K_PATH.exists():
        pytest.skip(f"Missing real 4K asset: {YOSEMITE_4K_PATH}")

    world = DynamicSpacetimeWorld(YOSEMITE_4K_PATH, seed=42)
    cinema = SpacetimeStructuralCinema()
    session = cinema.ingest(world, sample_times=(0.0, 0.5, 1.0, 1.5))

    assert len(session.rewind_traces) == 3
    for rw in session.rewind_traces:
        assert rw.rewind_vector_error < 1e-12

    track_map = {t.track_id: t for t in session.tracks}
    cls_by_proto = {track_map[rw.track_id].prototype_label: rw.classification for rw in session.rewind_traces}
    assert cls_by_proto["anisotropic_cross_ply_veneer"] == "INTRINSIC_PERSISTENT_CROSS_PLY_STRUCTURE"
    assert cls_by_proto["anisotropic_phase_locked_timber"] == "PHASE_LOCKED_COHESIVE_STRUCTURE"
    assert cls_by_proto["isotropic_stiff_solid"] == "ISOTROPIC_INVARIANT_STRUCTURE"

    assert len(session.slow_motion_zooms) == 2
    for zoom in session.slow_motion_zooms:
        assert zoom.will_cleave
        assert zoom.tau_crit is not None
        assert abs(zoom.tau_crit - (10.0 / 18.0)) <= 0.12
        assert any(f.cleaved for f in zoom.frames)
        assert not zoom.frames[0].cleaved


def test_counterfactual_video_branches_match_physical_reality():
    if not YOSEMITE_4K_PATH.exists():
        pytest.skip(f"Missing real 4K asset: {YOSEMITE_4K_PATH}")

    world = DynamicSpacetimeWorld(YOSEMITE_4K_PATH, seed=42)
    cinema = SpacetimeStructuralCinema()
    session = cinema.ingest(world, sample_times=(0.0, 0.5, 1.0, 1.5))

    by_name = {}
    for cf in session.counterfactual_branches:
        assert cf.prediction_matched is True
        by_name.setdefault(cf.branch_name, []).append(cf)

    assert all(cf.predicted_cleavage and cf.actual_yield for cf in by_name["BranchA_WeakCleavage_t1"])
    assert all((not cf.predicted_cleavage) and (not cf.actual_yield) for cf in by_name["BranchB_StrongHold_t1"])
    # At future t=2.0, stale t=0 weak angle no longer cleaves, while rotor-tracked angle cleaves!
    assert all((not cf.predicted_cleavage) and (not cf.actual_yield) for cf in by_name["BranchC_StaleT0Angle_t2"])
    assert all(cf.predicted_cleavage and cf.actual_yield and (cf.actual_cleavage_flow_mag or 0.0) >= 1.0 for cf in by_name["BranchC_RotorTrackedAngle_t2"])


def test_spacetime_causal_field_multiscale_finite():
    if not YOSEMITE_4K_PATH.exists():
        pytest.skip(f"Missing real 4K asset: {YOSEMITE_4K_PATH}")

    world = DynamicSpacetimeWorld(YOSEMITE_4K_PATH, seed=42)
    cinema = SpacetimeStructuralCinema()
    session = cinema.ingest(world, sample_times=(0.0, 0.5, 1.0, 1.5))

    field = session.final_causal_field
    assert any(k.startswith("region_") for k in field.voxels)
    assert any(v.scale_level == 8.0 for k, v in field.voxels.items() if k.startswith("track_"))
    assert any(v.scale_level == 10.0 for k, v in field.voxels.items() if k.startswith("generator_"))
    assert any(v.scale_level == 12.0 for k, v in field.voxels.items() if k.startswith("reflection_"))
    assert any(v.scale_level == 14.0 for k, v in field.voxels.items() if k.startswith("rewind_"))
    assert any(v.scale_level == 14.0 for k, v in field.voxels.items() if k.startswith("counterfactual_"))
    assert len(field.beams) >= 12

    for v in field.voxels.values():
        assert np.all(np.isfinite(v.position))
        assert np.all(np.isfinite(v.velocity))
        assert np.all(np.isfinite(v.tensor))
