"""
test_imaginative_perceptual_recreator.py

End-to-end verification of:
1. Exact invertible cone-opponent codec (RMSE < 1e-5) & self-mapped rainbow prism
2. Regional saturation/chroma sweep toward Red (650nm) and Yellow (580nm) with tipping-point discovery
3. Full 360-deg Clifford Rainbow Rotor closure & algebraic reversibility
4. Luminance (EV) & Texture Roughness axis orthogonality under re-observation
5. Closed-loop autonomous goal re-creation (Imagine -> Re-Observe -> Refine -> Converge)
6. Cross-domain Game Engine PBR Shader & External Video CDL mapping + CausalField L2..L14
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from core.sensory.imaginative_perceptual_recreator import (
    ImaginativeExplorationSession,
    calibrate_perceptual_manifold,
    decode_opponent_to_srgb,
    run_full_imaginative_session,
)
from core.sensory.ocular_scale_organ import OcularScaleOrgan
from simulators.hidden_multiphysics_world import YOSEMITE_4K_PATH


@pytest.fixture(scope="module")
def yosemite_cliff_crop() -> np.ndarray:
    if not YOSEMITE_4K_PATH.exists():
        pytest.skip(f"Missing real 4K asset: {YOSEMITE_4K_PATH}")
    arr = np.array(Image.open(YOSEMITE_4K_PATH), dtype=np.uint8)
    return arr[600:1350, 1300:2400].copy()


@pytest.fixture(scope="module")
def yosemite_imagination_session(yosemite_cliff_crop: np.ndarray) -> ImaginativeExplorationSession:
    return run_full_imaginative_session(yosemite_cliff_crop)


def test_cone_opponent_codec_roundtrip_and_self_mapped_rainbow_prism(yosemite_cliff_crop: np.ndarray):
    eye = OcularScaleOrgan()
    obs = eye.observe(yosemite_cliff_crop, max_width=240)
    rec, clip_ratio = decode_opponent_to_srgb(
        obs.l1.luminance_a, obs.l1.rg_opponent, obs.l1.by_opponent, eps=eye.EPS
    )
    rmse = float(np.sqrt(np.mean((rec - obs.rgb_working) ** 2)))
    assert rmse < 1e-5
    assert clip_ratio < 1e-5

    manifold = calibrate_perceptual_manifold(eye, yosemite_cliff_crop)
    assert len(manifold.axes) == 4
    wls_check = [410, 450, 490, 530, 580, 605, 650]
    raw_angles = [manifold.anchor_angles_deg[w] for w in wls_check]
    unwrapped_deg = np.rad2deg(np.unwrap(np.deg2rad(raw_angles)))
    assert all(unwrapped_deg[i + 1] > unwrapped_deg[i] for i in range(len(unwrapped_deg) - 1))


def test_regional_chroma_sweep_toward_red_and_yellow_discovers_tipping_point(
    yosemite_imagination_session: ImaginativeExplorationSession,
):
    sess = yosemite_imagination_session
    for sweep in (sess.red_sweep, sess.yellow_sweep):
        dcs = [s.regional_delta_c for s in sweep.steps]
        assert dcs[0] < 1e-4
        assert dcs[-1] >= 0.45
        assert all(dcs[i + 1] >= dcs[i] - 1e-4 for i in range(len(dcs) - 1))
        assert sweep.min_l2_persistence_corr >= 0.80
        assert sweep.alpha_tipping_point is not None
        assert 0.0 < sweep.alpha_tipping_point <= 1.0
        # Target region shifted distinctly away from baseline luminance while preserving edges
        assert all(s.regional_delta_a < 0.35 for s in sweep.steps)


def test_full_360_rainbow_clifford_rotor_closure_and_reversibility(
    yosemite_imagination_session: ImaginativeExplorationSession,
):
    rb = yosemite_imagination_session.rainbow_sweep
    assert rb.closure_rmse_360 < 1e-5
    assert rb.rewind_rmse_120 < 1e-5
    assert rb.min_l2_persistence_corr >= 0.70


def test_luminance_and_roughness_axes_orthogonality_under_reobservation(
    yosemite_imagination_session: ImaginativeExplorationSession,
):
    ortho = yosemite_imagination_session.orthogonality_sweep
    assert all(dc < 0.05 for dc in ortho.luminance_delta_cs)
    assert all(corr > 0.90 for corr in ortho.luminance_persistence_corrs)

    energies = ortho.roughness_fine_energies
    assert all(energies[i + 1] > energies[i] for i in range(len(energies) - 1))
    assert all(corr > 0.85 for corr in ortho.roughness_persistence_corrs)


def test_closed_loop_autonomous_goal_recreation_converges_via_self_reobservation(
    yosemite_imagination_session: ImaginativeExplorationSession,
):
    goal = yosemite_imagination_session.goal_recreation
    assert goal.converged
    assert len(goal.iterations) >= 3
    init_err = goal.iterations[0].total_error
    end_err = goal.iterations[-1].total_error
    assert end_err < init_err * 0.45
    assert goal.iterations[-1].l2_persistence_corr >= 0.82


def test_cross_domain_pbr_video_mapping_and_causal_field_multiscale(
    yosemite_imagination_session: ImaginativeExplorationSession,
):
    sess = yosemite_imagination_session
    pbr = sess.pbr_spec
    vid = sess.video_grade_spec

    assert all(0.0 <= c <= 1.0 for c in pbr.albedo_linear_rgb)
    assert 0.05 <= pbr.microfacet_roughness <= 0.95
    assert 0.0 <= pbr.albedo_saturation <= 1.0
    assert "fracture" in pbr.domain_boundary_note.lower() or "cleavage" in pbr.domain_boundary_note.lower()

    assert all(0.25 <= s <= 4.0 for s in vid.cdl_slope_rgb)
    assert 0.2 <= vid.chroma_gain <= 4.0
    assert "radiometrically" in vid.domain_boundary_note.lower()

    field = sess.final_causal_field
    assert any(k.startswith("region_") for k in field.voxels)
    assert any(
        v.scale_level == 10.0 and (k.startswith("axis_") or k.startswith("rotor_"))
        for k, v in field.voxels.items()
    )
    assert any(v.scale_level == 12.0 and k.startswith("imagination_") for k, v in field.voxels.items())
    assert any(v.scale_level == 14.0 and k.startswith("transfer_") for k, v in field.voxels.items())
    assert len(field.beams) >= 15

    for v in field.voxels.values():
        assert np.all(np.isfinite(v.position))
        assert np.all(np.isfinite(v.velocity))
        assert np.all(np.isfinite(v.tensor))
