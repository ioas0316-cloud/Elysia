"""
test_ocular_scale_organ.py

Comprehensive pytest suite verifying OcularScaleOrgan (L0-L4), Circular
Re-Observation, VisualCausalReasoner, and CausalField projection strictly on
real unlabeled 4K camera photographs and physically faithful linear-irradiance
optical transforms.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import gaussian_filter1d

from core.sensory.ocular_scale_organ import (
    OcularScaleOrgan,
)
from core.sensory.visual_causal_reasoner import (
    VisualCausalReasoner,
)

YOSEMITE_PATH = Path(r"c:\Elysia\demos\real_world_assets\genuine_optical_4k_yosemite.jpg")
SECOND_PATH = Path(r"c:\Elysia\demos\real_world_assets\second_real_world_4k.jpg")


@pytest.fixture(scope="module")
def real_photos():
    if not YOSEMITE_PATH.exists():
        pytest.skip(f"Missing real photo: {YOSEMITE_PATH}")
    if not SECOND_PATH.exists():
        pytest.skip(f"Missing second photo: {SECOND_PATH}")
    yosemite = np.array(Image.open(YOSEMITE_PATH), dtype=np.uint8)
    second = np.array(Image.open(SECOND_PATH), dtype=np.uint8)
    return yosemite, second


def _scale_linear_irradiance(rgb_u8: np.ndarray, scale_mask: np.ndarray | float) -> np.ndarray:
    """Apply optical dimming/shadow in physical linear irradiance space (gamma 2.2)."""
    srgb = rgb_u8.astype(np.float32) / 255.0
    lin = np.power(np.clip(srgb, 0.0, 1.0), 2.2)
    if np.ndim(scale_mask) == 2:
        lin = lin * scale_mask[..., None]
    else:
        lin = lin * float(scale_mask)
    out_srgb = np.power(np.clip(lin, 0.0, 1.0), 1.0 / 2.2)
    return np.clip(np.round(out_srgb * 255.0), 0, 255).astype(np.uint8)


def _make_half_plane_shadow_mask(h: int, w: int, x_edge: int, dark_factor: float = 0.42) -> np.ndarray:
    mask = np.ones(w, dtype=np.float32)
    mask[x_edge:] = dark_factor
    # Soft 3px-scale penumbra in 4K (smooth over ~10px in crop)
    soft_1d = gaussian_filter1d(mask, sigma=4.0)
    return np.broadcast_to(soft_1d[None, :], (h, w)).copy()


def test_l0_l1_multiplicative_illumination_invariance(real_photos):
    yosemite, _ = real_photos
    dimmed = _scale_linear_irradiance(yosemite, 0.35)

    organ = OcularScaleOrgan()
    reasoner = VisualCausalReasoner(organ)
    obs1 = organ.observe(yosemite)
    obs2 = organ.observe(dimmed)

    temporal = organ.observe_transition(obs1, obs2)
    report = reasoner.compare_observations(obs1, obs2, temporal)

    assert report.l0_linear_rmse > 0.10
    assert temporal.mean_delta_a < -0.50
    assert temporal.mean_delta_c < 0.03
    assert temporal.structural_correlation > 0.95
    assert temporal.event_classification == "GLOBAL_ILLUMINATION_CHANGE"
    assert report.first_divergence_layer == "L0/L1_LUMINANCE"


def test_l2_scale_coupled_bond_cleavage_prevents_fragmentation(real_photos):
    yosemite, _ = real_photos
    organ = OcularScaleOrgan()
    obs = organ.observe(yosemite)

    assert obs.l2.single_scale_fragment_count > 5 * obs.l2.hierarchical_region_count
    assert obs.l2.bond_lattice.single_scale_cleaved_ratio > obs.l2.bond_lattice.cleaved_bond_ratio
    assert obs.l2.coarse_region_count >= 2
    assert len(obs.regions) == obs.l2.hierarchical_region_count
    for reg in obs.regions:
        assert 0 <= reg.parent_coarse_id < obs.l2.coarse_region_count
        assert reg.child_fine_count >= 1


def test_circular_observation_reveals_hidden_substructure(real_photos):
    yosemite, _ = real_photos
    organ = OcularScaleOrgan()
    obs = organ.observe(yosemite)

    assert len(obs.circular_reobservations) >= 1
    assert any(
        c.adapted_gain_factor > 1.0
        and c.pass2_internal_cleaved_ratio > c.pass1_internal_cleaved_ratio
        and c.newly_revealed_subregion_count >= 1
        for c in obs.circular_reobservations
    )


def test_l3_shadow_vs_material_cleavage_and_ambiguity(real_photos):
    yosemite, _ = real_photos
    organ = OcularScaleOrgan()
    reasoner = VisualCausalReasoner(organ)

    # Real well-lit high-texture granite crop (540 x 960)
    crop_granite = yosemite[1620:2160, 960:1920]
    crop_slope = yosemite[540:1080, 2880:3840]

    # (a) Optical cast shadow: multiplicative 0.42x in linear irradiance
    h, w = crop_granite.shape[:2]
    shadow_mask = _make_half_plane_shadow_mask(h, w, x_edge=w // 2, dark_factor=0.42)
    shadowed_granite = _scale_linear_irradiance(crop_granite, shadow_mask)

    obs_shadow = organ.observe(shadowed_granite)
    shadow_edges = [
        b for b in obs_shadow.boundaries if b.optical_cause == "ILLUMINATION_SHADOW_CANDIDATE"
    ]
    assert len(shadow_edges) >= 1
    assert all(b.is_ambiguous for b in shadow_edges)

    hyps = reasoner.generate_hypotheses(obs_shadow)
    assert any(h_item.kind == "ACHROMATIC_SHADOW_VS_PIGMENT" and h_item.status == "OPEN" for h_item in hyps)
    assert any(
        h_item.kind == "MODALITY_BOUNDARY_NON_OPTICAL" and h_item.status == "REQUIRES_OTHER_MODALITY"
        for h_item in hyps
    )

    # (b) Real material splice: granite (left) vs distinct blue-green slope (right)
    material_splice = np.concatenate([crop_granite[:, : w // 2], crop_slope[:, w // 2 :]], axis=1)
    obs_mat = organ.observe(material_splice)
    mat_edges = [b for b in obs_mat.boundaries if b.optical_cause == "MATERIAL_BOUNDARY"]
    assert len(mat_edges) >= 1
    assert all(not b.is_ambiguous for b in mat_edges)


def test_l4_saccade_vs_independent_object_motion_and_occlusion_depth(real_photos):
    yosemite, _ = real_photos
    organ = OcularScaleOrgan()

    # (a) Rigid saccade via shifted 1350x2400 crop windows (10:1 -> 135x240 working)
    crop_a = yosemite[200:1550, 300:2700]
    # Camera shifts right +120px and up -60px => image features move left -12px, down +6px
    crop_b = yosemite[140:1490, 420:2820]

    obs_a = organ.observe(crop_a)
    obs_b = organ.observe(crop_b)
    temporal_sac = organ.observe_transition(obs_a, obs_b)

    assert temporal_sac.event_classification == "SELF_EYE_MOTION"
    vy, vx = temporal_sac.global_shift_yx
    assert abs(vy - 6.0) < 2.5
    assert abs(vx - (-12.0)) < 2.5

    # (b) Independent real-patch motion over real granite background + occlusion depth
    bg = yosemite[1620:2160, 960:1920].copy()  # 540x960 -> 135x240 (4x downsample)
    patch = yosemite[1150:1390, 300:540].copy()  # 240x240 dark forest patch (60x60 working)

    frame1 = bg.copy()
    frame2 = bg.copy()
    y0, x0 = 150, 320
    frame1[y0 : y0 + 240, x0 : x0 + 240] = patch
    # Shift ONLY the patch by +40px in 4K crop = +10px horizontally in 240-wide working frame
    frame2[y0 : y0 + 240, x0 + 40 : x0 + 280] = patch

    obs1 = organ.observe(frame1)
    obs2 = organ.observe(frame2)
    temporal_obj = organ.observe_transition(obs1, obs2)

    assert temporal_obj.event_classification == "INDEPENDENT_OBJECT_MOTION"
    assert int(np.sum(temporal_obj.moving_mask)) > 100
    assert temporal_obj.occlusion_depth_verdict == "MOVING_REGION_IN_FRONT_OF_BACKGROUND"
    assert temporal_obj.leading_occlusion_error > temporal_obj.interior_warp_error


def test_l5_moving_shadow_resolves_epistemic_hypothesis(real_photos):
    yosemite, _ = real_photos
    organ = OcularScaleOrgan()
    reasoner = VisualCausalReasoner(organ)

    crop_granite = yosemite[1620:2160, 960:1920]
    h, w = crop_granite.shape[:2]

    # Shadow moves right by +56px (+14px at 240 working width) over stationary granite texture
    shadow1 = _scale_linear_irradiance(
        crop_granite, _make_half_plane_shadow_mask(h, w, x_edge=480, dark_factor=0.42)
    )
    shadow2 = _scale_linear_irradiance(
        crop_granite, _make_half_plane_shadow_mask(h, w, x_edge=536, dark_factor=0.42)
    )

    obs1 = organ.observe(shadow1)
    obs2 = organ.observe(shadow2)
    temporal = organ.observe_transition(obs1, obs2)

    assert temporal.event_classification == "MOVING_CAST_SHADOW"
    assert temporal.texture_shift_under_luma_edge < 0.9

    hyps = reasoner.generate_hypotheses(obs1)
    verified = reasoner.verify_hypotheses_with_transition(hyps, obs1, obs2, temporal)
    shadow_hyps = [h_item for h_item in verified if h_item.kind == "ACHROMATIC_SHADOW_VS_PIGMENT"]
    assert len(shadow_hyps) >= 1
    assert all(h_item.status == "CONFIRMED" for h_item in shadow_hyps)

    field_obj = reasoner.project_to_causal_field(obs1, temporal=temporal, verified_hypotheses=verified)
    region_beams = [
        b
        for b in field_obj.beams
        if b.source_id.startswith("region_") and b.target_id.startswith("region_")
    ]
    assert any(b.strength >= 2.0 and not b.is_broken for b in region_beams)


def test_cross_scene_attribution_and_four_domain_audit(real_photos):
    yosemite, second = real_photos
    organ = OcularScaleOrgan()
    reasoner = VisualCausalReasoner(organ)

    obs_yos = organ.observe(yosemite)
    obs_sec = organ.observe(second)
    obs_dim = organ.observe(_scale_linear_irradiance(yosemite, 0.35))

    obs_sac_a = organ.observe(yosemite[200:1550, 300:2700])
    obs_sac_b = organ.observe(yosemite[140:1490, 420:2820])

    # Completely disjoint 4K sub-scenes (sky ridge vs dark lower forest cliff)
    obs_sub1 = organ.observe(yosemite[0:1080, 0:1920])
    obs_sub2 = organ.observe(yosemite[1080:2160, 1920:3840])

    rep_reencode = reasoner.compare_observations(obs_yos, obs_sec)
    rep_dim = reasoner.compare_observations(obs_yos, obs_dim)
    rep_sac = reasoner.compare_observations(obs_sac_a, obs_sac_b)
    rep_disjoint = reasoner.compare_observations(obs_sub1, obs_sub2)

    assert rep_reencode.l4_event_classification == "STATIC_IDENTICAL"
    assert rep_reencode.first_divergence_layer == "NONE_INVARIANT"

    assert rep_dim.l4_event_classification == "GLOBAL_ILLUMINATION_CHANGE"
    assert rep_dim.first_divergence_layer == "L0/L1_LUMINANCE"

    assert rep_sac.l4_event_classification == "SELF_EYE_MOTION"
    assert rep_sac.first_divergence_layer == "L4_GLOBAL_MOTION"

    assert rep_disjoint.l4_event_classification == "SCENE_CHANGE"
    assert rep_disjoint.first_divergence_layer == "L1_CHROMATICITY_AND_L2_STRUCTURE"

    audit = reasoner.audit_organ_causality(obs_yos)
    assert "l0_valid_irradiance_ratio" in audit.vision_domain
    assert audit.structure_domain["fragmentation_reduction_ratio"] > 5.0
    assert "base_yield_threshold" in audit.code_domain
    assert "referential_contract" in audit.language_domain
    assert len(audit.modality_limits) >= 3


def test_causal_field_scale_projection_and_step(real_photos):
    yosemite, _ = real_photos
    organ = OcularScaleOrgan()
    reasoner = VisualCausalReasoner(organ)

    bg = yosemite[1620:2160, 960:1920].copy()
    patch = yosemite[1150:1390, 300:540].copy()
    frame1 = bg.copy()
    frame2 = bg.copy()
    frame1[150:390, 320:560] = patch
    frame2[150:390, 360:600] = patch

    obs1 = organ.observe(frame1)
    obs2 = organ.observe(frame2)
    temporal = organ.observe_transition(obs1, obs2)

    field_obj = reasoner.project_to_causal_field(obs1, temporal=temporal)

    coarse_voxels = [v for k, v in field_obj.voxels.items() if k.startswith("coarse_")]
    region_voxels = [v for k, v in field_obj.voxels.items() if k.startswith("region_")]
    assert len(coarse_voxels) >= 1
    assert len(region_voxels) >= 2
    assert all(v.position.shape == (3,) for v in region_voxels)
    assert all(v.tensor.shape == (6,) for v in region_voxels)

    containment_beams = [
        b
        for b in field_obj.beams
        if b.source_id.startswith("region_") and b.target_id.startswith("coarse_")
    ]
    assert len(containment_beams) == len(region_voxels)
    assert all(b.strength == 3.0 for b in containment_beams)

    motion_broken_beams = [b for b in field_obj.beams if b.is_broken]
    assert len(motion_broken_beams) >= 1

    # Ensure CausalField state remains finite after step(dt=0.05) in project_to_causal_field
    for v in field_obj.voxels.values():
        assert np.all(np.isfinite(v.position))
        assert np.all(np.isfinite(v.velocity))
        assert np.all(np.isfinite(v.tensor))
