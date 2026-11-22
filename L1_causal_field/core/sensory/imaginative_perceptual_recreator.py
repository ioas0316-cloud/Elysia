"""
imaginative_perceptual_recreator.py

Closed-Loop Perceptual Axis Self-Mapping, Imaginative Re-Observation,
Re-Creation & Cross-Domain Principle Mapping
=====================================================================
Empowers ELYSIA to self-define continuous perceptual axes directly from raw
OcularScaleOrgan L0/L1/L2 physics (rather than hardcoded color tables), imagine
controlled excursions along those axes (e.g. pulling a cliff's chroma toward
Red or Yellow, sweeping a 360-deg Clifford Rainbow Rotor, or modulating texture
roughness and EV exposure), re-observe her own imagined outputs through
OcularScaleOrgan.observe() to verify whether structural boundaries hold or tip,
converge via closed-loop re-creation feedback, and transfer the resulting rules
into Game Engine PBR shader specs and External Video CDL grading parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from core.physics.causal_field import CausalField, InformationVoxel
from core.physics.clifford_spacetime_rotor import CliffordSpacetimeRotor
from core.sensory.ocular_scale_organ import (
    _SRGB_TO_LMS,
    LayerProvenance,
    OcularScaleOrgan,
    OpticalSceneObservation,
)
from core.sensory.visual_causal_reasoner import (
    FourDomainCausalAudit,
    SceneComparisonReport,
    VisualCausalReasoner,
)

_LMS_TO_SRGB = np.linalg.inv(_SRGB_TO_LMS).astype(np.float64)


def wavelength_nm_to_stimulus_srgb(lam_nm: float) -> np.ndarray:
    """
    Generate an external physical visible-wavelength prism stimulus in sRGB [0, 1]
    spanning 380..700 nm (with gentle broadband pedestal so all cones stay non-zero).
    """
    lam = float(np.clip(lam_nm, 380.0, 700.0))
    if lam < 440.0:
        r, g, b = -(lam - 440.0) / 60.0, 0.0, 1.0
    elif lam < 490.0:
        r, g, b = 0.0, (lam - 440.0) / 50.0, 1.0
    elif lam < 510.0:
        r, g, b = 0.0, 1.0, -(lam - 510.0) / 20.0
    elif lam < 580.0:
        r, g, b = (lam - 510.0) / 70.0, 1.0, 0.0
    elif lam < 645.0:
        r, g, b = 1.0, -(lam - 645.0) / 65.0, 0.0
    else:
        r, g, b = 1.0, 0.0, 0.0

    rgb = np.array([r, g, b], dtype=np.float32) * 0.75 + 0.15
    return np.clip(rgb, 0.0, 1.0)


def decode_opponent_to_srgb(
    a: np.ndarray, rg: np.ndarray, by: np.ndarray, eps: float = 1e-3
) -> Tuple[np.ndarray, float]:
    """
    Invert L1 retinal opponent channels back to sRGB [0, 1].

    Forward (OcularScaleOrgan):
      a  = 0.5 * (logL + logM)
      rg = logL - logM
      by = logS - a
    Inverse:
      logL = a + 0.5 * rg
      logM = a - 0.5 * rg
      logS = a + by
    Returns (srgb_hw3, gamut_clip_ratio).
    """
    a64 = np.asarray(a, dtype=np.float64)
    rg64 = np.asarray(rg, dtype=np.float64)
    by64 = np.asarray(by, dtype=np.float64)

    log_l = np.clip(a64 + 0.5 * rg64, -12.0, 6.0)
    log_m = np.clip(a64 - 0.5 * rg64, -12.0, 6.0)
    log_s = np.clip(a64 + by64, -12.0, 6.0)

    lms = np.maximum(0.0, np.exp(np.stack([log_l, log_m, log_s], axis=-1)) - float(eps))
    lin_raw = np.einsum("hwk,ck->hwc", lms, _LMS_TO_SRGB)

    clipped_mask = np.any((lin_raw < 0.0) | (lin_raw > 1.0), axis=-1)
    gamut_clip_ratio = float(np.mean(clipped_mask))

    lin_safe = np.clip(lin_raw, 0.0, 1.0)
    srgb = np.power(lin_safe, 1.0 / 2.2).astype(np.float32)
    return srgb, gamut_clip_ratio


@dataclass
class SelfDefinedAxisSpec:
    """Grounded continuous perceptual axis built from observed sensory responses."""

    axis_id: str
    name: str
    provenance: LayerProvenance
    metric_unit: str
    empirical_span: Tuple[float, float]


@dataclass
class PerceptualAxisManifold:
    """Self-calibrated continuous rainbow rotor, chroma, luminance, and roughness axes."""

    sigma_rg: float
    sigma_by: float
    prism_wavelengths_nm: List[float]
    prism_angles_deg: List[float]
    prism_u_coords: List[Tuple[float, float]]
    anchor_angles_deg: Dict[int, float]
    anchor_unit_vecs: Dict[int, np.ndarray]
    axes: List[SelfDefinedAxisSpec]

    def spectral_unit_vec(self, wavelength_nm: float) -> np.ndarray:
        """Interpolate unit direction vector along self-discovered rainbow manifold."""
        wls = np.asarray(self.prism_wavelengths_nm, dtype=np.float64)
        angs = np.deg2rad(np.asarray(self.prism_angles_deg, dtype=np.float64))
        w = float(np.clip(wavelength_nm, wls[0], wls[-1]))
        # Circularly unwrap angles for smooth linear interpolation
        angs_u = np.unwrap(angs)
        ang_w = float(np.interp(w, wls, angs_u))
        return np.array([np.cos(ang_w), np.sin(ang_w)], dtype=np.float32)

    def spectral_angle_deg(self, wavelength_nm: float) -> float:
        uv = self.spectral_unit_vec(wavelength_nm)
        return float(np.rad2deg(np.arctan2(uv[1], uv[0])) % 360.0)


@dataclass
class PerceptualImaginationOp:
    """Continuous multi-axis excursion applied in grounded opponent/scale space."""

    hue_rotor_deg: float = 0.0
    chroma_scale: float = 1.0
    target_wavelength_nm: Optional[float] = None
    target_chroma_pull: float = 0.0
    chroma_dissolve_frac: float = 0.0
    custom_u_shift: Optional[Tuple[float, float]] = None
    delta_ev: float = 0.0
    roughness_scale: float = 1.0


@dataclass
class ChromaSweepStep:
    alpha: float
    imagined_rgb: np.ndarray
    reobs: OpticalSceneObservation
    comparison: SceneComparisonReport
    regional_mean_u_rg: float
    regional_mean_u_by: float
    regional_delta_c: float
    regional_delta_a: float
    material_boundary_count: int
    shadow_boundary_count: int
    gamut_clip_ratio: float


@dataclass
class RegionalChromaSweepResult:
    region_id: int
    target_wavelength_nm: float
    target_name: str
    steps: List[ChromaSweepStep]
    alpha_tipping_point: Optional[float]
    min_l2_persistence_corr: float


@dataclass
class RainbowRotorStep:
    angle_deg: float
    reobs: OpticalSceneObservation
    cone_rmse_vs_orig: float
    l2_persistence_corr: float
    mean_u_rg: float
    mean_u_by: float
    gamut_clip_ratio: float


@dataclass
class RainbowRotorSweepResult:
    steps: List[RainbowRotorStep]
    closure_rmse_360: float
    rewind_rmse_120: float
    min_l2_persistence_corr: float


@dataclass
class OrthogonalitySweepResult:
    luminance_evs: List[float]
    luminance_delta_cs: List[float]
    luminance_persistence_corrs: List[float]
    roughness_scales: List[float]
    roughness_fine_energies: List[float]
    roughness_persistence_corrs: List[float]


@dataclass
class GoalIterationRecord:
    iteration: int
    cur_u_rg: float
    cur_u_by: float
    cur_fine_texture: float
    color_error: float
    roughness_error: float
    total_error: float
    l2_persistence_corr: float
    gamut_clip_ratio: float


@dataclass
class GoalRecreationTrace:
    region_id: int
    target_wavelength_nm: float
    target_chroma_norm: float
    target_roughness_ratio: float
    target_u_coords: Tuple[float, float]
    target_fine_texture: float
    iterations: List[GoalIterationRecord]
    final_imagined_rgb: np.ndarray
    final_reobs: OpticalSceneObservation
    final_op: PerceptualImaginationOp
    converged: bool


@dataclass
class GameEnginePBRMaterialSpec:
    """Downstream Game Engine physically based shader mapping from sensory axes."""

    albedo_linear_rgb: Tuple[float, float, float]
    albedo_hue_deg: float
    albedo_saturation: float
    exposure_ev: float
    microfacet_roughness: float
    anisotropy_angle_deg: float
    anisotropy_strength: float
    domain_boundary_note: str


@dataclass
class VideoColorGradeSpec:
    """Downstream External Video ASC-CDL & hue-rotor grade mapping from sensory axes."""

    cdl_slope_rgb: Tuple[float, float, float]
    cdl_offset_ev: float
    hue_rotor_deg: float
    chroma_gain: float
    detail_sharpen_gain: float
    domain_boundary_note: str


@dataclass
class ImaginativeExplorationSession:
    """Complete verified session binding sensory manifold, re-observations, transfers, and CausalField."""

    orig_obs: OpticalSceneObservation
    manifold: PerceptualAxisManifold
    target_region_id: int
    red_sweep: RegionalChromaSweepResult
    yellow_sweep: RegionalChromaSweepResult
    rainbow_sweep: RainbowRotorSweepResult
    orthogonality_sweep: OrthogonalitySweepResult
    goal_recreation: GoalRecreationTrace
    pbr_spec: GameEnginePBRMaterialSpec
    video_grade_spec: VideoColorGradeSpec
    final_causal_field: CausalField
    organ_audit: FourDomainCausalAudit


def calibrate_perceptual_manifold(
    eye: OcularScaleOrgan, real_scene_rgb: np.ndarray
) -> PerceptualAxisManifold:
    """
    Feed real external optical wavelength stimuli (380..700 nm) and natural imagery
    through OcularScaleOrgan so Elysia discovers her own whitening metric and rainbow rotor axes.
    """
    wls = [float(x) for x in np.linspace(410.0, 650.0, 17)]
    raw_rg: List[float] = []
    raw_by: List[float] = []
    for wl in wls:
        stim = wavelength_nm_to_stimulus_srgb(wl)
        patch = np.broadcast_to(stim, (64, 64, 3)).copy()
        obs_wl = eye.observe(patch, max_width=64)
        raw_rg.append(float(np.mean(obs_wl.l1.rg_opponent)))
        raw_by.append(float(np.mean(obs_wl.l1.by_opponent)))

    sigma_rg = max(float(np.std(raw_rg)), 0.15)
    sigma_by = max(float(np.std(raw_by)), 0.15)

    prism_u: List[Tuple[float, float]] = []
    prism_angles: List[float] = []
    for rg_v, by_v in zip(raw_rg, raw_by):
        u_rg = rg_v / sigma_rg
        u_by = by_v / sigma_by
        prism_u.append((float(u_rg), float(u_by)))
        prism_angles.append(float(np.rad2deg(np.arctan2(u_by, u_rg)) % 360.0))

    anchor_wls = [410, 450, 490, 530, 580, 605, 650]
    anchor_angles: Dict[int, float] = {}
    anchor_vecs: Dict[int, np.ndarray] = {}
    angs_unwrapped = np.unwrap(np.deg2rad(prism_angles))
    for awl in anchor_wls:
        ang_rad = float(np.interp(float(awl), np.asarray(wls), angs_unwrapped))
        anchor_angles[awl] = float(np.rad2deg(ang_rad) % 360.0)
        anchor_vecs[awl] = np.array([np.cos(ang_rad), np.sin(ang_rad)], dtype=np.float32)

    scene_obs = eye.observe(real_scene_rgb, max_width=240)
    a_min, a_max = float(np.min(scene_obs.l1.luminance_a)), float(np.max(scene_obs.l1.luminance_a))
    tex_max = float(np.max(scene_obs.l2.fine_texture_map))

    axes = [
        SelfDefinedAxisSpec(
            axis_id="axis_luminance",
            name="Log-Photoreceptor Luminance Axis (EV)",
            provenance=LayerProvenance(
                layer_id="SELF_AXIS_LUMA",
                name="Additive Log-Cone Exposure Axis",
                domain="Vision",
                causal_mechanism="Translates achromatic log(L*M)^0.5 along EV stops while preserving opponent ratios.",
                why_necessary="Changes scene exposure cleanly without distorting hue or structural boundaries.",
                key_metrics={"a_min": a_min, "a_max": a_max},
            ),
            metric_unit="EV (log2 irradiance)",
            empirical_span=(-0.7, 0.7),
        ),
        SelfDefinedAxisSpec(
            axis_id="rotor_rainbow_hue",
            name="Whitened Cone-Opponent Clifford Rainbow Rotor",
            provenance=LayerProvenance(
                layer_id="SELF_AXIS_HUE_ROTOR",
                name="Continuous Spectral Chromatic Rotor",
                domain="Vision + Structure",
                causal_mechanism="Clifford Cl(3,0) bivector rotor R(theta) = cos(t/2) - e12*sin(t/2) on whitened (RG, BY).",
                why_necessary="Preserves periodic 360-deg rainbow continuity and exact algebraic rewind R~ v' R.",
                key_metrics={"sigma_rg": sigma_rg, "sigma_by": sigma_by, "red_deg": anchor_angles[650], "yellow_deg": anchor_angles[580]},
            ),
            metric_unit="deg [0, 360)",
            empirical_span=(0.0, 360.0),
        ),
        SelfDefinedAxisSpec(
            axis_id="axis_chroma",
            name="Radial Opponent Chroma / Saturation Axis",
            provenance=LayerProvenance(
                layer_id="SELF_AXIS_CHROMA",
                name="Radial Opponent Distance from Achromatic Neutral",
                domain="Vision",
                causal_mechanism="Scales or pulls whitened chromatic vector u=(u_rg, u_by) radially or toward spectral anchors.",
                why_necessary="Separates pure neutral stone from chromatic pigment intensity without touching structure.",
                key_metrics={"prism_radius_mean": float(np.mean([np.hypot(x, y) for x, y in prism_u]))},
            ),
            metric_unit="whitened opponent units",
            empirical_span=(0.0, 1.5),
        ),
        SelfDefinedAxisSpec(
            axis_id="axis_texture_roughness",
            name="Scale-Separated Fine Texture Roughness Axis",
            provenance=LayerProvenance(
                layer_id="SELF_AXIS_ROUGHNESS",
                name="High-Pass Luminance Micro-Contrast Gain",
                domain="Vision + Structure",
                causal_mechanism="Scales high-pass band a_detail = a - G_sigma4(a) independently of chromatic channels.",
                why_necessary="Modulates tactile/optical surface roughness without shifting color temperature.",
                key_metrics={"scene_fine_texture_max": tex_max},
            ),
            metric_unit="dimensionless gain",
            empirical_span=(0.1, 2.0),
        ),
    ]

    return PerceptualAxisManifold(
        sigma_rg=sigma_rg,
        sigma_by=sigma_by,
        prism_wavelengths_nm=wls,
        prism_angles_deg=prism_angles,
        prism_u_coords=prism_u,
        anchor_angles_deg=anchor_angles,
        anchor_unit_vecs=anchor_vecs,
        axes=axes,
    )


class ImaginativeReobservationEngine:
    """Executes closed-loop perceptual excursions and re-observes them via OcularScaleOrgan."""

    def __init__(
        self,
        eye: Optional[OcularScaleOrgan] = None,
        reasoner: Optional[VisualCausalReasoner] = None,
    ) -> None:
        self.eye = eye if eye is not None else OcularScaleOrgan()
        self.reasoner = reasoner if reasoner is not None else VisualCausalReasoner(self.eye)

    def apply_imagination(
        self,
        orig_obs: OpticalSceneObservation,
        manifold: PerceptualAxisManifold,
        op: PerceptualImaginationOp,
        region_id: Optional[int] = None,
    ) -> Tuple[np.ndarray, float]:
        a = orig_obs.l1.luminance_a.astype(np.float64)
        rg = orig_obs.l1.rg_opponent.astype(np.float64)
        by = orig_obs.l1.by_opponent.astype(np.float64)
        h, w = a.shape

        if region_id is not None:
            hard = (orig_obs.region_labels == region_id).astype(np.float32)
            mask = np.clip(gaussian_filter(hard, sigma=1.5), 0.0, 1.0).astype(np.float64)
            region_bool = orig_obs.region_labels == region_id
        else:
            mask = np.ones((h, w), dtype=np.float64)
            region_bool = np.ones((h, w), dtype=bool)

        u_rg = rg / manifold.sigma_rg
        u_by = by / manifold.sigma_by

        # 1. Hue Rotor via Clifford Cl(3,0) Sandwich Product R v R~
        if abs(op.hue_rotor_deg) > 1e-8:
            z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            rotor = CliffordSpacetimeRotor.create_rotor_3d(
                z_axis, float(np.deg2rad(op.hue_rotor_deg))
            )
            vecs = np.stack([u_rg.ravel(), u_by.ravel(), np.zeros(h * w, dtype=np.float64)], axis=-1)
            rotated = CliffordSpacetimeRotor.sandwich_transform_vector_3d(rotor, vecs)
            u_rg = rotated[:, 0].reshape(h, w)
            u_by = rotated[:, 1].reshape(h, w)

        # 2. Radial Chroma Scaling
        u_rg = u_rg * float(op.chroma_scale)
        u_by = u_by * float(op.chroma_scale)

        # 3. Spectral Pull Toward Self-Mapped Wavelength Anchor or Custom Closed-Loop Shift
        if op.target_wavelength_nm is not None and abs(op.target_chroma_pull) > 1e-8:
            t_vec = manifold.spectral_unit_vec(op.target_wavelength_nm).astype(np.float64)
            reg_u_rg0 = float(np.mean(u_rg[region_bool])) if np.any(region_bool) else 0.0
            reg_u_by0 = float(np.mean(u_by[region_bool])) if np.any(region_bool) else 0.0
            u_rg = u_rg + float(op.target_chroma_pull) * float(t_vec[0]) - float(op.chroma_dissolve_frac) * reg_u_rg0
            u_by = u_by + float(op.target_chroma_pull) * float(t_vec[1]) - float(op.chroma_dissolve_frac) * reg_u_by0

        if op.custom_u_shift is not None:
            u_rg = u_rg + float(op.custom_u_shift[0])
            u_by = u_by + float(op.custom_u_shift[1])

        rg_mod = u_rg * manifold.sigma_rg
        by_mod = u_by * manifold.sigma_by

        # 4. Luminance (EV) & Multi-Scale Texture Roughness
        if abs(op.roughness_scale - 1.0) > 1e-8:
            a_base = gaussian_filter(a, sigma=4.0)
            a_detail = a - a_base
            a_mod = a_base + float(op.roughness_scale) * a_detail
        else:
            a_mod = a.copy()

        if abs(op.delta_ev) > 1e-8:
            a_mod = a_mod + float(op.delta_ev) * float(np.log(2.0))

        # Blend smoothly by regional mask
        a_out = a + mask * (a_mod - a)
        rg_out = rg + mask * (rg_mod - rg)
        by_out = by + mask * (by_mod - by)

        return decode_opponent_to_srgb(a_out, rg_out, by_out, eps=self.eye.EPS)

    def sweep_regional_chroma_toward_spectrum(
        self,
        orig_obs: OpticalSceneObservation,
        manifold: PerceptualAxisManifold,
        region_id: int,
        target_wavelength_nm: float,
        target_name: str,
        alphas: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        max_chroma_pull: float = 0.85,
    ) -> RegionalChromaSweepResult:
        orig_mask = orig_obs.region_labels == region_id
        if not np.any(orig_mask):
            raise ValueError(f"Region {region_id} not present in observation.")

        orig_u_rg = float(np.mean(orig_obs.l1.rg_opponent[orig_mask])) / manifold.sigma_rg
        orig_u_by = float(np.mean(orig_obs.l1.by_opponent[orig_mask])) / manifold.sigma_by
        orig_a_mean = float(np.mean(orig_obs.l1.luminance_a[orig_mask]))
        orig_mat_cnt = sum(1 for b in orig_obs.boundaries if b.optical_cause == "MATERIAL_BOUNDARY")

        steps: List[ChromaSweepStep] = []
        tipping_alpha: Optional[float] = None
        min_pers = 1.0

        for alpha in alphas:
            op = PerceptualImaginationOp(
                target_wavelength_nm=target_wavelength_nm,
                target_chroma_pull=float(alpha) * float(max_chroma_pull),
                chroma_dissolve_frac=float(alpha) * 0.65,
            )
            imagined_rgb, clip_ratio = self.apply_imagination(
                orig_obs, manifold, op, region_id=region_id
            )
            reobs = self.eye.observe(imagined_rgb, max_width=orig_obs.shape_hw[1])
            comp = self.reasoner.compare_observations(orig_obs, reobs)

            cur_u_rg = float(np.mean(reobs.l1.rg_opponent[orig_mask])) / manifold.sigma_rg
            cur_u_by = float(np.mean(reobs.l1.by_opponent[orig_mask])) / manifold.sigma_by
            cur_a_mean = float(np.mean(reobs.l1.luminance_a[orig_mask]))

            reg_dc = float(np.hypot(cur_u_rg - orig_u_rg, cur_u_by - orig_u_by))
            reg_da = float(abs(cur_a_mean - orig_a_mean))
            mat_cnt = sum(1 for b in reobs.boundaries if b.optical_cause == "MATERIAL_BOUNDARY")
            shad_cnt = sum(
                1 for b in reobs.boundaries if b.optical_cause == "ILLUMINATION_SHADOW_CANDIDATE"
            )
            min_pers = min(min_pers, comp.l2_persistence_corr)

            if tipping_alpha is None and alpha > 0.0:
                if (reg_dc >= 0.35 or (mat_cnt - orig_mat_cnt) >= 1) and comp.l2_persistence_corr >= 0.80:
                    tipping_alpha = float(alpha)

            steps.append(
                ChromaSweepStep(
                    alpha=float(alpha),
                    imagined_rgb=imagined_rgb,
                    reobs=reobs,
                    comparison=comp,
                    regional_mean_u_rg=cur_u_rg,
                    regional_mean_u_by=cur_u_by,
                    regional_delta_c=reg_dc,
                    regional_delta_a=reg_da,
                    material_boundary_count=mat_cnt,
                    shadow_boundary_count=shad_cnt,
                    gamut_clip_ratio=clip_ratio,
                )
            )

        return RegionalChromaSweepResult(
            region_id=region_id,
            target_wavelength_nm=float(target_wavelength_nm),
            target_name=target_name,
            steps=steps,
            alpha_tipping_point=tipping_alpha,
            min_l2_persistence_corr=float(min_pers),
        )

    def sweep_full_rainbow_rotor(
        self,
        orig_obs: OpticalSceneObservation,
        manifold: PerceptualAxisManifold,
        angles_deg: Tuple[float, ...] = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0, 360.0),
    ) -> RainbowRotorSweepResult:
        orig_luma_op = np.stack(
            [orig_obs.l1.luminance_a, orig_obs.l1.rg_opponent, orig_obs.l1.by_opponent], axis=-1
        )
        steps: List[RainbowRotorStep] = []
        min_pers = 1.0

        for deg in angles_deg:
            op = PerceptualImaginationOp(hue_rotor_deg=float(deg))
            img_rgb, clip_ratio = self.apply_imagination(orig_obs, manifold, op, region_id=None)
            reobs = self.eye.observe(img_rgb, max_width=orig_obs.shape_hw[1])
            comp = self.reasoner.compare_observations(orig_obs, reobs)
            min_pers = min(min_pers, comp.l2_persistence_corr)

            cur_op = np.stack(
                [reobs.l1.luminance_a, reobs.l1.rg_opponent, reobs.l1.by_opponent], axis=-1
            )
            rmse = float(np.sqrt(np.mean((cur_op - orig_luma_op) ** 2)))
            steps.append(
                RainbowRotorStep(
                    angle_deg=float(deg),
                    reobs=reobs,
                    cone_rmse_vs_orig=rmse,
                    l2_persistence_corr=comp.l2_persistence_corr,
                    mean_u_rg=float(np.mean(reobs.l1.rg_opponent)) / manifold.sigma_rg,
                    mean_u_by=float(np.mean(reobs.l1.by_opponent)) / manifold.sigma_by,
                    gamut_clip_ratio=clip_ratio,
                )
            )

        # Closure at 360 deg + explicit algebraic scale-rotor rewind inside sRGB gamut: S^-1 o S
        closure_rmse = steps[-1].cone_rmse_vs_orig if abs(steps[-1].angle_deg - 360.0) < 1e-4 else 1.0

        rgb_120, _ = self.apply_imagination(
            orig_obs,
            manifold,
            PerceptualImaginationOp(hue_rotor_deg=120.0, chroma_scale=0.5),
            region_id=None,
        )
        obs_120 = self.eye.observe(rgb_120, max_width=orig_obs.shape_hw[1])
        rgb_rewound, _ = self.apply_imagination(
            obs_120,
            manifold,
            PerceptualImaginationOp(hue_rotor_deg=-120.0, chroma_scale=2.0),
            region_id=None,
        )
        obs_rewound = self.eye.observe(rgb_rewound, max_width=orig_obs.shape_hw[1])
        rew_op = np.stack(
            [obs_rewound.l1.luminance_a, obs_rewound.l1.rg_opponent, obs_rewound.l1.by_opponent],
            axis=-1,
        )
        rewind_rmse = float(np.sqrt(np.mean((rew_op - orig_luma_op) ** 2)))

        return RainbowRotorSweepResult(
            steps=steps,
            closure_rmse_360=closure_rmse,
            rewind_rmse_120=rewind_rmse,
            min_l2_persistence_corr=float(min_pers),
        )

    def sweep_luminance_and_roughness(
        self,
        orig_obs: OpticalSceneObservation,
        manifold: PerceptualAxisManifold,
        region_id: Optional[int] = None,
    ) -> OrthogonalitySweepResult:
        eval_mask = (
            (orig_obs.region_labels == region_id)
            if region_id is not None
            else np.ones(orig_obs.shape_hw, dtype=bool)
        )
        lum_evs = [-0.6, -0.3, 0.0, 0.25, 0.5]
        lum_dcs: List[float] = []
        lum_corrs: List[float] = []
        for ev in lum_evs:
            rgb, _ = self.apply_imagination(
                orig_obs, manifold, PerceptualImaginationOp(delta_ev=ev), region_id=region_id
            )
            reobs = self.eye.observe(rgb, max_width=orig_obs.shape_hw[1])
            comp = self.reasoner.compare_observations(orig_obs, reobs)
            lum_dcs.append(float(comp.l1_delta_c))
            lum_corrs.append(float(comp.l2_persistence_corr))

        rough_scales = [0.1, 0.5, 1.0, 1.5, 2.0]
        rough_energies: List[float] = []
        rough_corrs: List[float] = []
        for rs in rough_scales:
            rgb, _ = self.apply_imagination(
                orig_obs, manifold, PerceptualImaginationOp(roughness_scale=rs), region_id=region_id
            )
            reobs = self.eye.observe(rgb, max_width=orig_obs.shape_hw[1])
            comp = self.reasoner.compare_observations(orig_obs, reobs)
            rough_energies.append(float(np.mean(reobs.l2.fine_texture_map[eval_mask])))
            rough_corrs.append(float(comp.l2_persistence_corr))

        return OrthogonalitySweepResult(
            luminance_evs=lum_evs,
            luminance_delta_cs=lum_dcs,
            luminance_persistence_corrs=lum_corrs,
            roughness_scales=rough_scales,
            roughness_fine_energies=rough_energies,
            roughness_persistence_corrs=rough_corrs,
        )

    def autonomous_goal_recreation(
        self,
        orig_obs: OpticalSceneObservation,
        manifold: PerceptualAxisManifold,
        region_id: int,
        target_wavelength_nm: float = 605.0,
        target_chroma_norm: float = 0.72,
        target_roughness_ratio: float = 1.40,
        max_iters: int = 6,
    ) -> GoalRecreationTrace:
        """
        True closed-loop feedback: Imagine -> Re-observe through OcularScaleOrgan ->
        Compute vector discrepancy on self-mapped axes -> Refine parameters until convergence.
        """
        orig_mask = orig_obs.region_labels == region_id
        if not np.any(orig_mask):
            raise ValueError(f"Region {region_id} not present.")

        orig_fine_tex = float(np.mean(orig_obs.l2.fine_texture_map[orig_mask]))
        target_fine_tex = float(orig_fine_tex * target_roughness_ratio)

        t_dir = manifold.spectral_unit_vec(target_wavelength_nm).astype(np.float64)
        u_goal = (float(target_chroma_norm * t_dir[0]), float(target_chroma_norm * t_dir[1]))

        du_rg = 0.0
        du_by = 0.0
        rough_scale = 1.0

        iterations: List[GoalIterationRecord] = []
        final_rgb = orig_obs.rgb_working.copy()
        final_reobs = orig_obs
        final_op = PerceptualImaginationOp()

        for it in range(max_iters + 1):
            op = PerceptualImaginationOp(
                custom_u_shift=(du_rg, du_by),
                roughness_scale=rough_scale,
            )
            img_rgb, clip_ratio = self.apply_imagination(
                orig_obs, manifold, op, region_id=region_id
            )
            reobs = self.eye.observe(img_rgb, max_width=orig_obs.shape_hw[1])
            comp = self.reasoner.compare_observations(orig_obs, reobs)

            cur_u_rg = float(np.mean(reobs.l1.rg_opponent[orig_mask])) / manifold.sigma_rg
            cur_u_by = float(np.mean(reobs.l1.by_opponent[orig_mask])) / manifold.sigma_by
            cur_tex = float(np.mean(reobs.l2.fine_texture_map[orig_mask]))

            err_rg = u_goal[0] - cur_u_rg
            err_by = u_goal[1] - cur_u_by
            color_err = float(np.hypot(err_rg, err_by))
            rough_err = float(abs(target_fine_tex - cur_tex) / max(target_fine_tex, 1e-4))
            tot_err = float(np.hypot(color_err, rough_err))

            iterations.append(
                GoalIterationRecord(
                    iteration=it,
                    cur_u_rg=cur_u_rg,
                    cur_u_by=cur_u_by,
                    cur_fine_texture=cur_tex,
                    color_error=color_err,
                    roughness_error=rough_err,
                    total_error=tot_err,
                    l2_persistence_corr=comp.l2_persistence_corr,
                    gamut_clip_ratio=clip_ratio,
                )
            )
            final_rgb = img_rgb
            final_reobs = reobs
            final_op = op

            if it < max_iters:
                # Damped closed-loop proportional update driven solely by re-observed state
                du_rg += 0.85 * err_rg
                du_by += 0.85 * err_by
                rel_r = (target_fine_tex - cur_tex) / max(target_fine_tex, 1e-4)
                rough_scale = float(np.clip(rough_scale * (1.0 + 0.70 * rel_r), 0.15, 3.0))

        init_err = iterations[0].total_error
        end_err = iterations[-1].total_error
        converged = bool(
            end_err < init_err * 0.45 and iterations[-1].l2_persistence_corr >= 0.82
        )

        return GoalRecreationTrace(
            region_id=region_id,
            target_wavelength_nm=float(target_wavelength_nm),
            target_chroma_norm=float(target_chroma_norm),
            target_roughness_ratio=float(target_roughness_ratio),
            target_u_coords=u_goal,
            target_fine_texture=target_fine_tex,
            iterations=iterations,
            final_imagined_rgb=final_rgb,
            final_reobs=final_reobs,
            final_op=final_op,
            converged=converged,
        )


class CrossDomainPrincipleMapper:
    """
    Translates self-mapped continuous perceptual & structural axes into external
    Game Engine PBR Shader uniforms and External Video ASC-CDL / Hue-Rotor grades,
    preserving honest domain-separation notes.
    """

    @staticmethod
    def to_game_engine_pbr(
        orig_obs: OpticalSceneObservation,
        manifold: PerceptualAxisManifold,
        goal_trace: GoalRecreationTrace,
        anisotropy_angle_deg: float = 30.0,
        anisotropy_strength: float = 0.72,
    ) -> GameEnginePBRMaterialSpec:
        mask = orig_obs.region_labels == goal_trace.region_id
        reobs = goal_trace.final_reobs
        a_reg = np.array([[float(np.mean(reobs.l1.luminance_a[mask]))]], dtype=np.float64)
        rg_reg = np.array([[float(np.mean(reobs.l1.rg_opponent[mask]))]], dtype=np.float64)
        by_reg = np.array([[float(np.mean(reobs.l1.by_opponent[mask]))]], dtype=np.float64)
        srgb_patch, _ = decode_opponent_to_srgb(a_reg, rg_reg, by_reg)
        lin_rgb = np.power(np.clip(srgb_patch[0, 0], 0.0, 1.0), 2.2)

        u_rg, u_by = goal_trace.iterations[-1].cur_u_rg, goal_trace.iterations[-1].cur_u_by
        hue_deg = float(np.rad2deg(np.arctan2(u_by, u_rg)) % 360.0)
        sat = float(np.clip(np.hypot(u_rg, u_by) / 1.5, 0.0, 1.0))

        # Roughness from observed fine texture ratio mapped smoothly into [0.05, 0.95] BRDF
        rough_val = float(
            np.clip(0.25 + 0.35 * goal_trace.iterations[-1].cur_fine_texture, 0.05, 0.95)
        )
        orig_a = float(np.mean(orig_obs.l1.luminance_a[mask]))
        ev_shift = float((float(a_reg[0, 0]) - orig_a) / np.log(2.0))

        return GameEnginePBRMaterialSpec(
            albedo_linear_rgb=(float(lin_rgb[0]), float(lin_rgb[1]), float(lin_rgb[2])),
            albedo_hue_deg=hue_deg,
            albedo_saturation=sat,
            exposure_ev=ev_shift,
            microfacet_roughness=rough_val,
            anisotropy_angle_deg=float(anisotropy_angle_deg % 180.0),
            anisotropy_strength=float(np.clip(anisotropy_strength, 0.0, 1.0)),
            domain_boundary_note=(
                "PBR microfacet roughness & tangent anisotropy govern surface photon BRDF reflection only; "
                "physical bulk fracture/yield still requires separate cohesive tensile/cleavage lattice bindings."
            ),
        )

    @staticmethod
    def to_video_color_grade(
        orig_obs: OpticalSceneObservation,
        manifold: PerceptualAxisManifold,
        goal_trace: GoalRecreationTrace,
    ) -> VideoColorGradeSpec:
        mask = orig_obs.region_labels == goal_trace.region_id
        orig_a = float(np.mean(orig_obs.l1.luminance_a[mask]))
        orig_rg = float(np.mean(orig_obs.l1.rg_opponent[mask]))
        orig_by = float(np.mean(orig_obs.l1.by_opponent[mask]))
        orig_srgb, _ = decode_opponent_to_srgb(
            np.array([[orig_a]]), np.array([[orig_rg]]), np.array([[orig_by]])
        )
        orig_lin = np.maximum(np.power(orig_srgb[0, 0], 2.2), 1e-3)

        reobs = goal_trace.final_reobs
        fin_a = float(np.mean(reobs.l1.luminance_a[mask]))
        fin_rg = float(np.mean(reobs.l1.rg_opponent[mask]))
        fin_by = float(np.mean(reobs.l1.by_opponent[mask]))
        fin_srgb, _ = decode_opponent_to_srgb(
            np.array([[fin_a]]), np.array([[fin_rg]]), np.array([[fin_by]])
        )
        fin_lin = np.power(fin_srgb[0, 0], 2.2)

        slope = np.clip(fin_lin / orig_lin, 0.25, 4.0)

        u0_ang = np.arctan2(orig_by / manifold.sigma_by, orig_rg / manifold.sigma_rg)
        u1_ang = np.arctan2(fin_by / manifold.sigma_by, fin_rg / manifold.sigma_rg)
        hue_delta = float(np.rad2deg(np.arctan2(np.sin(u1_ang - u0_ang), np.cos(u1_ang - u0_ang))))

        c0 = max(float(np.hypot(orig_rg / manifold.sigma_rg, orig_by / manifold.sigma_by)), 0.1)
        c1 = float(np.hypot(fin_rg / manifold.sigma_rg, fin_by / manifold.sigma_by))
        chroma_gain = float(np.clip(c1 / c0, 0.2, 4.0))

        return VideoColorGradeSpec(
            cdl_slope_rgb=(float(slope[0]), float(slope[1]), float(slope[2])),
            cdl_offset_ev=float((fin_a - orig_a) / np.log(2.0)),
            hue_rotor_deg=hue_delta,
            chroma_gain=chroma_gain,
            detail_sharpen_gain=float(goal_trace.final_op.roughness_scale),
            domain_boundary_note=(
                "ASC-CDL slope/offset & Clifford hue rotor transform incoming footage radiometrically without "
                "inventing new 3D physical scene geometry; persistence invariance protects region boundaries."
            ),
        )


def run_full_imaginative_session(
    real_scene_rgb: np.ndarray,
    target_region_id: Optional[int] = None,
) -> ImaginativeExplorationSession:
    """End-to-end orchestration building manifold, sweeps, closed-loop recreation, transfers, and CausalField."""
    eye = OcularScaleOrgan()
    reasoner = VisualCausalReasoner(eye)
    engine = ImaginativeReobservationEngine(eye=eye, reasoner=reasoner)

    orig_obs = eye.observe(real_scene_rgb, max_width=240)
    manifold = calibrate_perceptual_manifold(eye, real_scene_rgb)

    if target_region_id is None:
        # Select most salient non-backdrop coherent region in [0.05, 0.40] area range
        cands = [r for r in orig_obs.regions if 0.05 <= r.area_ratio <= 0.40]
        if not cands:
            cands = [r for r in orig_obs.regions if r.region_id != 0] or [orig_obs.regions[0]]
        cands.sort(key=lambda r: (-r.area_ratio, r.region_id))
        target_region_id = cands[0].region_id

    red_sweep = engine.sweep_regional_chroma_toward_spectrum(
        orig_obs, manifold, region_id=target_region_id, target_wavelength_nm=650.0, target_name="Crimson Red (650nm)"
    )
    yellow_sweep = engine.sweep_regional_chroma_toward_spectrum(
        orig_obs, manifold, region_id=target_region_id, target_wavelength_nm=580.0, target_name="Golden Yellow (580nm)"
    )
    rainbow_sweep = engine.sweep_full_rainbow_rotor(orig_obs, manifold)
    ortho_sweep = engine.sweep_luminance_and_roughness(
        orig_obs, manifold, region_id=target_region_id
    )
    goal_trace = engine.autonomous_goal_recreation(
        orig_obs,
        manifold,
        region_id=target_region_id,
        target_wavelength_nm=605.0,
        target_chroma_norm=0.72,
        target_roughness_ratio=1.40,
        max_iters=6,
    )

    pbr_spec = CrossDomainPrincipleMapper.to_game_engine_pbr(orig_obs, manifold, goal_trace)
    video_spec = CrossDomainPrincipleMapper.to_video_color_grade(orig_obs, manifold, goal_trace)

    # Multi-Scale CausalField Projection (L2 -> L10 Self-Mapped Axes -> L12 Re-Observations -> L14 Transfers)
    field_obj = reasoner.project_to_causal_field(orig_obs)

    axis_vids: List[str] = []
    for idx, ax in enumerate(manifold.axes):
        vid = ax.axis_id
        axis_vids.append(vid)
        field_obj.add_voxel(
            InformationVoxel(
                id=vid,
                content={"type": "self_mapped_perceptual_axis", "name": ax.name, "unit": ax.metric_unit},
                tensor=np.array(
                    [float(ax.empirical_span[0]), float(ax.empirical_span[1]), float(idx + 1), manifold.sigma_rg, manifold.sigma_by, 1.0],
                    dtype=np.float32,
                ),
                mass=1.8,
                position=np.array([0.15 + 0.23 * idx, 0.86, 0.9], dtype=np.float32),
                scale_level=10.0,
            )
        )
        target_vid = f"region_{target_region_id}"
        if target_vid in field_obj.voxels:
            field_obj.link_voxels(target_vid, vid, strength=2.2)

    imag_specs = [
        (
            "imagination_red_sweep",
            np.array(
                [
                    red_sweep.steps[-1].regional_delta_c,
                    red_sweep.min_l2_persistence_corr,
                    red_sweep.alpha_tipping_point or 1.0,
                    red_sweep.steps[-1].gamut_clip_ratio,
                    650.0 / 700.0,
                    1.0,
                ],
                dtype=np.float32,
            ),
            0.22,
        ),
        (
            "imagination_yellow_sweep",
            np.array(
                [
                    yellow_sweep.steps[-1].regional_delta_c,
                    yellow_sweep.min_l2_persistence_corr,
                    yellow_sweep.alpha_tipping_point or 1.0,
                    yellow_sweep.steps[-1].gamut_clip_ratio,
                    580.0 / 700.0,
                    1.0,
                ],
                dtype=np.float32,
            ),
            0.50,
        ),
        (
            "imagination_goal_recreation",
            np.array(
                [
                    goal_trace.iterations[-1].total_error,
                    goal_trace.iterations[-1].l2_persistence_corr,
                    goal_trace.final_op.roughness_scale,
                    goal_trace.target_chroma_norm,
                    1.0 if goal_trace.converged else 0.0,
                    float(len(goal_trace.iterations)),
                ],
                dtype=np.float32,
            ),
            0.78,
        ),
    ]
    imag_vids: List[str] = []
    for vid, vec, px in imag_specs:
        imag_vids.append(vid)
        field_obj.add_voxel(
            InformationVoxel(
                id=vid,
                content={"type": "imagined_reobservation", "id": vid},
                tensor=vec,
                mass=2.0,
                position=np.array([px, 0.94, 1.2], dtype=np.float32),
                scale_level=12.0,
            )
        )
        for ax_v in axis_vids:
            field_obj.link_voxels(ax_v, vid, strength=1.7)

    transfer_specs = [
        (
            "transfer_game_pbr_shader",
            np.array(
                [
                    pbr_spec.albedo_linear_rgb[0],
                    pbr_spec.albedo_linear_rgb[1],
                    pbr_spec.albedo_linear_rgb[2],
                    pbr_spec.microfacet_roughness,
                    pbr_spec.albedo_hue_deg / 360.0,
                    pbr_spec.anisotropy_strength,
                ],
                dtype=np.float32,
            ),
            0.35,
        ),
        (
            "transfer_video_grade_cdl",
            np.array(
                [
                    video_spec.cdl_slope_rgb[0],
                    video_spec.cdl_slope_rgb[1],
                    video_spec.cdl_slope_rgb[2],
                    video_spec.chroma_gain,
                    video_spec.hue_rotor_deg / 180.0,
                    video_spec.detail_sharpen_gain,
                ],
                dtype=np.float32,
            ),
            0.65,
        ),
    ]
    for vid, vec, px in transfer_specs:
        field_obj.add_voxel(
            InformationVoxel(
                id=vid,
                content={"type": "cross_domain_transfer", "id": vid},
                tensor=vec,
                mass=2.2,
                position=np.array([px, 1.0, 1.5], dtype=np.float32),
                scale_level=14.0,
            )
        )
        for im_v in imag_vids:
            field_obj.link_voxels(im_v, vid, strength=1.5)

    field_obj.step(dt=0.05)
    audit = reasoner.audit_organ_causality(orig_obs)

    return ImaginativeExplorationSession(
        orig_obs=orig_obs,
        manifold=manifold,
        target_region_id=target_region_id,
        red_sweep=red_sweep,
        yellow_sweep=yellow_sweep,
        rainbow_sweep=rainbow_sweep,
        orthogonality_sweep=ortho_sweep,
        goal_recreation=goal_trace,
        pbr_spec=pbr_spec,
        video_grade_spec=video_spec,
        final_causal_field=field_obj,
        organ_audit=audit,
    )
