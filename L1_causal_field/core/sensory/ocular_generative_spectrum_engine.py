"""
ocular_generative_spectrum_engine.py

ELYSIA Ocular Axis-Spectrum Factorization, Reproduction, Modulation & Recreation
=================================================================================
Extends OcularScaleOrgan and ActiveOcularMechanicsEngine from passive/active
perception into bidirectional causal synthesis — without VLMs, LLMs, or neural
diffusion models:

1. Axis-Spectrum Factorization: Decomposes perceived reality into orthogonal
   physical-cognitive axes:
   - Axis 1: Cleaved Boundary & Region Topology Mold (L2/L3 optical cleavage)
   - Axis 2: Illumination / Luminance Field DoG Spectrum across 5 spatial scales
   - Axis 3: Chromatic Opponent Axes (RG = log(L/M), BY = log(S) - A)
   - Axis 4: Multi-Scale Energy & Orientation Envelope (L2 differential tensors)
2. Bidirectional Retinal Inversion: Analytically inverts opponent & log-LMS
   transduction back to linear irradiance and sRGB images.
3. Orthogonal Modulation: Adjusts illumination, chromatic opponent angle/gain,
   and individual spatial frequency bands independently.
4. Cross-Scene Recreation & Closed-Loop Self-Verification: Combines one scene's
   structural boundary mold with another scene's luminance-stratified chromatic
   and multi-scale energy spectrum, then immediately re-observes its own creation
   through OcularScaleOrgan to verify structural preservation and domain transfer
   inside CausalField(6D).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from core.physics.causal_field import CausalField, InformationVoxel
from core.sensory.active_ocular_mechanics_engine import (
    ActiveOcularMechanicsEngine,
    FovealEngram,
    _extract_optical_descriptor,
)
from core.sensory.ocular_scale_organ import (
    OcularScaleOrgan,
    OpticalSceneObservation,
    _SRGB_TO_LMS,
    _normalize_image_array,
)


@dataclass
class OcularCognitiveFactorization:
    """Orthogonal perceptual axes & multi-scale spectral bands extracted from an image."""

    source_name: str
    shape_hw: Tuple[int, int]
    source_rgb_working_u8: np.ndarray
    obs: OpticalSceneObservation
    # Axis 1: Boundary & region topology mold
    region_labels: np.ndarray
    boundary_persistence: np.ndarray
    material_boundary_mask: np.ndarray
    region_stats: List[Dict[str, Any]]
    # Axis 2: Illumination / luminance DoG spectrum
    luma_base: np.ndarray
    luma_bands: List[np.ndarray]
    # Axis 3: Chromatic opponent axes & DoG spectra
    rg_base: np.ndarray
    rg_bands: List[np.ndarray]
    by_base: np.ndarray
    by_bands: List[np.ndarray]
    global_rg_mean_std: Tuple[float, float]
    global_by_mean_std: Tuple[float, float]
    # Axis 4: Multi-scale energy & orientation descriptor
    scale_band_energies: np.ndarray
    orientation_spectrum: np.ndarray
    chromatic_signature: np.ndarray
    orientation_anisotropy: float
    vector_6d: np.ndarray
    voxel_id: str


@dataclass
class GenerativeVerificationResult:
    """Result of synthesis plus immediate closed-loop re-observation via OcularScaleOrgan."""

    operation_name: str
    mode: str
    synthesized_rgb: np.ndarray
    reobserved_obs: OpticalSceneObservation
    reobserved_vector_6d: np.ndarray
    reobserved_scale_profile: np.ndarray
    reobserved_chromatic_signature: np.ndarray
    structural_corr_with_source: float
    optical_resonance_with_source: float
    optical_resonance_with_palette_donor: Optional[float]
    delta_luma_mean: float
    delta_chroma_mean: float
    psnr_db: Optional[float]
    voxel_id: str


def _decompose_telescoping_bands(
    field: np.ndarray, scales: Tuple[float, ...]
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Exact telescoping Gaussian/DoG decomposition:
    base + sum(bands) == field identically (modulo float32 precision).
    """
    smoothed = [gaussian_filter(field, sigma=float(s)).astype(np.float32) for s in scales]
    bands: List[np.ndarray] = []
    prev = field.astype(np.float32)
    for sm in smoothed:
        bands.append((prev - sm).astype(np.float32))
        prev = sm
    return smoothed[-1], bands


def _recompose_telescoping_bands(
    base: np.ndarray,
    bands: List[np.ndarray],
    band_gains: Optional[Tuple[float, ...]] = None,
) -> np.ndarray:
    out = base.astype(np.float32).copy()
    for i, b in enumerate(bands):
        gain = float(band_gains[i]) if band_gains is not None and i < len(band_gains) else 1.0
        out += gain * b
    return out


def _compute_persistence_correlation(
    obs_a: OpticalSceneObservation, obs_b: OpticalSceneObservation
) -> float:
    p1 = obs_a.l2.persistence_map.ravel()
    p2 = obs_b.l2.persistence_map.ravel()
    if p1.size != p2.size or p1.size == 0:
        return 0.0
    if np.std(p1) < 1e-6 or np.std(p2) < 1e-6:
        return 1.0 if np.allclose(p1, p2) else 0.0
    return float(np.corrcoef(p1, p2)[0, 1])


def _fact_as_engram_proxy(fact: OcularCognitiveFactorization) -> FovealEngram:
    """Wrap factorization summary into FovealEngram shape for shared resonance scoring."""
    h, w = fact.shape_hw
    return FovealEngram(
        engram_id=fact.source_name,
        scene_name=fact.source_name,
        fixation_idx=0,
        gaze_yx_norm=(0.5, 0.5),
        gaze_yx_px=(h // 2, w // 2),
        fovea_box_yxyx=(0, 0, h, w),
        vector_6d=fact.vector_6d,
        scale_profile=fact.scale_band_energies,
        orientation_spectrum=fact.orientation_spectrum,
        chromatic_signature=fact.chromatic_signature,
        cleaved_bond_ratio=float(fact.obs.l2.bond_lattice.cleaved_bond_ratio),
        material_boundary_count=int(np.sum(fact.material_boundary_mask)),
        region_count=len(fact.region_stats),
        salience_peak=1.0,
        recognition_status="FACTORIZATION",
        matched_engram_id=None,
        resonance_score=1.0,
        voxel_id=fact.voxel_id,
    )


class OcularGenerativeSpectrumEngine:
    """
    Axis-Spectrum Factorization, Faithful Reproduction, Orthogonal Modulation,
    and Cross-Scene Recreation over pure retinal physics + CausalField (6D).
    """

    def __init__(self, organ: Optional[OcularScaleOrgan] = None) -> None:
        self.organ = organ if organ is not None else OcularScaleOrgan()
        self.eye = ActiveOcularMechanicsEngine(organ=self.organ)
        self.causal_field: CausalField = self.eye.causal_field
        self._inv_srgb_to_lms = np.linalg.inv(_SRGB_TO_LMS).astype(np.float32)
        self._op_counter = 0

    def _inverse_retinal_to_srgb(
        self, a: np.ndarray, rg: np.ndarray, by: np.ndarray
    ) -> np.ndarray:
        """
        Analytically invert retinal opponent channels -> log-LMS -> linear-LMS -> sRGB.
        Inverse of OcularScaleOrgan._transduce_photoreceptors & _compute_retinal_opponency.
        """
        log_l = np.clip(a + 0.5 * rg, -9.5, 2.0).astype(np.float32)
        log_m = np.clip(a - 0.5 * rg, -9.5, 2.0).astype(np.float32)
        log_s = np.clip(a + by, -9.5, 2.0).astype(np.float32)

        log_lms = np.stack([log_l, log_m, log_s], axis=-1)
        lms = np.maximum(np.exp(log_lms) - float(self.organ.EPS), 0.0).astype(np.float32)
        linear_rgb = np.einsum("hwk,ck->hwc", lms, self._inv_srgb_to_lms).astype(
            np.float32
        )
        linear_rgb = np.clip(linear_rgb, 0.0, 1.0)
        srgb = np.power(linear_rgb, 1.0 / 2.2)
        return np.clip(np.round(srgb * 255.0), 0.0, 255.0).astype(np.uint8)

    def _register_voxel(
        self,
        voxel_id: str,
        content: Dict[str, Any],
        vec6: np.ndarray,
        scale_prof: np.ndarray,
        chroma_sig: np.ndarray,
        parent_ids: Optional[List[str]] = None,
    ) -> None:
        pos_6d = np.array(
            [
                float(vec6[0]),
                float(vec6[1]),
                float(vec6[2]),
                float(np.clip(0.5 + chroma_sig[0], 0.0, 1.0)),
                float(np.clip(0.5 + chroma_sig[2], 0.0, 1.0)),
                float(vec6[4]),
            ],
            dtype=np.float32,
        )
        c_red = float(max(0.05, vec6[4]))
        c_blue = float(max(0.05, vec6[2]))
        c_yellow = float(max(0.05, vec6[0]))
        c_tot = c_red + c_blue + c_yellow
        voxel = InformationVoxel(
            id=voxel_id,
            content=content,
            tensor=vec6.astype(np.float32).copy(),
            mass=float(1.2 + np.sum(vec6)),
            position=pos_6d,
            velocity=np.zeros(6, dtype=np.float32),
            potential=float(np.linalg.norm(vec6)),
            chromatic_vector=np.array(
                [c_red / c_tot, c_blue / c_tot, c_yellow / c_tot], dtype=np.float32
            ),
            scale_level=float(scale_prof[0] + scale_prof[1]),
        )
        self.causal_field.add_voxel(voxel)
        if parent_ids:
            for pid in parent_ids:
                if pid in self.causal_field.voxels:
                    self.causal_field.link_voxels(pid, voxel_id, strength=0.85)

    def factorize(
        self, rgb_uint8: np.ndarray, source_name: str, max_width: int = 240
    ) -> OcularCognitiveFactorization:
        """
        Observe an external RGB image and factorize it into orthogonal optical-cognitive axes
        and DoG spatial-frequency spectra.
        """
        working_f = _normalize_image_array(rgb_uint8, max_width=max_width)
        working_u8 = np.clip(np.round(working_f * 255.0), 0.0, 255.0).astype(np.uint8)
        obs = self.organ.observe(working_u8, max_width=max_width)

        luma_base, luma_bands = _decompose_telescoping_bands(
            obs.l1.luminance_a, OcularScaleOrgan.SCALES
        )
        rg_base, rg_bands = _decompose_telescoping_bands(
            obs.l1.rg_opponent, OcularScaleOrgan.SCALES
        )
        by_base, by_bands = _decompose_telescoping_bands(
            obs.l1.by_opponent, OcularScaleOrgan.SCALES
        )

        valid = obs.l0.valid_mask
        if not np.any(valid):
            valid = np.ones(obs.shape_hw, dtype=bool)
        rg_vals = obs.l1.rg_opponent[valid]
        by_vals = obs.l1.by_opponent[valid]
        global_rg_mean_std = (float(np.mean(rg_vals)), float(np.std(rg_vals)))
        global_by_mean_std = (float(np.mean(by_vals)), float(np.std(by_vals)))

        vec6, scale_prof, orient_spec, chroma_sig, _ = _extract_optical_descriptor(obs)

        # Build luminance-ranked per-region statistics for Axis 1 topology mold
        sorted_regions = sorted(obs.regions, key=lambda r: r.mean_a)
        n_reg = max(1, len(sorted_regions) - 1)
        region_stats: List[Dict[str, Any]] = []
        for idx, r in enumerate(sorted_regions):
            rmask = (obs.region_labels == r.region_id) & valid
            if not np.any(rmask):
                rmask = obs.region_labels == r.region_id
            std_a = float(np.std(obs.l1.luminance_a[rmask])) if np.any(rmask) else 0.0
            std_rg = float(np.std(obs.l1.rg_opponent[rmask])) if np.any(rmask) else 0.0
            std_by = float(np.std(obs.l1.by_opponent[rmask])) if np.any(rmask) else 0.0
            region_stats.append(
                {
                    "region_id": int(r.region_id),
                    "area_ratio": float(r.area_ratio),
                    "mean_a": float(r.mean_a),
                    "mean_rg": float(r.mean_rg),
                    "mean_by": float(r.mean_by),
                    "std_a": std_a,
                    "std_rg": std_rg,
                    "std_by": std_by,
                    "rank_luma": float(idx / n_reg) if len(sorted_regions) > 1 else 0.5,
                }
            )

        self._op_counter += 1
        voxel_id = f"voxel_src_{source_name}_{self._op_counter:04d}"
        self._register_voxel(
            voxel_id=voxel_id,
            content={"mode": "SOURCE_FACTORIZATION", "source_name": source_name},
            vec6=vec6,
            scale_prof=scale_prof,
            chroma_sig=chroma_sig,
        )

        return OcularCognitiveFactorization(
            source_name=source_name,
            shape_hw=obs.shape_hw,
            source_rgb_working_u8=working_u8,
            obs=obs,
            region_labels=obs.region_labels.copy(),
            boundary_persistence=obs.l2.persistence_map.copy(),
            material_boundary_mask=(obs.cause_map == 3),
            region_stats=region_stats,
            luma_base=luma_base,
            luma_bands=luma_bands,
            rg_base=rg_base,
            rg_bands=rg_bands,
            by_base=by_base,
            by_bands=by_bands,
            global_rg_mean_std=global_rg_mean_std,
            global_by_mean_std=global_by_mean_std,
            scale_band_energies=scale_prof,
            orientation_spectrum=orient_spec,
            chromatic_signature=chroma_sig,
            orientation_anisotropy=float(vec6[2]),
            vector_6d=vec6,
            voxel_id=voxel_id,
        )

    def _verify_and_record(
        self,
        fact: OcularCognitiveFactorization,
        synth_rgb: np.ndarray,
        operation_name: str,
        mode: str,
        palette_donor: Optional[OcularCognitiveFactorization] = None,
        compute_psnr: bool = False,
    ) -> GenerativeVerificationResult:
        """Closed-loop re-observation of ELYSIA's own generated RGB output through OcularScaleOrgan."""
        h, w = fact.shape_hw
        reobs = self.organ.observe(synth_rgb, max_width=w)
        re_vec6, re_scale, re_orient, re_chroma, _ = _extract_optical_descriptor(reobs)

        struct_corr = _compute_persistence_correlation(fact.obs, reobs)
        src_proxy = _fact_as_engram_proxy(fact)
        res_source = self.eye._compute_structural_resonance(
            re_vec6, re_scale, re_orient, re_chroma, src_proxy
        )

        res_donor: Optional[float] = None
        parent_ids = [fact.voxel_id]
        if palette_donor is not None:
            donor_proxy = _fact_as_engram_proxy(palette_donor)
            res_donor = self.eye._compute_structural_resonance(
                re_vec6, re_scale, re_orient, re_chroma, donor_proxy
            )
            parent_ids.append(palette_donor.voxel_id)

        valid = fact.obs.l0.valid_mask & reobs.l0.valid_mask
        if int(np.sum(valid)) < 24:
            valid = np.ones((h, w), dtype=bool)
        delta_luma = float(np.mean(reobs.l1.luminance_a[valid] - fact.obs.l1.luminance_a[valid]))
        d_rg = reobs.l1.rg_opponent[valid] - fact.obs.l1.rg_opponent[valid]
        d_by = reobs.l1.by_opponent[valid] - fact.obs.l1.by_opponent[valid]
        delta_chroma = float(np.mean(np.hypot(d_rg, d_by)))

        psnr_val: Optional[float] = None
        if compute_psnr and synth_rgb.shape == fact.source_rgb_working_u8.shape:
            mse = float(
                np.mean(
                    (
                        synth_rgb.astype(np.float32)
                        - fact.source_rgb_working_u8.astype(np.float32)
                    )
                    ** 2
                )
            )
            psnr_val = 99.0 if mse < 1e-8 else float(10.0 * np.log10((255.0 ** 2) / mse))

        self._op_counter += 1
        voxel_id = f"voxel_gen_{operation_name}_{self._op_counter:04d}"
        self._register_voxel(
            voxel_id=voxel_id,
            content={
                "mode": mode,
                "operation_name": operation_name,
                "source_name": fact.source_name,
                "structural_corr": struct_corr,
            },
            vec6=re_vec6,
            scale_prof=re_scale,
            chroma_sig=re_chroma,
            parent_ids=parent_ids,
        )

        return GenerativeVerificationResult(
            operation_name=operation_name,
            mode=mode,
            synthesized_rgb=synth_rgb,
            reobserved_obs=reobs,
            reobserved_vector_6d=re_vec6,
            reobserved_scale_profile=re_scale,
            reobserved_chromatic_signature=re_chroma,
            structural_corr_with_source=struct_corr,
            optical_resonance_with_source=res_source,
            optical_resonance_with_palette_donor=res_donor,
            delta_luma_mean=delta_luma,
            delta_chroma_mean=delta_chroma,
            psnr_db=psnr_val,
            voxel_id=voxel_id,
        )

    def reconstruct(
        self, fact: OcularCognitiveFactorization, name: Optional[str] = None
    ) -> GenerativeVerificationResult:
        """Reproduce the image purely from its factorized axes & spectral bands."""
        a_rec = _recompose_telescoping_bands(fact.luma_base, fact.luma_bands)
        rg_rec = _recompose_telescoping_bands(fact.rg_base, fact.rg_bands)
        by_rec = _recompose_telescoping_bands(fact.by_base, fact.by_bands)
        synth_rgb = self._inverse_retinal_to_srgb(a_rec, rg_rec, by_rec)
        op_name = name or f"{fact.source_name}_reconstruct"
        return self._verify_and_record(
            fact=fact,
            synth_rgb=synth_rgb,
            operation_name=op_name,
            mode="RECONSTRUCTION",
            compute_psnr=True,
        )

    def modulate(
        self,
        fact: OcularCognitiveFactorization,
        name: str,
        luma_shift: float = 0.0,
        luma_contrast: float = 1.0,
        rg_shift: float = 0.0,
        by_shift: float = 0.0,
        chroma_rotation_deg: float = 0.0,
        chroma_gain: float = 1.0,
        band_gains: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0),
    ) -> GenerativeVerificationResult:
        """
        Continuously modulate orthogonal cognitive dials (illumination, chromatic opponent
        rotation/gain, and multi-scale spatial frequency bands) while keeping non-target axes intact.
        """
        a_bands = [float(luma_contrast) * b for b in fact.luma_bands]
        a_mod = _recompose_telescoping_bands(
            fact.luma_base + float(luma_shift), a_bands, band_gains=band_gains
        )

        rg_raw = _recompose_telescoping_bands(
            fact.rg_base, fact.rg_bands, band_gains=band_gains
        )
        by_raw = _recompose_telescoping_bands(
            fact.by_base, fact.by_bands, band_gains=band_gains
        )

        theta = float(np.deg2rad(chroma_rotation_deg))
        cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
        rg_rot = cos_t * rg_raw - sin_t * by_raw
        by_rot = sin_t * rg_raw + cos_t * by_raw

        rg_mod = float(chroma_gain) * rg_rot + float(rg_shift)
        by_mod = float(chroma_gain) * by_rot + float(by_shift)

        if abs(luma_shift) > 0.10 and abs(chroma_rotation_deg) < 1e-3 and abs(chroma_gain - 1.0) < 1e-3:
            mode = "ILLUMINATION_MODULATION"
        elif abs(chroma_rotation_deg) > 1e-3 or abs(rg_shift) > 0.05 or abs(by_shift) > 0.05:
            mode = "CHROMATIC_MODULATION"
        else:
            mode = "SPECTRUM_MODULATION"

        synth_rgb = self._inverse_retinal_to_srgb(a_mod, rg_mod, by_mod)
        return self._verify_and_record(
            fact=fact,
            synth_rgb=synth_rgb,
            operation_name=name,
            mode=mode,
        )

    def recreate_cross_scene(
        self,
        structure_fact: OcularCognitiveFactorization,
        palette_fact: OcularCognitiveFactorization,
        name: str,
        chroma_blend: float = 0.88,
        spectrum_blend: float = 0.55,
        luma_mood_blend: float = 0.30,
    ) -> GenerativeVerificationResult:
        """
        Fuse Scene A's boundary/region mold with Scene B's luminance-stratified chromatic
        opponent atmosphere and multi-scale energy spectrum to recreate a new coherent image.
        """
        # 1. Transfer multi-scale spectral energy profile smoothly onto Scene A's geometry
        src_e = np.maximum(structure_fact.scale_band_energies, 1e-4)
        pal_e = np.maximum(palette_fact.scale_band_energies, 1e-4)
        ratio_e = np.clip(np.sqrt(pal_e / src_e), 0.45, 2.0)
        band_gains = tuple(
            float((1.0 - spectrum_blend) * 1.0 + spectrum_blend * r)
            for r in ratio_e
        )

        a_src_mean = float(np.mean(structure_fact.luma_base))
        a_pal_mean = float(np.mean(palette_fact.luma_base))
        base_shift = float(luma_mood_blend) * (a_pal_mean - a_src_mean)

        # Keep Scene A's spatial variation around luma_base mean + spectral bands
        a_new = _recompose_telescoping_bands(
            structure_fact.luma_base + base_shift,
            structure_fact.luma_bands,
            band_gains=band_gains,
        )

        # 2. Luminance-stratified chromatic opponent transfer in (RG, BY)
        s_valid = structure_fact.obs.l0.valid_mask
        if int(np.sum(s_valid)) < 24:
            s_valid = np.ones(structure_fact.shape_hw, dtype=bool)
        p_valid = palette_fact.obs.l0.valid_mask
        if int(np.sum(p_valid)) < 24:
            p_valid = np.ones(palette_fact.shape_hw, dtype=bool)

        a_s_all = structure_fact.obs.l1.luminance_a
        p10, p90 = np.percentile(a_s_all[s_valid], [10.0, 90.0])
        luma_norm_s = np.clip((a_s_all - p10) / max(1e-4, float(p90 - p10)), 0.0, 1.0).astype(
            np.float32
        )

        # Anchor palette colors at 3 luminance strata (shadow, midtone, highlight)
        a_p_val = palette_fact.obs.l1.luminance_a[p_valid]
        rg_p_val = palette_fact.obs.l1.rg_opponent[p_valid]
        by_p_val = palette_fact.obs.l1.by_opponent[p_valid]
        q33, q67 = np.percentile(a_p_val, [33.3, 66.7])

        shadow_mask = a_p_val <= q33
        mid_mask = (a_p_val > q33) & (a_p_val <= q67)
        high_mask = a_p_val > q67

        anchor_rg = np.array(
            [
                float(np.mean(rg_p_val[shadow_mask])) if np.any(shadow_mask) else palette_fact.global_rg_mean_std[0],
                float(np.mean(rg_p_val[mid_mask])) if np.any(mid_mask) else palette_fact.global_rg_mean_std[0],
                float(np.mean(rg_p_val[high_mask])) if np.any(high_mask) else palette_fact.global_rg_mean_std[0],
            ],
            dtype=np.float32,
        )
        anchor_by = np.array(
            [
                float(np.mean(by_p_val[shadow_mask])) if np.any(shadow_mask) else palette_fact.global_by_mean_std[0],
                float(np.mean(by_p_val[mid_mask])) if np.any(mid_mask) else palette_fact.global_by_mean_std[0],
                float(np.mean(by_p_val[high_mask])) if np.any(high_mask) else palette_fact.global_by_mean_std[0],
            ],
            dtype=np.float32,
        )
        anchor_knots = np.array([0.15, 0.50, 0.85], dtype=np.float32)

        target_rg_mean_map = np.interp(luma_norm_s, anchor_knots, anchor_rg).astype(np.float32)
        target_by_mean_map = np.interp(luma_norm_s, anchor_knots, anchor_by).astype(np.float32)

        rg_s_full = _recompose_telescoping_bands(
            structure_fact.rg_base, structure_fact.rg_bands, band_gains=band_gains
        )
        by_s_full = _recompose_telescoping_bands(
            structure_fact.by_base, structure_fact.by_bands, band_gains=band_gains
        )

        mu_rg_s, sig_rg_s = structure_fact.global_rg_mean_std
        mu_by_s, sig_by_s = structure_fact.global_by_mean_std
        _, sig_rg_p = palette_fact.global_rg_mean_std
        _, sig_by_p = palette_fact.global_by_mean_std

        scale_rg = float(np.clip(sig_rg_p / max(0.035, sig_rg_s), 0.40, 1.85))
        scale_by = float(np.clip(sig_by_p / max(0.035, sig_by_s), 0.40, 1.85))

        rg_transferred = target_rg_mean_map + scale_rg * (rg_s_full - mu_rg_s)
        by_transferred = target_by_mean_map + scale_by * (by_s_full - mu_by_s)

        rg_new = (1.0 - float(chroma_blend)) * rg_s_full + float(chroma_blend) * rg_transferred
        by_new = (1.0 - float(chroma_blend)) * by_s_full + float(chroma_blend) * by_transferred

        synth_rgb = self._inverse_retinal_to_srgb(a_new, rg_new, by_new)
        return self._verify_and_record(
            fact=structure_fact,
            synth_rgb=synth_rgb,
            operation_name=name,
            mode="CROSS_SCENE_RECREATION",
            palette_donor=palette_fact,
        )
