"""
active_ocular_mechanics_engine.py

ELYSIA Active Ocular Mechanics Engine (VLM-Free, LLM-Free)
============================================================
Extends OcularScaleOrgan (L0-L4) from passive frame processing into a continuous
active ocular system grounded in physical-informational optics and the 4 Continuities:

1. Relationship: Peripheral salience is computed from continuous retinal opponent
   contrast (DoG center-surround), multi-scale boundary persistence, chromatic
   opponency (RG/BY), and temporal flow velocity rather than brittle rules.
2. Connectivity: Consecutive foveal fixations bind spatially inside CausalField (6D)
   via ConnectivityBeam structures, turning disjoint crops into a topological scene map.
3. Mobility: Gaze moves with saccadic momentum modulated by continuous Gaussian
   Inhibition-of-Return (IOR) relaxation wells and optical flow velocity fields.
4. Informational Continuity: Bridge discrete resolutions (wide periphery <-> dense
   high-res fovea) and temporal transitions (video/live screen motion tracking)
   to memorize and re-recognize invariant optical structures under darkening and shifts.

Zero external VLMs, zero LLMs, zero filename hints, zero synthetic placeholders.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from core.physics.causal_field import CausalField, InformationVoxel
from core.sensory.ocular_scale_organ import (
    OcularScaleOrgan,
    OpticalSceneObservation,
    TemporalVisualAnalysis,
)


@dataclass
class FovealEngram:
    """Illumination-invariant optical topology captured during a single foveal fixation."""

    engram_id: str
    scene_name: str
    fixation_idx: int
    gaze_yx_norm: Tuple[float, float]
    gaze_yx_px: Tuple[int, int]
    fovea_box_yxyx: Tuple[int, int, int, int]
    vector_6d: np.ndarray
    scale_profile: np.ndarray
    orientation_spectrum: np.ndarray
    chromatic_signature: np.ndarray
    cleaved_bond_ratio: float
    material_boundary_count: int
    region_count: int
    salience_peak: float
    recognition_status: str
    matched_engram_id: Optional[str]
    resonance_score: float
    voxel_id: str


@dataclass
class SceneScanResult:
    """Complete active peripheral-to-foveal scan outcome for a real image."""

    scene_name: str
    image_shape_hw: Tuple[int, int]
    peripheral_obs: OpticalSceneObservation
    salience_map: np.ndarray
    saccade_path_yx: List[Tuple[int, int]]
    foveal_engrams: List[FovealEngram]
    foveal_crops_rgb: List[np.ndarray]
    foveal_obs_list: List[OpticalSceneObservation]
    periphery_vs_fovea_detail_gain: float


@dataclass
class VideoTrackingResult:
    """Motion-guided active eye tracking outcome across a real video or screen stream."""

    video_name: str
    frames_processed: int
    frame_events: List[Dict[str, Any]]
    motion_gaze_path_yx: List[Tuple[int, int]]
    sample_frame_rgb: np.ndarray
    sample_flow_mag: np.ndarray
    sample_moving_mask: np.ndarray
    tracked_engrams: List[FovealEngram] = field(default_factory=list)


def _compute_radial_foveal_weights(
    obs: OpticalSceneObservation,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build smooth radial retinal weights so center structures dominate over edge crop shear."""
    h, w = obs.shape_hw
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    r2 = yy * yy + xx * xx
    interior = r2 <= 0.88
    valid = obs.l0.valid_mask & interior
    if int(np.sum(valid)) < 24:
        valid = interior
    w_radial = np.exp(-1.5 * r2).astype(np.float32) * valid.astype(np.float32)
    w_sum = float(np.sum(w_radial))
    if w_sum > 1e-8:
        w_norm = w_radial / w_sum
    else:
        w_norm = np.full((h, w), 1.0 / float(h * w), dtype=np.float32)
    return w_norm, valid


def _extract_optical_descriptor(
    obs: OpticalSceneObservation,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Extract illumination-invariant multi-scale, orientation, opponent, and 6D topology vectors."""
    w_norm, _ = _compute_radial_foveal_weights(obs)

    rg = obs.l1.rg_opponent
    by = obs.l1.by_opponent
    mean_rg = float(np.sum(w_norm * rg))
    mean_by = float(np.sum(w_norm * by))
    std_rg = float(np.sqrt(max(0.0, float(np.sum(w_norm * (rg - mean_rg) ** 2)))))
    std_by = float(np.sqrt(max(0.0, float(np.sum(w_norm * (by - mean_by) ** 2)))))
    chromatic_signature = np.array(
        [mean_rg, std_rg, mean_by, std_by], dtype=np.float32
    )

    # Multi-scale differential energy profile across all spatial Gaussian scales
    raw_scales = []
    for sigma in OcularScaleOrgan.SCALES:
        lvl = obs.l2.levels[sigma]
        e_sig = float(np.sum(w_norm * (lvl.grad_total * float(sigma))))
        raw_scales.append(max(0.0, e_sig))
    scale_arr = np.array(raw_scales, dtype=np.float32)
    scale_sum = float(np.sum(scale_arr))
    if scale_sum > 1e-8:
        scale_profile = (scale_arr / scale_sum).astype(np.float32)
    else:
        scale_profile = np.full(len(OcularScaleOrgan.SCALES), 0.2, dtype=np.float32)

    # Orientation spectrum from sigma=2.0 structure tensor weighted by persistence and coherence
    lvl2 = obs.l2.levels[2.0]
    orient = lvl2.orientation
    orient_weight = w_norm * (lvl2.grad_total + 1e-4) * (0.25 + lvl2.coherence)
    bins = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 9, dtype=np.float32)
    hist, _ = np.histogram(
        orient.ravel(), bins=bins, weights=orient_weight.ravel()
    )
    hist = hist.astype(np.float32)
    # Circularly smooth and roll peak orientation to index 0 for rotation resilience
    hist_smooth = (
        0.15 * np.roll(hist, -1) + 0.70 * hist + 0.15 * np.roll(hist, 1)
    ).astype(np.float32)
    peak_idx = int(np.argmax(hist_smooth))
    hist_rolled = np.roll(hist_smooth, -peak_idx)
    h_sum = float(np.sum(hist_rolled))
    if h_sum > 1e-8:
        orientation_spectrum = (hist_rolled / h_sum).astype(np.float32)
    else:
        orientation_spectrum = np.full(8, 0.125, dtype=np.float32)

    # Bounded 6D optical causal state vector [0, 1]^6
    cleaved_ratio = float(obs.l2.bond_lattice.cleaved_bond_ratio)
    cleaved_scaled = float(np.clip(cleaved_ratio * 5.0, 0.0, 1.0))

    n_bound = len(obs.boundaries)
    mat_count = sum(
        1 for b in obs.boundaries if b.optical_cause == "MATERIAL_BOUNDARY"
    )
    mat_ratio = float(mat_count / max(1, n_bound)) if n_bound > 0 else 0.0

    anisotropy = float(
        np.clip(
            0.5 * np.sum(w_norm * obs.l2.levels[1.0].coherence)
            + 0.5 * np.sum(w_norm * lvl2.coherence),
            0.0,
            1.0,
        )
    )
    chroma_contrast = float(np.clip(np.hypot(std_rg, std_by) * 2.5, 0.0, 1.0))
    fine_ratio = float(np.clip(scale_profile[0] + scale_profile[1], 0.0, 1.0))

    if obs.regions:
        areas = np.array([r.pixel_count for r in obs.regions], dtype=np.float32)
        comps = np.array([r.compactness for r in obs.regions], dtype=np.float32)
        w_comp = float(np.sum(areas * comps) / max(1.0, float(np.sum(areas))))
    else:
        w_comp = 0.5
    region_compactness = float(np.clip(w_comp, 0.0, 1.0))

    vector_6d = np.array(
        [
            cleaved_scaled,
            mat_ratio,
            anisotropy,
            chroma_contrast,
            fine_ratio,
            region_compactness,
        ],
        dtype=np.float32,
    )

    stats = {
        "cleaved_bond_ratio": cleaved_ratio,
        "material_boundary_count": int(mat_count),
        "region_count": int(len(obs.regions)),
    }
    return vector_6d, scale_profile, orientation_spectrum, chromatic_signature, stats


class ActiveOcularMechanicsEngine:
    """
    Active Fovea-Saccade, Structural Engram Memory, and Motion-Tracking Eye Organ
    built directly over OcularScaleOrgan optical physics and CausalField (6D).
    """

    def __init__(
        self,
        recognition_threshold: float = 0.88,
        organ: Optional[OcularScaleOrgan] = None,
    ) -> None:
        self.organ = organ if organ is not None else OcularScaleOrgan()
        self.causal_field = CausalField(dimensions=6)
        self.recognition_threshold = float(recognition_threshold)
        self.engram_memory: Dict[str, FovealEngram] = {}
        self.scene_links: List[Dict[str, Any]] = []
        self._engram_counter = 0

    def _compute_salience_field(
        self,
        obs: OpticalSceneObservation,
        motion_mag: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute continuous retinal salience from:
        - Scale-space boundary persistence (L2)
        - Retinal ON/OFF center-surround DoG energy (L1)
        - Chromatic opponency magnitude RG/BY (L1)
        - Optional temporal motion residual (L4)
        """
        h, w = obs.shape_hw

        def _norm01(arr: np.ndarray) -> np.ndarray:
            arr_f = np.asarray(arr, dtype=np.float32)
            lo = float(np.min(arr_f))
            hi = float(np.percentile(arr_f, 99.0))
            if hi - lo < 1e-6:
                return np.zeros_like(arr_f, dtype=np.float32)
            return np.clip((arr_f - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

        s_persist = _norm01(obs.l2.persistence_map)
        dog_energy = np.abs(obs.l1.dog_fine) + 0.55 * np.abs(obs.l1.dog_coarse)
        s_dog = _norm01(dog_energy)
        chroma_mag = np.hypot(obs.l1.rg_opponent, obs.l1.by_opponent)
        s_chroma = _norm01(chroma_mag)

        if motion_mag is not None and motion_mag.shape == (h, w):
            s_motion = _norm01(motion_mag)
            raw_sal = (
                0.28 * s_persist
                + 0.22 * s_dog
                + 0.12 * s_chroma
                + 0.38 * s_motion
            )
        else:
            raw_sal = 0.45 * s_persist + 0.35 * s_dog + 0.20 * s_chroma

        sig = max(1.5, 0.015 * float(min(h, w)))
        sal_smooth = gaussian_filter(raw_sal, sigma=sig).astype(np.float32)

        # Suppress extreme 6% border rim so saccades anchor on scene contents
        mh = max(2, int(round(0.06 * h)))
        mw = max(2, int(round(0.06 * w)))
        border_mask = np.ones((h, w), dtype=np.float32)
        border_mask[:mh, :] = 0.05
        border_mask[h - mh :, :] = 0.05
        border_mask[:, :mw] = 0.05
        border_mask[:, w - mw :] = 0.05
        sal_smooth *= border_mask

        mx = float(np.max(sal_smooth))
        if mx > 1e-8:
            sal_smooth = (sal_smooth / mx).astype(np.float32)
        return sal_smooth

    def _compute_structural_resonance(
        self,
        cand_vec6: np.ndarray,
        cand_scale: np.ndarray,
        cand_orient: np.ndarray,
        cand_chroma: np.ndarray,
        mem: FovealEngram,
    ) -> float:
        """Continuous structural isomorphism between two optical observations (zero pixel MSE)."""
        sim_scale = float(
            np.clip(
                np.sum(np.sqrt(np.maximum(cand_scale * mem.scale_profile, 0.0))),
                0.0,
                1.0,
            )
        )
        sim_orient = float(
            np.clip(
                np.sum(
                    np.sqrt(
                        np.maximum(cand_orient * mem.orientation_spectrum, 0.0)
                    )
                ),
                0.0,
                1.0,
            )
        )
        chroma_dist = float(np.linalg.norm(cand_chroma - mem.chromatic_signature))
        sim_chroma = float(np.exp(-chroma_dist / 0.14))

        vec6_dist = float(np.linalg.norm(cand_vec6 - mem.vector_6d))
        sim_vec6 = float(np.exp(-vec6_dist / 0.35))

        resonance = (
            0.34 * sim_chroma
            + 0.26 * sim_scale
            + 0.22 * sim_vec6
            + 0.18 * sim_orient
        )
        return float(np.clip(resonance, 0.0, 1.0))

    def _match_against_memory(
        self,
        vec6: np.ndarray,
        scale_prof: np.ndarray,
        orient_spec: np.ndarray,
        chroma_sig: np.ndarray,
        exclude_scene: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """Rank all stored CausalField foveal engrams by structural resonance."""
        scored: List[Tuple[str, float]] = []
        for eid, mem in self.engram_memory.items():
            if exclude_scene is not None and mem.scene_name == exclude_scene:
                continue
            score = self._compute_structural_resonance(
                vec6, scale_prof, orient_spec, chroma_sig, mem
            )
            scored.append((eid, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def _register_engram_voxel(
        self,
        engram: FovealEngram,
        prev_voxel_id: Optional[str],
    ) -> None:
        """Bind an optical FovealEngram as a 6D InformationVoxel in CausalField."""
        pos_6d = np.array(
            [
                engram.gaze_yx_norm[0],
                engram.gaze_yx_norm[1],
                float(engram.vector_6d[4]),
                float(np.clip(0.5 + engram.chromatic_signature[0], 0.0, 1.0)),
                float(np.clip(0.5 + engram.chromatic_signature[2], 0.0, 1.0)),
                float(engram.vector_6d[0]),
            ],
            dtype=np.float32,
        )
        # Chromatic analogical balance: Red(Flux=fine energy), Blue(Order=anisotropy), Yellow(Entropy=cleavage)
        c_red = float(max(0.05, engram.vector_6d[4]))
        c_blue = float(max(0.05, engram.vector_6d[2]))
        c_yellow = float(max(0.05, engram.vector_6d[0]))
        c_sum = c_red + c_blue + c_yellow
        chromatic_vec = np.array(
            [c_red / c_sum, c_blue / c_sum, c_yellow / c_sum], dtype=np.float32
        )

        mass = float(
            max(
                0.5,
                1.0
                + 2.0 * engram.salience_peak
                + 0.02 * float(engram.region_count),
            )
        )
        voxel = InformationVoxel(
            id=engram.voxel_id,
            content={
                "engram_id": engram.engram_id,
                "scene_name": engram.scene_name,
                "fixation_idx": engram.fixation_idx,
                "gaze_yx_px": engram.gaze_yx_px,
                "status": engram.recognition_status,
            },
            tensor=engram.vector_6d.copy(),
            mass=mass,
            position=pos_6d,
            velocity=np.zeros(6, dtype=np.float32),
            potential=float(engram.salience_peak),
            chromatic_vector=chromatic_vec,
            scale_level=float(np.mean(engram.scale_profile[:2])),
        )
        self.causal_field.add_voxel(voxel)

        if prev_voxel_id is not None:
            strength = float(0.6 + 0.4 * max(0.2, engram.vector_6d[2]))
            self.causal_field.link_voxels(prev_voxel_id, engram.voxel_id, strength=strength)
            self.scene_links.append(
                {
                    "source": prev_voxel_id,
                    "target": engram.voxel_id,
                    "scene_name": engram.scene_name,
                    "strength": strength,
                }
            )

    def _extract_fovea_patch_coords(
        self,
        full_h: int,
        full_w: int,
        target_y_norm: float,
        target_x_norm: float,
        fovea_radius_ratio: float,
    ) -> Tuple[int, int, Tuple[int, int, int, int]]:
        """Map normalized retinal fixation coordinates to native full-resolution fovea crop."""
        radius = max(48, int(round(float(min(full_h, full_w)) * fovea_radius_ratio)))
        max_r = max(24, min(full_h, full_w) // 2 - 2)
        radius = min(radius, max_r)

        ny = int(np.clip(round(target_y_norm * (full_h - 1)), radius, full_h - radius - 1))
        nx = int(np.clip(round(target_x_norm * (full_w - 1)), radius, full_w - radius - 1))

        y0 = max(0, ny - radius)
        y1 = min(full_h, ny + radius)
        x0 = max(0, nx - radius)
        x1 = min(full_w, nx + radius)
        return ny, nx, (y0, x0, y1, x1)

    def scan_image(
        self,
        rgb_uint8: np.ndarray,
        scene_name: str,
        n_fixations: int = 5,
        periphery_max_width: int = 220,
        fovea_radius_ratio: float = 0.14,
        fovea_max_width: int = 160,
        motion_mag: Optional[np.ndarray] = None,
    ) -> SceneScanResult:
        """
        Execute genuine active eye mechanics on an image:
        wide peripheral glimpse -> salience & IOR field -> saccadic gaze shifts ->
        native full-res foveal re-observation -> CausalField voxel/beam binding.
        """
        rgb_arr = np.asarray(rgb_uint8)
        if rgb_arr.ndim != 3 or rgb_arr.shape[2] < 3:
            raise ValueError(f"Expected HxWx3 RGB array, got shape {rgb_arr.shape}")
        rgb_arr = rgb_arr[..., :3]
        full_h, full_w = rgb_arr.shape[:2]

        periph_obs = self.organ.observe(rgb_arr, max_width=periphery_max_width)
        ph, pw = periph_obs.shape_hw
        salience_map = self._compute_salience_field(periph_obs, motion_mag=motion_mag)

        ior_field = np.ones((ph, pw), dtype=np.float32)
        gaze_y, gaze_x = ph * 0.5, pw * 0.5
        vel_y, vel_x = 0.0, 0.0
        ior_sigma = max(4.0, 0.14 * float(min(ph, pw)))
        yy_grid, xx_grid = np.mgrid[0:ph, 0:pw].astype(np.float32)

        saccade_path_yx: List[Tuple[int, int]] = []
        foveal_engrams: List[FovealEngram] = []
        foveal_crops_rgb: List[np.ndarray] = []
        foveal_obs_list: List[OpticalSceneObservation] = []
        detail_gains: List[float] = []
        prev_voxel_id: Optional[str] = None

        for k in range(max(1, int(n_fixations))):
            effective_potential = salience_map * ior_field
            ty, tx = np.unravel_index(
                int(np.argmax(effective_potential)), (ph, pw)
            )
            sal_peak = float(salience_map[ty, tx])

            # Continuous saccadic eye momentum & position update
            vel_y = 0.25 * vel_y + 0.75 * (float(ty) - gaze_y)
            vel_x = 0.25 * vel_x + 0.75 * (float(tx) - gaze_x)
            gaze_y = float(np.clip(gaze_y + vel_y, 0.0, ph - 1.0))
            gaze_x = float(np.clip(gaze_x + vel_x, 0.0, pw - 1.0))

            ty_norm = float(gaze_y / max(1.0, ph - 1.0))
            tx_norm = float(gaze_x / max(1.0, pw - 1.0))

            ny, nx, (y0, x0, y1, x1) = self._extract_fovea_patch_coords(
                full_h=full_h,
                full_w=full_w,
                target_y_norm=ty_norm,
                target_x_norm=tx_norm,
                fovea_radius_ratio=fovea_radius_ratio,
            )
            saccade_path_yx.append((ny, nx))

            fovea_crop = rgb_arr[y0:y1, x0:x1].copy()
            fovea_obs = self.organ.observe(fovea_crop, max_width=fovea_max_width)
            vec6, scale_prof, orient_spec, chroma_sig, stats = (
                _extract_optical_descriptor(fovea_obs)
            )

            # Detail gain: optical structure resolved in high-res fovea vs same region in periphery
            py0 = int(np.clip(round((y0 / float(full_h)) * ph), 0, ph - 2))
            py1 = int(np.clip(round((y1 / float(full_h)) * ph), py0 + 2, ph))
            px0 = int(np.clip(round((x0 / float(full_w)) * pw), 0, pw - 2))
            px1 = int(np.clip(round((x1 / float(full_w)) * pw), px0 + 2, pw))

            periph_regs_box = max(
                1, int(len(np.unique(periph_obs.region_labels[py0:py1, px0:px1])))
            )
            fovea_regs = max(1, int(stats["region_count"]))

            p_h_box = periph_obs.l2.bond_lattice.horiz_cleaved[py0:py1, px0 : px1 - 1]
            p_v_box = periph_obs.l2.bond_lattice.vert_cleaved[py0 : py1 - 1, px0:px1]
            periph_edges = float(np.sum(p_h_box) + np.sum(p_v_box))

            f_edges = float(
                np.sum(fovea_obs.l2.bond_lattice.horiz_cleaved)
                + np.sum(fovea_obs.l2.bond_lattice.vert_cleaved)
            )
            reg_gain = float(fovea_regs) / float(periph_regs_box)
            edge_gain = (f_edges + 1.0) / (periph_edges + 1.0)
            detail_gains.append(0.5 * reg_gain + 0.5 * edge_gain)

            matches = self._match_against_memory(
                vec6, scale_prof, orient_spec, chroma_sig
            )
            best_id: Optional[str] = None
            best_score = 0.0
            if matches:
                best_id, best_score = matches[0]
            rec_status = (
                "RECOGNIZED_STRUCTURE"
                if best_score >= self.recognition_threshold
                else "NOVEL_STRUCTURE"
            )

            self._engram_counter += 1
            engram_id = f"{scene_name}_fix{k + 1:02d}_{self._engram_counter:04d}"
            voxel_id = f"voxel_{engram_id}"

            engram = FovealEngram(
                engram_id=engram_id,
                scene_name=scene_name,
                fixation_idx=k,
                gaze_yx_norm=(ny / float(max(1, full_h - 1)), nx / float(max(1, full_w - 1))),
                gaze_yx_px=(ny, nx),
                fovea_box_yxyx=(y0, x0, y1, x1),
                vector_6d=vec6,
                scale_profile=scale_prof,
                orientation_spectrum=orient_spec,
                chromatic_signature=chroma_sig,
                cleaved_bond_ratio=float(stats["cleaved_bond_ratio"]),
                material_boundary_count=int(stats["material_boundary_count"]),
                region_count=int(stats["region_count"]),
                salience_peak=sal_peak,
                recognition_status=rec_status,
                matched_engram_id=best_id,
                resonance_score=best_score,
                voxel_id=voxel_id,
            )

            self.engram_memory[engram_id] = engram
            self._register_engram_voxel(engram, prev_voxel_id)
            prev_voxel_id = voxel_id

            foveal_engrams.append(engram)
            foveal_crops_rgb.append(fovea_crop)
            foveal_obs_list.append(fovea_obs)

            # Gaussian Inhibition-of-Return (IOR) relaxation
            ior_field = 1.0 - 0.85 * (1.0 - ior_field)
            dist2 = (yy_grid - float(ty)) ** 2 + (xx_grid - float(tx)) ** 2
            ior_trough = 1.0 - 0.94 * np.exp(-dist2 / (2.0 * (ior_sigma ** 2)))
            ior_field = (ior_field * ior_trough).astype(np.float32)

        mean_detail_gain = float(np.mean(detail_gains)) if detail_gains else 1.0
        return SceneScanResult(
            scene_name=scene_name,
            image_shape_hw=(full_h, full_w),
            peripheral_obs=periph_obs,
            salience_map=salience_map,
            saccade_path_yx=saccade_path_yx,
            foveal_engrams=foveal_engrams,
            foveal_crops_rgb=foveal_crops_rgb,
            foveal_obs_list=foveal_obs_list,
            periphery_vs_fovea_detail_gain=mean_detail_gain,
        )

    def recognize_perturbed_patch(
        self,
        patch_rgb: np.ndarray,
        query_name: str,
        fovea_max_width: int = 160,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Observe an unseen/darkened/shifted visual patch cleanly via OcularScaleOrgan and
        retrieve the highest-resonance FovealEngram in CausalField memory (without mutating memory).
        """
        obs = self.organ.observe(patch_rgb, max_width=fovea_max_width)
        vec6, scale_prof, orient_spec, chroma_sig, _ = _extract_optical_descriptor(
            obs
        )
        ranked = self._match_against_memory(
            vec6, scale_prof, orient_spec, chroma_sig
        )
        top_matches = [
            {
                "engram_id": eid,
                "scene_name": self.engram_memory[eid].scene_name,
                "resonance": float(score),
            }
            for eid, score in ranked[:top_k]
        ]
        return {
            "query_name": query_name,
            "best_engram_id": top_matches[0]["engram_id"] if top_matches else None,
            "best_scene_name": top_matches[0]["scene_name"] if top_matches else None,
            "best_resonance": top_matches[0]["resonance"] if top_matches else 0.0,
            "top_matches": top_matches,
        }

    def observe_frame_sequence(
        self,
        frames_rgb: List[np.ndarray],
        stream_name: str,
        max_width: int = 220,
        fovea_radius_ratio: float = 0.16,
    ) -> VideoTrackingResult:
        """
        Run continuous temporal optical flow + motion-guided foveal saccades across a
        ordered list of RGB frames (shared by real video files and live screen capture).
        """
        if len(frames_rgb) < 2:
            raise ValueError("Need at least 2 frames for temporal ocular tracking")

        obs_list = [
            self.organ.observe(frame, max_width=max_width) for frame in frames_rgb
        ]

        frame_events: List[Dict[str, Any]] = []
        motion_gaze_path_yx: List[Tuple[int, int]] = []
        tracked_engrams: List[FovealEngram] = []

        sample_frame = frames_rgb[0]
        sample_flow_mag = np.zeros(obs_list[0].shape_hw, dtype=np.float32)
        sample_moving_mask = np.zeros(obs_list[0].shape_hw, dtype=bool)
        best_flow_energy = -1.0
        prev_voxel_id: Optional[str] = None

        for idx in range(1, len(frames_rgb)):
            obs_prev = obs_list[idx - 1]
            obs_curr = obs_list[idx]
            curr_rgb = frames_rgb[idx]
            full_h, full_w = curr_rgb.shape[:2]

            temporal: TemporalVisualAnalysis = self.organ.observe_transition(
                obs_prev, obs_curr
            )
            sal_map = self._compute_salience_field(
                obs_curr, motion_mag=temporal.residual_flow_mag
            )
            ph, pw = obs_curr.shape_hw

            # Focus fovea directly on peak moving mask if coherent object motion exists, else salience peak
            if np.any(temporal.moving_mask):
                mov_score = sal_map * (0.25 + temporal.moving_mask.astype(np.float32))
                ty, tx = np.unravel_index(int(np.argmax(mov_score)), (ph, pw))
            else:
                ty, tx = np.unravel_index(int(np.argmax(sal_map)), (ph, pw))

            ny, nx, (y0, x0, y1, x1) = self._extract_fovea_patch_coords(
                full_h=full_h,
                full_w=full_w,
                target_y_norm=float(ty) / max(1.0, ph - 1.0),
                target_x_norm=float(tx) / max(1.0, pw - 1.0),
                fovea_radius_ratio=fovea_radius_ratio,
            )
            motion_gaze_path_yx.append((ny, nx))

            fovea_crop = curr_rgb[y0:y1, x0:x1]
            fovea_obs = self.organ.observe(fovea_crop, max_width=140)
            vec6, scale_prof, orient_spec, chroma_sig, stats = (
                _extract_optical_descriptor(fovea_obs)
            )
            matches = self._match_against_memory(
                vec6, scale_prof, orient_spec, chroma_sig
            )
            best_id, best_score = matches[0] if matches else (None, 0.0)

            self._engram_counter += 1
            engram_id = f"{stream_name}_t{idx:02d}_{self._engram_counter:04d}"
            engram = FovealEngram(
                engram_id=engram_id,
                scene_name=stream_name,
                fixation_idx=idx - 1,
                gaze_yx_norm=(ny / float(max(1, full_h - 1)), nx / float(max(1, full_w - 1))),
                gaze_yx_px=(ny, nx),
                fovea_box_yxyx=(y0, x0, y1, x1),
                vector_6d=vec6,
                scale_profile=scale_prof,
                orientation_spectrum=orient_spec,
                chromatic_signature=chroma_sig,
                cleaved_bond_ratio=float(stats["cleaved_bond_ratio"]),
                material_boundary_count=int(stats["material_boundary_count"]),
                region_count=int(stats["region_count"]),
                salience_peak=float(sal_map[ty, tx]),
                recognition_status=(
                    "RECOGNIZED_STRUCTURE"
                    if best_score >= self.recognition_threshold
                    else "NOVEL_STRUCTURE"
                ),
                matched_engram_id=best_id,
                resonance_score=float(best_score),
                voxel_id=f"voxel_{engram_id}",
            )
            self.engram_memory[engram_id] = engram
            self._register_engram_voxel(engram, prev_voxel_id)
            prev_voxel_id = engram.voxel_id
            tracked_engrams.append(engram)

            flow_energy = float(np.mean(temporal.residual_flow_mag)) + (
                0.5 if temporal.event_classification == "INDEPENDENT_OBJECT_MOTION" else 0.0
            )
            if flow_energy > best_flow_energy:
                best_flow_energy = flow_energy
                sample_frame = curr_rgb.copy()
                sample_flow_mag = temporal.residual_flow_mag.copy()
                sample_moving_mask = temporal.moving_mask.copy()

            frame_events.append(
                {
                    "transition_idx": idx,
                    "event_classification": temporal.event_classification,
                    "occlusion_depth_verdict": temporal.occlusion_depth_verdict,
                    "structural_correlation": float(temporal.structural_correlation),
                    "global_shift_yx": [
                        float(temporal.global_shift_yx[0]),
                        float(temporal.global_shift_yx[1]),
                    ],
                    "mean_residual_flow": float(np.mean(temporal.residual_flow_mag)),
                    "max_residual_flow": float(np.max(temporal.residual_flow_mag)),
                    "moving_fraction": float(np.mean(temporal.moving_mask)),
                    "gaze_yx_px": [ny, nx],
                    "causal_explanation": temporal.causal_explanation,
                }
            )

        return VideoTrackingResult(
            video_name=stream_name,
            frames_processed=len(frames_rgb),
            frame_events=frame_events,
            motion_gaze_path_yx=motion_gaze_path_yx,
            sample_frame_rgb=sample_frame,
            sample_flow_mag=sample_flow_mag,
            sample_moving_mask=sample_moving_mask,
            tracked_engrams=tracked_engrams,
        )

    def observe_video_stream(
        self,
        video_path: Path | str,
        max_frames: int = 8,
        frame_stride: int = 8,
        max_width: int = 220,
    ) -> VideoTrackingResult:
        """Decode real frames from a video file (BGR->RGB) and track motion with active fovea."""
        vpath = Path(video_path)
        if not vpath.exists():
            raise FileNotFoundError(f"Video file not found: {vpath}")

        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video stream: {vpath}")

        frames_rgb: List[np.ndarray] = []
        frame_idx = 0
        try:
            while len(frames_rgb) < max_frames:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                if frame_idx % max(1, frame_stride) == 0:
                    frames_rgb.append(
                        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).copy()
                    )
                frame_idx += 1
        finally:
            cap.release()

        if len(frames_rgb) < 2:
            raise RuntimeError(
                f"Decoded only {len(frames_rgb)} frames from {vpath}; need at least 2."
            )

        return self.observe_frame_sequence(
            frames_rgb=frames_rgb,
            stream_name=vpath.stem,
            max_width=max_width,
        )

    def observe_live_screen(
        self,
        n_frames: int = 3,
        interval_sec: float = 0.4,
        max_width: int = 240,
    ) -> Dict[str, Any]:
        """Capture live desktop screen frames directly into the optical eye (no external VLM)."""
        from PIL import ImageGrab

        captures: List[np.ndarray] = []
        for idx in range(max(2, int(n_frames))):
            img = ImageGrab.grab()
            captures.append(np.array(img.convert("RGB"), dtype=np.uint8))
            if idx < n_frames - 1 and interval_sec > 0:
                time.sleep(interval_sec)

        initial_scan = self.scan_image(
            captures[0],
            scene_name="live_pc_screen",
            n_fixations=4,
            periphery_max_width=max_width,
        )
        tracking = self.observe_frame_sequence(
            frames_rgb=captures,
            stream_name="live_pc_screen_motion",
            max_width=max_width,
        )
        return {
            "initial_scan": initial_scan,
            "tracking": tracking,
        }
