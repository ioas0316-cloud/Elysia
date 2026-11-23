"""
ocular_topological_composition_engine.py

ELYSIA Topological Mold Composition & From-Scratch Novel Scene Recreation
==========================================================================
Extends OcularGenerativeSpectrumEngine so ELYSIA no longer merely reproduces
or modulates a single given image: it harvests autonomous OcularPrimitiveAtoms
from diverse real-world scenes via active saccadic vision, synthesizes blank
retinal atmospheric fields from causal statistics, and weaves atoms together
through multi-scale Burt-Adelson DoG pyramid blending into brand-new scenes —
then closes the loop by re-observing its own novel creations inside CausalField(6D).

Zero VLMs, zero LLMs, zero neural diffusion models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from core.physics.causal_field import CausalField, InformationVoxel
from core.sensory.active_ocular_mechanics_engine import (
    ActiveOcularMechanicsEngine,
    FovealEngram,
    _extract_optical_descriptor,
)
from core.sensory.ocular_generative_spectrum_engine import (
    OcularCognitiveFactorization,
    OcularGenerativeSpectrumEngine,
    _compute_persistence_correlation,
    _decompose_telescoping_bands,
    _recompose_telescoping_bands,
)
from core.sensory.ocular_scale_organ import (
    OcularScaleOrgan,
    OpticalSceneObservation,
)


@dataclass
class OcularPrimitiveAtom:
    """Autonomous multi-band optical object/texture atom harvested by active foveation."""

    atom_id: str
    source_scene: str
    engram: FovealEngram
    factorization: OcularCognitiveFactorization
    organic_mask: np.ndarray
    chromatic_role: str
    voxel_id: str


@dataclass
class AtomPlacement:
    """Spatial & photometric coupling specification for placing an atom onto a blank mold."""

    atom_id: str
    center_yx_norm: Tuple[float, float]
    scale_ratio: float
    opacity: float = 0.95
    luma_offset: float = 0.0
    chroma_harmonize: float = 0.50


@dataclass
class ComposedSceneResult:
    """Novel scene synthesized from blank canvas + causal provenance & novelty verification."""

    composition_name: str
    canvas_shape_hw: Tuple[int, int]
    synthesized_rgb: np.ndarray
    reobserved_obs: OpticalSceneObservation
    placements: List[AtomPlacement]
    atom_provenance_recalls: List[Dict[str, Any]]
    novelty_vs_sources: Dict[str, float]
    cleaved_bond_ratio: float
    material_boundary_count: int
    region_count: int
    voxel_id: str
    placed_boxes_yxyx: List[Tuple[int, int, int, int]] = field(default_factory=list)


def _build_organic_support_mask(obs: OpticalSceneObservation) -> np.ndarray:
    """
    Create a smooth organic optical support mask in [0, 1] combining a raised-cosine
    retinal taper (strictly < 0.05 along all edges so zero rectangular seams appear)
    with local cleaved-boundary persistence.
    """
    h, w = obs.shape_hw
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    r = np.hypot(yy, xx)

    # Cosine taper: 1.0 for r <= 0.25, smoothly descending to 0.0 by r = 0.92
    t = np.clip((r - 0.25) / 0.67, 0.0, 1.0)
    radial = (0.5 * (1.0 + np.cos(np.pi * t))).astype(np.float32)

    # Extra edge-margin envelope over outer 12% rows/cols
    edge_y = np.minimum(
        np.linspace(0.0, 1.0, h, dtype=np.float32) / 0.12,
        np.linspace(1.0, 0.0, h, dtype=np.float32) / 0.12,
    )[:, None]
    edge_x = np.minimum(
        np.linspace(0.0, 1.0, w, dtype=np.float32) / 0.12,
        np.linspace(1.0, 0.0, w, dtype=np.float32) / 0.12,
    )[None, :]
    edge_env = np.clip(np.minimum(edge_y, edge_x), 0.0, 1.0)

    persist = obs.l2.persistence_map.astype(np.float32)
    p_hi = float(np.percentile(persist, 95.0)) + 1e-6
    persist_norm = np.clip(persist / p_hi, 0.0, 1.0)
    support = (0.68 + 0.32 * gaussian_filter(persist_norm, sigma=2.5)).astype(
        np.float32
    )

    raw_mask = np.clip(radial * edge_env * support, 0.0, 1.0).astype(np.float32)
    smooth_mask = gaussian_filter(raw_mask, sigma=max(1.2, 0.015 * float(min(h, w))))
    # Re-enforce smooth cosine envelope so outermost border pixels stay well below 0.05
    final_mask = np.clip(smooth_mask * radial * edge_env, 0.0, 1.0).astype(np.float32)
    return final_mask


def _classify_chromatic_role(vec_rgb_ryb: np.ndarray) -> str:
    idx = int(np.argmax(vec_rgb_ryb))
    if idx == 0:
        return "FLUX_DYNAMIC"
    if idx == 1:
        return "ORDER_STRUCTURAL"
    return "ENTROPY_ATMOSPHERIC"


class OcularTopologicalCompositionEngine:
    """
    Topological Composition Engine: harvests real foveal atoms across scenes,
    generates blank ambient retinal worlds, and composes novel coherent scenes
    using scale-progressive DoG pyramid blending + CausalField graph binding.
    """

    def __init__(self, organ: Optional[OcularScaleOrgan] = None) -> None:
        self.organ = organ if organ is not None else OcularScaleOrgan()
        self.gen = OcularGenerativeSpectrumEngine(organ=self.organ)
        self.eye: ActiveOcularMechanicsEngine = self.gen.eye
        self.causal_field: CausalField = self.gen.causal_field
        self.atoms: Dict[str, OcularPrimitiveAtom] = {}
        self.source_obs: Dict[str, OpticalSceneObservation] = {}
        self._comp_counter = 0

    def harvest_atoms_from_scene(
        self,
        rgb_uint8: np.ndarray,
        scene_name: str,
        n_fixations: int = 4,
        periphery_max_width: int = 220,
        fovea_radius_ratio: float = 0.18,
        atom_size: int = 128,
    ) -> List[OcularPrimitiveAtom]:
        """Scan a real scene actively and factorize each fixation into a reusable optical atom."""
        scan = self.eye.scan_image(
            rgb_uint8,
            scene_name=scene_name,
            n_fixations=n_fixations,
            periphery_max_width=periphery_max_width,
            fovea_radius_ratio=fovea_radius_ratio,
            fovea_max_width=atom_size,
        )
        self.source_obs[scene_name] = scan.peripheral_obs

        harvested: List[OcularPrimitiveAtom] = []
        for engram, crop_rgb in zip(scan.foveal_engrams, scan.foveal_crops_rgb):
            fact = self.gen.factorize(
                crop_rgb,
                source_name=f"atom_{engram.engram_id}",
                max_width=atom_size,
            )
            organic_mask = _build_organic_support_mask(fact.obs)
            eng_voxel = self.causal_field.voxels.get(engram.voxel_id)
            if eng_voxel is not None and eng_voxel.chromatic_vector is not None:
                role = _classify_chromatic_role(eng_voxel.chromatic_vector)
            else:
                role = "ORDER_STRUCTURAL"

            if (
                engram.voxel_id in self.causal_field.voxels
                and fact.voxel_id in self.causal_field.voxels
            ):
                self.causal_field.link_voxels(engram.voxel_id, fact.voxel_id, strength=0.95)

            atom = OcularPrimitiveAtom(
                atom_id=engram.engram_id,
                source_scene=scene_name,
                engram=engram,
                factorization=fact,
                organic_mask=organic_mask,
                chromatic_role=role,
                voxel_id=fact.voxel_id,
            )
            self.atoms[atom.atom_id] = atom
            harvested.append(atom)

        return harvested

    def _build_blank_ambient_canvas(
        self,
        canvas_hw: Tuple[int, int],
        atmosphere_atom: OcularPrimitiveAtom,
        gradient_strength: float = 0.35,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Synthesize a continuous blank retinal atmosphere from statistical luminance/opponent
        moments and low-frequency harmonic waves (zero copied source pixels).
        """
        h, w = canvas_hw
        fact = atmosphere_atom.factorization
        mean_a = float(np.mean(fact.luma_base))
        mean_rg, std_rg = fact.global_rg_mean_std
        mean_by, std_by = fact.global_by_mean_std

        yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
        xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]

        # Continuous vertical horizon luminance gradient + soft smooth harmonic mist
        wave_a = 0.08 * np.sin(np.pi * xx) * np.cos(0.5 * np.pi * yy)
        a_bg = (mean_a - float(gradient_strength) * yy + wave_a).astype(np.float32)

        wave_rg = 0.18 * std_rg * np.cos(0.7 * np.pi * yy + 0.4 * xx)
        wave_by = 0.22 * std_by * np.sin(0.6 * np.pi * yy - 0.3 * xx)
        rg_bg = np.full((h, w), mean_rg, dtype=np.float32) + wave_rg.astype(np.float32)
        by_bg = np.full((h, w), mean_by, dtype=np.float32) + wave_by.astype(np.float32)

        a_bg = gaussian_filter(a_bg, sigma=6.0).astype(np.float32)
        rg_bg = gaussian_filter(rg_bg, sigma=6.0).astype(np.float32)
        by_bg = gaussian_filter(by_bg, sigma=6.0).astype(np.float32)
        return a_bg, rg_bg, by_bg

    def compose_novel_scene(
        self,
        composition_name: str,
        placements: List[AtomPlacement],
        atmosphere_atom_id: str,
        canvas_hw: Tuple[int, int] = (240, 320),
        global_luma_shift: float = 0.0,
        global_chroma_rot_deg: float = 0.0,
        harmonize_atmosphere: float = 0.40,
    ) -> ComposedSceneResult:
        """
        Build a brand-new visual composition on a blank ambient canvas using multi-scale
        Burt-Adelson DoG pyramid blending, then verify novelty & component recall closed-loop.
        """
        if atmosphere_atom_id not in self.atoms:
            raise KeyError(f"Unknown atmosphere atom: {atmosphere_atom_id}")
        if not placements:
            raise ValueError("At least one AtomPlacement is required")

        h_c, w_c = int(canvas_hw[0]), int(canvas_hw[1])
        atmo_atom = self.atoms[atmosphere_atom_id]
        atmo_rg_mean = atmo_atom.factorization.global_rg_mean_std[0]
        atmo_by_mean = atmo_atom.factorization.global_by_mean_std[0]

        a_can_init, rg_can_init, by_can_init = self._build_blank_ambient_canvas(
            (h_c, w_c), atmo_atom
        )
        scales = OcularScaleOrgan.SCALES  # (1.0, 2.0, 4.0, 8.0, 16.0)
        can_a_base, can_a_bands = _decompose_telescoping_bands(a_can_init, scales)
        can_rg_base, can_rg_bands = _decompose_telescoping_bands(rg_can_init, scales)
        can_by_base, can_by_bands = _decompose_telescoping_bands(by_can_init, scales)

        # Progressive mask sigmas per DoG band index [B0(fine)..B4(coarse)] + base
        band_mask_sigmas = (1.2, 2.0, 3.5, 6.0, 9.0)
        base_mask_sigma = 10.5

        placed_boxes: List[Tuple[int, int, int, int]] = []
        placed_atom_objs: List[OcularPrimitiveAtom] = []

        for pl in placements:
            if pl.atom_id not in self.atoms:
                raise KeyError(f"Unknown placed atom: {pl.atom_id}")
            atom = self.atoms[pl.atom_id]
            fact = atom.factorization
            placed_atom_objs.append(atom)

            ah_src, aw_src = fact.shape_hw
            target_short = max(44, int(round(pl.scale_ratio * float(min(h_c, w_c)))))
            aspect = float(aw_src) / max(1.0, float(ah_src))
            if aspect >= 1.0:
                th = target_short
                tw = max(44, int(round(target_short * aspect)))
            else:
                tw = target_short
                th = max(44, int(round(target_short / aspect)))

            a_src_full = _recompose_telescoping_bands(fact.luma_base, fact.luma_bands)
            rg_src_full = _recompose_telescoping_bands(fact.rg_base, fact.rg_bands)
            by_src_full = _recompose_telescoping_bands(fact.by_base, fact.by_bands)

            a_resized = cv2.resize(a_src_full, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32)
            rg_resized = cv2.resize(rg_src_full, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32)
            by_resized = cv2.resize(by_src_full, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32)
            m_resized = np.clip(
                cv2.resize(atom.organic_mask, (tw, th), interpolation=cv2.INTER_LINEAR),
                0.0,
                1.0,
            ).astype(np.float32) * float(np.clip(pl.opacity, 0.0, 1.0))

            # Atmosphere harmonization in invariant retinal opponent & luminance coordinates
            harm_w = float(np.clip(pl.chroma_harmonize * harmonize_atmosphere, 0.0, 0.85))
            atom_rg_mean = fact.global_rg_mean_std[0]
            atom_by_mean = fact.global_by_mean_std[0]
            rg_resized += harm_w * (atmo_rg_mean - atom_rg_mean)
            by_resized += harm_w * (atmo_by_mean - atom_by_mean)
            a_resized += float(pl.luma_offset)

            cy = int(round(np.clip(pl.center_yx_norm[0], 0.05, 0.95) * (h_c - 1)))
            cx = int(round(np.clip(pl.center_yx_norm[1], 0.05, 0.95) * (w_c - 1)))

            py0 = cy - th // 2
            py1 = py0 + th
            px0 = cx - tw // 2
            px1 = px0 + tw

            cy0 = max(0, py0)
            cy1 = min(h_c, py1)
            cx0 = max(0, px0)
            cx1 = min(w_c, px1)
            if cy1 <= cy0 or cx1 <= cx0:
                continue

            sy0 = cy0 - py0
            sy1 = sy0 + (cy1 - cy0)
            sx0 = cx0 - px0
            sx1 = sx0 + (cx1 - cx0)
            placed_boxes.append((cy0, cx0, cy1, cx1))

            # Embed onto canvas background (never 0.0 black-as-white log artifact) before DoG split
            curr_a_est = _recompose_telescoping_bands(can_a_base, can_a_bands)
            curr_rg_est = _recompose_telescoping_bands(can_rg_base, can_rg_bands)
            curr_by_est = _recompose_telescoping_bands(can_by_base, can_by_bands)

            layer_a = curr_a_est.copy()
            layer_rg = curr_rg_est.copy()
            layer_by = curr_by_est.copy()
            layer_a[cy0:cy1, cx0:cx1] = a_resized[sy0:sy1, sx0:sx1]
            layer_rg[cy0:cy1, cx0:cx1] = rg_resized[sy0:sy1, sx0:sx1]
            layer_by[cy0:cy1, cx0:cx1] = by_resized[sy0:sy1, sx0:sx1]

            mask_layer = np.zeros((h_c, w_c), dtype=np.float32)
            mask_layer[cy0:cy1, cx0:cx1] = m_resized[sy0:sy1, sx0:sx1]

            lay_a_base, lay_a_bands = _decompose_telescoping_bands(layer_a, scales)
            lay_rg_base, lay_rg_bands = _decompose_telescoping_bands(layer_rg, scales)
            lay_by_base, lay_by_bands = _decompose_telescoping_bands(layer_by, scales)

            m_base = np.clip(gaussian_filter(mask_layer, sigma=base_mask_sigma), 0.0, 1.0)
            can_a_base = (1.0 - m_base) * can_a_base + m_base * lay_a_base
            can_rg_base = (1.0 - m_base) * can_rg_base + m_base * lay_rg_base
            can_by_base = (1.0 - m_base) * can_by_base + m_base * lay_by_base

            for b_idx, sig_m in enumerate(band_mask_sigmas):
                m_b = np.clip(gaussian_filter(mask_layer, sigma=sig_m), 0.0, 1.0)
                can_a_bands[b_idx] = (
                    (1.0 - m_b) * can_a_bands[b_idx] + m_b * lay_a_bands[b_idx]
                )
                can_rg_bands[b_idx] = (
                    (1.0 - m_b) * can_rg_bands[b_idx] + m_b * lay_rg_bands[b_idx]
                )
                can_by_bands[b_idx] = (
                    (1.0 - m_b) * can_by_bands[b_idx] + m_b * lay_by_bands[b_idx]
                )

        a_final = _recompose_telescoping_bands(can_a_base, can_a_bands) + float(
            global_luma_shift
        )
        rg_raw = _recompose_telescoping_bands(can_rg_base, can_rg_bands)
        by_raw = _recompose_telescoping_bands(can_by_base, can_by_bands)

        if abs(global_chroma_rot_deg) > 1e-4:
            theta = float(np.deg2rad(global_chroma_rot_deg))
            c_th, s_th = float(np.cos(theta)), float(np.sin(theta))
            rg_final = c_th * rg_raw - s_th * by_raw
            by_final = s_th * rg_raw + c_th * by_raw
        else:
            rg_final, by_final = rg_raw, by_raw

        synth_rgb = self.gen._inverse_retinal_to_srgb(a_final, rg_final, by_final)

        # Closed-loop optical re-observation
        reobs = self.organ.observe(synth_rgb, max_width=w_c)

        # 1. Verify genuine structural novelty against full original source scenes
        novelty_vs_sources: Dict[str, float] = {}
        p_comp = reobs.l2.persistence_map.astype(np.float32)
        for src_name, src_obs in self.source_obs.items():
            p_src_resized = cv2.resize(
                src_obs.l2.persistence_map.astype(np.float32),
                (w_c, h_c),
                interpolation=cv2.INTER_AREA,
            )
            v1 = p_comp.ravel()
            v2 = p_src_resized.ravel()
            if np.std(v1) > 1e-6 and np.std(v2) > 1e-6:
                corr = float(np.corrcoef(v1, v2)[0, 1])
            else:
                corr = 0.0
            novelty_vs_sources[src_name] = corr

        # 2. Verify component provenance recall inside the composed scene
        placed_ids = {pl.atom_id for pl in placements}
        atom_provenance_recalls: List[Dict[str, Any]] = []
        for pl, box in zip(placements, placed_boxes):
            y0, x0, y1, x1 = box
            crop = synth_rgb[y0:y1, x0:x1]
            if crop.shape[0] < 16 or crop.shape[1] < 16:
                continue
            c_obs = self.organ.observe(crop, max_width=120)
            c_vec6, c_sc, c_or, c_ch, _ = _extract_optical_descriptor(c_obs)

            true_atom = self.atoms[pl.atom_id]
            self_res = self.eye._compute_structural_resonance(
                c_vec6, c_sc, c_or, c_ch, true_atom.engram
            )

            non_placed_scores = [
                self.eye._compute_structural_resonance(
                    c_vec6, c_sc, c_or, c_ch, other.engram
                )
                for oid, other in self.atoms.items()
                if oid not in placed_ids
            ]
            mean_distractor = (
                float(np.mean(non_placed_scores)) if non_placed_scores else 0.0
            )
            all_scores = sorted(
                [
                    (
                        oid,
                        self.eye._compute_structural_resonance(
                            c_vec6, c_sc, c_or, c_ch, a_obj.engram
                        ),
                    )
                    for oid, a_obj in self.atoms.items()
                ],
                key=lambda kv: kv[1],
                reverse=True,
            )
            rank = next(
                (i + 1 for i, (oid, _) in enumerate(all_scores) if oid == pl.atom_id),
                len(all_scores),
            )
            atom_provenance_recalls.append(
                {
                    "atom_id": pl.atom_id,
                    "source_scene": true_atom.source_scene,
                    "self_resonance": float(self_res),
                    "mean_non_placed_distractor": mean_distractor,
                    "margin_over_mean_distractor": float(self_res - mean_distractor),
                    "rank_among_all_atoms": int(rank),
                    "placed_box_yxyx": list(box),
                }
            )

        # Register composed scene in CausalField(6D) linked to every parent atom
        comp_vec6, comp_sc, _, comp_ch, comp_stats = _extract_optical_descriptor(reobs)
        self._comp_counter += 1
        comp_voxel_id = f"voxel_comp_{composition_name}_{self._comp_counter:04d}"
        self.gen._register_voxel(
            voxel_id=comp_voxel_id,
            content={
                "mode": "TOPOLOGICAL_COMPOSITION",
                "composition_name": composition_name,
                "atom_count": len(placed_atom_objs),
                "novelty_vs_sources": novelty_vs_sources,
            },
            vec6=comp_vec6,
            scale_prof=comp_sc,
            chroma_sig=comp_ch,
            parent_ids=[a.voxel_id for a in placed_atom_objs],
        )

        return ComposedSceneResult(
            composition_name=composition_name,
            canvas_shape_hw=(h_c, w_c),
            synthesized_rgb=synth_rgb,
            reobserved_obs=reobs,
            placements=placements,
            atom_provenance_recalls=atom_provenance_recalls,
            novelty_vs_sources=novelty_vs_sources,
            cleaved_bond_ratio=float(comp_stats["cleaved_bond_ratio"]),
            material_boundary_count=int(comp_stats["material_boundary_count"]),
            region_count=int(comp_stats["region_count"]),
            voxel_id=comp_voxel_id,
            placed_boxes_yxyx=placed_boxes,
        )
