"""
ocular_scale_organ.py

ELYSIA Multi-Scale Ocular Sensory Organ (L0-L4)
=================================================
Implements an optically grounded visual organ whose judgments emerge from
its physical and mathematical structure—retinal transduction, opponent
channels, continuous scale space, causal optical bond cleavage, circular
local re-observation, and temporal flow/occlusion analysis—without hardcoding
scene semantics or flattening distinct layers into one metric.

Strict Four-Domain Separation:
- Vision Domain: Linear irradiance, log-LMS photoreceptors, opponency,
  DoG center-surround, multi-scale gradients, illumination-invariant optical flow.
- Structure / Principle Domain: Optical bond lattice (stress, anisotropic
  grain cohesion, scale-coupled yield, iterative crack-tip redistribution),
  graph connected components, and Pass 2 local dynamic-range re-adaptation.
- Code Domain: Discrete numpy/scipy/cv2/skimage grid operators and explicit
  thresholds documented in LayerProvenance.
- Language Domain: Category strings are referential labels pointing back to
  measured metrics and causal provenance, never standalone causes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# Hunt-Pointer-Estevez sRGB -> LMS cone sensitivity matrix
_SRGB_TO_LMS = np.array(
    [
        [0.31399, 0.63951, 0.04650],
        [0.15537, 0.75789, 0.08670],
        [0.01775, 0.10944, 0.87257],
    ],
    dtype=np.float32,
)

# Cause map codes
CAUSE_HOMOGENEOUS = 0
CAUSE_SURFACE_TEXTURE = 1
CAUSE_ILLUMINATION_SHADOW_CANDIDATE = 2
CAUSE_MATERIAL_BOUNDARY = 3


@dataclass
class LayerProvenance:
    """Traces why and how a given optical layer produced its state."""

    layer_id: str
    name: str
    domain: str
    causal_mechanism: str
    why_necessary: str
    key_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PhotoreceptorField:
    """L0: Linear irradiance and logarithmic LMS cone responses."""

    linear_rgb: np.ndarray
    lms: np.ndarray
    log_l: np.ndarray
    log_m: np.ndarray
    log_s: np.ndarray
    valid_mask: np.ndarray
    provenance: LayerProvenance


@dataclass
class RetinalOpponentField:
    """L1: Achromatic luminance and illumination-canceling opponent channels."""

    luminance_a: np.ndarray
    rg_opponent: np.ndarray
    by_opponent: np.ndarray
    dog_fine: np.ndarray
    dog_coarse: np.ndarray
    provenance: LayerProvenance


@dataclass
class ScaleLevelFeatures:
    """L2: Differential optical geometry at a single Gaussian spatial scale."""

    sigma: float
    grad_a: np.ndarray
    grad_c: np.ndarray
    grad_total: np.ndarray
    orientation: np.ndarray
    coherence: np.ndarray
    normalized_energy: np.ndarray


@dataclass
class OpticalBondLattice:
    """Structural cleavage lattice connecting adjacent spatial cells."""

    shape_hw: Tuple[int, int]
    horiz_stress: np.ndarray
    vert_stress: np.ndarray
    horiz_yield: np.ndarray
    vert_yield: np.ndarray
    horiz_cleaved: np.ndarray
    vert_cleaved: np.ndarray
    crack_propagation_steps: int
    cleaved_bond_ratio: float
    single_scale_cleaved_ratio: float


@dataclass
class ScaleSpaceHierarchy:
    """L2: Cross-scale persistence, fine texture residual, and bond cleavage."""

    scales: Tuple[float, ...]
    levels: Dict[float, ScaleLevelFeatures]
    persistence_map: np.ndarray
    fine_texture_map: np.ndarray
    bond_lattice: OpticalBondLattice
    single_scale_fragment_count: int
    hierarchical_region_count: int
    coarse_region_count: int
    fragmentation_ratio: float
    provenance: LayerProvenance


@dataclass
class BoundaryEdge:
    """L3: Adjacency edge between two cleaved coherent regions."""

    region_a: int
    region_b: int
    length_px: int
    mean_delta_a: float
    mean_delta_c: float
    chroma_to_luma_ratio: float
    texture_discontinuity: float
    persistence: float
    mean_bond_stress: float
    mean_bond_yield: float
    optical_cause: str
    is_ambiguous: bool
    causal_explanation: str


@dataclass
class VisualRegion:
    """L3: Coherent region formed by surviving optical bonds."""

    region_id: int
    pixel_count: int
    area_ratio: float
    centroid_yx: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    compactness: float
    mean_a: float
    mean_rg: float
    mean_by: float
    chroma_std: float
    fine_texture_energy: float
    boundary_persistence: float
    parent_coarse_id: int
    child_fine_count: int
    micro_inclusion_count: int


@dataclass
class CircularReObservation:
    """Pass 2 closed-loop local adaptation inside a Pass 1 cleaved region."""

    parent_region_id: int
    pass1_mean_a: float
    pass1_contrast_std: float
    adapted_gain_factor: float
    pass1_internal_cleaved_ratio: float
    pass2_internal_cleaved_ratio: float
    newly_revealed_subregion_count: int
    newly_revealed_texture_energy: float
    causal_explanation: str


@dataclass
class OpticalSceneObservation:
    """Full single-frame multi-layer observation record."""

    shape_hw: Tuple[int, int]
    rgb_working: np.ndarray
    l0: PhotoreceptorField
    l1: RetinalOpponentField
    l2: ScaleSpaceHierarchy
    cause_map: np.ndarray
    region_labels: np.ndarray
    coarse_labels: np.ndarray
    fine_fragment_labels: np.ndarray
    regions: List[VisualRegion]
    boundaries: List[BoundaryEdge]
    ambiguities: List[Dict[str, Any]]
    circular_reobservations: List[CircularReObservation]
    provenance_chain: List[LayerProvenance]


@dataclass
class TemporalVisualAnalysis:
    """L4: Frame-to-frame optical transition, flow, and occlusion analysis."""

    flow_yx: np.ndarray
    global_shift_yx: Tuple[float, float]
    residual_flow_mag: np.ndarray
    mean_delta_a: float
    std_delta_a: float
    mean_delta_c: float
    structural_correlation: float
    event_classification: str
    moving_mask: np.ndarray
    occlusion_depth_verdict: str
    leading_occlusion_error: float
    interior_warp_error: float
    texture_shift_under_luma_edge: float
    causal_explanation: str


def _normalize_image_array(rgb_image: np.ndarray, max_width: int) -> np.ndarray:
    arr = np.asarray(rgb_image)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"Expected HxWx3 RGB array, got shape {arr.shape}")
    arr = arr[..., :3]
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
        if float(np.max(arr)) > 1.5:
            arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    h, w = arr.shape[:2]
    if max_width > 0 and w > max_width:
        new_h = max(32, int(round(h * (max_width / float(w)))))
        arr = cv2.resize(arr, (max_width, new_h), interpolation=cv2.INTER_AREA)
        arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
    return arr


def _connected_components_from_bonds(
    horiz_cleaved: np.ndarray, vert_cleaved: np.ndarray
) -> Tuple[int, np.ndarray]:
    """Compute grid connected components across surviving (non-cleaved) bonds."""
    h = vert_cleaved.shape[0] + 1
    w = horiz_cleaved.shape[1] + 1
    n_nodes = h * w
    node_ids = np.arange(n_nodes, dtype=np.int32).reshape(h, w)

    surv_h = ~horiz_cleaved
    u_h = node_ids[:, :-1][surv_h]
    v_h = node_ids[:, 1:][surv_h]

    surv_v = ~vert_cleaved
    u_v = node_ids[:-1, :][surv_v]
    v_v = node_ids[1:, :][surv_v]

    rows = np.concatenate([u_h, v_h, u_v, v_v])
    cols = np.concatenate([v_h, u_h, v_v, u_v])
    if rows.size == 0:
        return n_nodes, node_ids.copy()

    data = np.ones(rows.shape[0], dtype=np.int8)
    graph = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    n_comp, labels_flat = connected_components(graph, directed=False, return_labels=True)
    return int(n_comp), labels_flat.reshape(h, w)


def _bond_crack_neighbor_counts(
    h_cleaved: np.ndarray, v_cleaved: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Count neighboring cleaved bonds around every horizontal and vertical bond."""
    hh, hw = h_cleaved.shape  # H, W-1
    vh, vw = v_cleaved.shape  # H-1, W

    h_int = h_cleaved.astype(np.int32)
    v_int = v_cleaved.astype(np.int32)

    cnt_h = np.zeros_like(h_int)
    # Collinear vertical crack neighbors of horizontal bond (y, x)
    cnt_h[1:, :] += h_int[:-1, :]
    cnt_h[:-1, :] += h_int[1:, :]
    # Orthogonal corners meeting horizontal bond (y, x) endpoints
    cnt_h[:-1, :] += v_int[:, :-1] + v_int[:, 1:]
    cnt_h[1:, :] += v_int[:, :-1] + v_int[:, 1:]

    cnt_v = np.zeros_like(v_int)
    # Collinear horizontal crack neighbors of vertical bond (y, x)
    cnt_v[:, 1:] += v_int[:, :-1]
    cnt_v[:, :-1] += v_int[:, 1:]
    # Orthogonal corners meeting vertical bond (y, x) endpoints
    cnt_v[:, :-1] += h_int[:-1, :] + h_int[1:, :]
    cnt_v[:, 1:] += h_int[:-1, :] + h_int[1:, :]

    return cnt_h, cnt_v


def _extract_raw_adjacent_pairs(labels: np.ndarray) -> List[Tuple[int, int]]:
    pairs = set()
    diff_h = labels[:, :-1] != labels[:, 1:]
    if np.any(diff_h):
        a = labels[:, :-1][diff_h]
        b = labels[:, 1:][diff_h]
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        for u, v in zip(lo.tolist(), hi.tolist()):
            pairs.add((u, v))
    diff_v = labels[:-1, :] != labels[1:, :]
    if np.any(diff_v):
        a = labels[:-1, :][diff_v]
        b = labels[1:, :][diff_v]
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        for u, v in zip(lo.tolist(), hi.tolist()):
            pairs.add((u, v))
    return sorted(pairs)


class OcularScaleOrgan:
    """Genuine L0-L4 multi-scale visual organ using causal bond cleavage."""

    SCALES: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0)
    EPS: float = 1e-3
    BASE_YIELD: float = 1.05
    CRACK_PROPAGATION_STEPS: int = 3

    def observe(self, rgb_image: np.ndarray, max_width: int = 240) -> OpticalSceneObservation:
        """Process a raw RGB frame through L0->L1->L2(cleavage)->L3->Pass2 circular loop."""
        rgb_working = _normalize_image_array(rgb_image, max_width=max_width)
        h, w = rgb_working.shape[:2]

        l0 = self._transduce_photoreceptors(rgb_working)
        l1 = self._compute_retinal_opponency(l0)
        l2, fine_labels, region_labels, coarse_labels, micro_counts = (
            self._build_scale_space_and_bond_lattice(l1)
        )
        regions, boundaries, ambiguities, cause_map = self._disentangle_optical_causes(
            l0=l0,
            l1=l1,
            l2=l2,
            region_labels=region_labels,
            coarse_labels=coarse_labels,
            fine_labels=fine_labels,
            micro_counts=micro_counts,
        )
        circular_reobservations = self._run_circular_reobservation(
            l1=l1, l2=l2, region_labels=region_labels, regions=regions
        )

        circular_prov = LayerProvenance(
            layer_id="L3_CIRCULAR",
            name="Circular Local Re-Observation (Pass 2)",
            domain="Vision + Structure",
            causal_mechanism=(
                "Use Pass 1 cleaved region masks to re-calibrate local dynamic-range "
                "gain and re-evaluate internal optical bond stress vs yield inside dark/low-contrast zones."
            ),
            why_necessary=(
                "Observation conditions change after structural division: global dynamic range "
                "crushes low-contrast details inside shadows or large regions."
            ),
            key_metrics={
                "reobserved_regions": len(circular_reobservations),
                "total_newly_revealed_subregions": int(
                    sum(r.newly_revealed_subregion_count for r in circular_reobservations)
                ),
            },
        )
        provenance_chain = [
            l0.provenance,
            l1.provenance,
            l2.provenance,
            LayerProvenance(
                layer_id="L3",
                name="Causal Boundary & Ambiguity Disentanglement",
                domain="Vision + Structure",
                causal_mechanism=(
                    "Measure adjacent 4px border strips for luminance jump (Delta A), "
                    "chromaticity jump (Delta C), and fine-texture discontinuity; flag achromatic jumps as ambiguous."
                ),
                why_necessary=(
                    "Separate material boundaries from illumination shadow candidates without assuming pigment invariance."
                ),
                key_metrics={
                    "region_count": len(regions),
                    "boundary_edges": len(boundaries),
                    "material_boundaries": sum(1 for b in boundaries if b.optical_cause == "MATERIAL_BOUNDARY"),
                    "shadow_candidates": sum(
                        1 for b in boundaries if b.optical_cause == "ILLUMINATION_SHADOW_CANDIDATE"
                    ),
                },
            ),
            circular_prov,
        ]

        return OpticalSceneObservation(
            shape_hw=(h, w),
            rgb_working=rgb_working,
            l0=l0,
            l1=l1,
            l2=l2,
            cause_map=cause_map,
            region_labels=region_labels,
            coarse_labels=coarse_labels,
            fine_fragment_labels=fine_labels,
            regions=regions,
            boundaries=boundaries,
            ambiguities=ambiguities,
            circular_reobservations=circular_reobservations,
            provenance_chain=provenance_chain,
        )

    def _transduce_photoreceptors(self, rgb_working: np.ndarray) -> PhotoreceptorField:
        linear_rgb = np.power(np.clip(rgb_working, 0.0, 1.0), 2.2).astype(np.float32)
        lms = np.einsum("hwc,kc->hwk", linear_rgb, _SRGB_TO_LMS).astype(np.float32)
        log_lms = np.log(np.maximum(lms, 1e-4) + self.EPS).astype(np.float32)
        valid_mask = (lms[..., 0] + lms[..., 1]) > 0.03

        prov = LayerProvenance(
            layer_id="L0",
            name="Photoreceptor Linear-Log LMS Transduction",
            domain="Vision",
            causal_mechanism="Undo sRGB gamma (2.2), project to Hunt-Pointer-Estevez LMS cones, apply log(LMS + 1e-3).",
            why_necessary="Convert multiplicative irradiance k*R into additive log(k) + log(R) while gating noise floor.",
            key_metrics={
                "working_hw": tuple(rgb_working.shape[:2]),
                "valid_irradiance_ratio": float(np.mean(valid_mask)),
                "mean_linear_luma": float(np.mean(0.5 * (lms[..., 0] + lms[..., 1]))),
            },
        )
        return PhotoreceptorField(
            linear_rgb=linear_rgb,
            lms=lms,
            log_l=log_lms[..., 0],
            log_m=log_lms[..., 1],
            log_s=log_lms[..., 2],
            valid_mask=valid_mask,
            provenance=prov,
        )

    def _compute_retinal_opponency(self, l0: PhotoreceptorField) -> RetinalOpponentField:
        a = (0.5 * (l0.log_l + l0.log_m)).astype(np.float32)
        rg = (l0.log_l - l0.log_m).astype(np.float32)
        by = (l0.log_s - a).astype(np.float32)

        dog_fine = (gaussian_filter(a, 1.0) - gaussian_filter(a, 2.5)).astype(np.float32)
        dog_coarse = (gaussian_filter(a, 4.0) - gaussian_filter(a, 10.0)).astype(np.float32)

        prov = LayerProvenance(
            layer_id="L1",
            name="Retinal Chromatic Opponency & Center-Surround DoG",
            domain="Vision",
            causal_mechanism="Subtract log-cone channels (L-M, S-(L+M)/2) and take bandpass DoG on luminance A.",
            why_necessary=(
                "Cancel uniform multiplicative illumination k(x,y) identically in RG/BY channels and "
                "suppress global exposure DC in bandpass DoG."
            ),
            key_metrics={
                "luminance_std": float(np.std(a[l0.valid_mask])) if np.any(l0.valid_mask) else 0.0,
                "rg_std": float(np.std(rg[l0.valid_mask])) if np.any(l0.valid_mask) else 0.0,
                "by_std": float(np.std(by[l0.valid_mask])) if np.any(l0.valid_mask) else 0.0,
                "dog_fine_std": float(np.std(dog_fine)),
            },
        )
        return RetinalOpponentField(
            luminance_a=a,
            rg_opponent=rg,
            by_opponent=by,
            dog_fine=dog_fine,
            dog_coarse=dog_coarse,
            provenance=prov,
        )

    def _build_scale_space_and_bond_lattice(
        self, l1: RetinalOpponentField
    ) -> Tuple[ScaleSpaceHierarchy, np.ndarray, np.ndarray, np.ndarray, Dict[int, int]]:
        levels: Dict[float, ScaleLevelFeatures] = {}
        smoothed_fields: Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        for sigma in self.SCALES:
            a_s = gaussian_filter(l1.luminance_a, sigma=sigma)
            rg_s = gaussian_filter(l1.rg_opponent, sigma=sigma)
            by_s = gaussian_filter(l1.by_opponent, sigma=sigma)
            smoothed_fields[sigma] = (a_s, rg_s, by_s)

            gy_a, gx_a = np.gradient(a_s)
            grad_a = np.hypot(gy_a, gx_a).astype(np.float32)

            gy_rg, gx_rg = np.gradient(rg_s)
            gy_by, gx_by = np.gradient(by_s)
            grad_c = np.sqrt(gy_rg**2 + gx_rg**2 + gy_by**2 + gx_by**2).astype(np.float32)

            grad_total = np.sqrt(grad_a**2 + 2.0 * grad_c**2).astype(np.float32)

            tensor_sig = max(1.0, 0.8 * sigma)
            jxx = gaussian_filter(gx_a**2 + 2.0 * (gx_rg**2 + gx_by**2), tensor_sig)
            jyy = gaussian_filter(gy_a**2 + 2.0 * (gy_rg**2 + gy_by**2), tensor_sig)
            jxy = gaussian_filter(gx_a * gy_a + 2.0 * (gx_rg * gy_rg + gx_by * gy_by), tensor_sig)

            trace = jxx + jyy
            diff = jxx - jyy
            disc = np.sqrt(diff**2 + 4.0 * jxy**2)
            lam1 = 0.5 * (trace + disc)
            lam2 = 0.5 * (trace - disc)
            coherence = np.clip((lam1 - lam2) / (lam1 + lam2 + 1e-8), 0.0, 1.0).astype(np.float32)
            orientation = (0.5 * np.arctan2(2.0 * jxy, diff)).astype(np.float32)

            raw_e = sigma * grad_total
            p80 = float(np.percentile(raw_e, 80)) + 1e-6
            norm_e = np.clip(raw_e / p80, 0.0, 3.0).astype(np.float32)

            levels[sigma] = ScaleLevelFeatures(
                sigma=sigma,
                grad_a=grad_a,
                grad_c=grad_c,
                grad_total=grad_total,
                orientation=orientation,
                coherence=coherence,
                normalized_energy=norm_e,
            )

        e1 = levels[1.0].normalized_energy
        e2 = levels[2.0].normalized_energy
        e4 = levels[4.0].normalized_energy
        e8 = levels[8.0].normalized_energy

        coarse_support = np.power(np.maximum(e2 * e4 * e8, 0.0), 1.0 / 3.0).astype(np.float32)
        persistence_map = np.sqrt(np.maximum(e1 * coarse_support, 0.0)).astype(np.float32)
        fine_texture_map = np.maximum(0.0, e1 - coarse_support).astype(np.float32)

        h, w = l1.luminance_a.shape
        # Causal Optical Bond Lattice cleavage
        bond_lattice, region_labels, micro_counts = self._cleave_optical_bond_lattice(
            h=h,
            w=w,
            persistence_map=persistence_map,
            coarse_support=coarse_support,
            smoothed_sigma2=smoothed_fields[2.0],
            level_sigma2=levels[2.0],
            e1=e1,
        )

        # Single-scale ablation (for strict fragmentation comparison)
        fine_labels, single_fragment_count = self._single_scale_bond_cleavage(e1=e1)

        # Coarse-only parent envelope cleavage
        coarse_labels, coarse_region_count = self._coarse_envelope_cleavage(
            e8=e8, smoothed_sigma8=smoothed_fields[8.0], h=h, w=w
        )

        hier_count = int(region_labels.max()) + 1 if region_labels.size > 0 else 0
        frag_ratio = float(single_fragment_count / max(1, hier_count))

        prov = LayerProvenance(
            layer_id="L2",
            name="Continuous Scale Space & Causal Optical Bond Lattice",
            domain="Vision + Structure",
            causal_mechanism=(
                "Evaluate Gaussian scales sigma in (1,2,4,8,16); compute anisotropic grain cohesion "
                "along local tangents, scale-coupled yield threshold, and 3-step crack-tip stress concentration."
            ),
            why_necessary=(
                "Without coarse-scale yield reinforcement, fine texture shatters a continuous surface into "
                f"{single_fragment_count} disconnected fragments instead of {hier_count} coherent regions."
            ),
            key_metrics={
                "single_scale_fragment_count": single_fragment_count,
                "hierarchical_region_count": hier_count,
                "coarse_region_count": coarse_region_count,
                "fragmentation_ratio": frag_ratio,
                "cleaved_bond_ratio": bond_lattice.cleaved_bond_ratio,
                "single_scale_cleaved_ratio": bond_lattice.single_scale_cleaved_ratio,
            },
        )

        hierarchy = ScaleSpaceHierarchy(
            scales=self.SCALES,
            levels=levels,
            persistence_map=persistence_map,
            fine_texture_map=fine_texture_map,
            bond_lattice=bond_lattice,
            single_scale_fragment_count=single_fragment_count,
            hierarchical_region_count=hier_count,
            coarse_region_count=coarse_region_count,
            fragmentation_ratio=frag_ratio,
            provenance=prov,
        )
        return hierarchy, fine_labels, region_labels, coarse_labels, micro_counts

    def _cleave_optical_bond_lattice(
        self,
        h: int,
        w: int,
        persistence_map: np.ndarray,
        coarse_support: np.ndarray,
        smoothed_sigma2: Tuple[np.ndarray, np.ndarray, np.ndarray],
        level_sigma2: ScaleLevelFeatures,
        e1: np.ndarray,
    ) -> Tuple[OpticalBondLattice, np.ndarray, Dict[int, int]]:
        a2, rg2, by2 = smoothed_sigma2
        theta2 = level_sigma2.orientation
        coh2 = level_sigma2.coherence

        # Anisotropic tangent components: tangent = (-sin(theta), cos(theta))
        abs_tx = np.abs(-np.sin(theta2))
        abs_ty = np.abs(np.cos(theta2))

        grain_h = 0.5 * (coh2[:, :-1] * abs_tx[:, :-1] + coh2[:, 1:] * abs_tx[:, 1:])
        grain_v = 0.5 * (coh2[:-1, :] * abs_ty[:-1, :] + coh2[1:, :] * abs_ty[1:, :])

        persist_h = 0.5 * (persistence_map[:, :-1] + persistence_map[:, 1:])
        persist_v = 0.5 * (persistence_map[:-1, :] + persistence_map[1:, :])

        coarse_h = 0.5 * (coarse_support[:, :-1] + coarse_support[:, 1:])
        coarse_v = 0.5 * (coarse_support[:-1, :] + coarse_support[1:, :])

        jump_h = np.sqrt(
            (a2[:, :-1] - a2[:, 1:]) ** 2
            + 2.0 * ((rg2[:, :-1] - rg2[:, 1:]) ** 2 + (by2[:, :-1] - by2[:, 1:]) ** 2)
        )
        jump_v = np.sqrt(
            (a2[:-1, :] - a2[1:, :]) ** 2
            + 2.0 * ((rg2[:-1, :] - rg2[1:, :]) ** 2 + (by2[:-1, :] - by2[1:, :]) ** 2)
        )
        p80_jump = float(np.percentile(np.concatenate([jump_h.ravel(), jump_v.ravel()]), 80)) + 1e-6

        stress_h = (persist_h * np.clip(0.5 + 0.5 * (jump_h / p80_jump), 0.4, 2.0)).astype(np.float32)
        stress_v = (persist_v * np.clip(0.5 + 0.5 * (jump_v / p80_jump), 0.4, 2.0)).astype(np.float32)

        # Coarse cohesion only reinforces flat/textured interiors (>= 1.0 factor)
        coarse_coh_h = 1.0 / np.clip(0.45 + 0.55 * coarse_h, 0.45, 1.0)
        coarse_coh_v = 1.0 / np.clip(0.45 + 0.55 * coarse_v, 0.45, 1.0)

        yield_h = (self.BASE_YIELD * coarse_coh_h * (1.0 + 0.35 * grain_h)).astype(np.float32)
        yield_v = (self.BASE_YIELD * coarse_coh_v * (1.0 + 0.35 * grain_v)).astype(np.float32)

        # Iterative crack-tip stress concentration
        cleaved_h = stress_h > yield_h
        cleaved_v = stress_v > yield_v
        for _ in range(self.CRACK_PROPAGATION_STEPS):
            cnt_h, cnt_v = _bond_crack_neighbor_counts(cleaved_h, cleaved_v)
            eff_h = np.where(cnt_h >= 2, stress_h * (1.0 + 0.25 * np.minimum(cnt_h, 3)), stress_h)
            eff_v = np.where(cnt_v >= 2, stress_v * (1.0 + 0.25 * np.minimum(cnt_v, 3)), stress_v)
            new_h = cleaved_h | (eff_h > yield_h)
            new_v = cleaved_v | (eff_v > yield_v)
            if np.array_equal(new_h, cleaved_h) and np.array_equal(new_v, cleaved_v):
                break
            cleaved_h, cleaved_v = new_h, new_v

        # Connected components + micro-inclusion binding
        _, raw_labels = _connected_components_from_bonds(cleaved_h, cleaved_v)
        min_area = max(30, int(0.004 * h * w))
        region_labels, micro_counts = self._bind_micro_inclusions(raw_labels, min_area=min_area)

        total_bonds = cleaved_h.size + cleaved_v.size
        cleaved_ratio = float((np.sum(cleaved_h) + np.sum(cleaved_v)) / max(1, total_bonds))

        single_h = 0.5 * (e1[:, :-1] + e1[:, 1:]) > 0.72
        single_v = 0.5 * (e1[:-1, :] + e1[1:, :]) > 0.72
        single_ratio = float((np.sum(single_h) + np.sum(single_v)) / max(1, total_bonds))

        lattice = OpticalBondLattice(
            shape_hw=(h, w),
            horiz_stress=stress_h,
            vert_stress=stress_v,
            horiz_yield=yield_h,
            vert_yield=yield_v,
            horiz_cleaved=cleaved_h,
            vert_cleaved=cleaved_v,
            crack_propagation_steps=self.CRACK_PROPAGATION_STEPS,
            cleaved_bond_ratio=cleaved_ratio,
            single_scale_cleaved_ratio=single_ratio,
        )
        return lattice, region_labels, micro_counts

    def _bind_micro_inclusions(
        self, raw_labels: np.ndarray, min_area: int
    ) -> Tuple[np.ndarray, Dict[int, int]]:
        """Absorb isolated micro-flecks below min_area into nearest coherent parent regions."""
        from scipy.ndimage import distance_transform_edt

        counts = np.bincount(raw_labels.ravel())
        large_ids = np.where(counts >= min_area)[0]
        if large_ids.size == 0:
            # Fallback for very flat scenes: keep up to 2 largest components
            order = np.argsort(counts)[::-1]
            large_ids = order[: min(2, len(order))]

        is_large = np.isin(raw_labels, large_ids)
        if np.all(is_large):
            remapped = np.searchsorted(large_ids, raw_labels).astype(np.int32)
            return remapped, {i: 0 for i in range(len(large_ids))}

        _, indices = distance_transform_edt(~is_large, return_indices=True)
        bound_raw = np.where(is_large, raw_labels, raw_labels[indices[0], indices[1]])

        # Count how many micro components were bound into each surviving parent
        small_mask = ~is_large
        small_ids = raw_labels[small_mask]
        parent_ids = bound_raw[small_mask]
        micro_counts_raw: Dict[int, int] = {int(lid): 0 for lid in large_ids.tolist()}
        seen_pairs = set()
        for sid, pid in zip(small_ids.tolist(), parent_ids.tolist()):
            if (sid, pid) not in seen_pairs:
                seen_pairs.add((sid, pid))
                micro_counts_raw[pid] = micro_counts_raw.get(pid, 0) + 1

        unique_parents = np.sort(large_ids)
        remapped = np.searchsorted(unique_parents, bound_raw).astype(np.int32)
        micro_counts = {
            idx: micro_counts_raw.get(int(pid), 0) for idx, pid in enumerate(unique_parents.tolist())
        }
        return remapped, micro_counts

    def _single_scale_bond_cleavage(self, e1: np.ndarray) -> Tuple[np.ndarray, int]:
        """Single-scale ablation: fine-scale sigma=1.0 bonds without coarse yield reinforcement."""
        single_h = 0.5 * (e1[:, :-1] + e1[:, 1:]) > 0.72
        single_v = 0.5 * (e1[:-1, :] + e1[1:, :]) > 0.72
        _, fine_labels = _connected_components_from_bonds(single_h, single_v)
        counts = np.bincount(fine_labels.ravel())
        frag_count = int(np.sum(counts >= 4))
        return fine_labels, frag_count

    def _coarse_envelope_cleavage(
        self,
        e8: np.ndarray,
        smoothed_sigma8: Tuple[np.ndarray, np.ndarray, np.ndarray],
        h: int,
        w: int,
    ) -> Tuple[np.ndarray, int]:
        a8, rg8, by8 = smoothed_sigma8
        jh = np.sqrt((a8[:, :-1] - a8[:, 1:]) ** 2 + 2.0 * ((rg8[:, :-1] - rg8[:, 1:]) ** 2 + (by8[:, :-1] - by8[:, 1:]) ** 2))
        jv = np.sqrt((a8[:-1, :] - a8[1:, :]) ** 2 + 2.0 * ((rg8[:-1, :] - rg8[1:, :]) ** 2 + (by8[:-1, :] - by8[1:, :]) ** 2))
        p80 = float(np.percentile(np.concatenate([jh.ravel(), jv.ravel()]), 80)) + 1e-6

        s8h = (0.5 * (e8[:, :-1] + e8[:, 1:]) * (0.5 + jh / p80)).astype(np.float32)
        s8v = (0.5 * (e8[:-1, :] + e8[1:, :]) * (0.5 + jv / p80)).astype(np.float32)
        ch = s8h > (1.15 * self.BASE_YIELD)
        cv_bonds = s8v > (1.15 * self.BASE_YIELD)
        for _ in range(self.CRACK_PROPAGATION_STEPS):
            nh, nv = _bond_crack_neighbor_counts(ch, cv_bonds)
            ch = ch | (np.where(nh >= 2, s8h * (1.0 + 0.30 * np.minimum(nh, 3)), s8h) > (1.15 * self.BASE_YIELD))
            cv_bonds = cv_bonds | (np.where(nv >= 2, s8v * (1.0 + 0.30 * np.minimum(nv, 3)), s8v) > (1.15 * self.BASE_YIELD))
        _, raw_coarse = _connected_components_from_bonds(ch, cv_bonds)
        coarse_labels, _ = self._bind_micro_inclusions(raw_coarse, min_area=max(120, int(0.02 * h * w)))
        return coarse_labels, int(coarse_labels.max()) + 1

    def _disentangle_optical_causes(
        self,
        l0: PhotoreceptorField,
        l1: RetinalOpponentField,
        l2: ScaleSpaceHierarchy,
        region_labels: np.ndarray,
        coarse_labels: np.ndarray,
        fine_labels: np.ndarray,
        micro_counts: Dict[int, int],
    ) -> Tuple[List[VisualRegion], List[BoundaryEdge], List[Dict[str, Any]], np.ndarray]:
        h, w = region_labels.shape
        total_px = float(h * w)
        n_regions = int(region_labels.max()) + 1

        a2 = gaussian_filter(l1.luminance_a, 1.5)
        rg2 = gaussian_filter(l1.rg_opponent, 1.5)
        by2 = gaussian_filter(l1.by_opponent, 1.5)
        persistence = l2.persistence_map
        fine_tex = l2.fine_texture_map

        # Build perimeter length and persistence per region from cleaved bonds
        cleaved_h = l2.bond_lattice.horiz_cleaved
        cleaved_v = l2.bond_lattice.vert_cleaved
        border_mask = np.zeros((h, w), dtype=bool)
        border_mask[:, :-1] |= region_labels[:, :-1] != region_labels[:, 1:]
        border_mask[:, 1:] |= region_labels[:, :-1] != region_labels[:, 1:]
        border_mask[:-1, :] |= region_labels[:-1, :] != region_labels[1:, :]
        border_mask[1:, :] |= region_labels[:-1, :] != region_labels[1:, :]

        regions: List[VisualRegion] = []
        for rid in range(n_regions):
            rmask = region_labels == rid
            px_cnt = int(np.sum(rmask))
            if px_cnt == 0:
                continue
            ys, xs = np.where(rmask)
            y0, y1 = int(ys.min()), int(ys.max())
            x0, x1 = int(xs.min()), int(xs.max())
            b_px = rmask & border_mask
            perim = float(max(1, int(np.sum(b_px))))
            compactness = float((4.0 * np.pi * px_cnt) / (perim**2))

            v_rmask = rmask & l0.valid_mask
            if not np.any(v_rmask):
                v_rmask = rmask

            parent_vals = coarse_labels[rmask]
            parent_id = int(np.bincount(parent_vals).argmax())
            child_fines = np.unique(fine_labels[rmask])

            regions.append(
                VisualRegion(
                    region_id=rid,
                    pixel_count=px_cnt,
                    area_ratio=float(px_cnt / total_px),
                    centroid_yx=(float(np.mean(ys)), float(np.mean(xs))),
                    bbox=(y0, x0, y1, x1),
                    compactness=compactness,
                    mean_a=float(np.mean(l1.luminance_a[rmask])),
                    mean_rg=float(np.mean(l1.rg_opponent[v_rmask])),
                    mean_by=float(np.mean(l1.by_opponent[v_rmask])),
                    chroma_std=float(
                        np.sqrt(np.var(l1.rg_opponent[v_rmask]) + np.var(l1.by_opponent[v_rmask]))
                    ),
                    fine_texture_energy=float(np.mean(fine_tex[rmask])),
                    boundary_persistence=float(np.mean(persistence[b_px])) if np.any(b_px) else 0.0,
                    parent_coarse_id=parent_id,
                    child_fine_count=int(child_fines.size),
                    micro_inclusion_count=int(micro_counts.get(rid, 0)),
                )
            )

        # Evaluate every shared region-to-region boundary along 4px adjacent strips
        raw_pairs = _extract_raw_adjacent_pairs(region_labels)
        boundaries: List[BoundaryEdge] = []
        ambiguities: List[Dict[str, Any]] = []

        edge_cause_grid = np.zeros((h, w), dtype=np.int32)
        struct4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

        for ra, rb in raw_pairs:
            # Shared interface bonds
            bh = ((region_labels[:, :-1] == ra) & (region_labels[:, 1:] == rb)) | (
                (region_labels[:, :-1] == rb) & (region_labels[:, 1:] == ra)
            )
            bv = ((region_labels[:-1, :] == ra) & (region_labels[1:, :] == rb)) | (
                (region_labels[:-1, :] == rb) & (region_labels[1:, :] == ra)
            )
            shared_len = int(np.sum(bh) + np.sum(bv))
            if shared_len < 4:
                continue

            interface_px = np.zeros((h, w), dtype=bool)
            interface_px[:, :-1] |= bh
            interface_px[:, 1:] |= bh
            interface_px[:-1, :] |= bv
            interface_px[1:, :] |= bv

            strip_dil = binary_dilation(interface_px, structure=struct4, iterations=4)
            side_a = strip_dil & (region_labels == ra)
            side_b = strip_dil & (region_labels == rb)
            if not np.any(side_a) or not np.any(side_b):
                continue

            local_delta_a = float(abs(np.median(a2[side_a]) - np.median(a2[side_b])))
            # Chromaticity measured on pixels with usable cone SNR if possible
            va = side_a & l0.valid_mask
            vb = side_b & l0.valid_mask
            ca = va if int(np.sum(va)) >= 3 else side_a
            cb = vb if int(np.sum(vb)) >= 3 else side_b
            drg = float(np.median(rg2[ca]) - np.median(rg2[cb]))
            dby = float(np.median(by2[ca]) - np.median(by2[cb]))
            local_delta_c = float(np.hypot(drg, dby))

            ratio_cl = float(local_delta_c / (local_delta_a + 1e-4))
            tex_a = float(np.mean(fine_tex[side_a]))
            tex_b = float(np.mean(fine_tex[side_b]))
            local_tex_diff = float(abs(tex_a - tex_b) / (max(tex_a, tex_b) + 1e-4))

            # Bond stress & yield along this interface
            stresses = []
            yields = []
            if np.any(bh):
                stresses.append(l2.bond_lattice.horiz_stress[bh])
                yields.append(l2.bond_lattice.horiz_yield[bh])
            if np.any(bv):
                stresses.append(l2.bond_lattice.vert_stress[bv])
                yields.append(l2.bond_lattice.vert_yield[bv])
            mean_stress = float(np.mean(np.concatenate(stresses))) if stresses else 0.0
            mean_yield = float(np.mean(np.concatenate(yields))) if yields else 0.0
            mean_pers = float(np.mean(persistence[interface_px]))

            if local_delta_c >= 0.06 or ratio_cl >= 0.22 or local_tex_diff >= 0.45:
                cause = "MATERIAL_BOUNDARY"
                is_ambiguous = False
                explanation = (
                    f"Chromatic opponent shift dC={local_delta_c:.3f} (C/L={ratio_cl:.2f}) or "
                    f"texture jump dTex={local_tex_diff:.2f} indicates distinct optical surface properties."
                )
                edge_cause_grid[interface_px] = np.maximum(
                    edge_cause_grid[interface_px], CAUSE_MATERIAL_BOUNDARY
                )
            elif local_delta_a >= 0.12 and ratio_cl < 0.22 and local_tex_diff < 0.45:
                cause = "ILLUMINATION_SHADOW_CANDIDATE"
                is_ambiguous = True
                explanation = (
                    f"Strong achromatic luminance step dA={local_delta_a:.3f} with preserved chromaticity "
                    f"(dC={local_delta_c:.3f}, C/L={ratio_cl:.2f}) and continuous texture (dTex={local_tex_diff:.2f}); "
                    "cannot distinguish cast shadow from neutral pigment change in a static frame."
                )
                edge_cause_grid[interface_px] = np.maximum(
                    edge_cause_grid[interface_px], CAUSE_ILLUMINATION_SHADOW_CANDIDATE
                )
                ambiguities.append(
                    {
                        "type": "ACHROMATIC_SHADOW_VS_PIGMENT",
                        "region_a": ra,
                        "region_b": rb,
                        "delta_a": local_delta_a,
                        "delta_c": local_delta_c,
                        "chroma_to_luma_ratio": ratio_cl,
                        "texture_discontinuity": local_tex_diff,
                        "explanation": explanation,
                    }
                )
            else:
                cause = "WEAK_CONTINUOUS_TRANSITION"
                is_ambiguous = False
                explanation = (
                    f"Low optical jump across boundary (dA={local_delta_a:.3f}, dC={local_delta_c:.3f})."
                )

            boundaries.append(
                BoundaryEdge(
                    region_a=ra,
                    region_b=rb,
                    length_px=shared_len,
                    mean_delta_a=local_delta_a,
                    mean_delta_c=local_delta_c,
                    chroma_to_luma_ratio=ratio_cl,
                    texture_discontinuity=local_tex_diff,
                    persistence=mean_pers,
                    mean_bond_stress=mean_stress,
                    mean_bond_yield=mean_yield,
                    optical_cause=cause,
                    is_ambiguous=is_ambiguous,
                    causal_explanation=explanation,
                )
            )

        # Build pixelwise optical cause map combining L2 persistence/texture and L3 edge labels
        cause_map = np.full((h, w), CAUSE_HOMOGENEOUS, dtype=np.int32)
        tex_mask = (fine_tex > 0.55) & (persistence < 0.85)
        cause_map[tex_mask] = CAUSE_SURFACE_TEXTURE

        # Mark persistent gradients around σ=2 not on region interfaces as fine structural boundaries
        grad_a2 = l2.levels[2.0].grad_a
        grad_c2 = l2.levels[2.0].grad_c
        pers_edge = (persistence >= 0.85) & (edge_cause_grid == 0)
        mat_edge = pers_edge & ((grad_c2 / (grad_a2 + 1e-4)) >= 0.22)
        shad_edge = pers_edge & ((grad_c2 / (grad_a2 + 1e-4)) < 0.22) & (grad_a2 > 0.04)
        cause_map[shad_edge] = CAUSE_ILLUMINATION_SHADOW_CANDIDATE
        cause_map[mat_edge] = CAUSE_MATERIAL_BOUNDARY

        # Apply region-interface classifications prominently
        mat_iface = edge_cause_grid == CAUSE_MATERIAL_BOUNDARY
        shad_iface = edge_cause_grid == CAUSE_ILLUMINATION_SHADOW_CANDIDATE
        cause_map[shad_iface] = CAUSE_ILLUMINATION_SHADOW_CANDIDATE
        cause_map[mat_iface] = CAUSE_MATERIAL_BOUNDARY

        return regions, boundaries, ambiguities, cause_map

    def _run_circular_reobservation(
        self,
        l1: RetinalOpponentField,
        l2: ScaleSpaceHierarchy,
        region_labels: np.ndarray,
        regions: List[VisualRegion],
    ) -> List[CircularReObservation]:
        """
        Circular Principle (Pass 2):
        Inside major cleaved regions, re-calibrate local dynamic-range gain using
        the region's internal luminance dispersion and re-evaluate optical bond
        cleavage inside the region interior.
        """
        if not regions:
            return []

        scene_a_med = float(np.median(l1.luminance_a))
        global_a_std = float(np.std(l1.luminance_a)) + 1e-4
        candidates = [r for r in regions if r.area_ratio >= 0.04]
        # Prioritize darker or lower-contrast regions where global dynamic range crushed details
        candidates.sort(
            key=lambda r: (0 if r.mean_a < scene_a_med else 1, -r.area_ratio)
        )
        selected = candidates[:3]
        if not selected and regions:
            selected = sorted(regions, key=lambda r: -r.area_ratio)[:1]

        reobservations: List[CircularReObservation] = []
        bond_lat = l2.bond_lattice
        global_grad_p80 = float(np.percentile(l2.levels[1.0].grad_total, 80)) + 1e-6
        global_jmp_p80 = float(np.percentile(l2.levels[2.0].grad_total, 80)) + 1e-6

        for reg in selected:
            rmask = region_labels == reg.region_id
            interior = binary_erosion(rmask, iterations=2)
            if int(np.sum(interior)) < 40:
                interior = rmask

            local_vals = l1.luminance_a[interior]
            local_mean = float(np.mean(local_vals))
            local_std = float(np.std(local_vals)) + 1e-4
            gain = float(np.clip(global_a_std / local_std, 1.2, 6.0))

            # Pass 1 internal cleaved bond ratio strictly inside this region
            ih = interior[:, :-1] & interior[:, 1:]
            iv = interior[:-1, :] & interior[1:, :]
            total_int_bonds = int(np.sum(ih) + np.sum(iv))
            if total_int_bonds == 0:
                continue
            pass1_cleaved = (
                int(np.sum(bond_lat.horiz_cleaved & ih) + np.sum(bond_lat.vert_cleaved & iv))
                / float(total_int_bonds)
            )

            # Re-normalize local sensory field under adapted gain and evaluate against Pass 1 global baseline
            a_local = (l1.luminance_a - local_mean) * gain
            a_local_s = gaussian_filter(a_local, sigma=1.2)
            rg_local = (l1.rg_opponent - reg.mean_rg) * min(gain, 3.0)
            by_local = (l1.by_opponent - reg.mean_by) * min(gain, 3.0)
            rg_s = gaussian_filter(rg_local, sigma=1.2)
            by_s = gaussian_filter(by_local, sigma=1.2)

            gy_l, gx_l = np.gradient(a_local_s)
            gy_r, gx_r = np.gradient(rg_s)
            gy_b, gx_b = np.gradient(by_s)
            grad_loc = np.sqrt(
                gy_l**2 + gx_l**2 + 2.0 * (gy_r**2 + gx_r**2 + gy_b**2 + gx_b**2)
            )
            energy_loc = np.clip(grad_loc / global_grad_p80, 0.0, 5.0)

            jh_loc = np.sqrt(
                (a_local_s[:, :-1] - a_local_s[:, 1:]) ** 2
                + 2.0 * ((rg_s[:, :-1] - rg_s[:, 1:]) ** 2 + (by_s[:, :-1] - by_s[:, 1:]) ** 2)
            )
            jv_loc = np.sqrt(
                (a_local_s[:-1, :] - a_local_s[1:, :]) ** 2
                + 2.0 * ((rg_s[:-1, :] - rg_s[1:, :]) ** 2 + (by_s[:-1, :] - by_s[1:, :]) ** 2)
            )

            stress_h2 = 0.5 * (energy_loc[:, :-1] + energy_loc[:, 1:]) * np.clip(0.5 + 0.5 * (jh_loc / global_jmp_p80), 0.4, 2.5)
            stress_v2 = 0.5 * (energy_loc[:-1, :] + energy_loc[1:, :]) * np.clip(0.5 + 0.5 * (jv_loc / global_jmp_p80), 0.4, 2.5)

            cleaved_h2 = (stress_h2 > self.BASE_YIELD) & ih
            cleaved_v2 = (stress_v2 > self.BASE_YIELD) & iv
            pass2_cleaved = (
                int(np.sum(cleaved_h2) + np.sum(cleaved_v2)) / float(total_int_bonds)
            )

            # Count newly coherent internal sub-regions (>= 15px) within the parent interior
            # Cut outside bonds so components stay partitioned inside this region's interior
            outside_h = cleaved_h2 | ~ih
            outside_v = cleaved_v2 | ~iv
            _, sub_labels = _connected_components_from_bonds(outside_h, outside_v)
            int_sub = sub_labels[interior]
            sub_counts = np.bincount(int_sub.ravel())
            subregion_count = int(np.sum(sub_counts >= 15))
            newly_revealed = max(0, subregion_count - 1)
            revealed_tex_energy = float(np.mean(grad_loc[interior]))

            explanation = (
                f"Pass 1 cleaved Region {reg.region_id} (area={reg.area_ratio:.2%}, "
                f"mean_A={local_mean:.2f}); adapting local dynamic-range gain by {gain:.2f}x "
                f"raised internal bond cleavage from {pass1_cleaved:.3f} to {pass2_cleaved:.3f} "
                f"and revealed {newly_revealed} additional internal sub-structure(s)."
            )
            reobservations.append(
                CircularReObservation(
                    parent_region_id=reg.region_id,
                    pass1_mean_a=local_mean,
                    pass1_contrast_std=local_std,
                    adapted_gain_factor=gain,
                    pass1_internal_cleaved_ratio=float(pass1_cleaved),
                    pass2_internal_cleaved_ratio=float(pass2_cleaved),
                    newly_revealed_subregion_count=newly_revealed,
                    newly_revealed_texture_energy=revealed_tex_energy,
                    causal_explanation=explanation,
                )
            )

        return reobservations

    def observe_transition(
        self, obs1: OpticalSceneObservation, obs2: OpticalSceneObservation
    ) -> TemporalVisualAnalysis:
        """
        L4 Temporal Analysis:
        Separates global illumination changes, moving cast shadows, uniform eye
        saccades, independent object motion with occlusion depth ordering, and
        unrelated scene changes using illumination-invariant DoG optical flow.
        """
        if obs1.shape_hw != obs2.shape_hw:
            raise ValueError(
                f"Frame dimensions must match for temporal analysis: {obs1.shape_hw} vs {obs2.shape_hw}"
            )
        h, w = obs1.shape_hw

        valid = obs1.l0.valid_mask & obs2.l0.valid_mask
        if not np.any(valid):
            valid = np.ones((h, w), dtype=bool)

        delta_a = obs2.l1.luminance_a - obs1.l1.luminance_a
        delta_rg = obs2.l1.rg_opponent - obs1.l1.rg_opponent
        delta_by = obs2.l1.by_opponent - obs1.l1.by_opponent
        delta_c = np.hypot(delta_rg, delta_by)

        margin = max(12, min(20, min(h, w) // 6))
        interior = np.zeros((h, w), dtype=bool)
        interior[margin : h - margin, margin : w - margin] = True
        eval_mask = valid & interior
        if not np.any(eval_mask):
            eval_mask = interior

        mean_delta_a = float(np.mean(delta_a[eval_mask]))
        std_delta_a = float(np.std(delta_a[eval_mask]))
        mean_delta_c = float(np.mean(delta_c[eval_mask]))

        # Persistence structural correlation across interior
        p1 = obs1.l2.persistence_map[interior].ravel()
        p2 = obs2.l2.persistence_map[interior].ravel()
        if np.std(p1) > 1e-6 and np.std(p2) > 1e-6:
            structural_corr = float(np.corrcoef(p1, p2)[0, 1])
        else:
            structural_corr = 1.0 if np.allclose(p1, p2) else 0.0

        # 1. Global rigid translation alignment via illumination-invariant DoG phase correlation
        s1 = (obs1.l1.dog_fine + 0.6 * obs1.l1.dog_coarse).astype(np.float64)
        s2 = (obs2.l1.dog_fine + 0.6 * obs2.l1.dog_coarse).astype(np.float64)
        hann = cv2.createHanningWindow((w, h), cv2.CV_64F)
        (dx_pc, dy_pc), pc_resp = cv2.phaseCorrelate(s1 * hann, s2 * hann)
        pc_speed = float(np.hypot(dy_pc, dx_pc))

        # 2. Multi-band illumination-invariant field for dense local optical flow
        def _to_multiband_u8(obs_item: OpticalSceneObservation) -> np.ndarray:
            mb = (
                obs_item.l1.dog_fine
                + 0.8 * obs_item.l1.dog_coarse
                + 0.6 * (obs_item.l1.rg_opponent - float(np.median(obs_item.l1.rg_opponent)))
                + 0.6 * (obs_item.l1.by_opponent - float(np.median(obs_item.l1.by_opponent)))
            )
            return np.clip(127.5 + mb * 180.0, 0.0, 255.0).astype(np.uint8)

        u1 = _to_multiband_u8(obs1)
        u2 = _to_multiband_u8(obs2)
        flow_xy_12 = cv2.calcOpticalFlowFarneback(
            u1, u2, None, pyr_scale=0.5, levels=5, winsize=19, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        flow_yx = np.stack([flow_xy_12[..., 1], flow_xy_12[..., 0]], axis=-1).astype(np.float32)

        int_vy = flow_yx[interior, 0]
        int_vx = flow_yx[interior, 1]
        global_vy = float(np.median(int_vy))
        global_vx = float(np.median(int_vx))
        global_speed = float(np.hypot(global_vy, global_vx))

        residual_mag = np.hypot(flow_yx[..., 0] - global_vy, flow_yx[..., 1] - global_vx).astype(
            np.float32
        )

        # Check rigid saccade continuity before local flow thresholding
        saccade_aligned_corr = 0.0
        if pc_speed >= 1.5 and pc_resp >= 0.25:
            warp_m = np.array([[1.0, 0.0, -dx_pc], [0.0, 1.0, -dy_pc]], dtype=np.float32)
            s2_aligned = cv2.warpAffine(
                s2.astype(np.float32), warp_m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
            a1_vec = s1[interior].ravel()
            a2_vec = s2_aligned[interior].ravel()
            if np.std(a1_vec) > 1e-6 and np.std(a2_vec) > 1e-6:
                saccade_aligned_corr = float(np.corrcoef(a1_vec, a2_vec)[0, 1])

        # Non-uniform luminance shift band (potential cast shadow band)
        med_da = float(np.median(delta_a[interior]))
        luma_changed = (np.abs(delta_a - med_da) > 0.20) & interior
        luma_changed_ratio = float(np.mean(luma_changed[interior]))

        raw_flow_mag = np.hypot(flow_yx[..., 0], flow_yx[..., 1])
        if np.any(luma_changed):
            tex_shift_under_luma = float(np.median(raw_flow_mag[luma_changed]))
            band_delta_c = float(np.mean(delta_c[luma_changed]))
            d1_band = obs1.l1.dog_fine[luma_changed]
            d2_band = obs2.l1.dog_fine[luma_changed]
            if np.std(d1_band) > 1e-6 and np.std(d2_band) > 1e-6:
                band_dog_corr = float(np.corrcoef(d1_band, d2_band)[0, 1])
            else:
                band_dog_corr = 0.0
        else:
            tex_shift_under_luma = float(np.median(raw_flow_mag[interior]))
            band_delta_c = mean_delta_c
            band_dog_corr = 1.0

        raw_moving_mask = (residual_mag > 2.0) & interior
        moving_frac = float(np.mean(raw_moving_mask[interior]))

        empty_mask = np.zeros((h, w), dtype=bool)
        effective_global_shift = (global_vy, global_vx)

        # Classification waterfall respecting each mechanism's domain invariants
        if saccade_aligned_corr >= 0.75 and pc_speed >= 1.5:
            event_cls = "SELF_EYE_MOTION"
            effective_global_shift = (float(dy_pc), float(dx_pc))
            flow_yx[..., 0] = float(dy_pc)
            flow_yx[..., 1] = float(dx_pc)
            residual_mag = np.zeros((h, w), dtype=np.float32)
            moving_mask = empty_mask
            occ_verdict = "GLOBAL_RETINAL_TRANSLATION"
            lead_err = 0.0
            int_err = 0.0
            explanation = (
                f"Uniform full-field DoG translation verified by phase correlation "
                f"(global_shift_yx=({dy_pc:+.2f}, {dx_pc:+.2f})px, pc_resp={pc_resp:.3f}, "
                f"post_align_corr={saccade_aligned_corr:.3f})."
            )

        elif (
            abs(mean_delta_a) < 0.03
            and std_delta_a < 0.04
            and mean_delta_c < 0.03
            and structural_corr > 0.98
            and global_speed < 0.5
            and moving_frac < 0.02
        ):
            event_cls = "STATIC_IDENTICAL"
            moving_mask = empty_mask
            occ_verdict = "NOT_APPLICABLE"
            lead_err = 0.0
            int_err = 0.0
            explanation = (
                f"Negligible retinal shift (mean dA={mean_delta_a:.4f}, dC={mean_delta_c:.4f}) "
                f"and nearly zero motion (global={global_speed:.2f}px, corr={structural_corr:.3f})."
            )

        elif (
            abs(mean_delta_a) > 0.15
            and std_delta_a < (0.35 * abs(mean_delta_a) + 0.08)
            and mean_delta_c < 0.05
            and structural_corr > 0.90
            and global_speed < 1.0
            and pc_speed < 1.5
        ):
            event_cls = "GLOBAL_ILLUMINATION_CHANGE"
            moving_mask = empty_mask
            occ_verdict = "NOT_APPLICABLE"
            lead_err = 0.0
            int_err = 0.0
            explanation = (
                f"Uniform log-luminance shift (mean dA={mean_delta_a:+.3f}, std={std_delta_a:.3f}) "
                f"with preserved opponent chromaticity (dC={mean_delta_c:.4f}) and "
                f"scale-space persistence (corr={structural_corr:.3f})."
            )

        elif (
            luma_changed_ratio > 0.02
            and band_delta_c < 0.06
            and tex_shift_under_luma < 0.9
            and moving_frac < 0.02
            and (band_dog_corr > 0.55 or structural_corr > 0.85)
            and global_speed < 1.0
        ):
            event_cls = "MOVING_CAST_SHADOW"
            moving_mask = empty_mask
            occ_verdict = "NON_OCCLUDING_ILLUMINATION_WAVE"
            lead_err = 0.0
            int_err = 0.0
            explanation = (
                f"Localized luminance edge displaced across {luma_changed_ratio:.1%} of interior while "
                f"local band chromaticity stayed invariant (band_dC={band_delta_c:.4f}) and "
                f"illumination-invariant surface texture stayed stationary (flow={tex_shift_under_luma:.2f}px)."
            )

        elif global_speed >= 1.5 and moving_frac < 0.12 and mean_delta_c < 0.10:
            event_cls = "SELF_EYE_MOTION"
            moving_mask = empty_mask
            occ_verdict = "GLOBAL_RETINAL_TRANSLATION"
            lead_err = 0.0
            int_err = 0.0
            explanation = (
                f"Uniform full-field optical flow detected (global_shift_yx=({global_vy:+.2f}, {global_vx:+.2f})px, "
                f"speed={global_speed:.2f}px) with low residual motion fraction ({moving_frac:.1%})."
            )

        elif 0.01 <= moving_frac <= 0.65 and global_speed < 1.5 and structural_corr > 0.40:
            event_cls = "INDEPENDENT_OBJECT_MOTION"
            mov_blob = self._keep_largest_coherent_blob(raw_moving_mask)
            # Isolate moving core from Farneback smoothing halo
            p90_spd = float(np.percentile(raw_flow_mag[mov_blob], 90)) if np.any(mov_blob) else 0.0
            core_cand = mov_blob & (raw_flow_mag >= 0.55 * p90_spd)
            moving_mask = self._keep_largest_coherent_blob(core_cand) if int(np.sum(core_cand)) >= 25 else mov_blob

            occ_verdict, lead_err, int_err = self._evaluate_occlusion_depth(
                u1=u1, u2=u2, flow_xy_12=flow_xy_12, moving_mask=moving_mask, interior=interior
            )
            obj_vy = float(np.median(flow_yx[moving_mask, 0]))
            obj_vx = float(np.median(flow_yx[moving_mask, 1]))
            explanation = (
                f"Isolated coherent region ({np.mean(moving_mask[interior]):.1%} of interior) moving at "
                f"velocity ({obj_vy:+.2f}, {obj_vx:+.2f})px against stationary background (global={global_speed:.2f}px). "
                f"Occlusion test: {occ_verdict} (leading_rim_err={lead_err:.2f} vs core_conserv_err={int_err:.2f})."
            )

        else:
            event_cls = "SCENE_CHANGE"
            moving_mask = empty_mask
            occ_verdict = "DISJOINT_SCENES"
            lead_err = 0.0
            int_err = 0.0
            explanation = (
                f"Cross-frame structural divergence (corr={structural_corr:.3f}, "
                f"mean_dC={mean_delta_c:.3f}, global_shift={global_speed:.2f}px) exceeds continuous transform bounds."
            )

        return TemporalVisualAnalysis(
            flow_yx=flow_yx,
            global_shift_yx=effective_global_shift,
            residual_flow_mag=residual_mag,
            mean_delta_a=mean_delta_a,
            std_delta_a=std_delta_a,
            mean_delta_c=mean_delta_c,
            structural_correlation=structural_corr,
            event_classification=event_cls,
            moving_mask=moving_mask,
            occlusion_depth_verdict=occ_verdict,
            leading_occlusion_error=lead_err,
            interior_warp_error=int_err,
            texture_shift_under_luma_edge=tex_shift_under_luma,
            causal_explanation=explanation,
        )

    def _keep_largest_coherent_blob(self, mask: np.ndarray) -> np.ndarray:
        n_labels, cc_labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        if n_labels <= 1:
            return mask
        counts = np.bincount(cc_labels.ravel())
        counts[0] = 0
        best = int(np.argmax(counts))
        return cc_labels == best

    def _evaluate_occlusion_depth(
        self,
        u1: np.ndarray,
        u2: np.ndarray,
        flow_xy_12: np.ndarray,
        moving_mask: np.ndarray,
        interior: np.ndarray,
    ) -> Tuple[str, float, float]:
        """Compare moving-core texture conservation error vs leading-rim background occlusion error."""
        if int(np.sum(moving_mask)) < 25:
            return "INSUFFICIENT_AREA", 0.0, 0.0

        h, w = u1.shape
        eroded_int = binary_erosion(moving_mask, iterations=3)
        if int(np.sum(eroded_int)) < 15:
            eroded_int = moving_mask

        vy = float(np.median(flow_xy_12[moving_mask, 1]))
        vx = float(np.median(flow_xy_12[moving_mask, 0]))
        speed = float(np.hypot(vy, vx))
        if speed < 1.0:
            return "AMBIGUOUS_ZERO_VELOCITY", 0.0, 0.0

        # Texture conservation inside the moving core: u1(p) vs u2(p + v)
        ys_i, xs_i = np.where(eroded_int)
        ty = np.clip(np.round(ys_i + vy).astype(np.int32), 0, h - 1)
        tx = np.clip(np.round(xs_i + vx).astype(np.int32), 0, w - 1)
        core_conserv_err = float(np.mean(np.abs(u1[ys_i, xs_i].astype(np.float32) - u2[ty, tx].astype(np.float32))))

        # Leading-rim pixels in Frame 1 that are overwritten by the shifted core in Frame 2
        ys_c, xs_c = np.where(moving_mask)
        sy = np.clip(np.round(ys_c + vy).astype(np.int32), 0, h - 1)
        sx = np.clip(np.round(xs_c + vx).astype(np.int32), 0, w - 1)
        shifted_core = np.zeros_like(moving_mask)
        shifted_core[sy, sx] = True
        leading_rim = shifted_core & (~moving_mask) & interior

        if int(np.sum(leading_rim)) < 10:
            return "LEADING_RIM_OUT_OF_FRAME", 0.0, core_conserv_err

        lead_err = float(np.mean(np.abs(u1[leading_rim].astype(np.float32) - u2[leading_rim].astype(np.float32))))

        if lead_err > core_conserv_err * 1.35:
            verdict = "MOVING_REGION_IN_FRONT_OF_BACKGROUND"
        else:
            verdict = "AMBIGUOUS_DEPTH_ORDER"
        return verdict, lead_err, core_conserv_err
