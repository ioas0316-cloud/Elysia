"""
visual_causal_reasoner.py

Cognitive Causal Reasoning & CausalField Projection Bridge for OcularScaleOrgan
===============================================================================
Bridges raw multi-layer optical observations (L0-L4) to explicit four-domain
auditing, cross-scene causal attribution, active epistemic hypothesis testing,
and multi-scale CausalField voxel/beam projection.

Strict Four-Domain Separation:
- Vision Domain: Radiometric/optical invariants (photons, opponency, flow, occlusion).
- Structure / Principle Domain: Bond lattice cleavage, crack-tip propagation,
  circular local adaptation, and CausalField connectivity graph.
- Code Domain: Concrete array operations, thresholds, and execution parameters.
- Language Domain: Human-readable identifiers pointing to measured evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.physics.causal_field import CausalField, ConnectivityBeam, InformationVoxel
from core.sensory.ocular_scale_organ import (
    BoundaryEdge,
    OcularScaleOrgan,
    OpticalSceneObservation,
    TemporalVisualAnalysis,
)


@dataclass
class FourDomainCausalAudit:
    """Explicit separation of Vision, Structure, Code, Language, and Modality Limits."""

    vision_domain: Dict[str, Any]
    structure_domain: Dict[str, Any]
    code_domain: Dict[str, Any]
    language_domain: Dict[str, Any]
    modality_limits: List[str]


@dataclass
class SceneComparisonReport:
    """Layer-by-layer causal comparison explaining what stayed invariant vs what diverged."""

    l0_linear_rmse: float
    l1_delta_a: float
    l1_delta_c: float
    l2_persistence_corr: float
    l3_region_count_diff: int
    l4_event_classification: str
    first_divergence_layer: str
    same_invariants: List[str]
    diverged_causes: List[str]
    causal_attribution: str


@dataclass
class EpistemicHypothesis:
    """Testable causal hypothesis emitted from optical ambiguity or partial evidence."""

    hypothesis_id: str
    kind: str
    region_pair: Optional[Tuple[int, int]]
    prediction: str
    required_action_or_test: str
    status: str  # OPEN | CONFIRMED | FALSIFIED | REQUIRES_OTHER_MODALITY
    evidence: Dict[str, Any] = field(default_factory=dict)


class VisualCausalReasoner:
    """Connects optical layer outputs to testable cognition and CausalField topology."""

    def __init__(self, organ: Optional[OcularScaleOrgan] = None):
        self.organ = organ if organ is not None else OcularScaleOrgan()

    def audit_organ_causality(self, obs: OpticalSceneObservation) -> FourDomainCausalAudit:
        """Audit an observation across all four distinct causal domains + optical limits."""
        mat_cnt = sum(1 for b in obs.boundaries if b.optical_cause == "MATERIAL_BOUNDARY")
        shad_cnt = sum(
            1 for b in obs.boundaries if b.optical_cause == "ILLUMINATION_SHADOW_CANDIDATE"
        )
        weak_cnt = sum(
            1 for b in obs.boundaries if b.optical_cause == "WEAK_CONTINUOUS_TRANSITION"
        )

        vision_domain = {
            "working_resolution_hw": obs.shape_hw,
            "l0_valid_irradiance_ratio": obs.l0.provenance.key_metrics["valid_irradiance_ratio"],
            "l0_mean_linear_luma": obs.l0.provenance.key_metrics["mean_linear_luma"],
            "l1_luminance_std": obs.l1.provenance.key_metrics["luminance_std"],
            "l1_rg_std": obs.l1.provenance.key_metrics["rg_std"],
            "l1_by_std": obs.l1.provenance.key_metrics["by_std"],
            "l3_material_boundaries": mat_cnt,
            "l3_shadow_candidate_boundaries": shad_cnt,
            "l3_weak_transitions": weak_cnt,
            "static_ambiguities_count": len(obs.ambiguities),
            "note": (
                "Measured strictly from incoming photon radiance (RGB linearized by gamma 2.2) "
                "and spatial/chromatic derivatives."
            ),
        }

        structure_domain = {
            "single_scale_fragment_count": obs.l2.single_scale_fragment_count,
            "hierarchical_region_count": obs.l2.hierarchical_region_count,
            "coarse_region_count": obs.l2.coarse_region_count,
            "fragmentation_reduction_ratio": obs.l2.fragmentation_ratio,
            "single_scale_cleaved_bond_ratio": obs.l2.bond_lattice.single_scale_cleaved_ratio,
            "scale_coupled_cleaved_bond_ratio": obs.l2.bond_lattice.cleaved_bond_ratio,
            "crack_propagation_steps": obs.l2.bond_lattice.crack_propagation_steps,
            "total_micro_inclusions_bound": sum(r.micro_inclusion_count for r in obs.regions),
            "circular_pass2_reobservations": len(obs.circular_reobservations),
            "circular_newly_revealed_subregions": sum(
                c.newly_revealed_subregion_count for c in obs.circular_reobservations
            ),
            "note": (
                "Regions emerge from surviving bonds in the optical lattice after anisotropic "
                "grain cohesion and scale-coupled yield thresholding, not a post-hoc contour slice."
            ),
        }

        code_domain = {
            "scales_px": obs.l2.scales,
            "log_epsilon": self.organ.EPS,
            "base_yield_threshold": self.organ.BASE_YIELD,
            "crack_propagation_steps": self.organ.CRACK_PROPAGATION_STEPS,
            "discretization_grid": obs.shape_hw,
            "library_primitives": [
                "numpy.gradient",
                "scipy.ndimage.gaussian_filter",
                "scipy.sparse.csgraph.connected_components",
                "cv2.calcOpticalFlowFarneback",
            ],
            "note": (
                "Numeric thresholds (e.g. BASE_YIELD=1.05, C/L>=0.22) are discrete code "
                "approximations calibrating the digital lattice, NOT physical material laws."
            ),
        }

        language_domain = {
            "semantic_tokens_used": [
                "MATERIAL_BOUNDARY",
                "ILLUMINATION_SHADOW_CANDIDATE",
                "WEAK_CONTINUOUS_TRANSITION",
                "CircularReObservation",
            ],
            "referential_contract": (
                "These strings are descriptive labels indexing measured optical/structural metrics "
                "above; assigning a label never alters lattice physics or proves external physical identity."
            ),
        }

        modality_limits = [
            "Optical irradiance alone cannot measure mechanical mass, rigidity, or Young's modulus (Pa).",
            "Static achromatic boundaries cannot decisively separate a neutral pigment edge from a cast shadow without illumination shift or motion.",
            "Hidden rear geometry (3D back-faces) and internal material density are unobserved by surface reflectance photons.",
            "Thermal temperature cannot be inferred in standard visible-band sRGB imagery.",
        ]

        return FourDomainCausalAudit(
            vision_domain=vision_domain,
            structure_domain=structure_domain,
            code_domain=code_domain,
            language_domain=language_domain,
            modality_limits=modality_limits,
        )

    def compare_observations(
        self,
        obs1: OpticalSceneObservation,
        obs2: OpticalSceneObservation,
        temporal: Optional[TemporalVisualAnalysis] = None,
    ) -> SceneComparisonReport:
        """Trace layer-by-layer similarity vs divergence between two real frames."""
        if temporal is None:
            temporal = self.organ.observe_transition(obs1, obs2)

        l0_rmse = float(np.sqrt(np.mean((obs1.l0.linear_rgb - obs2.l0.linear_rgb) ** 2)))
        region_diff = len(obs2.regions) - len(obs1.regions)
        evt = temporal.event_classification

        same_invariants: List[str] = []
        diverged_causes: List[str] = []

        if evt == "STATIC_IDENTICAL":
            first_div = "NONE_INVARIANT"
            same_invariants.extend(
                [
                    f"L0 linear irradiance preserved (RMSE={l0_rmse:.4f})",
                    f"L1 chromaticity preserved (dC={temporal.mean_delta_c:.4f})",
                    f"L2 cross-scale persistence invariant (corr={temporal.structural_correlation:.3f})",
                    f"L4 retinal flow ~0 ({np.hypot(*temporal.global_shift_yx):.2f}px)",
                ]
            )
            attribution = (
                f"Both observations represent the same stationary optical field "
                f"(L0 RMSE={l0_rmse:.4f}, L2 corr={temporal.structural_correlation:.3f})."
            )
        elif evt == "GLOBAL_ILLUMINATION_CHANGE":
            first_div = "L0/L1_LUMINANCE"
            same_invariants.extend(
                [
                    f"L1 opponent chromaticity invariant (dC={temporal.mean_delta_c:.4f})",
                    f"L2 cross-scale structure preserved (corr={temporal.structural_correlation:.3f})",
                    f"L4 stationary geometry (global_shift={np.hypot(*temporal.global_shift_yx):.2f}px)",
                ]
            )
            diverged_causes.extend(
                [
                    f"L0 linear irradiance scaled (RMSE={l0_rmse:.4f})",
                    f"L1 uniform log-luminance shifted (mean dA={temporal.mean_delta_a:+.3f}, std={temporal.std_delta_a:.3f})",
                ]
            )
            attribution = (
                f"Diverged at L0/L1 luminance by global multiplicative exposure factor (~exp({temporal.mean_delta_a:+.2f})="
                f"{np.exp(temporal.mean_delta_a):.2f}x) while physical chromaticity (dC={temporal.mean_delta_c:.4f}) "
                f"and L2 structure (corr={temporal.structural_correlation:.3f}) stayed invariant."
            )
        elif evt == "MOVING_CAST_SHADOW":
            first_div = "L1_LOCAL_LUMINANCE"
            same_invariants.extend(
                [
                    f"L1 opponent chromaticity invariant (dC={temporal.mean_delta_c:.4f})",
                    f"L1 DoG fine surface texture stationary under edge (flow={temporal.texture_shift_under_luma_edge:.2f}px)",
                ]
            )
            diverged_causes.append(
                f"L1 localized luminance edge displaced without moving underlying surface texture."
            )
            attribution = (
                "Diverged strictly in local illumination: the luminance step shifted spatially while "
                f"underlying high-pass DoG surface texture stayed stationary (flow={temporal.texture_shift_under_luma_edge:.2f}px) "
                "and chromatic opponency remained unchanged."
            )
        elif evt == "SELF_EYE_MOTION":
            first_div = "L4_GLOBAL_MOTION"
            same_invariants.extend(
                [
                    f"L1 scene chromatic distribution preserved (dC={temporal.mean_delta_c:.4f})",
                    f"Rigid full-field optical translation (residual non-rigid fraction < 12%)",
                ]
            )
            diverged_causes.append(
                f"L4 uniform retinal shift global_shift_yx=({temporal.global_shift_yx[0]:+.2f}, {temporal.global_shift_yx[1]:+.2f})px."
            )
            attribution = (
                f"Diverged at L4 via rigid full-frame eye/camera translation "
                f"({temporal.global_shift_yx[0]:+.2f}, {temporal.global_shift_yx[1]:+.2f})px; scene content is continuous."
            )
        elif evt == "INDEPENDENT_OBJECT_MOTION":
            first_div = "L4_LOCAL_OBJECT_MOTION"
            same_invariants.append("L4 stationary background frame (global residual < 1.5px)")
            diverged_causes.extend(
                [
                    f"L4 coherent isolated region moving independently ({temporal.causal_explanation})",
                    f"L4 occlusion depth: {temporal.occlusion_depth_verdict} "
                    f"(leading_err={temporal.leading_occlusion_error:.2f} > interior_err={temporal.interior_warp_error:.2f})",
                ]
            )
            attribution = temporal.causal_explanation
        else:  # SCENE_CHANGE
            first_div = "L1_CHROMATICITY_AND_L2_STRUCTURE"
            diverged_causes.extend(
                [
                    f"L0 linear field changed (RMSE={l0_rmse:.4f})",
                    f"L1 opponent chromaticity diverged (dC={temporal.mean_delta_c:.3f})",
                    f"L2 cross-scale persistence broke (corr={temporal.structural_correlation:.3f})",
                ]
            )
            attribution = (
                f"Disjoint scene change starting at L1/L2 (dC={temporal.mean_delta_c:.3f}, "
                f"L2 structural_corr={temporal.structural_correlation:.3f}); cannot be explained by lighting or rigid motion."
            )

        return SceneComparisonReport(
            l0_linear_rmse=l0_rmse,
            l1_delta_a=temporal.mean_delta_a,
            l1_delta_c=temporal.mean_delta_c,
            l2_persistence_corr=temporal.structural_correlation,
            l3_region_count_diff=region_diff,
            l4_event_classification=evt,
            first_divergence_layer=first_div,
            same_invariants=same_invariants,
            diverged_causes=diverged_causes,
            causal_attribution=attribution,
        )

    def generate_hypotheses(self, obs: OpticalSceneObservation) -> List[EpistemicHypothesis]:
        """Generate falsifiable predictions from ambiguous optical observations + modality limits."""
        hypotheses: List[EpistemicHypothesis] = []
        idx = 0

        for edge in obs.boundaries:
            if edge.optical_cause == "ILLUMINATION_SHADOW_CANDIDATE":
                idx += 1
                hypotheses.append(
                    EpistemicHypothesis(
                        hypothesis_id=f"H_{idx:02d}_SHADOW",
                        kind="ACHROMATIC_SHADOW_VS_PIGMENT",
                        region_pair=(edge.region_a, edge.region_b),
                        prediction=(
                            "If illumination/light source moves, the luminance boundary between Regions "
                            f"{edge.region_a} and {edge.region_b} will shift while high-pass DoG surface texture "
                            "remains stationary (< 0.9 px flow) and RG/BY chromaticity stays invariant."
                        ),
                        required_action_or_test=(
                            "Observe temporal transition under shadow/illumination displacement (MOVING_CAST_SHADOW)."
                        ),
                        status="OPEN",
                        evidence={
                            "delta_a": edge.mean_delta_a,
                            "delta_c": edge.mean_delta_c,
                            "chroma_to_luma_ratio": edge.chroma_to_luma_ratio,
                            "texture_discontinuity": edge.texture_discontinuity,
                        },
                    )
                )

        if obs.boundaries:
            # Take highest-persistence boundary for static depth-order ambiguity
            strongest = max(obs.boundaries, key=lambda b: b.persistence)
            idx += 1
            hypotheses.append(
                EpistemicHypothesis(
                    hypothesis_id=f"H_{idx:02d}_DEPTH",
                    kind="STATIC_DEPTH_ORDER_UNKNOWN",
                    region_pair=(strongest.region_a, strongest.region_b),
                    prediction=(
                        f"Static single frame cannot determine whether Region {strongest.region_a} or "
                        f"Region {strongest.region_b} lies closer in depth; relative object motion will produce "
                        "higher photometric/flow-consistency occlusion error on the background leading rim than interior."
                    ),
                    required_action_or_test=(
                        "Observe independent relative motion or parallax shift (INDEPENDENT_OBJECT_MOTION)."
                    ),
                    status="OPEN",
                    evidence={
                        "boundary_length_px": strongest.length_px,
                        "persistence": strongest.persistence,
                    },
                )
            )

        for circ in obs.circular_reobservations:
            if circ.newly_revealed_subregion_count > 0:
                idx += 1
                hypotheses.append(
                    EpistemicHypothesis(
                        hypothesis_id=f"H_{idx:02d}_CIRCULAR",
                        kind="CIRCULAR_HIDDEN_SUBSTRUCTURE",
                        region_pair=(circ.parent_region_id, circ.parent_region_id),
                        prediction=(
                            f"Local gain adaptation ({circ.adapted_gain_factor:.2f}x) inside Region "
                            f"{circ.parent_region_id} revealed {circ.newly_revealed_subregion_count} finer sub-structures."
                        ),
                        required_action_or_test="Higher-resolution fixation or spatial zoom on Region interior.",
                        status="CONFIRMED",
                        evidence={
                            "pass1_cleaved": circ.pass1_internal_cleaved_ratio,
                            "pass2_cleaved": circ.pass2_internal_cleaved_ratio,
                            "newly_revealed_subregions": circ.newly_revealed_subregion_count,
                        },
                    )
                )

        idx += 1
        hypotheses.append(
            EpistemicHypothesis(
                hypothesis_id=f"H_{idx:02d}_MODALITY",
                kind="MODALITY_BOUNDARY_NON_OPTICAL",
                region_pair=None,
                prediction=(
                    "Cleaved optical regions indicate reflectance/illumination discontinuities, NOT physical "
                    "mechanical yield stress, tensile bond strength, mass, or surface temperature."
                ),
                required_action_or_test=(
                    "Couple visual boundary coordinates with motor-tactile force probe (efference vs displacement)."
                ),
                status="REQUIRES_OTHER_MODALITY",
                evidence={
                    "optical_regions": len(obs.regions),
                    "optical_cleaved_bond_ratio": obs.l2.bond_lattice.cleaved_bond_ratio,
                },
            )
        )

        return hypotheses

    def verify_hypotheses_with_transition(
        self,
        hypotheses: List[EpistemicHypothesis],
        obs1: OpticalSceneObservation,
        obs2: OpticalSceneObservation,
        temporal: Optional[TemporalVisualAnalysis] = None,
    ) -> List[EpistemicHypothesis]:
        """Resolve OPEN optical hypotheses using observed L4 temporal transitions."""
        if temporal is None:
            temporal = self.organ.observe_transition(obs1, obs2)

        updated: List[EpistemicHypothesis] = []
        evt = temporal.event_classification

        for hyp in hypotheses:
            h_copy = EpistemicHypothesis(
                hypothesis_id=hyp.hypothesis_id,
                kind=hyp.kind,
                region_pair=hyp.region_pair,
                prediction=hyp.prediction,
                required_action_or_test=hyp.required_action_or_test,
                status=hyp.status,
                evidence=dict(hyp.evidence),
            )

            if h_copy.kind == "ACHROMATIC_SHADOW_VS_PIGMENT":
                if evt == "MOVING_CAST_SHADOW":
                    h_copy.status = "CONFIRMED"
                    h_copy.evidence["resolution"] = "CONFIRMED_CAST_SHADOW"
                    h_copy.evidence["texture_shift_under_luma_edge"] = (
                        temporal.texture_shift_under_luma_edge
                    )
                    h_copy.evidence["mean_delta_c"] = temporal.mean_delta_c
                elif evt == "INDEPENDENT_OBJECT_MOTION":
                    h_copy.status = "FALSIFIED"
                    h_copy.evidence["resolution"] = "MOVING_MATERIAL_SURFACE"

            elif h_copy.kind == "STATIC_DEPTH_ORDER_UNKNOWN":
                if (
                    evt == "INDEPENDENT_OBJECT_MOTION"
                    and temporal.occlusion_depth_verdict == "MOVING_REGION_IN_FRONT_OF_BACKGROUND"
                ):
                    h_copy.status = "CONFIRMED"
                    h_copy.evidence["occlusion_depth_verdict"] = temporal.occlusion_depth_verdict
                    h_copy.evidence["leading_occlusion_error"] = temporal.leading_occlusion_error
                    h_copy.evidence["interior_warp_error"] = temporal.interior_warp_error

            updated.append(h_copy)

        return updated

    def project_to_causal_field(
        self,
        obs: OpticalSceneObservation,
        temporal: Optional[TemporalVisualAnalysis] = None,
        verified_hypotheses: Optional[List[EpistemicHypothesis]] = None,
    ) -> CausalField:
        """
        Project hierarchical optical regions, coarse parent envelopes, and optical
        bond cleavage / healing causality into ELYSIA's CausalField.
        """
        field_obj = CausalField(dimensions=3)
        h, w = obs.shape_hw

        shadow_confirmed = any(
            hyp.kind == "ACHROMATIC_SHADOW_VS_PIGMENT" and hyp.status == "CONFIRMED"
            for hyp in (verified_hypotheses or [])
        ) or (temporal is not None and temporal.event_classification == "MOVING_CAST_SHADOW")

        # 1. Add coarse macro-envelope voxels (scale_level = 16.0)
        coarse_ids = sorted(int(c) for c in np.unique(obs.coarse_labels).tolist())
        for cid in coarse_ids:
            cmask = obs.coarse_labels == cid
            px_cnt = int(np.sum(cmask))
            if px_cnt == 0:
                continue
            ys, xs = np.where(cmask)
            cy_norm = float(np.mean(ys)) / max(1.0, float(h))
            cx_norm = float(np.mean(xs)) / max(1.0, float(w))
            v_coarse = InformationVoxel(
                id=f"coarse_{cid}",
                content={"type": "coarse_envelope", "coarse_id": cid, "pixel_count": px_cnt},
                tensor=np.array(
                    [
                        float(np.mean(obs.l1.luminance_a[cmask])),
                        float(np.mean(obs.l1.rg_opponent[cmask])),
                        float(np.mean(obs.l1.by_opponent[cmask])),
                        float(np.mean(obs.l2.fine_texture_map[cmask])),
                        1.0,
                        float(np.mean(obs.l2.persistence_map[cmask])),
                    ],
                    dtype=np.float32,
                ),
                mass=max(0.5, float(px_cnt / (h * w)) * 12.0),
                position=np.array([cx_norm, cy_norm, 1.0], dtype=np.float32),
                scale_level=16.0,
            )
            field_obj.add_voxel(v_coarse)

        # 2. Add coherent L2/L3 region voxels + scale-containment beams to parent coarse voxels
        for reg in obs.regions:
            cy_norm = reg.centroid_yx[0] / max(1.0, float(h))
            cx_norm = reg.centroid_yx[1] / max(1.0, float(w))
            sig_vec = np.array(
                [
                    reg.mean_a,
                    reg.mean_rg,
                    reg.mean_by,
                    reg.fine_texture_energy,
                    reg.compactness,
                    reg.boundary_persistence,
                ],
                dtype=np.float32,
            )
            v_reg = InformationVoxel(
                id=f"region_{reg.region_id}",
                content={
                    "type": "coherent_region",
                    "region_id": reg.region_id,
                    "parent_coarse_id": reg.parent_coarse_id,
                    "child_fine_count": reg.child_fine_count,
                },
                tensor=sig_vec,
                mass=max(0.1, reg.area_ratio * 10.0 * (1.0 + reg.boundary_persistence)),
                position=np.array([cx_norm, cy_norm, 0.3], dtype=np.float32),
                scale_level=max(1.0, float(reg.boundary_persistence)),
            )
            field_obj.add_voxel(v_reg)

            parent_vid = f"coarse_{reg.parent_coarse_id}"
            if parent_vid in field_obj.voxels:
                field_obj.link_voxels(v_reg.id, parent_vid, strength=3.0)

        # 3. Compute region overlap with temporal moving mask if independent motion occurred
        region_moving_overlap: Dict[int, float] = {}
        if (
            temporal is not None
            and temporal.event_classification == "INDEPENDENT_OBJECT_MOTION"
            and np.any(temporal.moving_mask)
        ):
            for reg in obs.regions:
                rmask = obs.region_labels == reg.region_id
                ov = float(np.mean(temporal.moving_mask[rmask])) if np.any(rmask) else 0.0
                region_moving_overlap[reg.region_id] = ov

        # 4. Add region-to-region connectivity beams honoring optical causality
        for edge in obs.boundaries:
            vid_a = f"region_{edge.region_a}"
            vid_b = f"region_{edge.region_b}"
            if vid_a not in field_obj.voxels or vid_b not in field_obj.voxels:
                continue

            is_motion_torn = False
            if region_moving_overlap:
                oa = region_moving_overlap.get(edge.region_a, 0.0)
                ob = region_moving_overlap.get(edge.region_b, 0.0)
                if (oa >= 0.30 and ob < 0.15) or (ob >= 0.30 and oa < 0.15):
                    is_motion_torn = True

            if is_motion_torn:
                strength = 0.01
                broken = True
            elif edge.optical_cause == "ILLUMINATION_SHADOW_CANDIDATE":
                strength = 2.5 if shadow_confirmed else 1.0
                broken = False
            elif edge.optical_cause == "MATERIAL_BOUNDARY":
                strength = 0.15
                broken = False
            else:
                strength = 1.2
                broken = False

            field_obj.link_voxels(vid_a, vid_b, strength=strength)
            beam = field_obj.beams[-1]
            beam.current_tension = edge.mean_bond_stress
            beam.break_threshold = max(0.05, edge.mean_bond_yield)
            beam.is_broken = broken

        # 5. Circular motion-induced cleavage for camouflaged foregrounds whose static edge held
        if (
            temporal is not None
            and temporal.event_classification == "INDEPENDENT_OBJECT_MOTION"
            and np.any(temporal.moving_mask)
            and not any(b.is_broken for b in field_obj.beams)
        ):
            m_mask = temporal.moving_mask
            ys_m, xs_m = np.where(m_mask)
            cy_m = float(np.mean(ys_m)) / max(1.0, float(h))
            cx_m = float(np.mean(xs_m)) / max(1.0, float(w))
            bg_mask = ~m_mask
            # Dominant enclosing background region in Pass 1
            bg_candidates = [obs.region_labels[m_mask]]
            parent_reg_id = int(np.bincount(obs.region_labels[m_mask].ravel()).argmax())
            parent_coarse = int(np.bincount(obs.coarse_labels[m_mask].ravel()).argmax())
            fg_area_ratio = float(np.sum(m_mask) / max(1.0, float(h * w)))

            v_fg = InformationVoxel(
                id="region_motion_fg",
                content={
                    "type": "motion_cleaved_region",
                    "pixel_count": int(np.sum(m_mask)),
                    "parent_coarse_id": parent_coarse,
                },
                tensor=np.array(
                    [
                        float(np.mean(obs.l1.luminance_a[m_mask])),
                        float(np.mean(obs.l1.rg_opponent[m_mask])),
                        float(np.mean(obs.l1.by_opponent[m_mask])),
                        float(np.mean(obs.l2.fine_texture_map[m_mask])),
                        0.75,
                        float(np.mean(obs.l2.persistence_map[m_mask])),
                    ],
                    dtype=np.float32,
                ),
                mass=max(0.1, fg_area_ratio * 10.0),
                position=np.array([cx_m, cy_m, 0.3], dtype=np.float32),
                scale_level=2.0,
            )
            field_obj.add_voxel(v_fg)
            if f"coarse_{parent_coarse}" in field_obj.voxels:
                field_obj.link_voxels(v_fg.id, f"coarse_{parent_coarse}", strength=3.0)
            if f"region_{parent_reg_id}" in field_obj.voxels:
                field_obj.link_voxels(v_fg.id, f"region_{parent_reg_id}", strength=0.01)
                torn_beam = field_obj.beams[-1]
                torn_beam.current_tension = float(np.mean(temporal.residual_flow_mag[m_mask]))
                torn_beam.break_threshold = 1.0
                torn_beam.is_broken = True

        # Advance CausalField continuously with physical dynamics
        field_obj.step(dt=0.05)
        return field_obj
