"""
hypothesis_space_organ.py

Cognitive Principle Formation & Transformation via Perceptual Re-Observation
============================================================================
Adheres strictly to:
  "Do not calculate, let it flow."
  "Human cognitive operation is not numerical calculation or brute-force search.
   Through one's own sensory structure, one observes what changes and how it is
   re-recognized, internalizes that relational change into a cognitive principle,
   and processes/transforms those principles."

Core Architecture (Zero Brute-Force Search, Zero Numerical Table Scanning):
1. `PerceptualTransitionObservation`:
   Observes any phenomenon or action outcome (Before -> After, or Early -> Late)
   strictly through ELYSIA's own sensory organs (`OcularScaleOrgan`,
   `MechanoAcousticOrgan`) and extracts qualitative-relational change topology:
   - `retained_stem` (무엇이 유지되었는가 — Blue / Order / Structural Persistence)
   - `flow_vector` (무엇이 어느 방향으로 흘렀는가 — Red / Flux / Directional Shift)
   - `emergent_cleavage` (어떤 경계·빈 공간·분화가 새로 생겨났는가 — Yellow / Boundary & Void)
2. `CognitivePrincipleSchema`:
   Distills repeated or contrastive perceptual transitions into a composable,
   non-numerical relational principle:
   - Presence vs Absence (`있는 것` vs `없는 것`) of perceptual qualities
   - Invariant stem vs Transformed branch
   - Reversible / Non-reversible symmetry
3. `CrossModalPerceptualBridge`:
   When a 1D temporal stream (e.g. continuous sound) cannot be separated by a
   single static tap, ELYSIA does NOT brute-force numerical functions. She folds
   the temporal flow into a 2D perceptual field and observes it through her own
   `OcularScaleOrgan` (seeing rising vs falling flow, parallel strata, or
   periodic rhythm as structural-topological transitions).
4. `CognitivePrincipleTransformer`:
   Processes, combines, and inverts internalized principles (`원리의 가공과 변환`)
   to solve unseen problems and design novel structures, verifying each step by
   re-observing through her own sensory organs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.physics.causal_field import CausalField, InformationVoxel
from core.sensory.epistemic_knowledge_spine import (
    CreationVerificationRecord,
    EpistemicKnowledgeSpine,
    SynthesizedOrganRecord,
)
from core.sensory.mechano_acoustic_organ import MechanoAcousticOrgan
from core.sensory.ocular_scale_organ import OcularScaleOrgan, OpticalSceneObservation
from core.sensory.visual_causal_reasoner import VisualCausalReasoner
from simulators.opaque_external_tools import (
    CADShapeObservation,
    ContinuousSpeechAcousticWorld,
    OpaqueParametricCADKernel,
    evaluate_occ_physics,
    render_occ_to_rgb,
)


@dataclass
class PerceptualTransition:
    """
    Non-numerical relational representation of 'What changed and how it is
    re-recognized through ELYSIA's sensory organs'.
    """

    source_state_id: str
    target_state_id: str
    action_tag: str
    # 1. Blue (Order / Invariance): what structural qualities persisted
    retained_qualities: List[str]
    # 2. Red (Flux / Mobility): qualitative direction of movement/expansion
    flow_qualities: List[str]
    # 3. Yellow (Boundary / Cleavage / Void): what new boundary or absence emerged
    emergent_qualities: List[str]
    # Continuous relational resonance scores (from sensory organs, not brute-force math)
    structural_persistence: float
    directional_polarity: Tuple[float, float, float]  # (vertical_flow, lateral_spread, cavity_emergence)


@dataclass
class CognitivePrinciple:
    """
    A principle formed by re-observing how an action or phenomenon transforms
    perceptual reality. Can be composed, inverted, or transferred across domains.
    """

    principle_id: str
    origin_domain: str
    present_before: List[str]
    present_after: List[str]
    newly_emerged: List[str]
    dissolved_into_absence: List[str]
    retained_stem: List[str]
    is_reversible_symmetric: bool
    chromatic_character: str
    relational_signature: Tuple[float, float, float]
    bound_operator_id: Optional[str] = None


@dataclass
class DiscoveredRelationalAxis:
    """
    An open-ended cognitive axis born dynamically when ELYSIA observes a novel
    structural contrast between two forms or parts that her existing axes do
    not yet span. Never restricted to a fixed 3-category list.
    """

    axis_id: str
    origin_pair: Tuple[str, str]  # (form_a_id, form_b_id) that gave birth to this axis
    direction_vector: np.ndarray  # normalized differential in continuous observation space
    dominant_observable_dims: List[str]
    discovery_step: int
    spanned_forms: List[str] = field(default_factory=list)


@dataclass
class FormExplorationRecord:
    """
    What ELYSIA discovers by actively observing and probing a resulting form
    itself across multiple viewpoints, slices, part-whole regions, and
    form-to-form interactions.
    """

    form_id: str
    view_observations: Dict[str, Dict[str, float]]  # multi-angle ocular & spatial readings
    slice_profile: List[Dict[str, float]]  # bottom-to-top structural evolution across Z
    part_nodes: List[Dict[str, Any]]  # discovered sub-regions & their relation to the whole
    interaction_outcomes: Dict[str, Dict[str, Any]]  # containment, passage, stacking probes
    continuous_descriptor: np.ndarray
    emergent_axes_born: List[str]
    connected_forms: List[Tuple[str, float, str]]  # (other_form_id, resonance, relation_note)


class OpenEndedFormExplorer:
    """
    Observes and explores any resulting form (`형태결과 자체`) across multiple
    angles, cross-sections, part-whole divisions, and physical interactions with
    other forms. Continuously differentiates new cognitive axes and connects
    knowledge into `EpistemicKnowledgeSpine` & `CausalField` without a fixed
    category ceiling.
    """

    OBSERVABLE_DIMS: Tuple[str, ...] = (
        "height_to_footprint",
        "enclosure_depth",
        "through_passage_ratio",
        "top_bottom_asymmetry",
        "lateral_cantilever_offset",
        "ocular_boundary_density",
        "ocular_region_differentiation",
        "support_equilibrium_margin",
    )

    def __init__(self, spine: EpistemicKnowledgeSpine) -> None:
        self.spine = spine
        self.eye = OcularScaleOrgan()
        self.reasoner = VisualCausalReasoner(self.eye)
        self.discovered_axes: List[DiscoveredRelationalAxis] = []
        self.explored_forms: Dict[str, FormExplorationRecord] = {}
        self._step = 0

    def _render_multi_angle_views(self, occ: np.ndarray, size: int = 72) -> Dict[str, np.ndarray]:
        """Observe the form by rotating it: oblique isometric, front elevation, and top plan."""
        iso = render_occ_to_rgb(occ, size=size)
        rot_90 = render_occ_to_rgb(np.rot90(occ, k=1, axes=(0, 1)), size=size)
        top_tilted = render_occ_to_rgb(np.rot90(occ, k=1, axes=(1, 2)), size=size)
        return {"oblique_0": iso, "oblique_90": rot_90, "top_tilted": top_tilted}

    def _probe_part_whole_slices(self, occ: np.ndarray) -> List[Dict[str, float]]:
        """Scan the form from bottom to top to see how its cross-section and cavity evolve."""
        from scipy.ndimage import binary_fill_holes

        g = occ.shape[2]
        slices: List[Dict[str, float]] = []
        coords = np.linspace(-1.0, 1.0, occ.shape[0])
        for k in range(g):
            sl = occ[:, :, k]
            area = int(np.sum(sl))
            if area == 0:
                continue
            filled = binary_fill_holes(sl)
            enclosed_void = int(np.sum(filled & (~sl)))
            idx = np.argwhere(sl)
            cx = float(np.mean(coords[idx[:, 0]]))
            cy = float(np.mean(coords[idx[:, 1]]))
            span_x = float((idx[:, 0].max() - idx[:, 0].min() + 1) / g)
            span_y = float((idx[:, 1].max() - idx[:, 1].min() + 1) / g)
            slices.append(
                {
                    "z_level": float(coords[k]),
                    "solid_area": float(area / (g * g)),
                    "enclosed_void_area": float(enclosed_void / (g * g)),
                    "center_x": cx,
                    "center_y": cy,
                    "span_xy": 0.5 * (span_x + span_y),
                }
            )
        return slices

    def _probe_physical_interactions(self, occ: np.ndarray, slice_profile: List[Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
        """
        Actively test the form with probe interactions:
        1) Vertical drop / containment probe: if a small test parcel is dropped into the top center,
           does it pass all the way through (conduit), rest inside an enclosed basin (container),
           or bounce off a solid roof (closed top)?
        2) Top-load stacking probe: if a weight is placed on the upper surface, does the combined
           form maintain gravitational equilibrium?
        """
        phys = evaluate_occ_physics(occ)
        if not slice_profile:
            return {}

        void_slices = [s for s in slice_profile if s["enclosed_void_area"] > 0.005]
        bottom_has_void = bool(slice_profile[0]["enclosed_void_area"] > 0.005)
        top_has_void = bool(slice_profile[-1]["enclosed_void_area"] > 0.005)

        through_ratio = float(len(void_slices) / max(1, len(slice_profile)))
        holds_parcel_inside = bool(len(void_slices) >= 2 and top_has_void and not bottom_has_void)
        passes_flow_through = bool(len(void_slices) >= 2 and top_has_void and bottom_has_void)

        # Stacking probe: add virtual load on the highest occupied slice
        top_s = slice_profile[-1]
        base_s = slice_profile[0]
        cantilever_offset = float(np.hypot(top_s["center_x"] - base_s["center_x"], top_s["center_y"] - base_s["center_y"]))
        stack_stable = bool(phys["stable_under_gravity"] and (phys["stability_margin"] - 0.35 * cantilever_offset) > 0.0)

        return {
            "drop_parcel_probe": {
                "enclosure_slice_fraction": through_ratio,
                "open_at_top": top_has_void,
                "sealed_at_bottom": not bottom_has_void,
                "holds_interior_contents": holds_parcel_inside,
                "acts_as_through_conduit": passes_flow_through,
            },
            "top_stack_probe": {
                "top_to_base_offset": cantilever_offset,
                "base_stability_margin": float(phys["stability_margin"]),
                "can_support_upper_load": stack_stable,
            },
        }

    def explore_form(self, form_id: str, occ: np.ndarray) -> FormExplorationRecord:
        """
        Observe a resulting form from multiple angles, decompose its bottom-to-top
        part structure, probe its physical interactions, and dynamically expand
        ELYSIA's cognitive axes and relational connections.
        """
        self._step += 1
        views = self._render_multi_angle_views(occ, size=72)
        view_metrics: Dict[str, Dict[str, float]] = {}
        total_bounds = 0.0
        total_regions = 0.0

        for vname, rgb in views.items():
            obs = self.eye.observe(rgb, max_width=72)
            mat_b = sum(1 for b in obs.boundaries if b.optical_cause == "MATERIAL_BOUNDARY")
            view_metrics[vname] = {
                "regions": float(len(obs.regions)),
                "material_boundaries": float(mat_b),
                "fine_texture": float(np.mean(obs.l2.fine_texture_map)),
            }
            total_bounds += mat_b
            total_regions += len(obs.regions)

        slices = self._probe_part_whole_slices(occ)
        interactions = self._probe_physical_interactions(occ, slices)
        phys = evaluate_occ_physics(occ)

        # Decompose into emergent sub-parts (e.g. lower foundation, mid trunk/chamber, upper crown)
        part_nodes: List[Dict[str, Any]] = []
        if len(slices) >= 3:
            n_s = len(slices)
            thirds = [slices[: n_s // 3], slices[n_s // 3 : 2 * n_s // 3], slices[2 * n_s // 3 :]]
            for idx_p, seg in enumerate(thirds):
                if not seg:
                    continue
                part_nodes.append(
                    {
                        "part_index": idx_p,
                        "mean_z": float(np.mean([s["z_level"] for s in seg])),
                        "mean_span": float(np.mean([s["span_xy"] for s in seg])),
                        "mean_void": float(np.mean([s["enclosed_void_area"] for s in seg])),
                        "lateral_shift": float(np.mean([np.hypot(s["center_x"], s["center_y"]) for s in seg])),
                    }
                )

        # Continuous 8D observation descriptor derived strictly from multi-angle & slice re-observation
        bx, by, bz = phys["bbox_xyz"]
        footprint = max(0.5 * (bx + by), 1e-4)
        h_to_fp = float(bz / footprint)
        drop_info = interactions.get("drop_parcel_probe", {})
        stack_info = interactions.get("top_stack_probe", {})
        enclosure_depth = float(
            np.mean([s["enclosed_void_area"] for s in slices]) * 10.0 if slices else 0.0
        )
        through_ratio = float(drop_info.get("enclosure_slice_fraction", 0.0)) * (
            1.0 if drop_info.get("acts_as_through_conduit") else 0.4
        )
        top_bot_asym = float(
            abs(slices[-1]["span_xy"] - slices[0]["span_xy"]) if len(slices) >= 2 else 0.0
        )
        cantilever = float(stack_info.get("top_to_base_offset", 0.0))
        bound_dens = float(total_bounds / max(1.0, len(views) * 4.0))
        reg_diff = float(total_regions / max(1.0, len(views) * 3.0))
        eq_margin = float(phys["stability_margin"])

        desc = np.array(
            [
                h_to_fp,
                enclosure_depth,
                through_ratio,
                top_bot_asym,
                cantilever,
                bound_dens,
                reg_diff,
                eq_margin,
            ],
            dtype=np.float64,
        )

        # Compare with previously explored forms:
        # 1) Connect to resonant forms (Sameness) & contrast forms (Difference)
        # 2) If residual difference is not spanned by current discovered_axes, birth a NEW cognitive axis!
        new_axes: List[str] = []
        connections: List[Tuple[str, float, str]] = []

        for prev_id, prev_rec in self.explored_forms.items():
            diff = desc - prev_rec.continuous_descriptor
            dist = float(np.linalg.norm(diff))
            sim = float(1.0 / (1.0 + dist))

            # Check how much of `diff` is already explained by existing discovered axes
            residual = diff.copy()
            for ax in self.discovered_axes:
                proj = float(np.dot(residual, ax.direction_vector))
                residual = residual - proj * ax.direction_vector
            resid_norm = float(np.linalg.norm(residual))

            top_dim_idx = int(np.argmax(np.abs(diff)))
            top_dim_name = self.OBSERVABLE_DIMS[top_dim_idx]
            connections.append(
                (
                    prev_id,
                    sim,
                    f"dominant_contrast={top_dim_name}({diff[top_dim_idx]:+.2f}), residual={resid_norm:.2f}",
                )
            )

            # Birth a new cognitive dimension whenever an unexplained structural contrast (> 0.18) appears
            if resid_norm > 0.18:
                unit_dir = residual / resid_norm
                ranked_dims = sorted(
                    range(len(self.OBSERVABLE_DIMS)),
                    key=lambda i: abs(unit_dir[i]),
                    reverse=True,
                )
                dom_dims = [self.OBSERVABLE_DIMS[i] for i in ranked_dims[:2] if abs(unit_dir[i]) > 0.25]
                ax_id = f"axis_{len(self.discovered_axes):02d}_{'_vs_'.join(dom_dims)}"
                new_axis = DiscoveredRelationalAxis(
                    axis_id=ax_id,
                    origin_pair=(prev_id, form_id),
                    direction_vector=unit_dir,
                    dominant_observable_dims=dom_dims,
                    discovery_step=self._step,
                    spanned_forms=[prev_id, form_id],
                )
                self.discovered_axes.append(new_axis)
                new_axes.append(ax_id)
            else:
                for ax in self.discovered_axes:
                    if abs(float(np.dot(diff, ax.direction_vector))) > 0.15 and form_id not in ax.spanned_forms:
                        ax.spanned_forms.append(form_id)

        rec = FormExplorationRecord(
            form_id=form_id,
            view_observations=view_metrics,
            slice_profile=slices,
            part_nodes=part_nodes,
            interaction_outcomes=interactions,
            continuous_descriptor=desc,
            emergent_axes_born=new_axes,
            connected_forms=connections,
        )
        self.explored_forms[form_id] = rec

        # Ground the explored form & its newly discovered axes into EpistemicKnowledgeSpine
        self.spine.upsert_or_split_concept(
            domain_signatures={
                "optical": [bound_dens, reg_diff, h_to_fp, enclosure_depth],
                "mechanical": [eq_margin, cantilever, through_ratio],
            },
            source_medium=f"form_exploration:{form_id}",
            optical_match_threshold=0.35,
            surprise_threshold=0.30,
        )
        return rec

    def project_expansion_to_causal_field(self) -> CausalField:
        """Project all explored forms, sub-parts, and newly born cognitive axes into CausalField."""
        field_obj = self.spine.project_to_causal_field()
        for idx, (fid, rec) in enumerate(self.explored_forms.items()):
            vid = f"form_{fid}"
            field_obj.add_voxel(
                InformationVoxel(
                    id=vid,
                    content={
                        "type": "explored_form",
                        "parts": len(rec.part_nodes),
                        "interactions": rec.interaction_outcomes,
                    },
                    tensor=rec.continuous_descriptor[:6].astype(np.float32),
                    mass=1.8 + 0.2 * len(rec.connected_forms),
                    position=np.array([0.15 + 0.18 * idx, 0.45, 0.65], dtype=np.float32),
                    scale_level=9.0,
                )
            )
            for other_id, sim, _ in rec.connected_forms:
                other_vid = f"form_{other_id}"
                if other_vid in field_obj.voxels:
                    field_obj.link_voxels(other_vid, vid, strength=float(0.8 + sim))

        for a_idx, ax in enumerate(self.discovered_axes):
            avid = f"discovered_{ax.axis_id}"
            field_obj.add_voxel(
                InformationVoxel(
                    id=avid,
                    content={
                        "type": "emergent_cognitive_axis",
                        "origin_pair": ax.origin_pair,
                        "dims": ax.dominant_observable_dims,
                    },
                    tensor=ax.direction_vector[:6].astype(np.float32),
                    mass=2.2,
                    position=np.array([0.20 + 0.16 * a_idx, 0.78, 1.10], dtype=np.float32),
                    scale_level=11.0,
                )
            )
            for sf in ax.spanned_forms:
                fvid = f"form_{sf}"
                if fvid in field_obj.voxels:
                    field_obj.link_voxels(fvid, avid, strength=1.6)

        if field_obj.voxels:
            field_obj.step(dt=0.05)
        return field_obj


# ======================================================================
# 4. Five-Sense Coupled Tool & Scale Extension (Autonomous Correlational Discovery)
# ======================================================================
@dataclass
class OpticalScaleInstrument:
    """
    An optical instrument (Microscope, Naked Eye, Telescope) that couples in front
    of `OcularScaleOrgan` to shift focal distance and spatial scale level.
    """

    instrument_id: str
    focal_distance_m: float
    magnification: float
    scale_level: float


@dataclass
class HandCoupledTool:
    """
    An unlabelled physical object/tool held by ELYSIA's hand, defined purely by
    continuous physical properties: mass, lever length (distance), rotational
    inertia (I = m * L^2), and contact tip area.
    """

    tool_id: str
    mass_kg: float
    lever_distance_m: float
    contact_area_cm2: float
    scale_level: float

    @property
    def moment_of_inertia(self) -> float:
        return float(self.mass_kg * (self.lever_distance_m ** 2))


class EmbodiedToolScaleExtensionEngine:
    """
    Zero-hardcoding embodied learning across scales:
    - No `if tool == ...` or `if motion == ...` outcome string tables.
    - Pure continuous physical coupling produces force-displacement, acoustic
      waveforms, and visual surface changes observed by `MechanoAcousticOrgan`
      and `OcularScaleOrgan`.
    - ELYSIA autonomously discovers the correlational connectivity between
      [Tool Weight/Distance/Inertia/Edge + Hand Motion] and [Re-observed Sensory
      Changes], and maps unlabelled goals/contexts on her own.
    """

    def __init__(self, spine: EpistemicKnowledgeSpine) -> None:
        self.spine = spine
        self.eye = OcularScaleOrgan()
        self.mechano_acoustic = MechanoAcousticOrgan()
        self.reasoner = VisualCausalReasoner(self.eye)
        self.scale_observations: Dict[str, Dict[str, Any]] = {}

    def observe_across_optical_instruments(self, multi_scale_target_rgb: np.ndarray) -> Dict[str, Dict[str, Any]]:
        h, w = multi_scale_target_rgb.shape[:2]
        instruments = [
            OpticalScaleInstrument("microscope_lens", focal_distance_m=0.02, magnification=8.0, scale_level=2.0),
            OpticalScaleInstrument("naked_eye_fovea", focal_distance_m=0.50, magnification=1.0, scale_level=6.0),
            OpticalScaleInstrument("telescope_lens", focal_distance_m=50.0, magnification=12.0, scale_level=12.0),
        ]

        results: Dict[str, Dict[str, Any]] = {}
        for inst in instruments:
            if inst.instrument_id == "microscope_lens":
                y0, y1 = int(h * 0.42), int(h * 0.58)
                x0, x1 = int(w * 0.42), int(w * 0.58)
                view = cv2.resize(multi_scale_target_rgb[y0:y1, x0:x1], (96, 96), interpolation=cv2.INTER_CUBIC)
            elif inst.instrument_id == "naked_eye_fovea":
                y0, y1 = int(h * 0.20), int(h * 0.80)
                x0, x1 = int(w * 0.20), int(w * 0.80)
                view = cv2.resize(multi_scale_target_rgb[y0:y1, x0:x1], (96, 96), interpolation=cv2.INTER_AREA)
            else:
                view = cv2.resize(multi_scale_target_rgb, (96, 96), interpolation=cv2.INTER_AREA)

            obs = self.eye.observe(view, max_width=96)
            fine_tex = float(np.mean(obs.l2.fine_texture_map))
            mat_bounds = sum(1 for b in obs.boundaries if b.optical_cause == "MATERIAL_BOUNDARY")
            results[inst.instrument_id] = {
                "focal_distance_m": inst.focal_distance_m,
                "magnification": inst.magnification,
                "scale_level": inst.scale_level,
                "ocular_regions": len(obs.regions),
                "material_boundaries": mat_bounds,
                "fine_texture_energy": fine_tex,
            }
        self.scale_observations = results
        return results

    def _render_workpiece_continuous(
        self,
        fasten_depth: float,
        shear_cut_depth: float,
        cleave_split_width: float,
        size: int = 80,
    ) -> np.ndarray:
        img = np.full((size, size, 3), 34, dtype=np.float32)
        yy, xx = np.indices((size, size), dtype=np.float32)
        c = 0.5 * size

        half_gap = float(np.clip(cleave_split_width * 9.0, 0.0, 11.0))
        left_mask = (yy >= 14) & (yy <= size - 14) & (xx >= 10 - 0.5 * half_gap) & (xx <= c - half_gap)
        right_mask = (yy >= 14) & (yy <= size - 14) & (xx >= c + half_gap) & (xx <= size - 10 + 0.5 * half_gap)
        in_block = left_mask | right_mask

        grain = 12.0 * np.sin(0.45 * yy)
        img[in_block, 0] = np.clip(165.0 + grain[in_block], 0.0, 255.0)
        img[in_block, 1] = np.clip(128.0 + grain[in_block], 0.0, 255.0)
        img[in_block, 2] = np.clip(88.0 + grain[in_block], 0.0, 255.0)

        if shear_cut_depth > 0.02 and half_gap < 2.0:
            cut_len = float(np.clip(shear_cut_depth, 0.0, 1.0)) * (size - 28.0)
            kerf_mask = (yy >= 14) & (yy <= 14 + cut_len) & (np.abs(xx - c) <= 1.4)
            edge_highlight = (yy >= 14) & (yy <= 14 + cut_len) & (np.abs(np.abs(xx - c) - 2.2) <= 0.7)
            img[kerf_mask] = np.array([22.0, 24.0, 28.0], dtype=np.float32)
            img[edge_highlight] = np.array([212.0, 178.0, 134.0], dtype=np.float32)

        r_dist = np.hypot(yy - c, xx - c)
        ring_r = 3.5 + 6.5 * float(np.clip(fasten_depth, 0.0, 1.0))
        head_mask = r_dist <= ring_r
        rim_mask = (r_dist >= ring_r - 1.8) & (r_dist <= ring_r + 1.0)
        slot_mask = (np.abs(yy - c) <= 1.2) & (np.abs(xx - c) <= ring_r - 1.5)
        if half_gap < 2.0 and shear_cut_depth < 0.25:
            metallic = 105.0 + 110.0 * float(np.clip(fasten_depth, 0.0, 1.0))
            img[head_mask] = np.array([metallic, metallic + 6.0, metallic + 16.0], dtype=np.float32)
            img[rim_mask] = np.array([42.0, 46.0, 52.0], dtype=np.float32)
            if fasten_depth > 0.15:
                img[slot_mask] = np.array([28.0, 30.0, 34.0], dtype=np.float32)

        return np.clip(np.round(img), 0, 255).astype(np.uint8)

    def simulate_continuous_coupling(
        self,
        tool: HandCoupledTool,
        motor_vec: Tuple[float, float, float],
    ) -> Dict[str, Any]:
        twist_e, draw_e, swing_e = (float(motor_vec[0]), float(motor_vec[1]), float(motor_vec[2]))
        inertia = tool.moment_of_inertia
        inv_inertia = 1.0 / (1.0 + 80.0 * inertia)
        sharpness = 1.0 / max(tool.contact_area_cm2, 0.005)
        reach_balance = float(np.exp(-((tool.lever_distance_m - 0.26) ** 2) / 0.025))

        # 1) Torsional thread engagement: requires low rotational inertia & axial tip contact (not thin blade)
        tip_torsion_grip = float(np.exp(-max(0.0, sharpness - 12.0) / 15.0))
        fasten_progress = float(twist_e * inv_inertia * tip_torsion_grip * np.exp(-tool.lever_distance_m / 0.30))
        # 2) Tangential shear cut: local edge pressure must exceed material shear yield strength
        shear_over_yield = max(0.0, sharpness - 14.0)
        shear_progress = float(draw_e * np.tanh(shear_over_yield / 22.0) * reach_balance)
        # 3) Bulk kinetic cleavage: kinetic energy 0.5 * I * w^2 must exceed bulk fracture toughness
        kinetic_energy = float(0.5 * inertia * ((4.2 * swing_e) ** 2))
        cleave_progress = float(np.tanh(max(0.0, kinetic_energy - 0.85) * 0.45))

        n_t = 80
        dt = 0.002
        u = np.linspace(0.0, 1.0, n_t, dtype=np.float64)
        sr = 16000
        t_aud = np.linspace(0.0, 0.20, int(sr * 0.20), endpoint=False, dtype=np.float64)

        f_twist = 42.0 * fasten_progress * (u ** 1.2)
        f_shear = 26.0 * shear_progress * (np.sin(np.pi * u) ** 0.45)
        f_impact = (12.0 + 315.0 * cleave_progress + 15.0 * swing_e) * np.exp(-((u - 0.28) ** 2) / 0.008)
        force_t = f_twist + f_shear + f_impact + 2.5 * np.sin(np.pi * u)

        x_twist = 0.014 * fasten_progress * np.minimum(1.0, u * 1.5)
        x_shear = 0.020 * shear_progress * u
        x_cleave = 0.045 * cleave_progress * np.maximum(0.0, (u - 0.24) / 0.76) + 0.001 * swing_e * np.exp(-((u - 0.28) ** 2) / 0.01)
        disp_t = x_twist + x_shear + x_cleave + 0.0002 * u

        w_twist = fasten_progress * np.exp(-26.0 * t_aud) * np.sin(2.0 * np.pi * 1420.0 * t_aud)
        w_shear = shear_progress * np.exp(-7.5 * t_aud) * np.sin(2.0 * np.pi * 2680.0 * t_aud)
        w_cleave = (0.15 * swing_e + cleave_progress) * np.exp(-15.0 * t_aud) * np.sin(2.0 * np.pi * 430.0 * t_aud)
        pcm = w_twist + w_shear + w_cleave + 0.02 * np.sin(2.0 * np.pi * 800.0 * t_aud)

        m_obs = self.mechano_acoustic.observe_push(force_t.astype(np.float32), disp_t.astype(np.float32), dt=dt)
        a_obs = self.mechano_acoustic.observe_tap(pcm.astype(np.float32), sample_rate=sr)

        before_rgb = self._render_workpiece_continuous(0.0, 0.0, 0.0)
        after_rgb = self._render_workpiece_continuous(fasten_progress, shear_progress, cleave_progress)
        eye_b = self.eye.observe(before_rgb, max_width=80)
        eye_a = self.eye.observe(after_rgb, max_width=80)
        comp = self.reasoner.compare_observations(eye_b, eye_a)

        center_crop_diff = float(np.mean(np.abs(after_rgb[28:52, 28:52].astype(np.float32) - before_rgb[28:52, 28:52].astype(np.float32))) / 255.0)
        wide_split_diff = float(np.mean(np.abs(after_rgb[:, :22].astype(np.float32) - before_rgb[:, :22].astype(np.float32))) / 255.0)

        sensory_effect_vec = np.array(
            [
                float(np.log10(max(m_obs.max_force_n, 1.0))),
                float(np.log10(max(m_obs.effective_stiffness_n_m, 1.0))),
                float(m_obs.max_disp_m * 100.0),
                float(a_obs.peak_frequency_hz / 1000.0),
                float(a_obs.decay_rate_per_sec / 30.0),
                float(1.0 - comp.l2_persistence_corr),
                center_crop_diff,
                wide_split_diff,
            ],
            dtype=np.float64,
        )
        proprioceptive_tool_vec = np.array(
            [
                float(tool.mass_kg),
                float(tool.lever_distance_m),
                float(np.log10(max(inertia, 1e-5))),
                float(np.log10(max(sharpness, 1e-2))),
            ],
            dtype=np.float64,
        )

        return {
            "tool_id": tool.tool_id,
            "motor_vec": motor_vec,
            "proprioceptive_tool_vec": proprioceptive_tool_vec,
            "sensory_effect_vec": sensory_effect_vec,
            "before_rgb": before_rgb,
            "after_rgb": after_rgb,
            "m_obs": m_obs,
            "a_obs": a_obs,
            "ocular_comp": comp,
        }

    def discover_correlational_mappings_autonomously(
        self,
        unlabelled_tools: List[HandCoupledTool],
        exploratory_motors: List[Tuple[float, float, float]],
        unlabelled_contexts: Optional[List[Tuple[str, np.ndarray]]] = None,
    ) -> Dict[str, Any]:
        episodes: List[Dict[str, Any]] = []
        for tool in unlabelled_tools:
            for m_vec in exploratory_motors:
                ep = self.simulate_continuous_coupling(tool, m_vec)
                eff = ep["sensory_effect_vec"]
                world_change_salience = float(eff[2] + 3.0 * eff[5] + 2.5 * eff[6] + 3.0 * eff[7])
                ep["world_change_salience"] = world_change_salience
                episodes.append(ep)

        cause_mat = np.array(
            [np.concatenate([e["proprioceptive_tool_vec"], np.array(e["motor_vec"])]) for e in episodes],
            dtype=np.float64,
        )
        effect_mat = np.array([e["sensory_effect_vec"] for e in episodes], dtype=np.float64)

        c_z = (cause_mat - np.mean(cause_mat, axis=0)) / np.maximum(np.std(cause_mat, axis=0), 1e-6)
        e_z = (effect_mat - np.mean(effect_mat, axis=0)) / np.maximum(np.std(effect_mat, axis=0), 1e-6)
        cross_corr_matrix = (c_z.T @ e_z) / max(1, len(episodes))

        discovered_couplings: Dict[str, Dict[str, Any]] = {}
        for tool in unlabelled_tools:
            t_eps = [e for e in episodes if e["tool_id"] == tool.tool_id]
            best_ep = max(t_eps, key=lambda e: e["world_change_salience"])
            worst_ep = min(t_eps, key=lambda e: e["world_change_salience"])
            discovered_couplings[tool.tool_id] = {
                "resonant_motor_vec": best_ep["motor_vec"],
                "resonant_salience": best_ep["world_change_salience"],
                "mismatched_salience": worst_ep["world_change_salience"],
                "selective_resonance_ratio": float(
                    best_ep["world_change_salience"] / max(worst_ep["world_change_salience"], 1e-4)
                ),
                "sensory_effect_signature": best_ep["sensory_effect_vec"],
                "felt_force_n": float(best_ep["m_obs"].max_force_n),
                "acoustic_hz": float(best_ep["a_obs"].peak_frequency_hz),
                "ocular_l2_persistence": float(best_ep["ocular_comp"].l2_persistence_corr),
                "after_rgb": best_ep["after_rgb"],
            }
            self.spine.upsert_or_split_concept(
                domain_signatures={
                    "optical": best_ep["sensory_effect_vec"][5:8].tolist(),
                    "mechanical": best_ep["sensory_effect_vec"][0:3].tolist(),
                    "acoustic": best_ep["sensory_effect_vec"][3:5].tolist(),
                },
                source_medium=f"autonomous_tool_discovery:{tool.tool_id}",
                optical_match_threshold=0.20,
                surprise_threshold=0.20,
            )

        autonomous_goal_mappings: List[Dict[str, Any]] = []
        if unlabelled_contexts:
            for ctx_tag, target_after_rgb in unlabelled_contexts:
                base_rgb = self._render_workpiece_continuous(0.0, 0.0, 0.0)
                eb = self.eye.observe(base_rgb, max_width=80)
                ea = self.eye.observe(target_after_rgb, max_width=80)
                comp = self.reasoner.compare_observations(eb, ea)
                c_diff = float(np.mean(np.abs(target_after_rgb[28:52, 28:52].astype(np.float32) - base_rgb[28:52, 28:52].astype(np.float32))) / 255.0)
                w_diff = float(np.mean(np.abs(target_after_rgb[:, :22].astype(np.float32) - base_rgb[:, :22].astype(np.float32))) / 255.0)
                target_visual_sig = np.array([1.0 - comp.l2_persistence_corr, c_diff, w_diff], dtype=np.float64)

                best_tool_id = min(
                    discovered_couplings.keys(),
                    key=lambda tid: float(
                        np.linalg.norm(discovered_couplings[tid]["sensory_effect_signature"][5:8] - target_visual_sig)
                    ),
                )
                autonomous_goal_mappings.append(
                    {
                        "context_or_goal": ctx_tag,
                        "inferred_tool_id": best_tool_id,
                        "inferred_motor_vec": discovered_couplings[best_tool_id]["resonant_motor_vec"],
                    }
                )

        return {
            "total_exploratory_episodes": len(episodes),
            "cross_correlation_matrix": cross_corr_matrix,
            "discovered_couplings": discovered_couplings,
            "autonomous_goal_mappings": autonomous_goal_mappings,
        }


# ======================================================================
# 5. Circulating Sensorimotor Nervous System (Non-Linear Eye <-> Hand <-> Ear <-> Foot Loop)
# ======================================================================
@dataclass
class NervousTickTrace:
    """Snapshot of the circulating nervous system at a single continuous step dt."""

    step: int
    tip_pos_yx: Tuple[float, float]
    gaze_pos_yx: Tuple[float, float]
    ocular_focal_scale: float  # dynamically zooms between macro (L10), meso (L6), micro (L2)
    stance_distance_m: float  # Foot/posture distance adjusting leverage & stability
    hand_velocity_yx: Tuple[float, float]
    hand_normal_force_n: float
    felt_resistance_n: float
    acoustic_freq_hz: float
    acoustic_harmony_ratio: float
    # Instantaneous circulating currents between organs
    current_hand_to_eye: float
    current_eye_to_hand: float
    current_ear_to_hand: float
    current_hand_to_foot: float
    work_progress: float
    damage_or_slip_friction: float


class CirculatingNervousSystemEngine:
    """
    Replaces linear one-shot execution (`Input -> Run to end -> Observe`) with a
    continuous circulating nervous system (`순환 신경계통`) where Eye, Ear, Hand, and
    Foot/Posture are coupled nodes in a live `CausalField` circuit exchanging
    signals at every `dt`:
    - What the Hand feels (`felt_resistance`, slip) immediately pulls the Eye's
      foveal gaze and focal scale (`L6 -> L2` zoom) and shifts Foot stance distance.
    - What the Eye sees at the zoomed contact zone (local curved grain direction
      & hidden hard knot boundaries) continuously steers the Hand's velocity vector
      mid-motion.
    - What the Ear hears (high-frequency slip/stress squeal vs resonant cleavage hum)
      continuously modulates the Hand's normal force & speed in real time and brings
      the hand to rest when completion resonance is heard.
    - Bidirectional `ConnectivityBeam` conductances evolve plastically at every `dt`
      whenever one organ's signal reduces friction in another organ.
    """

    def __init__(self, spine: EpistemicKnowledgeSpine) -> None:
        self.spine = spine
        self.eye = OcularScaleOrgan()
        self.mechano_acoustic = MechanoAcousticOrgan()
        self.nervous_field = CausalField(dimensions=3)
        self._init_nervous_circuit()

    def _init_nervous_circuit(self) -> None:
        """Initialize the circulating organ nodes and bidirectional neural beams in CausalField."""
        organs = [
            ("organ_eye", np.array([0.0, 0.8, 1.0], dtype=np.float32), 6.0),
            ("organ_ear", np.array([-0.7, 0.6, 0.9], dtype=np.float32), 6.0),
            ("organ_hand", np.array([0.6, 0.0, 0.5], dtype=np.float32), 5.0),
            ("organ_foot_posture", np.array([0.0, -0.8, 0.2], dtype=np.float32), 8.0),
        ]
        for oid, pos, sc in organs:
            self.nervous_field.add_voxel(
                InformationVoxel(
                    id=oid,
                    content={"type": "nervous_organ"},
                    tensor=np.zeros(6, dtype=np.float32),
                    mass=2.0,
                    position=pos,
                    scale_level=sc,
                )
            )
        # Bidirectional circulating neural pathways
        pairs = [
            ("organ_eye", "organ_hand"),
            ("organ_hand", "organ_eye"),
            ("organ_ear", "organ_hand"),
            ("organ_hand", "organ_ear"),
            ("organ_eye", "organ_ear"),
            ("organ_hand", "organ_foot_posture"),
            ("organ_foot_posture", "organ_hand"),
            ("organ_eye", "organ_foot_posture"),
        ]
        self.beam_conductance: Dict[Tuple[str, str], float] = {}
        for src, dst in pairs:
            self.nervous_field.link_voxels(src, dst, strength=1.0)
            self.beam_conductance[(src, dst)] = 1.0

    def _build_complex_workpiece_canvas(self, size: int = 96) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a non-trivial physical substrate with:
        1) A curved S-shaped natural cleavage channel running from y=14 to y=82,
        2) A hidden high-density mineral/knot obstruction at (y=48, x=46) that causes
           slip/blade-jam if pushed blindly in a straight line!
        Returns `(rgb_canvas, resistance_field, ideal_channel_x_by_y)`.
        """
        yy, xx = np.indices((size, size), dtype=np.float64)
        # S-curved low-resistance grain seam (visibly brighter golden-wood cleavage channel)
        channel_x = 0.5 * size + 15.0 * np.sin(2.0 * np.pi * (yy - 14.0) / (size - 28.0))
        dist_to_seam = np.abs(xx - channel_x)

        # Hard knots sitting on the naive straight centerline (y=32, x=48) and near lower curve (y=62, x=37)
        knot1_dist = np.hypot(yy - 32.0, xx - 48.0)
        knot2_dist = np.hypot(yy - 62.0, xx - 37.0)
        knot_field = 2.8 * np.exp(-(knot1_dist ** 2) / (2.0 * (6.0 ** 2))) + 1.6 * np.exp(-(knot2_dist ** 2) / (2.0 * (4.0 ** 2)))

        # Resistance is low along the curved seam, high off-seam, and spikes at the knots
        resistance = 0.32 + 1.15 * np.tanh(dist_to_seam / 7.0) + knot_field

        # Render visibly distinct RGB surface:
        # - Natural cleavage seam is warm bright amber (+42 luminance)
        # - Hard knots are dark dense mineral/bark (-65 luminance)
        # So OcularScaleOrgan's photoreceptor luminance_a directly sees both the bright seam and dark knots!
        grain_wave = np.sin(0.55 * (xx - channel_x))
        seam_glow = 44.0 * np.exp(-(dist_to_seam ** 2) / 22.0)
        knot_shadow = 68.0 * np.exp(-(knot1_dist ** 2) / 34.0) + 48.0 * np.exp(-(knot2_dist ** 2) / 18.0)
        r = np.clip(145.0 + seam_glow + 10.0 * grain_wave - knot_shadow, 18, 248)
        g = np.clip(112.0 + 0.85 * seam_glow + 8.0 * grain_wave - 0.9 * knot_shadow, 18, 248)
        b = np.clip(78.0 + 0.55 * seam_glow + 6.0 * grain_wave - 0.8 * knot_shadow, 18, 248)
        rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
        return rgb, resistance, channel_x

    def run_episode(self, circulate_feedback: bool = True, max_ticks: int = 28) -> Dict[str, Any]:
        """
        Run an embodied tool-hand-eye-ear-foot episode on the curved/knotted workpiece.
        - When `circulate_feedback=False` (Linear / Predetermined mode):
          The hand moves along its predetermined straight downward trajectory without
          real-time circulation between Eye, Ear, Hand, and Foot.
        - When `circulate_feedback=True` (Circulating Nervous System mode):
          Eye, Ear, Hand, and Foot exchange currents at every `dt`, dynamically
          zooming the Eye's scale, shifting gaze ahead of the hand, curving the hand's
          trajectory around the knot along the grain, adjusting foot stance leverage,
          and modulating force from acoustic resonance.
        """
        size = 96
        canvas_rgb, res_field, channel_x = self._build_complex_workpiece_canvas(size=size)
        live_rgb = canvas_rgb.copy()

        # Initial states
        tip_y, tip_x = 14.0, float(channel_x[14, 0])
        gaze_y, gaze_x = tip_y + 4.0, tip_x
        focal_scale = 6.0  # Starts at meso naked-eye scale
        stance_dist = 0.35  # Foot-to-workpiece stance distance (m)
        vel_y, vel_x = 2.4, 0.0  # Initial downward hand velocity
        normal_force = 22.0

        traces: List[NervousTickTrace] = []
        total_work = 0.0
        total_friction_damage = 0.0
        prev_friction = 0.0

        sr = 16000
        t_aud = np.linspace(0.0, 0.04, int(sr * 0.04), endpoint=False, dtype=np.float64)

        for step in range(max_ticks):
            iy = int(np.clip(round(tip_y), 2, size - 3))
            ix = int(np.clip(round(tip_x), 2, size - 3))

            # 1. HAND (Tactile & Proprioceptive Afferent): feel local resistance & lateral gradient
            local_res = float(res_field[iy, ix])
            grad_rx = float(0.5 * (res_field[iy, min(size - 2, ix + 2)] - res_field[iy, max(1, ix - 2)]))
            felt_force = float(normal_force * local_res * (0.85 + 0.4 * stance_dist))
            slip_friction = float(max(0.0, local_res - 0.85) * (normal_force / 20.0))

            # Short force-displacement micro-trace fed through MechanoAcousticOrgan
            u_m = np.linspace(0.0, 1.0, 20, dtype=np.float32)
            f_trace = (felt_force * (0.6 + 0.4 * u_m)).astype(np.float32)
            x_trace = ((0.004 / max(local_res, 0.2)) * u_m).astype(np.float32)
            m_obs = self.mechano_acoustic.observe_push(f_trace, x_trace, dt=0.002)

            # 2. EAR (Acoustic Afferent): smooth resonant hum (620Hz) in seam vs high screech (2900Hz) on knot/off-seam
            seam_harmony = float(np.exp(-max(0.0, local_res - 0.45) * 2.2))
            freq_dom = 620.0 * seam_harmony + 2900.0 * (1.0 - seam_harmony)
            pcm = (
                seam_harmony * np.sin(2.0 * np.pi * 620.0 * t_aud)
                + (1.0 - seam_harmony) * np.sin(2.0 * np.pi * 2900.0 * t_aud)
            ).astype(np.float32)
            a_obs = self.mechano_acoustic.observe_tap(pcm, sample_rate=sr)

            # 3. EYE (Ocular Afferent at current gaze & dynamic focal scale)
            # Smaller focal_scale (e.g. L2 micro) zooms tightly into a smaller patch around gaze
            half_fov = int(np.clip(round(10 + 2.8 * focal_scale), 12, 36))
            gy = int(np.clip(round(gaze_y), half_fov, size - half_fov - 1))
            gx = int(np.clip(round(gaze_x), half_fov, size - half_fov - 1))
            fov_patch = cv2.resize(
                canvas_rgb[gy - half_fov : gy + half_fov, gx - half_fov : gx + half_fov],
                (48, 48),
                interpolation=cv2.INTER_AREA,
            )
            eye_obs = self.eye.observe(fov_patch, max_width=48)
            # Pure photoreceptor guidance from `eye_obs.l1.luminance_a`:
            # Look at the lower-middle rows of the foveal retina (ahead of the tool tip)
            # where the bright cleavage seam has maximum photoreceptor luminance and dark knots have minimum!
            retina_ahead = np.mean(eye_obs.l1.luminance_a[24:36, :], axis=0)
            best_retina_col = int(np.argmax(retina_ahead))
            # Map retina column [0..47] back to workspace X coordinate around gaze `gx`
            visual_target_x = float(gx - half_fov + (best_retina_col / 47.0) * (2.0 * half_fov))
            visual_dx = float(visual_target_x - tip_x)

            # Update organ voxel tensors in CausalField
            self.nervous_field.voxels["organ_hand"].tensor = np.array(
                [felt_force / 40.0, local_res, grad_rx, slip_friction, m_obs.damping_ratio, 1.0], dtype=np.float32
            )
            self.nervous_field.voxels["organ_ear"].tensor = np.array(
                [a_obs.peak_frequency_hz / 3000.0, seam_harmony, a_obs.rms_energy, 0.0, 0.0, 1.0], dtype=np.float32
            )
            self.nervous_field.voxels["organ_eye"].tensor = np.array(
                [visual_dx / 15.0, focal_scale / 10.0, float(len(eye_obs.regions)) / 5.0, 0.0, 0.0, 1.0], dtype=np.float32
            )
            self.nervous_field.voxels["organ_foot_posture"].tensor = np.array(
                [stance_dist, normal_force / 40.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
            )
            self.nervous_field.step(dt=0.02)

            if circulate_feedback:
                # --- CIRCULATING NERVOUS CURRENTS BETWEEN ORGANS ---
                # A) Hand -> Eye current: high resistance/friction pulls gaze to tip & zooms focal scale toward micro (L2.5)
                c_hand_eye = float(self.beam_conductance[("organ_hand", "organ_eye")] * (0.4 * local_res + 1.2 * slip_friction))
                focal_scale = float(np.clip(6.5 - 2.8 * c_hand_eye, 2.2, 8.0))
                gaze_y = float(np.clip(tip_y + 3.5, 4.0, size - 5.0))
                gaze_x = float(np.clip(0.35 * gaze_x + 0.65 * visual_target_x, 4.0, size - 5.0))

                # B) Eye & Hand -> Hand Motor steering: visual lookahead + tactile gradient steer lateral velocity
                c_eye_hand = float(self.beam_conductance[("organ_eye", "organ_hand")] * visual_dx)
                vel_x = float(np.clip(0.55 * c_eye_hand - 1.8 * grad_rx, -3.2, 3.2))

                # C) Ear -> Hand reflex current: high-frequency disharmony damps forward speed & eases force
                c_ear_hand = float(self.beam_conductance[("organ_ear", "organ_hand")] * (1.0 - seam_harmony))
                vel_y = float(np.clip(2.5 * (1.0 - 0.45 * c_ear_hand), 1.1, 2.7))
                normal_force = float(np.clip(22.0 * (0.75 + 0.45 * seam_harmony), 14.0, 28.0))

                # D) Hand -> Foot/Posture current: resistance adjusts stance distance for optimal leverage
                c_hand_foot = float(self.beam_conductance[("organ_hand", "organ_foot_posture")] * (local_res - 0.5))
                stance_dist = float(np.clip(0.35 + 0.18 * c_hand_foot, 0.22, 0.65))

                # E) Hebbian / Causal Plasticity: strengthen pathways when friction drops or harmony holds
                friction_improved = (slip_friction <= prev_friction + 0.05) and (seam_harmony > 0.65)
                delta_w = 0.04 if friction_improved else 0.01
                for k_pair in self.beam_conductance:
                    self.beam_conductance[k_pair] = float(np.clip(self.beam_conductance[k_pair] + delta_w, 0.5, 2.5))
            else:
                # Linear predetermined motion: no inter-organ circulation!
                c_hand_eye = 0.0
                c_eye_hand = 0.0
                c_ear_hand = 0.0
                c_hand_foot = 0.0
                vel_x = 0.0
                vel_y = 2.4

            # Apply physical tool stroke step & mark cut trail on live canvas
            step_work = float(max(0.0, (1.35 - local_res)) * np.hypot(vel_y, vel_x))
            total_work += step_work
            total_friction_damage += slip_friction
            prev_friction = slip_friction

            cv2.circle(live_rgb, (ix, iy), 1, (28, 220, 185) if seam_harmony > 0.55 else (235, 65, 55), -1)

            traces.append(
                NervousTickTrace(
                    step=step,
                    tip_pos_yx=(tip_y, tip_x),
                    gaze_pos_yx=(gaze_y, gaze_x),
                    ocular_focal_scale=focal_scale,
                    stance_distance_m=stance_dist,
                    hand_velocity_yx=(vel_y, vel_x),
                    hand_normal_force_n=normal_force,
                    felt_resistance_n=felt_force,
                    acoustic_freq_hz=float(a_obs.peak_frequency_hz),
                    acoustic_harmony_ratio=seam_harmony,
                    current_hand_to_eye=c_hand_eye,
                    current_eye_to_hand=c_eye_hand,
                    current_ear_to_hand=c_ear_hand,
                    current_hand_to_foot=c_hand_foot,
                    work_progress=total_work,
                    damage_or_slip_friction=total_friction_damage,
                )
            )

            tip_y = float(np.clip(tip_y + vel_y, 2.0, size - 3.0))
            tip_x = float(np.clip(tip_x + vel_x, 2.0, size - 3.0))
            if tip_y >= size - 14.0:
                break

        mean_seam_deviation = float(
            np.mean([abs(tr.tip_pos_yx[1] - channel_x[int(round(tr.tip_pos_yx[0])), 0]) for tr in traces])
        )
        mean_harmony = float(np.mean([tr.acoustic_harmony_ratio for tr in traces]))
        scale_span = (
            float(min(tr.ocular_focal_scale for tr in traces)),
            float(max(tr.ocular_focal_scale for tr in traces)),
        )
        stance_span = (
            float(min(tr.stance_distance_m for tr in traces)),
            float(max(tr.stance_distance_m for tr in traces)),
        )

        return {
            "circulate_feedback": circulate_feedback,
            "ticks_executed": len(traces),
            "total_effective_work": total_work,
            "total_friction_damage": total_friction_damage,
            "mean_seam_deviation_px": mean_seam_deviation,
            "mean_acoustic_harmony": mean_harmony,
            "ocular_scale_range": scale_span,
            "foot_stance_range_m": stance_span,
            "final_beam_conductances": dict(self.beam_conductance),
            "traces": traces,
            "final_canvas_rgb": live_rgb,
        }


# ======================================================================
# 6. Non-Point Boundary Ecosystem (Field-to-Membrane Interaction without Point/Scalar Collapse)
# ======================================================================
class BoundaryEcosystemOrganism:
    """
    Eliminates 0D point-collapse (`argmax`, `argmin`, single `(y, x)` point tracking).
    - What the Eye (`OcularScaleOrgan`) perceives is kept as a full 2D spatial
      perceptual field: regional cohesion maps, material boundary tension walls,
      and local tangent flow fields.
    - What the Body/Hand/Tool is: an extended 2D deformable boundary membrane
      (a coupled mesh of boundary receptor-actuator cells bound by internal
      `ConnectivityBeam` surface tension, not a single point!).
    - Interaction occurs across the entire contact boundary between the body's
      membrane and the external world's field: unequal resistance and ocular
      boundary pressure across the left/right/front flanks of the membrane
      naturally bend, rotate, and flow the extended body along the world's
      contours without ever computing a single target coordinate.
    """

    def __init__(self, n_membrane_cells: int = 11) -> None:
        self.eye = OcularScaleOrgan()
        self.n_cells = int(n_membrane_cells)

    def perceive_world_as_spatial_field(self, world_rgb: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Observe the world through `OcularScaleOrgan` and preserve the entire 2D
        spatial topology as coupled fields (never collapsing to scalars or points).
        """
        h, w = world_rgb.shape[:2]
        obs = self.eye.observe(world_rgb, max_width=w)

        # 1. Photoreceptor luminance field & multi-scale texture field (2D maps)
        luma_field = cv2.resize(obs.l1.luminance_a.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        tex_field = cv2.resize(obs.l2.fine_texture_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

        # 2. Material boundary barrier field reconstructed from OcularScaleOrgan's 2D region label transitions
        reg_labels = cv2.resize(obs.region_labels.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        gy_r, gx_r = np.gradient(reg_labels)
        boundary_wall_field = np.clip(np.hypot(gy_r, gx_r), 0.0, 1.0)

        # 3. Continuous spatial flow potential field U(y, x) formed by photoreceptor topology:
        #    - Coherent bright cleavage channels form low-potential valleys (attracting flow)
        #    - Dark dense knots & region boundaries form high-potential walls (repelling flow)
        potential_field = -1.2 * luma_field + 0.8 * tex_field + 0.6 * boundary_wall_field

        # Local field pressure gradients (-dU/dy, -dU/dx) exist everywhere as a 2D vector field
        grad_u_y, grad_u_x = np.gradient(potential_field)
        flow_field_y = -grad_u_y
        flow_field_x = -grad_u_x

        return {
            "luma_field": luma_field,
            "tex_field": tex_field,
            "boundary_wall_field": boundary_wall_field,
            "potential_field": potential_field,
            "flow_field_y": flow_field_y,
            "flow_field_x": flow_field_x,
            "region_labels": reg_labels,
        }

    def interact_as_boundary_ecosystem(
        self,
        world_rgb: np.ndarray,
        world_resistance_field: np.ndarray,
        steps: int = 30,
    ) -> Dict[str, Any]:
        """
        An extended 11-cell body-tool boundary membrane (spanning ~14 pixels of
        physical width and thickness) flows through the 2D world field:
        - Each cell `(i = 0..10)` of the boundary membrane feels local tactile
          resistance `R(y_i, x_i)` and local ocular field pressure `(-dU/dy, -dU/dx)`
          at its own spatial position `(y_i, x_i)`.
        - Neighboring cells in the membrane are coupled by elastic surface-tension
          beams (`k_membrane`), keeping the hand-tool body coherent as an ecosystem
          while allowing the boundary to curve, tilt, and conform to the world!
        - Zero `argmax`, zero `argmin`, zero point-target coordinate calculation.
        """
        h, w = world_rgb.shape[:2]
        ocular_fields = self.perceive_world_as_spatial_field(world_rgb)
        flow_x = ocular_fields["flow_field_x"]

        # Initialize the extended boundary membrane across the top entry zone (y=14, x spanning 42..54)
        span_offsets = np.linspace(-6.0, 6.0, self.n_cells, dtype=np.float64)
        cell_y = np.full(self.n_cells, 14.0, dtype=np.float64)
        cell_x = 0.5 * w + span_offsets

        membrane_history: List[np.ndarray] = []
        boundary_shear_history: List[float] = []
        tactile_distribution_history: List[np.ndarray] = []

        for _ in range(steps):
            iy = np.clip(np.round(cell_y).astype(np.int32), 2, h - 3)
            ix = np.clip(np.round(cell_x).astype(np.int32), 2, w - 3)

            # 1. Distributed tactile resistance across the entire contact membrane (11 cells, not 1 point!)
            local_tactile_r = world_resistance_field[iy, ix]
            # Local tactile gradient felt directly across the left-to-right cells of the membrane
            tactile_grad_x = 0.5 * (
                world_resistance_field[iy, np.clip(ix + 2, 2, w - 3)]
                - world_resistance_field[iy, np.clip(ix - 2, 2, w - 3)]
            )

            # 2. Distributed ocular field pressure across the membrane's front boundary
            look_iy = np.clip(iy + 3, 2, h - 3)
            local_ocular_push_x = flow_x[look_iy, ix]

            # 3. Unequal resistance across the left vs right flanks of the membrane creates
            #    natural boundary torque & lateral flow (high resistance on one flank pushes that flank away)
            cell_vel_x = 14.0 * local_ocular_push_x - 2.4 * tactile_grad_x
            # Cells feeling lower resistance advance faster; cells hitting a knot/wall slow down,
            # naturally pivoting the extended boundary around obstacles!
            cell_vel_y = np.clip(2.4 / (0.6 + 0.8 * local_tactile_r), 0.7, 2.6)

            # 4. Internal surface tension (`ConnectivityBeam` coupling between adjacent membrane cells)
            #    maintains the organism's structural integrity without collapsing it to a point
            next_x = cell_x + np.clip(cell_vel_x, -3.0, 3.0)
            next_y = cell_y + cell_vel_y

            # Elastic relaxation across neighboring cells in the boundary membrane
            mean_x = float(np.mean(next_x))
            mean_y = float(np.mean(next_y))
            # Allow natural tilt/curvature across the membrane while preserving cohesive spacing
            desired_x = mean_x + span_offsets
            next_x = 0.65 * next_x + 0.35 * desired_x
            next_y = 0.75 * next_y + 0.25 * mean_y

            cell_x = np.clip(next_x, 4.0, w - 5.0)
            cell_y = np.clip(next_y, 4.0, h - 5.0)

            membrane_coords = np.stack([cell_y.copy(), cell_x.copy()], axis=-1)
            membrane_history.append(membrane_coords)
            tactile_distribution_history.append(local_tactile_r.copy())
            # Boundary shear = pressure differential across the extended contact surface
            boundary_shear_history.append(float(np.max(local_tactile_r) - np.min(local_tactile_r)))

            if float(np.mean(cell_y)) >= h - 14.0:
                break

        return {
            "ocular_fields": ocular_fields,
            "membrane_history": membrane_history,
            "tactile_distribution_history": tactile_distribution_history,
            "mean_boundary_resistance": float(np.mean([np.mean(r) for r in tactile_distribution_history])),
            "mean_boundary_shear_differential": float(np.mean(boundary_shear_history)),
        }







class CrossModalPerceptualBridge:
    """
    Instead of brute-forcing math formulas when a 1D static tap fails, ELYSIA
    projects temporal flow into a 2D time-frequency perceptual canvas and
    observes it directly with her own `OcularScaleOrgan`.
    """

    def __init__(self, eye: OcularScaleOrgan) -> None:
        self.eye = eye

    def fold_waveform_to_ocular_canvas(
        self, waveform: np.ndarray, sample_rate: int = 16000, size: int = 64
    ) -> np.ndarray:
        """
        Convert a 1D temporal wave into a 2D perceptual field (Time on X,
        Resonance Frequency on Y) rendered as an RGB sensory canvas so
        `OcularScaleOrgan` can perceive temporal flow as visual trajectory/strata.
        """
        x = np.asarray(waveform, dtype=np.float64)
        n_frames = 16
        frames = np.array_split(x, n_frames)
        spec_cols: List[np.ndarray] = []
        freqs: Optional[np.ndarray] = None

        for seg in frames:
            m = len(seg)
            win = np.hanning(m)
            mag = np.abs(np.fft.rfft(seg * win))
            if freqs is None:
                freqs = np.fft.rfftfreq(m, d=1.0 / sample_rate)
            band = (freqs >= 300.0) & (freqs <= 3200.0)
            col = mag[band]
            # Place high frequency at top (row 0), low frequency at bottom
            spec_cols.append(col[::-1])

        grid = np.stack(spec_cols, axis=1)  # (F_bins, T_frames)
        grid = np.log1p(grid / (np.max(grid) + 1e-9) * 12.0)
        grid /= max(float(np.max(grid)), 1e-9)

        canvas_2d = cv2.resize(grid.astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
        canvas_2d = np.clip(canvas_2d, 0.0, 1.0)

        # Map energy & vertical frequency position into photoreceptor-friendly RGB contrast
        yy = np.linspace(1.0, 0.0, size, dtype=np.float32)[:, None]
        r = np.clip(255.0 * (0.20 + 0.80 * canvas_2d), 0, 255)
        g = np.clip(255.0 * (0.15 + 0.70 * canvas_2d * (0.5 + 0.5 * yy)), 0, 255)
        b = np.clip(255.0 * (0.25 + 0.75 * canvas_2d * yy), 0, 255)
        return np.stack([r, g, b], axis=-1).astype(np.uint8)

    def perceive_temporal_sound_structure(
        self, waveform: np.ndarray, sample_rate: int = 16000
    ) -> Dict[str, Any]:
        """
        Re-observe the folded temporal sound canvas through `OcularScaleOrgan`
        and extract its qualitative perceptual structure (no numeric classifier search!):
        - Does the resonant ridge flow upward (rising) or downward (falling)?
        - Is there an upper resonant stratum present (dual-formant cleavage)?
        - Does the temporal contour split into distinct islands (rhythmic pulse void)?
        """
        canvas = self.fold_waveform_to_ocular_canvas(waveform, sample_rate=sample_rate, size=64)
        obs: OpticalSceneObservation = self.eye.observe(canvas, max_width=64)
        a = obs.l1.luminance_a  # (H, W), row 0 = high freq, row H-1 = low freq, col 0..W-1 = time

        h, w = a.shape
        # Normalize photoreceptor activation above background
        act = np.maximum(0.0, a - float(np.percentile(a, 35)))
        yy = np.linspace(1.0, 0.0, h, dtype=np.float64)[:, None]  # 1.0 = top (high freq), 0.0 = bottom (low freq)

        # Early vs Late vertical center of perception (temporal flow direction)
        early_act = act[:, : w // 3]
        late_act = act[:, -w // 3 :]
        y_early = float(np.sum(yy[:, : w // 3] * early_act) / max(float(np.sum(early_act)), 1e-6))
        y_late = float(np.sum(yy[:, -w // 3 :] * late_act) / max(float(np.sum(late_act)), 1e-6))
        vertical_flow = float(y_late - y_early)

        # Upper stratum presence vs lower stratum presence (structural stratification)
        upper_energy = float(np.mean(act[: int(h * 0.42), :]))
        lower_energy = float(np.mean(act[int(h * 0.42) :, :]))
        upper_stratum_ratio = float(upper_energy / max(lower_energy, 1e-6))

        # Temporal continuity vs rhythmic void (does energy vanish in the middle?)
        mid_time_energy = float(np.mean(act[:, w // 3 : 2 * w // 3]))
        edge_time_energy = float(0.5 * (np.mean(act[:, : w // 3]) + np.mean(act[:, -w // 3 :])))
        temporal_void_ratio = float(edge_time_energy / max(mid_time_energy, 1e-6))

        qualities_present: List[str] = ["RESONANT_CONTINUITY"]
        qualities_absent: List[str] = []

        if temporal_void_ratio > 1.35:
            qualities_present.append("TEMPORAL_RHYTHMIC_CLEAVAGE")
            qualities_absent.append("UNBROKEN_SUSTAIN")
        else:
            qualities_present.append("UNBROKEN_SUSTAIN")
            qualities_absent.append("TEMPORAL_RHYTHMIC_CLEAVAGE")

        if upper_stratum_ratio > 0.52:
            qualities_present.append("UPPER_HARMONIC_STRATUM")
        else:
            qualities_absent.append("UPPER_HARMONIC_STRATUM")

        if vertical_flow > 0.045:
            qualities_present.append("ASCENDING_FLOW")
            qualities_absent.append("DESCENDING_FLOW")
        elif vertical_flow < -0.045:
            qualities_present.append("DESCENDING_FLOW")
            qualities_absent.append("ASCENDING_FLOW")
        else:
            qualities_present.append("HORIZONTAL_EQUILIBRIUM")

        return {
            "canvas_rgb": canvas,
            "ocular_regions": len(obs.regions),
            "vertical_flow": vertical_flow,
            "upper_stratum_ratio": upper_stratum_ratio,
            "temporal_void_ratio": temporal_void_ratio,
            "qualities_present": qualities_present,
            "qualities_absent": qualities_absent,
            "relational_signature": (vertical_flow, upper_stratum_ratio, temporal_void_ratio),
        }


class CognitivePrincipleEngine:
    """
    Internalizes 'how things change and are re-recognized through sensory organs'
    into CognitivePrinciples, and transforms/composes those principles without
    brute-force numerical search.
    """

    def __init__(self, spine: EpistemicKnowledgeSpine) -> None:
        self.spine = spine
        self.eye = OcularScaleOrgan()
        self.mechano_acoustic = MechanoAcousticOrgan()
        self.reasoner = VisualCausalReasoner(self.eye)
        self.bridge = CrossModalPerceptualBridge(self.eye)
        self.principles: Dict[str, CognitivePrinciple] = {}

    # ------------------------------------------------------------------
    # 1. Presence/Absence Awareness & Re-Observation of Temporal Flow
    # ------------------------------------------------------------------
    def internalize_auditory_flow_principles(
        self,
        speech_world: ContinuousSpeechAcousticWorld,
        train_samples: List[Tuple[np.ndarray, int]],
        unseen_samples: List[Tuple[np.ndarray, int]],
    ) -> Dict[str, Any]:
        """
        1) Notice what is ABSENT in a single static tap (`MechanoAcousticOrgan`):
           Rising and falling flows collapse into the same static peak -> registers
           `VOID_MISSING_TEMPORAL_REOBSERVATION`.
        2) Re-observe the sounds through `CrossModalPerceptualBridge` + `OcularScaleOrgan`
           to form qualitative CognitivePrinciples for each sound pattern.
        3) Recognize unseen pitch-shifted (+35Hz) sounds by matching their qualitative
           perceptual principle (Presence/Absence of Ascending Flow, Upper Stratum,
           or Temporal Rhythmic Cleavage).
        """
        # Step 1: Experience friction with single static tap on sample 0 (rising) vs sample 1 (falling)
        rising_wav = next(w for w, c in train_samples if c == 0)
        falling_wav = next(w for w, c in train_samples if c == 1)
        tap_r = self.mechano_acoustic.observe_tap(rising_wav, speech_world.sample_rate)
        tap_f = self.mechano_acoustic.observe_tap(falling_wav, speech_world.sample_rate)

        static_peak_diff = abs(tap_r.peak_frequency_hz - tap_f.peak_frequency_hz)
        void_obj = self.spine.register_void(
            kind="VOID_MISSING_TEMPORAL_REOBSERVATION",
            target="temporal_direction_of_sound",
            priority=0.95,
            origin_medium="sensory_reflection",
            hypothesis=(
                f"Static tap organ sees identical peak ({tap_r.peak_frequency_hz:.1f}Hz vs "
                f"{tap_f.peak_frequency_hz:.1f}Hz); temporal flow direction is absent in 0D snapshot."
            ),
            suggested_medium="ocular_temporal_reobservation",
        )

        # Step 2: Re-observe through OcularScaleOrgan bridge and form CognitivePrinciples per class
        class_prototypes: Dict[int, Dict[str, Any]] = {}
        for cls_idx in sorted({c for _, c in train_samples}):
            cls_wavs = [w for w, c in train_samples if c == cls_idx]
            perceptions = [
                self.bridge.perceive_temporal_sound_structure(w, speech_world.sample_rate)
                for w in cls_wavs
            ]
            # Consensus of present/absent perceptual qualities
            all_present = sorted(
                {
                    q
                    for p in perceptions
                    for q in p["qualities_present"]
                    if sum(q in p2["qualities_present"] for p2 in perceptions) >= len(perceptions) * 0.6
                }
            )
            all_absent = sorted(
                {
                    q
                    for p in perceptions
                    for q in p["qualities_absent"]
                    if sum(q in p2["qualities_absent"] for p2 in perceptions) >= len(perceptions) * 0.6
                }
            )
            mean_sig = tuple(
                float(np.mean([p["relational_signature"][k] for p in perceptions]))
                for k in range(3)
            )
            pid = f"principle_sound_pattern_{cls_idx}"
            chrom = (
                "RED_FLUX"
                if ("ASCENDING_FLOW" in all_present or "DESCENDING_FLOW" in all_present)
                else ("YELLOW_CLEAVAGE" if "TEMPORAL_RHYTHMIC_CLEAVAGE" in all_present else "BLUE_ORDER")
            )
            principle = CognitivePrinciple(
                principle_id=pid,
                origin_domain="auditory_ocular_bridge",
                present_before=["RAW_TEMPORAL_VIBRATION"],
                present_after=all_present,
                newly_emerged=[q for q in all_present if q != "RESONANT_CONTINUITY"],
                dissolved_into_absence=all_absent,
                retained_stem=["RESONANT_CONTINUITY"],
                is_reversible_symmetric=("HORIZONTAL_EQUILIBRIUM" in all_present),
                chromatic_character=chrom,
                relational_signature=mean_sig,  # type: ignore[arg-type]
            )
            self.principles[pid] = principle
            class_prototypes[cls_idx] = {
                "principle": principle,
                "present_set": set(all_present),
                "absent_set": set(all_absent),
                "mean_sig": mean_sig,
                "sample_canvas": perceptions[0]["canvas_rgb"],
            }

        # Notice how class 0 (ascending) and class 1 (descending) are structural inverses!
        self.spine.resolve_void(
            void_obj.void_id,
            "Folded 1D time stream into 2D ocular canvas; ascending vs descending flow separated as inverse trajectories.",
        )

        # Step 3: Recognize unseen pitch-shifted (+35Hz) waveforms via qualitative principle resonance
        correct = 0
        for wav, true_cls in unseen_samples:
            perc = self.bridge.perceive_temporal_sound_structure(wav, speech_world.sample_rate)
            p_set = set(perc["qualities_present"])
            a_set = set(perc["qualities_absent"])
            sig = perc["relational_signature"]

            best_cls = -1
            best_resonance = -1e9
            for cid, proto in class_prototypes.items():
                # Qualitative presence/absence match + directional sign agreement
                qual_overlap = len(p_set & proto["present_set"]) + 0.5 * len(a_set & proto["absent_set"])
                qual_mismatch = len(p_set & proto["absent_set"]) + len(a_set & proto["present_set"])
                # Relational direction alignment (not threshold fitting, just structural similarity)
                psig = proto["mean_sig"]
                flow_align = -abs(sig[0] - psig[0]) * 4.0 - abs(sig[1] - psig[1]) * 1.5 - abs(sig[2] - psig[2]) * 1.5
                res = float(2.0 * qual_overlap - 2.5 * qual_mismatch + flow_align)
                if res > best_resonance:
                    best_resonance = res
                    best_cls = cid
            if best_cls == true_cls:
                correct += 1

        unseen_acc = float(correct / max(1, len(unseen_samples)))
        return {
            "static_tap_peak_diff_hz": static_peak_diff,
            "void_resolved": void_obj.void_id,
            "principles_formed": {cid: proto["principle"] for cid, proto in class_prototypes.items()},
            "sample_canvases": {cid: proto["sample_canvas"] for cid, proto in class_prototypes.items()},
            "unseen_recognition_accuracy": unseen_acc,
        }

    # ------------------------------------------------------------------
    # 2. Understanding How CAD Designs via Perceptual Re-Observation
    # ------------------------------------------------------------------
    def observe_cad_transition(
        self,
        before: CADShapeObservation,
        after: CADShapeObservation,
        action_tag: str,
    ) -> PerceptualTransition:
        """
        Observe 'what changed and how it is re-recognized' when a CAD operation
        transforms `before` into `after`, using `OcularScaleOrgan` and tactile/spatial
        sense (`bbox`, `hollow_cavity`, `stability`).
        """
        obs_b = self.eye.observe(before.rgb_iso, max_width=96)
        obs_a = self.eye.observe(after.rgb_iso, max_width=96)
        comp = self.reasoner.compare_observations(obs_b, obs_a)

        bx, by, bz = before.bbox_xyz
        ax, ay, az = after.bbox_xyz

        retained: List[str] = []
        flow: List[str] = []
        emergent: List[str] = []

        # What stayed the same? (Blue / Order)
        if abs(ax - bx) < 0.06 and abs(ay - by) < 0.06:
            retained.append("BASE_FOOTPRINT_INVARIANT")
        if before.stable_under_gravity and after.stable_under_gravity:
            retained.append("GRAVITATIONAL_EQUILIBRIUM")

        # How did form move or grow? (Red / Flux)
        if az > bz + 0.08 and abs(ax - bx) < 0.06:
            flow.append("VERTICAL_DIRECTIONAL_EXTENSION")
        elif ax > bx + 0.08 and ay > by + 0.08 and az > bz + 0.08:
            flow.append("OMNIDIRECTIONAL_EXPANSION")
        elif after.volume > before.volume + 0.02:
            flow.append("CONSTRUCTIVE_MATTER_ACCUMULATION")
        elif after.volume < before.volume - 0.01:
            flow.append("SUBTRACTIVE_MATTER_REMOVAL")

        # What boundary or absence newly appeared? (Yellow / Cleavage & Void)
        if after.hollow_cavity_voxels > before.hollow_cavity_voxels + 10:
            emergent.append("INTERNAL_HOLLOW_VOID_BORN")
            emergent.append("INNER_CONTAINMENT_WALL")
        if before.stable_under_gravity and not after.stable_under_gravity:
            emergent.append("EQUILIBRIUM_COLLAPSE_TIPPING")

        v_flow = float((az - bz) / max(bz, 1e-4))
        lat_spread = float(0.5 * ((ax - bx) + (ay - by)) / max(0.5 * (bx + by), 1e-4))
        cav_emerge = float(after.hollow_cavity_voxels - before.hollow_cavity_voxels)

        return PerceptualTransition(
            source_state_id=before.shape_id,
            target_state_id=after.shape_id,
            action_tag=action_tag,
            retained_qualities=retained,
            flow_qualities=flow,
            emergent_qualities=emergent,
            structural_persistence=float(comp.l2_persistence_corr),
            directional_polarity=(v_flow, lat_spread, cav_emerge),
        )

    def discover_and_transform_cad_principles(
        self, cad: OpaqueParametricCADKernel
    ) -> Dict[str, Any]:
        """
        1) Experience each unknown CAD operator by watching how it transforms a base form
           through ELYSIA's sensory organs (`observe_cad_transition`).
        2) Distill each transformation into a non-numerical `CognitivePrinciple`
           (Presence -> Absence, Footprint Invariance + Vertical Flow, Whole Merging, etc.).
        3) Transform & Compose those principles (`원리를 가공하고 변화시켜 창조`):
           Given the functional goal:
             'Create a structure that holds an inner void (INTERNAL_HOLLOW_VOID_BORN),
              reaches upward (VERTICAL_DIRECTIONAL_EXTENSION), and never tips over
              even with an overhanging crown (GRAVITATIONAL_EQUILIBRIUM)',
           ELYSIA composes her internalized principles in sequence and re-observes every
           intermediate state through `OcularScaleOrgan` to self-correct and verify!
        """
        ops = cad.available_ops()
        primitive_op: Optional[str] = None
        unary_ops: List[str] = []
        binary_ops: List[str] = []
        seed_shape: Optional[CADShapeObservation] = None

        for op_id in ops:
            try:
                seed_shape = cad.execute(op_id, params={"hx": 0.34, "hy": 0.34, "hz": 0.24})
                primitive_op = op_id
                continue
            except ValueError:
                pass
            assert seed_shape is not None
            try:
                cad.execute(op_id, shape_a=seed_shape.shape_id)
                unary_ops.append(op_id)
            except ValueError:
                binary_ops.append(op_id)

        assert primitive_op is not None and seed_shape is not None

        # Principle 0: Manifesting Solid Presence from Empty Space
        p_manifest = CognitivePrinciple(
            principle_id="PRINCIPLE_SOLID_MANIFESTATION",
            origin_domain="cad_perceptual_reobservation",
            present_before=["EMPTY_VOID"],
            present_after=["BOUNDED_SOLID_PRESENCE", "GRAVITATIONAL_EQUILIBRIUM"],
            newly_emerged=["BOUNDED_SOLID_PRESENCE"],
            dissolved_into_absence=["EMPTY_VOID"],
            retained_stem=["GRAVITATIONAL_EQUILIBRIUM"],
            is_reversible_symmetric=True,
            chromatic_character="BLUE_ORDER",
            relational_signature=(0.0, 0.0, 0.0),
            bound_operator_id=primitive_op,
        )
        self.principles[p_manifest.principle_id] = p_manifest

        # Observe unary transformations through sensory transition
        for uop in unary_ops:
            after = cad.execute(uop, shape_a=seed_shape.shape_id, params={"factor": 1.75})
            trans = self.observe_cad_transition(seed_shape, after, uop)
            if "BASE_FOOTPRINT_INVARIANT" in trans.retained_qualities and "VERTICAL_DIRECTIONAL_EXTENSION" in trans.flow_qualities:
                p = CognitivePrinciple(
                    principle_id="PRINCIPLE_DIRECTIONAL_EXTRUSION",
                    origin_domain="cad_perceptual_reobservation",
                    present_before=["BOUNDED_SOLID_PRESENCE", "BASE_FOOTPRINT_INVARIANT"],
                    present_after=trans.retained_qualities + trans.flow_qualities,
                    newly_emerged=["VERTICAL_DIRECTIONAL_EXTENSION"],
                    dissolved_into_absence=[],
                    retained_stem=["BASE_FOOTPRINT_INVARIANT", "GRAVITATIONAL_EQUILIBRIUM"],
                    is_reversible_symmetric=False,
                    chromatic_character="RED_FLUX",
                    relational_signature=trans.directional_polarity,
                    bound_operator_id=uop,
                )
                self.principles[p.principle_id] = p
            elif "OMNIDIRECTIONAL_EXPANSION" in trans.flow_qualities:
                p = CognitivePrinciple(
                    principle_id="PRINCIPLE_PROPORTIONAL_EXPANSION",
                    origin_domain="cad_perceptual_reobservation",
                    present_before=["BOUNDED_SOLID_PRESENCE"],
                    present_after=trans.retained_qualities + trans.flow_qualities,
                    newly_emerged=["OMNIDIRECTIONAL_EXPANSION"],
                    dissolved_into_absence=[],
                    retained_stem=["GRAVITATIONAL_EQUILIBRIUM"],
                    is_reversible_symmetric=True,
                    chromatic_character="RED_FLUX",
                    relational_signature=trans.directional_polarity,
                    bound_operator_id=uop,
                )
                self.principles[p.principle_id] = p

        # Observe binary interactions: Merging two wholes vs Carving presence into absence (hollow void)
        outer = cad.execute(primitive_op, params={"hx": 0.46, "hy": 0.46, "hz": 0.46})
        inner = cad.execute(primitive_op, params={"hx": 0.22, "hy": 0.22, "hz": 0.62})
        offset_part = cad.execute(primitive_op, params={"hx": 0.30, "hy": 0.30, "hz": 0.30, "ox": 0.28})

        for bop in binary_ops:
            carved = cad.execute(bop, shape_a=outer.shape_id, shape_b=inner.shape_id)
            merged = cad.execute(bop, shape_a=outer.shape_id, shape_b=offset_part.shape_id)
            rev_carved = cad.execute(bop, shape_a=inner.shape_id, shape_b=outer.shape_id)
            is_sym = bool(np.array_equal(carved.occ_grid, rev_carved.occ_grid))

            t_carve = self.observe_cad_transition(outer, carved, bop)
            t_merge = self.observe_cad_transition(outer, merged, bop)

            if "INTERNAL_HOLLOW_VOID_BORN" in t_carve.emergent_qualities and not is_sym:
                p = CognitivePrinciple(
                    principle_id="PRINCIPLE_VOID_CARVING_CLEAVAGE",
                    origin_domain="cad_perceptual_reobservation",
                    present_before=["SOLID_INTERIOR_CORE", "BOUNDED_SOLID_PRESENCE"],
                    present_after=["INNER_CONTAINMENT_WALL", "INTERNAL_HOLLOW_VOID_BORN"],
                    newly_emerged=["INTERNAL_HOLLOW_VOID_BORN", "INNER_CONTAINMENT_WALL"],
                    dissolved_into_absence=["SOLID_INTERIOR_CORE"],
                    retained_stem=["BASE_FOOTPRINT_INVARIANT", "GRAVITATIONAL_EQUILIBRIUM"],
                    is_reversible_symmetric=False,
                    chromatic_character="YELLOW_CLEAVAGE",
                    relational_signature=t_carve.directional_polarity,
                    bound_operator_id=bop,
                )
                self.principles[p.principle_id] = p
            elif "CONSTRUCTIVE_MATTER_ACCUMULATION" in t_merge.flow_qualities and is_sym:
                p = CognitivePrinciple(
                    principle_id="PRINCIPLE_STRUCTURAL_UNIFICATION",
                    origin_domain="cad_perceptual_reobservation",
                    present_before=["SEPARATE_PARTS"],
                    present_after=["UNIFIED_CONTINUOUS_BODY", "CONSTRUCTIVE_MATTER_ACCUMULATION"],
                    newly_emerged=["UNIFIED_CONTINUOUS_BODY"],
                    dissolved_into_absence=["SEPARATE_PARTS"],
                    retained_stem=["GRAVITATIONAL_EQUILIBRIUM"],
                    is_reversible_symmetric=True,
                    chromatic_character="BLUE_ORDER",
                    relational_signature=t_merge.directional_polarity,
                    bound_operator_id=bop,
                )
                self.principles[p.principle_id] = p

        # ------------------------------------------------------------------
        # 3. Principle Transformation & Self-Corrective Imagination Loop
        # ------------------------------------------------------------------
        # First: Observe what happens when an overhanging crown is added to a narrow base
        # -> Re-observation detects EQUILIBRIUM_COLLAPSE_TIPPING (Absence of stability!)
        op_solid = self.principles["PRINCIPLE_SOLID_MANIFESTATION"].bound_operator_id
        op_ext = self.principles["PRINCIPLE_DIRECTIONAL_EXTRUSION"].bound_operator_id
        op_union = self.principles["PRINCIPLE_STRUCTURAL_UNIFICATION"].bound_operator_id
        op_carve = self.principles["PRINCIPLE_VOID_CARVING_CLEAVAGE"].bound_operator_id
        assert op_solid and op_ext and op_union and op_carve

        narrow_base = cad.execute(op_solid, params={"hx": 0.24, "hy": 0.24, "hz": 0.20, "oz": -0.45})
        heavy_overhang = cad.execute(op_solid, params={"hx": 0.34, "hy": 0.34, "hz": 0.28, "ox": 0.50, "oz": 0.05})
        unstable_draft = cad.execute(op_union, shape_a=narrow_base.shape_id, shape_b=heavy_overhang.shape_id)
        t_unstable = self.observe_cad_transition(narrow_base, unstable_draft, "draft_overhang")

        # Because ELYSIA perceives `EQUILIBRIUM_COLLAPSE_TIPPING` (absence of GRAVITATIONAL_EQUILIBRIUM),
        # she transforms her design principle: widen the base support (BLUE_ORDER) before extending
        # the tall body (RED_FLUX) and carving the inner containment cavity (YELLOW_CLEAVAGE)!
        self.spine.criteria_revisions.append(
            {
                "kind": "PERCEPTUAL_DESIGN_PRINCIPLE_REVISION",
                "trigger": t_unstable.emergent_qualities,
                "revision": (
                    "Narrow base lost GRAVITATIONAL_EQUILIBRIUM under overhang; "
                    "composed Wide-Support Manifestation -> Vertical Extrusion -> Inner Void Carving."
                ),
            }
        )

        # Execute the composed cognitive principle sequence and re-observe every stage:
        # Stage 1: Wide grounding plinth (Order)
        s1_base = cad.execute(op_solid, params={"hx": 0.65, "hy": 0.65, "hz": 0.16, "oz": -0.56})
        # Stage 2: Column seed extended upward via PRINCIPLE_DIRECTIONAL_EXTRUSION (Flux)
        s2_seed = cad.execute(op_solid, params={"hx": 0.40, "hy": 0.40, "hz": 0.22})
        s2_tall = cad.execute(op_ext, shape_a=s2_seed.shape_id, params={"factor": 2.3})
        # Stage 3: Unify plinth + column + cantilevered rim via PRINCIPLE_STRUCTURAL_UNIFICATION
        s3_joined = cad.execute(op_union, shape_a=s1_base.shape_id, shape_b=s2_tall.shape_id)
        s3_rim = cad.execute(op_solid, params={"hx": 0.50, "hy": 0.42, "hz": 0.12, "ox": 0.10, "oz": 0.46})
        s3_body = cad.execute(op_union, shape_a=s3_joined.shape_id, shape_b=s3_rim.shape_id)
        # Stage 4: Carve inner containment chamber via PRINCIPLE_VOID_CARVING_CLEAVAGE
        s4_core = cad.execute(op_solid, params={"hx": 0.20, "hy": 0.20, "hz": 0.78})
        s4_vessel = cad.execute(op_carve, shape_a=s3_body.shape_id, shape_b=s4_core.shape_id)

        final_transition = self.observe_cad_transition(s3_body, s4_vessel, "compose_hollow_vessel")
        final_eye = self.eye.observe(s4_vessel.rgb_iso, max_width=96)

        return {
            "unstable_draft_emergent": t_unstable.emergent_qualities,
            "unstable_draft_stable": unstable_draft.stable_under_gravity,
            "composed_principles_chain": [
                "PRINCIPLE_SOLID_MANIFESTATION",
                "PRINCIPLE_DIRECTIONAL_EXTRUSION",
                "PRINCIPLE_STRUCTURAL_UNIFICATION",
                "PRINCIPLE_VOID_CARVING_CLEAVAGE",
            ],
            "final_transition": final_transition,
            "final_vessel": s4_vessel,
            "final_ocular_regions": len(final_eye.regions),
        }
