"""
active_epistemic_explorer.py

Active Self-Directing Multi-Organ Epistemic Explorer & Blank-Slate Correlation Graph
=====================================================================================
Turns ELYSIA's passive sensory organs into an active perceptual-cognitive loop:
1. EpistemicAttentionField continuously scores spatial regions and ambiguous
   optical boundaries by:
     - Prediction Surprise (failed expectation from past experience)
     - Static Optical Ambiguity (e.g. ILLUMINATION_SHADOW_CANDIDATE)
     - Cross-Modal Epistemic Uncertainty (unknown mechanics/acoustics)
     - Resolution Gap (unexplored native 1:1 foveal microstructure)
     - Minus Inhibition of Return (preventing perseveration on resolved targets)
2. CrossModalCorrelationGraph starts completely empty (prototypes = []) and
   learns visual -> (mechanical, acoustic) associations strictly from raw probe
   episodes, triggering Prototype Mitosis (splitting, never averaging) when
   counter-examples cause high surprise (> SURPRISE_THRESHOLD).
3. Motor-visual interaction (MECHANICAL_PUSH_AND_WATCH) resolves static cast
   shadow ambiguities by directly testing whether both halves co-translate as a
   single rigid physical body.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.physics.causal_field import CausalField, InformationVoxel
from core.sensory.mechano_acoustic_organ import (
    AcousticObservation,
    MechanicalObservation,
    MechanoAcousticOrgan,
)
from core.sensory.ocular_scale_organ import (
    BoundaryEdge,
    OcularScaleOrgan,
    OpticalSceneObservation,
    TemporalVisualAnalysis,
    VisualRegion,
)
from core.sensory.visual_causal_reasoner import (
    FourDomainCausalAudit,
    VisualCausalReasoner,
)
from simulators.hidden_multiphysics_world import SensorimotorWorldProtocol


@dataclass
class SensoryEpisode:
    """Record of one autonomous sensorimotor probe at a chosen spatial target."""

    step: int
    target_id: str
    target_yx_norm: Tuple[float, float]
    action_type: str
    visual_coarse_vec: np.ndarray
    visual_foveal_vec: Optional[np.ndarray]
    mechanical_obs: Optional[MechanicalObservation]
    acoustic_obs: Optional[AcousticObservation]
    motor_visual_transition: Optional[TemporalVisualAnalysis]
    predicted_mech_vec: Optional[np.ndarray]
    predicted_acoustic_vec: Optional[np.ndarray]
    prediction_surprise: float
    matched_prototype_id: Optional[str]


@dataclass
class CrossModalPrototype:
    """Empirically discovered cross-modal regularity learned from blank slate."""

    prototype_id: str
    visual_coarse_center: np.ndarray
    visual_foveal_center: Optional[np.ndarray]
    mech_center: Optional[np.ndarray]
    acoustic_center: Optional[np.ndarray]
    confirmation_count: int
    member_steps: List[int] = field(default_factory=list)
    parent_prototype_id: Optional[str] = None
    split_reason: Optional[str] = None


@dataclass
class CorrelationRefinementEvent:
    """Logged whenever a surprising counter-example splits an existing prototype."""

    step: int
    parent_prototype_id: str
    new_prototype_id: str
    surprise_error: float
    separating_axis_note: str


@dataclass
class AttentionCandidate:
    """Priority decomposition for one selectable spatial/boundary candidate."""

    target_id: str
    target_kind: str  # REGION | BOUNDARY
    y_norm: float
    x_norm: float
    surprise_score: float
    ambiguity_score: float
    cross_modal_uncertainty: float
    resolution_gap: float
    ior_penalty: float
    total_priority: float
    recommended_action: str
    rationale: str


@dataclass
class ExplorationStepDecision:
    """Selected autonomous action and its epistemic outcome at step t."""

    step: int
    chosen: AttentionCandidate
    episode: Optional[SensoryEpisode]
    outcome_summary: str


@dataclass
class ExplorationTrace:
    """Complete record of an autonomous multi-step exploration run."""

    steps: int
    periphery_obs: OpticalSceneObservation
    candidates_per_step: List[List[AttentionCandidate]]
    decisions: List[ExplorationStepDecision]
    episodes: List[SensoryEpisode]
    prototypes: List[CrossModalPrototype]
    refinements: List[CorrelationRefinementEvent]
    resolved_shadow_boundaries: Dict[Tuple[int, int], str]
    correlation_matrix: Dict[str, Any]
    final_causal_field: CausalField
    organ_audit: FourDomainCausalAudit


@dataclass
class _TargetTracker:
    target_id: str
    target_kind: str  # REGION | BOUNDARY
    y_norm: float
    x_norm: float
    region: Optional[VisualRegion] = None
    boundary: Optional[BoundaryEdge] = None
    side_a_region: Optional[VisualRegion] = None
    side_b_region: Optional[VisualRegion] = None
    foveated: bool = False
    mech_probed: bool = False
    acoustic_probed: bool = False
    boundary_push_tested: bool = False
    boundary_verdict: Optional[str] = None
    surprise_flag: float = 0.0
    ior_count: int = 0
    foveal_vec: Optional[np.ndarray] = None
    last_prototype_id: Optional[str] = None


def _encode_visual_coarse(reg: VisualRegion) -> np.ndarray:
    return np.array(
        [reg.mean_a, reg.mean_rg, reg.mean_by, reg.fine_texture_energy], dtype=np.float32
    )


def _encode_visual_foveal(foveal_obs: OpticalSceneObservation) -> np.ndarray:
    a = foveal_obs.l1.luminance_a
    if a.shape[0] >= 4 and a.shape[1] >= 4:
        lap = (
            np.abs(
                4.0 * a[1:-1, 1:-1]
                - a[:-2, 1:-1]
                - a[2:, 1:-1]
                - a[1:-1, :-2]
                - a[1:-1, 2:]
            )
            / 4.0
        )
        lap_hf = float(np.mean(lap))
    else:
        lap_hf = 0.0
    dog_std = float(np.std(foveal_obs.l1.dog_fine))
    rg_std = float(np.std(foveal_obs.l1.rg_opponent))
    by_std = float(np.std(foveal_obs.l1.by_opponent))
    return np.array([lap_hf, dog_std, rg_std, by_std], dtype=np.float32)


def _encode_mech(mech: MechanicalObservation) -> np.ndarray:
    return np.array(
        [
            float(np.log10(max(mech.effective_stiffness_n_m, 1.0))),
            float(mech.damping_ratio),
            float(mech.residual_plastic_ratio),
        ],
        dtype=np.float32,
    )


def _encode_acoustic(acoust: AcousticObservation) -> np.ndarray:
    return np.array(
        [
            float(np.log10(max(acoust.peak_frequency_hz, 20.0))),
            float(min(acoust.spectral_sharpness_q, 20.0) / 10.0),
            float(np.log10(max(acoust.decay_rate_per_sec, 1.0))),
        ],
        dtype=np.float32,
    )


class CrossModalCorrelationGraph:
    """
    Blank-slate empirical correlation learner connecting visual invariants to
    physical mechanical & acoustic measurements.
    """

    SURPRISE_THRESHOLD: float = 0.75
    MATCH_DIST_THRESHOLD: float = 0.65
    VISUAL_WEIGHTS = np.array([0.45, 2.5, 2.5, 1.4], dtype=np.float32)
    FOVEAL_WEIGHTS = np.array([4.0, 1.0, 0.5, 0.5], dtype=np.float32)

    def __init__(self) -> None:
        self.prototypes: List[CrossModalPrototype] = []
        self.refinements: List[CorrelationRefinementEvent] = []
        self._episodes: List[SensoryEpisode] = []

    def _visual_distance(
        self,
        coarse_vec: np.ndarray,
        proto: CrossModalPrototype,
        foveal_vec: Optional[np.ndarray] = None,
    ) -> float:
        d_coarse = (coarse_vec - proto.visual_coarse_center) * self.VISUAL_WEIGHTS
        dist_sq = float(np.sum(d_coarse**2))
        if foveal_vec is not None and proto.visual_foveal_center is not None:
            d_fov = (foveal_vec - proto.visual_foveal_center) * self.FOVEAL_WEIGHTS
            dist_sq += float(np.sum(d_fov**2))
        return float(np.sqrt(dist_sq))

    def predict_from_vision(
        self,
        visual_coarse_vec: np.ndarray,
        visual_foveal_vec: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, Optional[str]]:
        if not self.prototypes:
            return None, None, 1.0, None

        best_proto = None
        min_dist = 1e9
        for proto in self.prototypes:
            dist = self._visual_distance(visual_coarse_vec, proto, visual_foveal_vec)
            if dist < min_dist:
                min_dist = dist
                best_proto = proto

        assert best_proto is not None
        uncertainty = float(1.0 - np.exp(-min_dist / 0.55))
        return best_proto.mech_center, best_proto.acoustic_center, uncertainty, best_proto.prototype_id

    def update_foveal_signature(self, prototype_id: str, foveal_vec: np.ndarray) -> None:
        for proto in self.prototypes:
            if proto.prototype_id == prototype_id:
                if proto.visual_foveal_center is None:
                    proto.visual_foveal_center = foveal_vec.copy()
                else:
                    n = float(max(1, proto.confirmation_count))
                    proto.visual_foveal_center = (
                        proto.visual_foveal_center * (n - 1.0) + foveal_vec
                    ) / n
                return

    def update_with_episode(self, episode: SensoryEpisode) -> Tuple[float, Optional[CorrelationRefinementEvent]]:
        self._episodes.append(episode)
        if episode.mechanical_obs is None and episode.acoustic_obs is None:
            return 0.0, None

        actual_mech = _encode_mech(episode.mechanical_obs) if episode.mechanical_obs else None
        actual_acoust = _encode_acoustic(episode.acoustic_obs) if episode.acoustic_obs else None

        best_proto: Optional[CrossModalPrototype] = None
        min_dist = 1e9
        for proto in self.prototypes:
            d = self._visual_distance(episode.visual_coarse_vec, proto, episode.visual_foveal_vec)
            if d < min_dist:
                min_dist = d
                best_proto = proto

        # No prototype close enough -> form new independent regularity cluster
        if best_proto is None or min_dist > self.MATCH_DIST_THRESHOLD:
            new_id = f"proto_{len(self.prototypes)}"
            new_proto = CrossModalPrototype(
                prototype_id=new_id,
                visual_coarse_center=episode.visual_coarse_vec.copy(),
                visual_foveal_center=(
                    episode.visual_foveal_vec.copy() if episode.visual_foveal_vec is not None else None
                ),
                mech_center=actual_mech.copy() if actual_mech is not None else None,
                acoustic_center=actual_acoust.copy() if actual_acoust is not None else None,
                confirmation_count=1,
                member_steps=[episode.step],
            )
            self.prototypes.append(new_proto)
            episode.matched_prototype_id = new_id
            return 0.0, None

        # Compute surprise error against matched prototype
        errs = []
        if actual_mech is not None and best_proto.mech_center is not None:
            errs.append(float(np.linalg.norm(actual_mech - best_proto.mech_center)))
        if actual_acoust is not None and best_proto.acoustic_center is not None:
            errs.append(float(np.linalg.norm(actual_acoust - best_proto.acoustic_center)))
        surprise_err = float(max(errs)) if errs else 0.0

        # If prediction holds within tolerance -> confirm & refine running centroid
        if surprise_err <= self.SURPRISE_THRESHOLD:
            n = float(best_proto.confirmation_count)
            best_proto.visual_coarse_center = (
                best_proto.visual_coarse_center * n + episode.visual_coarse_vec
            ) / (n + 1.0)
            if episode.visual_foveal_vec is not None:
                if best_proto.visual_foveal_center is None:
                    best_proto.visual_foveal_center = episode.visual_foveal_vec.copy()
                else:
                    best_proto.visual_foveal_center = (
                        best_proto.visual_foveal_center * n + episode.visual_foveal_vec
                    ) / (n + 1.0)
            if actual_mech is not None:
                if best_proto.mech_center is None:
                    best_proto.mech_center = actual_mech.copy()
                else:
                    best_proto.mech_center = (best_proto.mech_center * n + actual_mech) / (n + 1.0)
            if actual_acoust is not None:
                if best_proto.acoustic_center is None:
                    best_proto.acoustic_center = actual_acoust.copy()
                else:
                    best_proto.acoustic_center = (best_proto.acoustic_center * n + actual_acoust) / (
                        n + 1.0
                    )
            best_proto.confirmation_count += 1
            best_proto.member_steps.append(episode.step)
            episode.matched_prototype_id = best_proto.prototype_id
            episode.prediction_surprise = surprise_err
            return surprise_err, None

        # Counter-example contradicts prototype -> DO NOT average away! Split (Mitosis).
        child_id = f"proto_{len(self.prototypes)}_split"
        sep_note = (
            f"Visual match to {best_proto.prototype_id} (d_vis={min_dist:.3f}) predicted "
            f"mech={best_proto.mech_center.round(2).tolist() if best_proto.mech_center is not None else None}, "
            f"but measured mech={actual_mech.round(2).tolist() if actual_mech is not None else None} "
            f"(surprise={surprise_err:.3f} > {self.SURPRISE_THRESHOLD}); split child prototype rather than averaging."
        )
        child_proto = CrossModalPrototype(
            prototype_id=child_id,
            visual_coarse_center=episode.visual_coarse_vec.copy(),
            visual_foveal_center=(
                episode.visual_foveal_vec.copy() if episode.visual_foveal_vec is not None else None
            ),
            mech_center=actual_mech.copy() if actual_mech is not None else None,
            acoustic_center=actual_acoust.copy() if actual_acoust is not None else None,
            confirmation_count=1,
            member_steps=[episode.step],
            parent_prototype_id=best_proto.prototype_id,
            split_reason=sep_note,
        )
        self.prototypes.append(child_proto)
        ref_evt = CorrelationRefinementEvent(
            step=episode.step,
            parent_prototype_id=best_proto.prototype_id,
            new_prototype_id=child_id,
            surprise_error=surprise_err,
            separating_axis_note=sep_note,
        )
        self.refinements.append(ref_evt)
        episode.matched_prototype_id = child_id
        episode.prediction_surprise = surprise_err
        return surprise_err, ref_evt

    def compute_empirical_correlation_matrix(self) -> Dict[str, Any]:
        """Compute empirical Pearson correlations across all multi-modal episodes."""
        valid = [
            ep
            for ep in self._episodes
            if ep.mechanical_obs is not None and ep.acoustic_obs is not None
        ]
        feature_names = [
            "vis_mean_a",
            "vis_mean_rg",
            "vis_mean_by",
            "vis_fine_tex",
            "mech_log10_k",
            "mech_damping",
            "acoust_log10_freq",
            "acoust_log10_decay",
        ]
        if len(valid) < 3:
            return {
                "sample_count": len(valid),
                "feature_names": feature_names,
                "matrix": None,
                "stiffness_vs_freq_r": 0.0,
            }

        rows = []
        for ep in valid:
            vc = ep.visual_coarse_vec
            m = _encode_mech(ep.mechanical_obs)  # type: ignore[arg-type]
            a = _encode_acoustic(ep.acoustic_obs)  # type: ignore[arg-type]
            rows.append([vc[0], vc[1], vc[2], vc[3], m[0], m[1], a[0], a[2]])

        mat = np.asarray(rows, dtype=np.float64)
        n_f = mat.shape[1]
        corr = np.eye(n_f, dtype=np.float64)
        for i in range(n_f):
            for j in range(i + 1, n_f):
                si = float(np.std(mat[:, i]))
                sj = float(np.std(mat[:, j]))
                if si > 1e-8 and sj > 1e-8:
                    r = float(np.corrcoef(mat[:, i], mat[:, j])[0, 1])
                else:
                    r = 0.0
                corr[i, j] = r
                corr[j, i] = r

        idx_k = feature_names.index("mech_log10_k")
        idx_f = feature_names.index("acoust_log10_freq")
        return {
            "sample_count": len(valid),
            "feature_names": feature_names,
            "matrix": corr,
            "stiffness_vs_freq_r": float(corr[idx_k, idx_f]),
        }


class EpistemicAttentionField:
    """Computes autonomous epistemic priority across visual regions and ambiguous edges."""

    def score_candidates(
        self,
        trackers: List[_TargetTracker],
        correlation_graph: CrossModalCorrelationGraph,
        supports_physical: bool,
        scene_median_a: float,
    ) -> List[AttentionCandidate]:
        candidates: List[AttentionCandidate] = []

        for tr in trackers:
            if tr.target_kind == "BOUNDARY":
                ambig = 1.6 if (tr.boundary is not None and not tr.boundary_push_tested) else 0.0
                surp = 0.0
                unc = 0.0
                res_gap = 0.0
                ior = 1.5 * tr.ior_count
                action = "MECHANICAL_PUSH_AND_WATCH" if supports_physical else "FOVEAL_FIXATE"
                rationale = (
                    f"Ambiguous achromatic edge ({tr.boundary.region_a}, {tr.boundary.region_b}) "
                    "requires motor-visual displacement test."
                    if ambig > 0
                    else "Boundary ambiguity resolved."
                )
            else:
                assert tr.region is not None
                reg = tr.region
                vis_coarse = _encode_visual_coarse(reg)
                _, _, pred_unc, matched_id = correlation_graph.predict_from_vision(
                    vis_coarse, tr.foveal_vec
                )
                surp = 3.0 * tr.surprise_flag
                ambig = 0.0

                if not supports_physical:
                    unc = 0.0
                elif not tr.mech_probed:
                    # Novel objects score highest; familiar unprobed objects stay high enough to verify!
                    unc = float(0.65 + 0.55 * pred_unc)
                else:
                    unc = 0.0

                if not tr.foveated:
                    dark_bonus = 0.35 if reg.mean_a < scene_median_a else 0.0
                    res_gap = float(0.45 + 0.30 * min(reg.fine_texture_energy, 1.5) + dark_bonus)
                else:
                    res_gap = 0.0

                ior = 1.3 * tr.ior_count

                # Choose which organ best resolves the highest active uncertainty on this region
                if tr.surprise_flag > 0.0 and not tr.foveated:
                    action = "FOVEAL_FIXATE"
                    rationale = (
                        f"Surprise capture ({tr.surprise_flag:.2f}): periphery failed to predict contact "
                        f"on {reg.region_id}; directing native 1:1 fovea to inspect microstructure."
                    )
                elif supports_physical and not tr.mech_probed:
                    action = "MECHANICAL_AND_ACOUSTIC_PROBE"
                    rationale = (
                        f"Region {reg.region_id} unprobed (vis uncertainty={pred_unc:.2f}"
                        + (f", matched {matched_id}" if matched_id else ", blank slate")
                        + "); probing physical contact & impact resonance."
                    )
                elif not tr.foveated:
                    action = "FOVEAL_FIXATE"
                    rationale = (
                        f"Region {reg.region_id} unexplored at 1:1 foveal resolution "
                        f"(fine_tex={reg.fine_texture_energy:.2f}, mean_A={reg.mean_a:.2f})."
                    )
                else:
                    action = "FOVEAL_FIXATE"
                    rationale = f"Region {reg.region_id} already inspected (IOR={ior:.2f})."

            total = float(surp + ambig + unc + res_gap - ior)
            candidates.append(
                AttentionCandidate(
                    target_id=tr.target_id,
                    target_kind=tr.target_kind,
                    y_norm=tr.y_norm,
                    x_norm=tr.x_norm,
                    surprise_score=surp,
                    ambiguity_score=ambig,
                    cross_modal_uncertainty=unc,
                    resolution_gap=res_gap,
                    ior_penalty=ior,
                    total_priority=total,
                    recommended_action=action,
                    rationale=rationale,
                )
            )

        candidates.sort(key=lambda c: (-c.total_priority, c.target_id))
        return candidates


class ActiveEpistemicExplorer:
    """Self-directing epistemic loop uniting vision, foveation, mechanics, acoustics, and CausalField."""

    def __init__(
        self,
        eye: Optional[OcularScaleOrgan] = None,
        mechano_acoustic: Optional[MechanoAcousticOrgan] = None,
        reasoner: Optional[VisualCausalReasoner] = None,
    ) -> None:
        self.eye = eye if eye is not None else OcularScaleOrgan()
        self.mechano_acoustic = (
            mechano_acoustic if mechano_acoustic is not None else MechanoAcousticOrgan()
        )
        self.reasoner = reasoner if reasoner is not None else VisualCausalReasoner(self.eye)
        self.attention = EpistemicAttentionField()

    def explore(
        self, world: SensorimotorWorldProtocol, num_steps: int = 12
    ) -> ExplorationTrace:
        periph_rgb = world.observe_periphery()
        periph_obs = self.eye.observe(periph_rgb)
        h, w = periph_obs.shape_hw
        supports_phys = world.supports_physical_probing()
        scene_med_a = float(np.median(periph_obs.l1.luminance_a))

        correlation_graph = CrossModalCorrelationGraph()

        # Build trackers for non-backdrop regions + ambiguous shadow candidate boundaries
        region_by_id: Dict[int, VisualRegion] = {r.region_id: r for r in periph_obs.regions}
        trackers: List[_TargetTracker] = []
        area_cutoff = 0.45 if supports_phys else 0.65

        for reg in periph_obs.regions:
            if reg.area_ratio > area_cutoff and len(periph_obs.regions) >= 4:
                continue
            yn = float(reg.centroid_yx[0] / max(1.0, float(h)))
            xn = float(reg.centroid_yx[1] / max(1.0, float(w)))
            trackers.append(
                _TargetTracker(
                    target_id=f"region_{reg.region_id}",
                    target_kind="REGION",
                    y_norm=yn,
                    x_norm=xn,
                    region=reg,
                )
            )

        for b_edge in periph_obs.boundaries:
            if b_edge.optical_cause != "ILLUMINATION_SHADOW_CANDIDATE":
                continue
            ra = region_by_id.get(b_edge.region_a)
            rb = region_by_id.get(b_edge.region_b)
            if ra is None or rb is None:
                continue
            # Prefer pushing from the higher-luminance side (less shadow suppression)
            push_reg = ra if ra.mean_a >= rb.mean_a else rb
            other_reg = rb if push_reg is ra else ra
            # Skip backdrop-against-body false shadow pairings if one side is backdrop
            if push_reg.area_ratio > 0.45 or other_reg.area_ratio > 0.45:
                continue
            yn = float(push_reg.centroid_yx[0] / max(1.0, float(h)))
            xn = float(push_reg.centroid_yx[1] / max(1.0, float(w)))
            trackers.append(
                _TargetTracker(
                    target_id=f"shadow_edge_{b_edge.region_a}_{b_edge.region_b}",
                    target_kind="BOUNDARY",
                    y_norm=yn,
                    x_norm=xn,
                    boundary=b_edge,
                    side_a_region=push_reg,
                    side_b_region=other_reg,
                )
            )

        tracker_map: Dict[str, _TargetTracker] = {t.target_id: t for t in trackers}
        candidates_per_step: List[List[AttentionCandidate]] = []
        decisions: List[ExplorationStepDecision] = []
        episodes: List[SensoryEpisode] = []
        resolved_shadows: Dict[Tuple[int, int], str] = {}

        for step_idx in range(1, num_steps + 1):
            cands = self.attention.score_candidates(
                trackers=trackers,
                correlation_graph=correlation_graph,
                supports_physical=supports_phys,
                scene_median_a=scene_med_a,
            )
            candidates_per_step.append(cands)
            if not cands:
                break

            top = cands[0]
            tr = tracker_map[top.target_id]
            episode: Optional[SensoryEpisode] = None

            if top.recommended_action == "MECHANICAL_PUSH_AND_WATCH" and supports_phys:
                assert tr.boundary is not None and tr.side_a_region is not None and tr.side_b_region is not None
                f_t, x_t, post_rgb = world.mechanical_push(tr.y_norm, tr.x_norm, force_n=15.0, dir_yx=(0.0, 1.0))
                mech_obs = self.mechano_acoustic.observe_push(f_t, x_t)
                post_obs = self.eye.observe(post_rgb)
                temporal = self.eye.observe_transition(periph_obs, post_obs)

                mask_a = periph_obs.region_labels == tr.side_a_region.region_id
                mask_b = periph_obs.region_labels == tr.side_b_region.region_id
                va = np.array([np.median(temporal.flow_yx[mask_a, 0]), np.median(temporal.flow_yx[mask_a, 1])])
                vb = np.array([np.median(temporal.flow_yx[mask_b, 0]), np.median(temporal.flow_yx[mask_b, 1])])
                spd_a = float(np.linalg.norm(va))
                spd_b = float(np.linalg.norm(vb))
                vel_diff = float(np.linalg.norm(va - vb))

                if spd_a > 2.5 and spd_b > 2.5 and vel_diff < 3.5:
                    verdict = "CONFIRMED_RIGID_UNIFIED_BODY_UNDER_SHADOW"
                elif spd_a > 2.5 and spd_b < 1.5:
                    verdict = "CONFIRMED_SEPARATE_MECHANICAL_BODIES"
                else:
                    verdict = "INCONCLUSIVE_LOW_DISPLACEMENT"

                key_pair = (min(tr.boundary.region_a, tr.boundary.region_b), max(tr.boundary.region_a, tr.boundary.region_b))
                resolved_shadows[key_pair] = verdict
                tr.boundary_push_tested = True
                tr.boundary_verdict = verdict
                tr.ior_count += 2

                vis_c = _encode_visual_coarse(tr.side_a_region)
                episode = SensoryEpisode(
                    step=step_idx,
                    target_id=tr.target_id,
                    target_yx_norm=(tr.y_norm, tr.x_norm),
                    action_type="MECHANICAL_PUSH_AND_WATCH",
                    visual_coarse_vec=vis_c,
                    visual_foveal_vec=None,
                    mechanical_obs=mech_obs,
                    acoustic_obs=None,
                    motor_visual_transition=temporal,
                    predicted_mech_vec=None,
                    predicted_acoustic_vec=None,
                    prediction_surprise=0.0,
                    matched_prototype_id=None,
                )
                episodes.append(episode)
                outcome_summary = (
                    f"{verdict}: side_a speed={spd_a:.2f}px, side_b speed={spd_b:.2f}px, |vA-vB|={vel_diff:.2f}px"
                )

            elif top.recommended_action == "MECHANICAL_AND_ACOUSTIC_PROBE" and supports_phys:
                assert tr.region is not None
                vis_c = _encode_visual_coarse(tr.region)
                pred_m, pred_a, _, matched_id = correlation_graph.predict_from_vision(vis_c, tr.foveal_vec)

                f_t, x_t, post_rgb = world.mechanical_push(tr.y_norm, tr.x_norm, force_n=15.0, dir_yx=(0.0, 1.0))
                mech_obs = self.mechano_acoustic.observe_push(f_t, x_t)
                wave = world.acoustic_tap(tr.y_norm, tr.x_norm, impulse_n=10.0)
                acoust_obs = self.mechano_acoustic.observe_tap(wave)

                tr.mech_probed = True
                tr.acoustic_probed = True

                episode = SensoryEpisode(
                    step=step_idx,
                    target_id=tr.target_id,
                    target_yx_norm=(tr.y_norm, tr.x_norm),
                    action_type="MECHANICAL_AND_ACOUSTIC_PROBE",
                    visual_coarse_vec=vis_c,
                    visual_foveal_vec=tr.foveal_vec,
                    mechanical_obs=mech_obs,
                    acoustic_obs=acoust_obs,
                    motor_visual_transition=None,
                    predicted_mech_vec=pred_m,
                    predicted_acoustic_vec=pred_a,
                    prediction_surprise=0.0,
                    matched_prototype_id=matched_id,
                )
                surprise_err, ref_evt = correlation_graph.update_with_episode(episode)
                tr.last_prototype_id = episode.matched_prototype_id
                episodes.append(episode)

                if ref_evt is not None and not tr.foveated:
                    # Bottom-up surprise hijacks next step attention without increasing IOR
                    tr.surprise_flag = surprise_err
                else:
                    tr.ior_count += 1

                outcome_summary = (
                    f"Probed mech(k={mech_obs.effective_stiffness_n_m:.0f} N/m, {mech_obs.mechanical_regime}), "
                    f"acoust(f={acoust_obs.peak_frequency_hz:.0f} Hz, {acoust_obs.acoustic_regime}), "
                    f"surprise={surprise_err:.3f}, proto={episode.matched_prototype_id}"
                    + (" [MITOSIS TRIGGERED]" if ref_evt else "")
                )

            else:  # FOVEAL_FIXATE
                fov_rgb = world.foveate(tr.y_norm, tr.x_norm, fov_frac=0.14)
                fov_obs = self.eye.observe(fov_rgb, max_width=240)
                fov_vec = _encode_visual_foveal(fov_obs)
                tr.foveated = True
                tr.foveal_vec = fov_vec
                if tr.last_prototype_id is not None:
                    correlation_graph.update_foveal_signature(tr.last_prototype_id, fov_vec)
                if tr.surprise_flag > 0.0:
                    tr.surprise_flag = 0.0
                tr.ior_count += 1

                vis_c = _encode_visual_coarse(tr.region) if tr.region is not None else np.zeros(4, dtype=np.float32)
                episode = SensoryEpisode(
                    step=step_idx,
                    target_id=tr.target_id,
                    target_yx_norm=(tr.y_norm, tr.x_norm),
                    action_type="FOVEAL_FIXATE",
                    visual_coarse_vec=vis_c,
                    visual_foveal_vec=fov_vec,
                    mechanical_obs=None,
                    acoustic_obs=None,
                    motor_visual_transition=None,
                    predicted_mech_vec=None,
                    predicted_acoustic_vec=None,
                    prediction_surprise=0.0,
                    matched_prototype_id=tr.last_prototype_id,
                )
                episodes.append(episode)
                outcome_summary = (
                    f"Foveated at ({tr.y_norm:.2f}, {tr.x_norm:.2f}): lap_hf={fov_vec[0]:.3f}, "
                    f"dog_std={fov_vec[1]:.3f}, foveal_regions={fov_obs.l2.hierarchical_region_count}"
                )

            decisions.append(
                ExplorationStepDecision(
                    step=step_idx,
                    chosen=top,
                    episode=episode,
                    outcome_summary=outcome_summary,
                )
            )

        corr_matrix = correlation_graph.compute_empirical_correlation_matrix()
        final_field = self._project_epistemic_graph_to_causal_field(
            periph_obs=periph_obs,
            correlation_graph=correlation_graph,
            trackers=trackers,
            resolved_shadows=resolved_shadows,
        )
        audit = self.reasoner.audit_organ_causality(periph_obs)

        return ExplorationTrace(
            steps=len(decisions),
            periphery_obs=periph_obs,
            candidates_per_step=candidates_per_step,
            decisions=decisions,
            episodes=episodes,
            prototypes=correlation_graph.prototypes,
            refinements=correlation_graph.refinements,
            resolved_shadow_boundaries=resolved_shadows,
            correlation_matrix=corr_matrix,
            final_causal_field=final_field,
            organ_audit=audit,
        )

    def _project_epistemic_graph_to_causal_field(
        self,
        periph_obs: OpticalSceneObservation,
        correlation_graph: CrossModalCorrelationGraph,
        trackers: List[_TargetTracker],
        resolved_shadows: Dict[Tuple[int, int], str],
    ) -> CausalField:
        field_obj = self.reasoner.project_to_causal_field(periph_obs)

        # Heal shadow boundaries proved to belong to a unified physical rigid body
        for beam in field_obj.beams:
            if beam.source_id.startswith("region_") and beam.target_id.startswith("region_"):
                ra = int(beam.source_id.split("_")[1])
                rb = int(beam.target_id.split("_")[1])
                key = (min(ra, rb), max(ra, rb))
                if resolved_shadows.get(key) == "CONFIRMED_RIGID_UNIFIED_BODY_UNDER_SHADOW":
                    beam.strength = 2.8
                    beam.is_broken = False

        # Add discovered cross-modal prototype voxels and bind probed regions to them
        for idx, proto in enumerate(correlation_graph.prototypes):
            p_vid = f"proto_{proto.prototype_id}"
            mech_part = (
                proto.mech_center
                if proto.mech_center is not None
                else np.zeros(3, dtype=np.float32)
            )
            ac_part = (
                proto.acoustic_center
                if proto.acoustic_center is not None
                else np.zeros(3, dtype=np.float32)
            )
            vec = np.concatenate([proto.visual_coarse_center[:3], mech_part[:2], ac_part[:1]]).astype(
                np.float32
            )
            v_proto = InformationVoxel(
                id=p_vid,
                content={
                    "type": "cross_modal_prototype",
                    "prototype_id": proto.prototype_id,
                    "confirmations": proto.confirmation_count,
                    "parent": proto.parent_prototype_id,
                },
                tensor=vec,
                mass=max(0.5, float(proto.confirmation_count)),
                position=np.array([0.2 + 0.2 * idx, 0.9, 0.8], dtype=np.float32),
                scale_level=8.0,
            )
            field_obj.add_voxel(v_proto)

        for tr in trackers:
            if tr.target_kind == "REGION" and tr.last_prototype_id is not None:
                r_vid = tr.target_id
                p_vid = f"proto_{tr.last_prototype_id}"
                if r_vid in field_obj.voxels and p_vid in field_obj.voxels:
                    field_obj.link_voxels(r_vid, p_vid, strength=2.2)

        field_obj.step(dt=0.05)
        return field_obj
