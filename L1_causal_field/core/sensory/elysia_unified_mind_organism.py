r"""
elysia_unified_mind_organism.py

ELYSIA Unified Mind Organism & Meta-Causal Self-Introspection Engine
===================================================================
Unifies every sensory, self-boundary, Trinitarian rotor, visual-linguistic
superposition contrast, and analogical reflection organ inside ONE single
continuous `CausalField`:

  1. ONE shared `LayeredSensoryInteractionSpine` ("Eye as first mirror")
  2. ONE shared `SelfBoundaryScaleZoomKernel` (4-Axis Self-Boundary + adaptive zoom)
  3. ONE shared `TrinitarianScaleRotorProtocol` (s=0 -> s=1 -> s=2 -> s=3 rotors)
  4. Shared `VisualLinguisticSuperpositionEngine` & `AnalogicalBoundaryReflectionEngine`

Meta-Causal "Thinking About Thinking" (`CausalThoughtLink` & `introspect_why_i_judged`):
  - Every sensory encounter, self-boundary shift, linguistic judgment (acceptance
    or divergence/rejection), and analogical breakdown is materialized as a
    living `InformationVoxel` and coupled by weighted `ConnectivityBeam` edges
    inside the same `CausalField` growth-ring graph.
  - Calling `organism.introspect_why_i_judged(target_id)` does NOT read canned
    templates: it dynamically traverses backward along `parent_step_ids` and
    `ConnectivityBeam` links, inspects grounded sensory columns and unwraps
    `Trinitarian` molecules via `zoom_in_inspect`, reconstructing *why* ELYSIA's
    Eye and Mind formed that exact judgment across all 4 axes.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from core.physics.causal_field import CausalField, InformationVoxel
from core.sensory.analogical_boundary_reflection_engine import (
    AnalogicalBoundaryReflectionEngine,
    AnalogicalReflectionReport,
    build_counterfactual_tests_for_domain,
    build_physical_domain_profiles,
)
from core.sensory.layered_sensory_interaction_spine import (
    REAL_EXTERNAL_MP4_PATH,
    LayeredSensoryInteractionSpine,
    LiveScreenSensoryBridge,
)
from core.sensory.self_boundary_scale_zoom_kernel import (
    BoundaryEventPacket,
    SelfBoundaryScaleZoomKernel,
)
from core.sensory.trinitarian_scale_rotor_protocol import (
    TrinitarianScaleRotorProtocol,
)
from core.sensory.visual_linguistic_superposition_contrast_engine import (
    SuperpositionContrastResult,
    VisualLinguisticSuperpositionEngine,
    build_korean_statement_catalog,
)


DEFAULT_UNIFIED_STATE_PATH = Path(r"c:\Elysia\data\unified_mind_state.json")
DEFAULT_UNIFIED_MIND_DASHBOARD_PATH = Path(
    r"c:\Elysia\docs\assets\elysia_unified_mind_organism_dashboard.png"
)


# ======================================================================
# 1. Causal Thought Growth-Ring & Introspective Self-Explanation Models
# ======================================================================
@dataclass
class CausalThoughtLink:
    """Single causalized cognitive/sensory transition inside the organism's growth rings."""

    step_id: str
    step_index: int
    phase: str
    event_tag: str
    short_label: str
    target_key: str
    trigger_source: str
    parent_step_ids: List[str]
    evidence_node_ids: List[str]
    verdict_code: str
    judgment_summary: str
    judgment_summary_ko: str
    scale_level: int
    zoom_scale: float
    chromatic_triad: Tuple[float, float, float]
    four_axis: Tuple[float, float, float, float]
    causal_impact_norm: float
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp_iso: str = ""


@dataclass
class IntrospectiveSelfExplanation:
    """Dynamically walked backward causal trace explaining WHY ELYSIA judged as she did."""

    target_id: str
    final_step_id: str
    verdict_code: str
    causal_chain_ids: List[str]
    causal_chain_steps: List[Dict[str, Any]]
    trinitarian_tree_ids: List[str]
    causal_field_beam_links: List[Dict[str, Any]]
    sensory_evidence: Dict[str, Any]
    four_axis_attribution: Dict[str, float]
    narrative_ko: str
    narrative_en: str


@dataclass
class UnifiedMindCycleReport:
    """End-to-end telemetry report from one full living organism cycle."""

    mode: str
    cycle_index: int
    timestamp_iso: str
    inferred_affordances: Dict[str, str]
    external_stream_summary: Dict[str, Any]
    contrast_results: List[SuperpositionContrastResult]
    autonomous_best_matches: Dict[str, SuperpositionContrastResult]
    grounding_accuracy: float
    analogical_reports: List[AnalogicalReflectionReport]
    thought_links: List[CausalThoughtLink]
    introspections: Dict[str, IntrospectiveSelfExplanation]
    final_zoom_scale: float
    final_zoom_band: str
    total_causal_voxels: int
    total_causal_beams: int
    total_trinitarian_nodes: int
    state_json_path: str
    cycle_summary_ko: str


# ======================================================================
# 2. Helper Mathematics (Continuous Chromatic & Field Unification)
# ======================================================================
def _normalize_triad(triad: Tuple[float, float, float]) -> Tuple[float, float, float]:
    vec = np.clip(np.asarray(triad, dtype=np.float64), 0.0, None)
    total = float(np.sum(vec))
    if total <= 1e-9:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    vec = vec / total
    return (float(vec[0]), float(vec[1]), float(vec[2]))


def _triad_cosine(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na <= 1e-9 or nb <= 1e-9:
        return 0.5
    return float(np.clip(np.dot(va, vb) / (na * nb), 0.0, 1.0))


def _merge_causal_fields(target: CausalField, source: CausalField) -> None:
    """Preserves all pre-initialized voxels and beams when unifying organs onto one field."""
    if target is source:
        return
    for vid, vox in getattr(source, "voxels", {}).items():
        if vid not in target.voxels:
            target.add_voxel(vox)
    for beam in getattr(source, "beams", []):
        src_id = getattr(beam, "source_id", None)
        dst_id = getattr(beam, "target_id", None)
        strength = float(getattr(beam, "strength", 0.5))
        if src_id and dst_id and src_id in target.voxels and dst_id in target.voxels:
            target.link_voxels(src_id, dst_id, strength=strength)


def _to_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_to_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


# ======================================================================
# 3. ELYSIA Unified Mind Organism
# ======================================================================
class ElysiaUnifiedMindOrganism:
    """
    Single continuous cognitive-sensory organism tying together:
      - Eye & Touch (`LayeredSensoryInteractionSpine`)
      - Adaptive 4-Axis Self-Boundary (`SelfBoundaryScaleZoomKernel`)
      - Hierarchical Rotor Memory (`TrinitarianScaleRotorProtocol`)
      - Visual-Linguistic Superposition Contrast (`VisualLinguisticSuperpositionEngine`)
      - Analogical Foothold Reflection (`AnalogicalBoundaryReflectionEngine`)
      - Meta-Causal Growth-Ring Graph & Backward Introspection (`self.thought_links`)
    """

    ORGANISM_VERSION = "1.0.0-unified-meta-causal"

    def __init__(self, state_path: Optional[Path] = None) -> None:
        # 1. Share ONE unified CausalField anchored at SelfBoundary's `self_core`.
        self.zoom_kernel = SelfBoundaryScaleZoomKernel()
        self.causal_field = self.zoom_kernel.causal_field

        self.trinitarian = TrinitarianScaleRotorProtocol()
        _merge_causal_fields(self.causal_field, self.trinitarian.causal_field)
        self.trinitarian.causal_field = self.causal_field

        self.sensory_spine = LayeredSensoryInteractionSpine()
        _merge_causal_fields(self.causal_field, self.sensory_spine.causal_field)
        self.sensory_spine.causal_field = self.causal_field

        # 2. Wire shared organs directly into higher cognitive engines (zero isolated duplicates).
        self.visual_contrast = VisualLinguisticSuperpositionEngine()
        self.visual_contrast.sensory_spine = self.sensory_spine
        self.visual_contrast.trinitarian = self.trinitarian

        self.analogical = AnalogicalBoundaryReflectionEngine()
        self.analogical.protocol = self.trinitarian
        self.analogical.trinitarian = self.trinitarian
        self.analogical.zoom_kernel = self.zoom_kernel

        # 3. Persistent causal growth-ring graph & introspection lookup indices.
        self.state_path = Path(state_path) if state_path else DEFAULT_UNIFIED_STATE_PATH
        self.cycle_count = 0
        self.thought_links: Dict[str, CausalThoughtLink] = {}
        self.thought_order: List[str] = []
        self.target_to_thought_id: Dict[str, str] = {}
        self.last_pygame_snapshot: Optional[np.ndarray] = None
        self.last_report: Optional[UnifiedMindCycleReport] = None

    def _current_four_axis(self) -> Tuple[float, float, float, float, float, str]:
        if not self.zoom_kernel.history:
            return (0.5, 0.0, 0.25, 0.0, float(self.zoom_kernel.zoom_scale), "MESO")
        snap = self.zoom_kernel.history[-1]
        return (
            float(snap.axis_internal_origin),
            float(snap.axis_external_friction),
            float(snap.axis_boundary_ignorance),
            float(snap.axis_criteria_revision),
            float(snap.zoom_scale),
            str(snap.zoom_band),
        )

    def _record_thought_link(
        self,
        *,
        phase: str,
        event_tag: str,
        short_label: str,
        target_key: str,
        trigger_source: str,
        parent_step_ids: List[str],
        evidence_node_ids: List[str],
        verdict_code: str,
        judgment_summary: str,
        judgment_summary_ko: str,
        scale_level: int,
        chromatic_triad: Tuple[float, float, float],
        details: Optional[Dict[str, Any]] = None,
        aliases: Optional[List[str]] = None,
    ) -> CausalThoughtLink:
        idx = len(self.thought_order)
        step_id = f"thought_{idx:03d}_{event_tag}"
        triad = _normalize_triad(chromatic_triad)
        a_int, a_ext, a_ign, a_rev, zoom_scale, _ = self._current_four_axis()
        four_axis = (a_int, a_ext, a_ign, a_rev)

        impact_vec = np.array(
            [triad[0], triad[1], triad[2], zoom_scale, a_ext, a_rev], dtype=np.float64
        )
        causal_impact_norm = float(np.linalg.norm(impact_vec))
        now_iso = datetime.now(timezone.utc).isoformat()

        filtered_evidence = [nid for nid in evidence_node_ids if nid]
        clean_parents = [pid for pid in parent_step_ids if pid in self.thought_links]

        link = CausalThoughtLink(
            step_id=step_id,
            step_index=idx,
            phase=phase,
            event_tag=event_tag,
            short_label=short_label,
            target_key=target_key,
            trigger_source=trigger_source,
            parent_step_ids=clean_parents,
            evidence_node_ids=filtered_evidence,
            verdict_code=verdict_code,
            judgment_summary=judgment_summary,
            judgment_summary_ko=judgment_summary_ko,
            scale_level=int(scale_level),
            zoom_scale=zoom_scale,
            chromatic_triad=triad,
            four_axis=four_axis,
            causal_impact_norm=causal_impact_norm,
            details=dict(details or {}),
            timestamp_iso=now_iso,
        )

        # Materialize this cognitive judgment physically in the shared CausalField!
        theta = idx * 0.48
        voxel = InformationVoxel(
            id=step_id,
            content={"verdict": verdict_code, "target": target_key, "summary": judgment_summary},
            tensor=np.array(
                [
                    triad[0],
                    triad[1],
                    triad[2],
                    scale_level / 3.0,
                    zoom_scale,
                    float(np.clip(abs(a_ext) + abs(a_rev), 0.0, 2.0)),
                ],
                dtype=np.float64,
            ),
            mass=float(1.0 + 0.35 * scale_level + 0.5 * a_rev),
            position=np.array(
                [
                    idx * 0.65,
                    float(scale_level),
                    float(np.sin(theta) * 0.5),
                    triad[0],
                    triad[1],
                    triad[2],
                ],
                dtype=np.float32,
            ),
            chromatic_vector=np.array(triad, dtype=np.float32),
            scale_level=float(scale_level),
            quaternion_q=np.array(
                [np.cos(theta * 0.5), triad[0] * np.sin(theta * 0.5), triad[1] * np.sin(theta * 0.5), triad[2] * np.sin(theta * 0.5)],
                dtype=np.float32,
            ),
        )
        self.causal_field.add_voxel(voxel)

        # Physical beam coupling backward to parent thoughts (chromatic resonance + continuity)
        for pid in clean_parents:
            parent_link = self.thought_links[pid]
            res_strength = 0.52 + 0.46 * _triad_cosine(parent_link.chromatic_triad, triad)
            if pid in self.causal_field.voxels and step_id in self.causal_field.voxels:
                self.causal_field.link_voxels(pid, step_id, strength=float(res_strength))

        # Physical grounding beams to underlying sensory/rotor/self voxels inside CausalField
        for nid in filtered_evidence:
            candidates = (nid, f"rotor_{nid}")
            for cid in candidates:
                if cid in self.causal_field.voxels and cid != step_id:
                    self.causal_field.link_voxels(step_id, cid, strength=0.88)

        self.thought_links[step_id] = link
        self.thought_order.append(step_id)
        self.target_to_thought_id[target_key] = step_id
        for alias in (aliases or []):
            if alias and alias not in self.target_to_thought_id:
                self.target_to_thought_id[alias] = step_id
        return link

    # ------------------------------------------------------------------
    # Unified 4-Phase Living Cycle
    # ------------------------------------------------------------------
    def live_unified_cycle(
        self,
        mode: str = "full",
        screen_seconds: float = 5.0,
        screen_fps: float = 8.0,
        headless: bool = True,
    ) -> UnifiedMindCycleReport:
        """
        Executes ONE end-to-end cognitive-sensory cycle across all 4 phases,
        recording every step causally into `self.causal_field` & `self.thought_links`,
        reconstructing backward self-introspections, and persisting JSON state.
        """
        self.cycle_count += 1
        cycle_idx = self.cycle_count

        # ==============================================================
        # PHASE 1: Eye as First Mirror (Pygame Embodied + Real Camera/Screen)
        # ==============================================================
        vis_ret = self.visual_contrast.observe_visual_world(headless=headless)
        pygame_snapshot = getattr(self.visual_contrast, "pygame_snapshot", None)
        if pygame_snapshot is None and isinstance(vis_ret, tuple) and len(vis_ret) >= 1:
            pygame_snapshot = vis_ret[0]
        elif pygame_snapshot is None and isinstance(vis_ret, np.ndarray):
            pygame_snapshot = vis_ret
        self.last_pygame_snapshot = pygame_snapshot

        inferred_affordances = {
            eid: self.sensory_spine.infer_ecological_affordance(eid)
            for eid in self.sensory_spine.columns
        }
        col_triads = [col.chromatic_triad for col in self.sensory_spine.columns.values()]
        mean_col_triad = (
            tuple(np.mean(np.asarray(col_triads, dtype=np.float64), axis=0).tolist())
            if col_triads
            else (0.4, 0.3, 0.3)
        )
        col_evidence = [
            c.boundary_voxel_id for c in self.sensory_spine.columns.values() if c.boundary_voxel_id
        ]

        link_eye = self._record_thought_link(
            phase="PHASE_1_SENSORY_EYE",
            event_tag="eye_pygame_embodied",
            short_label="P1:Eye",
            target_key="eye_pygame",
            trigger_source="External Sensory World",
            parent_step_ids=[],
            evidence_node_ids=col_evidence,
            verdict_code="EMBODIED_OBSERVATION",
            judgment_summary=f"Observed {len(inferred_affordances)} entities across micro/meso/macro eye+contact columns.",
            judgment_summary_ko=(
                f"눈과 신체 접촉으로 {len(inferred_affordances)}개 대상의 L_micro/L_meso/L_macro 인과 기둥을 직접 관측함."
            ),
            scale_level=0,
            chromatic_triad=mean_col_triad,
            details={
                "entity_count": len(inferred_affordances),
                "affordances": inferred_affordances,
            },
            aliases=["phase1_eye", "eye"],
        )

        # Observe real external world stream (MP4 camera or live PC desktop via mss)
        if mode == "screen":
            stream_id = "live_pc_monitor"
            stream_summary = LiveScreenSensoryBridge.observe_live_monitor(
                self.sensory_spine,
                duration_sec=screen_seconds,
                fps=screen_fps,
                monitor_index=1,
                stream_id=stream_id,
            )
        else:
            stream_id = "real_mp4_flower"
            stream_summary = LiveScreenSensoryBridge.observe_real_video(
                self.sensory_spine,
                video_path=REAL_EXTERNAL_MP4_PATH,
                max_frames=60,
                stream_id=stream_id,
            )

        s_micro = float(stream_summary.get("mean_micro_activity", 0.3))
        s_meso = float(stream_summary.get("mean_meso_activity", 0.4))
        s_macro = float(stream_summary.get("mean_macro_persistence", 0.5))
        s_coh = float(stream_summary.get("mean_beam_coherence", 0.6))
        link_stream = self._record_thought_link(
            phase="PHASE_1_SENSORY_EYE",
            event_tag="eye_external_stream",
            short_label="P1:Stream",
            target_key=stream_id,
            trigger_source="External Sensory World",
            parent_step_ids=[link_eye.step_id],
            evidence_node_ids=[],
            verdict_code="EXTERNAL_STREAM_ANCHORED",
            judgment_summary=(
                f"External stream {stream_id}: frames={stream_summary.get('frames_processed', 0)}, "
                f"beam_coherence={s_coh:.3f}."
            ),
            judgment_summary_ko=(
                f"외부 영상 스트림({stream_id}, {stream_summary.get('frames_processed', 0)}프레임)을 "
                f"동일 감각 척추로 수신하여 층간 결속도 {s_coh:.3f}를 확보함."
            ),
            scale_level=0,
            chromatic_triad=(s_micro + 0.1, s_macro + 0.1, s_meso + 0.1),
            details=dict(stream_summary),
            aliases=["external_stream", "real_mp4", "live_screen"],
        )

        # ==============================================================
        # PHASE 2: Self-Boundary Zoom & 4-Axis Epistemic Coupling
        # ==============================================================
        mean_mm = float(np.mean([c.beam_micro_meso for c in self.sensory_spine.columns.values()]))
        mean_mc = float(np.mean([c.beam_meso_macro for c in self.sensory_spine.columns.values()]))

        snap_couple = self.zoom_kernel.step_boundary_cycle(
            BoundaryEventPacket(
                domain_tag="unified_sensory_world",
                step_label="eye_stream_coupling",
                intent_active=True,
                agency_resonance={
                    "sensory_eye_spine": float(np.clip(mean_mm, 0.10, 0.96)),
                    f"stream_{stream_id}": float(np.clip(s_coh, 0.10, 0.96)),
                },
                local_anomaly_sharpness=float(np.clip(abs(s_micro - s_macro), 0.05, 0.90)),
                global_context_mismatch=float(np.clip(1.0 - s_coh, 0.05, 0.90)),
                work_units_full=100,
                work_units_executed=max(10, int(round(100.0 * (1.0 - 0.5 * mean_mc)))),
            )
        )
        link_zoom_sense = self._record_thought_link(
            phase="PHASE_2_BOUNDARY_ZOOM",
            event_tag="boundary_sensory_coupling",
            short_label="P2:Couple",
            target_key="boundary_sensory_coupling",
            trigger_source="Boundary Zoom Control",
            parent_step_ids=[link_eye.step_id, link_stream.step_id],
            evidence_node_ids=["ELYSIA_SELF_CORE"],
            verdict_code=f"BOUNDARY_{snap_couple.zoom_band}",
            judgment_summary=(
                f"Coupled sensory world into self-boundary zoom={snap_couple.zoom_scale:.2f} ({snap_couple.zoom_band})."
            ),
            judgment_summary_ko=(
                f"감각 기둥과 외부 스트림을 자기경계 핵에 연결하여 "
                f"줌 배율 {snap_couple.zoom_scale:.2f} ({snap_couple.zoom_band})로 균형을 맞춤."
            ),
            scale_level=1,
            chromatic_triad=(
                snap_couple.axis_external_friction + 0.1,
                snap_couple.axis_internal_origin + 0.1,
                snap_couple.axis_criteria_revision + 0.1,
            ),
            details={
                "zoom_scale": float(snap_couple.zoom_scale),
                "zoom_band": str(snap_couple.zoom_band),
                "boundary_event": str(snap_couple.boundary_event),
            },
        )

        # Focus zoom inward on the sharpest structural asymmetry observed among columns
        column_tensions = [
            abs(c.beam_micro_meso - c.beam_meso_macro)
            for c in self.sensory_spine.columns.values()
        ]
        max_tension = float(max(column_tensions)) if column_tensions else 0.45
        snap_focus = self.zoom_kernel.step_boundary_cycle(
            BoundaryEventPacket(
                domain_tag="unified_sensory_world",
                step_label="micro_meso_tension_zoom",
                intent_active=True,
                agency_resonance={
                    eid: float(np.clip(col.beam_micro_meso, 0.10, 0.95))
                    for eid, col in self.sensory_spine.columns.items()
                },
                local_anomaly_sharpness=float(np.clip(max_tension + 0.25, 0.15, 0.96)),
                global_context_mismatch=float(np.clip(max_tension + 0.15, 0.10, 0.92)),
                work_units_full=100,
                work_units_executed=22,
            )
        )
        link_zoom_focus = self._record_thought_link(
            phase="PHASE_2_BOUNDARY_ZOOM",
            event_tag="boundary_anomaly_zoom",
            short_label="P2:Zoom",
            target_key="boundary_anomaly_zoom",
            trigger_source="Boundary Zoom Control",
            parent_step_ids=[link_zoom_sense.step_id],
            evidence_node_ids=["ELYSIA_SELF_CORE"],
            verdict_code=f"FOCUS_{snap_focus.zoom_band}",
            judgment_summary=(
                f"Focused boundary zoom={snap_focus.zoom_scale:.2f} ({snap_focus.zoom_band}) "
                f"on max sensory layer tension={max_tension:.3f}."
            ),
            judgment_summary_ko=(
                f"감각 층간 최대 비대칭({max_tension:.3f})을 추적하기 위해 "
                f"자기경계를 {snap_focus.zoom_scale:.2f} ({snap_focus.zoom_band})로 정밀 초점 이동함."
            ),
            scale_level=1,
            chromatic_triad=(
                snap_focus.axis_external_friction + 0.1,
                snap_focus.axis_internal_origin + 0.1,
                snap_focus.axis_boundary_ignorance + snap_focus.axis_criteria_revision + 0.1,
            ),
            details={
                "max_sensory_tension": max_tension,
                "zoom_scale": float(snap_focus.zoom_scale),
                "zoom_band": str(snap_focus.zoom_band),
            },
        )

        # ==============================================================
        # PHASE 3: Visual-Linguistic Superposition & Contrast
        # ==============================================================
        catalog = build_korean_statement_catalog()
        probe_pairs = [
            ("ent_1", "stmt_red_wall", "P3:ent1"),
            ("ent_3", "stmt_red_wall", "P3:Illusion!"),
            ("ent_unseen_a", "stmt_red_wall", "P3:GrnWall"),
            ("ent_4", "stmt_red_gate", "P3:ent4"),
        ]
        contrast_results: List[SuperpositionContrastResult] = []
        contrast_step_ids: List[str] = []

        for eid, sid, short_lbl in probe_pairs:
            stmt = catalog[sid]
            res = self.visual_contrast.superpose_and_contrast(eid, stmt, bind_on_conviction=True)
            contrast_results.append(res)

            # Continuous 4-Axis Boundary update caused by superposition agreement/divergence
            if res.verdict == "CONVICTION_FULL_SYNC":
                self.zoom_kernel.step_boundary_cycle(
                    BoundaryEventPacket(
                        domain_tag="visual_linguistic_contrast",
                        step_label=f"conviction_{eid}_{sid}",
                        intent_active=True,
                        agency_resonance={
                            eid: float(np.clip(res.overall_conviction, 0.55, 0.98)),
                            f"lang_{sid}": float(np.clip(res.sync_macro, 0.55, 0.98)),
                        },
                        local_anomaly_sharpness=float(np.clip(1.0 - res.sync_micro, 0.02, 0.35)),
                        global_context_mismatch=float(np.clip(1.0 - res.sync_macro, 0.02, 0.35)),
                        work_units_full=64,
                        work_units_executed=16,
                    )
                )
            else:
                self.zoom_kernel.step_boundary_cycle(
                    BoundaryEventPacket(
                        domain_tag="visual_linguistic_contrast",
                        step_label=f"divergence_{eid}_{sid}",
                        intent_active=True,
                        agency_resonance={
                            eid: float(np.clip(res.overall_conviction, 0.08, 0.65)),
                            f"lang_{sid}": float(np.clip(min(res.sync_meso, res.sync_macro), 0.08, 0.65)),
                        },
                        local_anomaly_sharpness=float(np.clip(1.0 - min(res.sync_micro, res.sync_meso), 0.25, 0.98)),
                        global_context_mismatch=float(np.clip(1.0 - min(res.sync_meso, res.sync_macro), 0.25, 0.98)),
                        work_units_full=64,
                        work_units_executed=48,
                    )
                )

            col = self.sensory_spine.columns.get(eid)
            evidence_ids = ["ELYSIA_SELF_CORE"]
            if col and col.boundary_voxel_id:
                evidence_ids.append(col.boundary_voxel_id)
            if res.grounded_super_molecule_id:
                evidence_ids.append(res.grounded_super_molecule_id)

            c_triad = (
                (1.0 - res.sync_meso) + 0.15,
                res.overall_conviction + 0.15,
                float(len(res.diverged_layers)) * 0.35 + 0.10,
            )
            scale_lvl = 2 if res.grounded_super_molecule_id else 1
            aliases = [eid] if sid == "stmt_red_wall" or eid == "ent_4" else []

            link_c = self._record_thought_link(
                phase="PHASE_3_LANGUAGE_CONTRAST",
                event_tag=f"contrast_{eid}_{sid}",
                short_label=short_lbl,
                target_key=f"{eid}::{sid}",
                trigger_source="Linguistic Proposition",
                parent_step_ids=[link_eye.step_id, link_zoom_focus.step_id],
                evidence_node_ids=evidence_ids,
                verdict_code=res.verdict,
                judgment_summary=(
                    f"{eid} vs {sid}: {res.verdict} "
                    f"(micro={res.sync_micro:.2f}, meso={res.sync_meso:.2f}, macro={res.sync_macro:.2f})."
                ),
                judgment_summary_ko=res.explanation_ko,
                scale_level=scale_lvl,
                chromatic_triad=c_triad,
                details={
                    "entity_id": eid,
                    "statement_id": sid,
                    "statement_text_ko": stmt.text_ko,
                    "verdict": res.verdict,
                    "sync_micro": float(res.sync_micro),
                    "sync_meso": float(res.sync_meso),
                    "sync_macro": float(res.sync_macro),
                    "overall_conviction": float(res.overall_conviction),
                    "matched_layers": list(res.matched_layers),
                    "diverged_layers": list(res.diverged_layers),
                    "grounded_super_molecule_id": res.grounded_super_molecule_id,
                },
                aliases=aliases,
            )
            contrast_step_ids.append(link_c.step_id)

        # Autonomous best-statement selection across all 6 observed entities
        entity_ids = ["ent_1", "ent_2", "ent_3", "ent_4", "ent_unseen_a", "ent_unseen_b"]
        expected_map = {
            "ent_1": "stmt_red_wall",
            "ent_2": "stmt_red_crate",
            "ent_3": "stmt_red_illusion",
            "ent_4": "stmt_red_gate",
            "ent_unseen_a": "stmt_green_wall",
            "ent_unseen_b": "stmt_green_illusion",
        }
        autonomous_best_matches = {
            eid: self.visual_contrast.ground_entity_to_best_statement(eid, catalog)
            for eid in entity_ids
        }
        correct_groundings = sum(
            1
            for eid, expected_sid in expected_map.items()
            if autonomous_best_matches[eid].statement_id == expected_sid
            and autonomous_best_matches[eid].verdict == "CONVICTION_FULL_SYNC"
        )
        grounding_accuracy = float(correct_groundings / len(entity_ids))

        grounded_mols = [
            r.grounded_super_molecule_id
            for r in autonomous_best_matches.values()
            if r.grounded_super_molecule_id
        ]
        self._record_thought_link(
            phase="PHASE_3_LANGUAGE_CONTRAST",
            event_tag="linguistic_grounding_consensus",
            short_label="P3:6/6",
            target_key="linguistic_grounding_consensus",
            trigger_source="Linguistic Proposition",
            parent_step_ids=contrast_step_ids,
            evidence_node_ids=grounded_mols,
            verdict_code="GROUNDED_CONSENSUS",
            judgment_summary=(
                f"Autonomous visual-linguistic best grounding accuracy: {correct_groundings}/{len(entity_ids)}."
            ),
            judgment_summary_ko=(
                f"시각-신체 감각 기둥과 한국어 진술 목록 전체를 자율 포개어 "
                f"{correct_groundings}/{len(entity_ids)} ({grounding_accuracy * 100:.0f}%) 완벽 결속함."
            ),
            scale_level=2,
            chromatic_triad=(0.20, grounding_accuracy + 0.1, 0.30),
            details={
                "correct_groundings": correct_groundings,
                "total_entities": len(entity_ids),
                "grounding_accuracy": grounding_accuracy,
                "matches": {eid: r.statement_id for eid, r in autonomous_best_matches.items()},
            },
            aliases=["grounding_consensus"],
        )

        # ==============================================================
        # PHASE 4: Analogical Boundary Reflection & Criteria Self-Revision
        # ==============================================================
        profiles = build_physical_domain_profiles()
        foothold_water = profiles["Water_Flow"]
        foothold_node = self.analogical.anchor_lived_foothold(foothold_water)
        foothold_id = getattr(foothold_node, "node_id", "Water_Flow")

        link_water = self._record_thought_link(
            phase="PHASE_4_ANALOGY_REFLECTION",
            event_tag="foothold_water_flow",
            short_label="P4:H2O",
            target_key="Water_Flow",
            trigger_source="Lived Foothold",
            parent_step_ids=[link_eye.step_id, link_zoom_sense.step_id],
            evidence_node_ids=[foothold_id, "ELYSIA_SELF_CORE"],
            verdict_code="FOOTHOLD_ANCHORED",
            judgment_summary="Anchored lived physical Water_Flow domain as sensory foothold.",
            judgment_summary_ko=(
                "눈과 접촉으로 먼저 납득한 가시적 '물의 흐름(Water_Flow)'을 보이지 않는 원리를 재해석할 인지적 디딤돌로 고정함."
            ),
            scale_level=2,
            chromatic_triad=(0.35, 0.55, 0.15),
            details={"foothold_node_id": foothold_id},
            aliases=["water", "water_flow"],
        )

        analogical_targets = [
            ("Invisible_Electricity", "P4:Elec", "analogical_electricity", ["electricity", "elec"]),
            ("Invisible_Heat_Conduction", "P4:Heat", "analogical_heat", ["heat", "heat_conduction"]),
        ]
        analogical_reports: List[AnalogicalReflectionReport] = []

        for domain_id, short_lbl, ev_tag, dom_aliases in analogical_targets:
            rep = self.analogical.reflect_on_unknown_via_foothold(
                source=foothold_water,
                target=profiles[domain_id],
                counterfactual_tests=build_counterfactual_tests_for_domain(domain_id),
            )
            analogical_reports.append(rep)

            ev_nodes = ["ELYSIA_SELF_CORE", foothold_id]
            if rep.reinterpreted_molecule_id:
                ev_nodes.append(rep.reinterpreted_molecule_id)

            self._record_thought_link(
                phase="PHASE_4_ANALOGY_REFLECTION",
                event_tag=ev_tag,
                short_label=short_lbl,
                target_key=domain_id,
                trigger_source="Physical Counterfactual Anomaly",
                parent_step_ids=[link_water.step_id, link_zoom_focus.step_id],
                evidence_node_ids=ev_nodes,
                verdict_code="BOUNDARY_TEAR_REINTERPRETED",
                judgment_summary=(
                    f"{domain_id}: preserved {rep.aligned_probes}, broke at {rep.breakdown_probes}; "
                    f"accuracy {rep.naive_analogy_accuracy * 100:.0f}% -> {rep.reinterpreted_accuracy * 100:.0f}%."
                ),
                judgment_summary_ko=rep.reflection_summary_ko,
                scale_level=3,
                chromatic_triad=(
                    0.20 + 0.15 * len(rep.breakdown_probes),
                    0.45 + 0.25 * rep.reinterpreted_accuracy,
                    0.25 + 0.20 * len(rep.torn_analogical_beams),
                ),
                details={
                    "target_domain_id": rep.target_domain_id,
                    "aligned_probes": list(rep.aligned_probes),
                    "breakdown_probes": list(rep.breakdown_probes),
                    "torn_analogical_beams": list(rep.torn_analogical_beams),
                    "reinterpreted_molecule_id": rep.reinterpreted_molecule_id,
                    "naive_analogy_accuracy": float(rep.naive_analogy_accuracy),
                    "reinterpreted_accuracy": float(rep.reinterpreted_accuracy),
                },
                aliases=dom_aliases,
            )

        # ==============================================================
        # Meta-Causal Consolidation: Self-Introspect Key Judgments & Save State
        # ==============================================================
        introspection_targets = [
            "ent_1::stmt_red_wall",
            "ent_3::stmt_red_wall",
            "ent_unseen_a::stmt_red_wall",
            "ent_4::stmt_red_gate",
            "Invisible_Electricity",
            "Invisible_Heat_Conduction",
        ]
        introspections: Dict[str, IntrospectiveSelfExplanation] = {
            tid: self.introspect_why_i_judged(tid) for tid in introspection_targets
        }

        _, _, _, _, final_zoom, final_band = self._current_four_axis()
        now_iso = datetime.now(timezone.utc).isoformat()

        elec_rep = analogical_reports[0]
        heat_rep = analogical_reports[1]
        cycle_summary_ko = (
            f"[ELYSIA 통합 마음 유기체 사이클 #{cycle_idx}] "
            f"감각-언어 자율 결속 {correct_groundings}/{len(entity_ids)} ({grounding_accuracy * 100:.0f}%) | "
            f"외부 스트림({stream_id}) 층간 결속 {s_coh:.3f} | "
            f"물 디딤돌 유비 재해석: 전기 {elec_rep.naive_analogy_accuracy * 100:.0f}%->{elec_rep.reinterpreted_accuracy * 100:.0f}%, "
            f"열 {heat_rep.naive_analogy_accuracy * 100:.0f}%->{heat_rep.reinterpreted_accuracy * 100:.0f}% | "
            f"최종 줌={final_zoom:.2f} ({final_band}), 단일 CausalField 복셀={len(self.causal_field.voxels)} 빔={len(self.causal_field.beams)}"
        )

        report = UnifiedMindCycleReport(
            mode=mode,
            cycle_index=cycle_idx,
            timestamp_iso=now_iso,
            inferred_affordances=inferred_affordances,
            external_stream_summary=dict(stream_summary),
            contrast_results=contrast_results,
            autonomous_best_matches=autonomous_best_matches,
            grounding_accuracy=grounding_accuracy,
            analogical_reports=analogical_reports,
            thought_links=list(self.thought_links[sid] for sid in self.thought_order),
            introspections=introspections,
            final_zoom_scale=final_zoom,
            final_zoom_band=final_band,
            total_causal_voxels=len(self.causal_field.voxels),
            total_causal_beams=len(self.causal_field.beams),
            total_trinitarian_nodes=len(self.trinitarian.nodes),
            state_json_path=str(self.state_path),
            cycle_summary_ko=cycle_summary_ko,
        )
        self.last_report = report
        self.save_state(self.state_path, report=report)
        return report

    # ------------------------------------------------------------------
    # Dynamic Backward Meta-Causal Introspection ("Thinking About Thinking")
    # ------------------------------------------------------------------
    def _resolve_target_step_id(self, target_id: str) -> str:
        if target_id in self.thought_links:
            return target_id
        if target_id in self.target_to_thought_id:
            return self.target_to_thought_id[target_id]
        lowered = target_id.lower()
        for key, step_id in self.target_to_thought_id.items():
            if key.lower() == lowered:
                return step_id
        for sid in reversed(self.thought_order):
            link = self.thought_links[sid]
            if lowered in link.target_key.lower() or lowered in sid.lower():
                return sid
        raise KeyError(f"Unknown judgment or target_id for introspection: {target_id!r}")

    def introspect_why_i_judged(self, target_id: str) -> IntrospectiveSelfExplanation:
        """
        Walks BACKWARD dynamically through `self.thought_links`, `self.causal_field.beams`,
        `self.sensory_spine.columns`, and `self.trinitarian.zoom_in_inspect(...)`
        to reconstruct why ELYSIA's Eye and Mind arrived at this exact judgment.
        """
        final_step_id = self._resolve_target_step_id(target_id)

        # 1. Backward BFS along parent_step_ids to collect the full causal ancestry
        visited = set()
        queue = deque([final_step_id])
        while queue:
            cur_id = queue.popleft()
            if cur_id in visited or cur_id not in self.thought_links:
                continue
            visited.add(cur_id)
            for pid in self.thought_links[cur_id].parent_step_ids:
                if pid not in visited:
                    queue.append(pid)

        # Order chronologically from root perception -> self-boundary -> verdict
        chain_links = sorted(
            (self.thought_links[sid] for sid in visited),
            key=lambda lk: lk.step_index,
        )
        chain_ids = [lk.step_id for lk in chain_links]
        chain_id_set = set(chain_ids)

        chain_steps_payload: List[Dict[str, Any]] = [
            {
                "step_id": lk.step_id,
                "step_index": lk.step_index,
                "phase": lk.phase,
                "trigger_source": lk.trigger_source,
                "verdict_code": lk.verdict_code,
                "scale_level": lk.scale_level,
                "zoom_scale": lk.zoom_scale,
                "parent_step_ids": list(lk.parent_step_ids),
                "chromatic_triad": list(lk.chromatic_triad),
                "four_axis": list(lk.four_axis),
                "judgment_summary": lk.judgment_summary,
                "judgment_summary_ko": lk.judgment_summary_ko,
            }
            for lk in chain_links
        ]

        # 2. Unwrap every Trinitarian rotor molecule referenced along this chain
        trinitarian_tree_ids: List[str] = []
        seen_rotor_ids = set()
        for lk in chain_links:
            for nid in lk.evidence_node_ids:
                if nid in self.trinitarian.nodes:
                    for sub_node in self.trinitarian.zoom_in_inspect(nid):
                        if sub_node.node_id not in seen_rotor_ids:
                            seen_rotor_ids.add(sub_node.node_id)
                            trinitarian_tree_ids.append(
                                f"{sub_node.node_id}(s={sub_node.scale_level})"
                            )

        # 3. Inspect live ConnectivityBeam strengths touching this causal chain
        beam_links: List[Dict[str, Any]] = []
        for beam in getattr(self.causal_field, "beams", []):
            src_id = getattr(beam, "source_id", None)
            dst_id = getattr(beam, "target_id", None)
            if src_id in chain_id_set or dst_id in chain_id_set:
                beam_links.append(
                    {
                        "source_id": src_id,
                        "target_id": dst_id,
                        "strength": round(float(getattr(beam, "strength", 0.0)), 4),
                    }
                )

        # 4. Read live sensory column evidence directly from `self.sensory_spine`
        final_link = self.thought_links[final_step_id]
        sensory_evidence: Dict[str, Any] = {}
        entity_id = final_link.details.get("entity_id")
        if not entity_id and "::" in final_link.target_key:
            entity_id = final_link.target_key.split("::", 1)[0]
        if entity_id and entity_id in self.sensory_spine.columns:
            col = self.sensory_spine.columns[entity_id]
            sensory_evidence = {
                "entity_id": entity_id,
                "beam_micro_meso": float(col.beam_micro_meso),
                "beam_meso_macro": float(col.beam_meso_macro),
                "chromatic_triad": [float(x) for x in col.chromatic_triad],
                "boundary_voxel_id": col.boundary_voxel_id,
                "inferred_affordance": self.sensory_spine.infer_ecological_affordance(entity_id),
            }

        fa = final_link.four_axis
        four_axis_attr = {
            "internal_origin": float(fa[0]),
            "external_friction": float(fa[1]),
            "boundary_ignorance": float(fa[2]),
            "criteria_revision": float(fa[3]),
        }

        # 5. Synthesize Korean & English narrative strictly from observed chain + evidence
        chain_trace_str = " -> ".join(f"{lk.short_label}[{lk.verdict_code}]" for lk in chain_links)
        ko_parts: List[str] = [
            f"[자기-인과 역추적: {final_link.target_key}] 인과 사슬: {chain_trace_str}.",
            f"출발 근거: {chain_links[0].judgment_summary_ko}",
        ]
        if sensory_evidence:
            ko_parts.append(
                f"실측 눈/촉각 기둥({sensory_evidence['entity_id']}): "
                f"micro<->meso={sensory_evidence['beam_micro_meso']:.3f}, "
                f"meso<->macro={sensory_evidence['beam_meso_macro']:.3f}, "
                f"행동유도성={sensory_evidence['inferred_affordance']}."
            )
        if "sync_micro" in final_link.details:
            d = final_link.details
            ko_parts.append(
                f"언어-시각 대조 결과: 일치층={d['matched_layers']}, 괴리층={d['diverged_layers']} "
                f"(L_micro={d['sync_micro']:.2f}, L_meso={d['sync_meso']:.2f}, L_macro={d['sync_macro']:.2f})."
            )
        if "breakdown_probes" in final_link.details:
            d = final_link.details
            ko_parts.append(
                f"물 디딤돌 유비 반성: 보존 구조={d['aligned_probes']}, 파열 경계={d['breakdown_probes']}, "
                f"절단된 빔={d['torn_analogical_beams']}, "
                f"예측 정확도 {d['naive_analogy_accuracy'] * 100:.0f}% -> {d['reinterpreted_accuracy'] * 100:.0f}%."
            )
        if trinitarian_tree_ids:
            ko_parts.append(f"결속/분해된 삼위일체 로터 계층: {', '.join(trinitarian_tree_ids[:8])}.")
        ko_parts.append(
            f"판단 시점 4축 귀속: 내적원점={fa[0]:.2f}, 외부마찰={fa[1]:.2f}, 무지경계={fa[2]:.2f}, 기준수정={fa[3]:.2f} "
            f"(줌={final_link.zoom_scale:.2f})."
        )

        narrative_ko = " ".join(ko_parts)
        narrative_en = (
            f"[{final_link.target_key}] Back-traced {len(chain_links)} causal steps ({chain_trace_str}) "
            f"| beams={len(beam_links)} | rotors={len(trinitarian_tree_ids)} "
            f"| 4Axis=(Int:{fa[0]:.2f}, Ext:{fa[1]:.2f}, Ign:{fa[2]:.2f}, Rev:{fa[3]:.2f})."
        )

        return IntrospectiveSelfExplanation(
            target_id=target_id,
            final_step_id=final_step_id,
            verdict_code=final_link.verdict_code,
            causal_chain_ids=chain_ids,
            causal_chain_steps=chain_steps_payload,
            trinitarian_tree_ids=trinitarian_tree_ids,
            causal_field_beam_links=beam_links,
            sensory_evidence=sensory_evidence,
            four_axis_attribution=four_axis_attr,
            narrative_ko=narrative_ko,
            narrative_en=narrative_en,
        )

    # ------------------------------------------------------------------
    # Persistent State Serialization (`c:\Elysia\data\unified_mind_state.json`)
    # ------------------------------------------------------------------
    def save_state(
        self,
        path: Optional[Path] = None,
        report: Optional[UnifiedMindCycleReport] = None,
    ) -> Path:
        out_path = Path(path) if path else self.state_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        rep = report or self.last_report
        _, _, _, _, zoom_s, zoom_b = self._current_four_axis()

        payload: Dict[str, Any] = {
            "organism_version": self.ORGANISM_VERSION,
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            "cycle_count": self.cycle_count,
            "metrics": {
                "grounding_accuracy": float(rep.grounding_accuracy) if rep else 0.0,
                "final_zoom_scale": zoom_s,
                "final_zoom_band": zoom_b,
                "total_causal_voxels": len(self.causal_field.voxels),
                "total_causal_beams": len(self.causal_field.beams),
                "total_trinitarian_nodes": len(self.trinitarian.nodes),
                "zoom_history_length": len(self.zoom_kernel.history),
            },
            "sensory_columns_summary": {
                eid: {
                    "beam_micro_meso": float(col.beam_micro_meso),
                    "beam_meso_macro": float(col.beam_meso_macro),
                    "chromatic_triad": [float(v) for v in col.chromatic_triad],
                    "boundary_voxel_id": col.boundary_voxel_id,
                    "affordance": self.sensory_spine.infer_ecological_affordance(eid),
                }
                for eid, col in self.sensory_spine.columns.items()
            },
            "thought_growth_rings": [
                asdict(self.thought_links[sid]) for sid in self.thought_order
            ],
            "introspections": {
                k: asdict(v) for k, v in (rep.introspections.items() if rep else {})
            },
            "boundary_trajectory": [
                {
                    "step_index": i,
                    "domain_tag": getattr(s, "domain_tag", ""),
                    "boundary_event": str(s.boundary_event),
                    "zoom_scale": float(s.zoom_scale),
                    "zoom_band": str(s.zoom_band),
                    "internal_origin": float(s.axis_internal_origin),
                    "external_friction": float(s.axis_external_friction),
                    "boundary_ignorance": float(s.axis_boundary_ignorance),
                    "criteria_revision": float(s.axis_criteria_revision),
                }
                for i, s in enumerate(self.zoom_kernel.history)
            ],
        }
        if rep and rep.analogical_reports:
            for ar in rep.analogical_reports:
                payload["metrics"][f"{ar.target_domain_id}_accuracy_before"] = float(ar.naive_analogy_accuracy)
                payload["metrics"][f"{ar.target_domain_id}_accuracy_after"] = float(ar.reinterpreted_accuracy)

        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(_to_json_safe(payload), fh, ensure_ascii=False, indent=2)
        return out_path


# ======================================================================
# 4. 4-Panel Dark-Theme Unified Organism Dashboard Renderer
# ======================================================================
def render_unified_mind_dashboard(
    organism: ElysiaUnifiedMindOrganism,
    report: UnifiedMindCycleReport,
    output_path: Path = DEFAULT_UNIFIED_MIND_DASHBOARD_PATH,
) -> Path:
    """
    Renders a clean 2x2 dark-theme ASCII/English telemetry dashboard summarizing:
      1) Meta-Causal Thought Growth-Ring DAG (`self.thought_links` + `ConnectivityBeam` flow)
      2) Eye-to-Language Layer Sync Heatmap (`L_micro`, `L_meso`, `L_macro`)
      3) Continuous 4-Axis Self-Boundary & Scale Zoom Trajectory
      4) Analogical Foothold Reflection (`Water -> Electricity & Heat` accuracy + torn beams)
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(16.5, 11.0), dpi=150, facecolor="#0D1117")
    gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.28, top=0.90, bottom=0.07, left=0.07, right=0.95)

    fig.suptitle(
        "ELYSIA Unified Mind Organism: Eye-Language Contrast, Self-Boundary Zoom & Meta-Causal Introspection",
        color="#F0F6FC",
        fontsize=13.5,
        fontweight="bold",
        y=0.96,
    )

    def _style_ax(ax: plt.Axes, title: str) -> None:
        ax.set_facecolor("#161B22")
        for spine in ax.spines.values():
            spine.set_color("#30363D")
            spine.set_linewidth(1.1)
        ax.tick_params(colors="#C9D1D9", labelsize=8.5)
        ax.set_title(title, color="#F0F6FC", fontsize=10.5, fontweight="bold", pad=10)

    # ------------------------------------------------------------------
    # Panel 1 (Top-Left): Causal Thought Growth-Ring Graph
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    _style_ax(ax1, "1. Meta-Causal Thought Growth-Ring Graph (Self-Introspection DAG)")

    links = report.thought_links
    same_scale_counts: Dict[int, int] = {}
    coords: Dict[str, Tuple[float, float]] = {}
    for lk in links:
        occ = same_scale_counts.get(lk.scale_level, 0)
        same_scale_counts[lk.scale_level] = occ + 1
        y_jitter = ((occ % 3) - 1) * 0.19
        coords[lk.step_id] = (float(lk.step_index), float(lk.scale_level) + y_jitter)

    for lk in links:
        x2, y2 = coords[lk.step_id]
        for pid in lk.parent_step_ids:
            if pid in coords:
                x1, y1 = coords[pid]
                ax1.annotate(
                    "",
                    xy=(x2, y2),
                    xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="#58A6FF",
                        alpha=0.45,
                        lw=1.1,
                        shrinkA=7,
                        shrinkB=7,
                        connectionstyle="arc3,rad=0.10",
                    ),
                    zorder=1,
                )

    for lk in links:
        x, y = coords[lk.step_id]
        r, b, yl = lk.chromatic_triad
        dom_idx = int(np.argmax([r, b, yl]))
        face_col = ("#FF6B6B", "#58A6FF", "#F2C94C")[dom_idx]
        edge_col = "#3FB950" if "SYNC" in lk.verdict_code or "REINTERPRETED" in lk.verdict_code else "#F0F6FC"
        ax1.scatter(
            [x],
            [y],
            s=155 + 25 * lk.scale_level,
            color=face_col,
            edgecolors=edge_col,
            linewidths=1.4,
            zorder=3,
        )
        y_off = 0.25 if (lk.step_index % 2 == 0) else -0.28
        ax1.text(
            x,
            y + y_off,
            lk.short_label,
            color="#E6EDF3",
            fontsize=7.3,
            ha="center",
            va="center",
            zorder=4,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#0D1117", edgecolor="none", alpha=0.72),
        )

    ax1.set_xlim(-0.85, max(2.0, len(links) + 0.35))
    ax1.set_ylim(-0.68, 3.75)
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_yticklabels(["L0 Sensory", "L1 Boundary", "L2 Super", "L3 Analogy"])
    ax1.set_xlabel("Causal Thought Step Index (Chronological Growth Ring)", color="#C9D1D9", fontsize=9)
    ax1.grid(axis="both", color="#21262D", linestyle="--", linewidth=0.7, alpha=0.7)

    # ------------------------------------------------------------------
    # Panel 2 (Top-Right): Eye-to-Language Layer Sync Heatmap
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    _style_ax(ax2, "2. Eye-to-Language Layer Superposition Contrast (Sync Score)")

    mat = np.array(
        [[r.sync_micro, r.sync_meso, r.sync_macro] for r in report.contrast_results],
        dtype=np.float64,
    )
    im = ax2.imshow(mat, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")

    short_verdicts = {
        "CONVICTION_FULL_SYNC": "FULL SYNC",
        "DIVERGED_AT_MESO_MACRO": "DIV MESO+MACRO",
        "DIVERGED_AT_MICRO_ONLY": "DIV MICRO ONLY",
    }
    row_labels = [
        f"{r.entity_id} vs {r.statement_id.replace('stmt_', '')}\n[{short_verdicts.get(r.verdict, r.verdict)}]"
        for r in report.contrast_results
    ]
    ax2.set_yticks(range(len(row_labels)))
    ax2.set_yticklabels(row_labels, fontsize=8.0)
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(["L_micro\n(Color/Edge)", "L_meso\n(Interaction)", "L_macro\n(Persistence)"], fontsize=8.5)

    for ri in range(mat.shape[0]):
        for ci in range(mat.shape[1]):
            val = float(mat[ri, ci])
            txt_col = "#0D1117" if 0.28 <= val <= 0.88 else "#FFFFFF"
            ax2.text(
                ci,
                ri,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=txt_col,
                fontsize=9.5,
                fontweight="bold",
            )

    cbar = fig.colorbar(im, ax=ax2, fraction=0.045, pad=0.03)
    cbar.ax.tick_params(colors="#C9D1D9", labelsize=8)
    cbar.set_label("Layer Alignment", color="#C9D1D9", fontsize=8.5)

    # ------------------------------------------------------------------
    # Panel 3 (Bottom-Left): Persistent 4-Axis Self-Boundary & Zoom Trajectory
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    _style_ax(ax3, "3. Continuous 4-Axis Self-Boundary & Scale Zoom Trajectory")

    hist = organism.zoom_kernel.history
    xs = np.arange(len(hist), dtype=np.float64)
    a_int = [s.axis_internal_origin for s in hist]
    a_ext = [s.axis_external_friction for s in hist]
    a_ign = [s.axis_boundary_ignorance for s in hist]
    a_rev = [s.axis_criteria_revision for s in hist]
    zooms = [s.zoom_scale for s in hist]

    l1 = ax3.plot(xs, a_int, color="#58A6FF", lw=1.8, marker="o", ms=3.5, label="Internal Origin")
    l2 = ax3.plot(xs, a_ext, color="#FF7B72", lw=1.8, marker="s", ms=3.5, label="External Friction")
    l3 = ax3.plot(xs, a_ign, color="#D29922", lw=1.6, ls="--", label="Boundary Ignorance")
    l4 = ax3.plot(xs, a_rev, color="#3FB950", lw=1.8, marker="^", ms=3.5, label="Criteria Revision")

    ax3.set_ylim(-0.05, 1.15)
    ax3.set_xlabel("Self-Boundary Cycle Step (Phase 2 -> Phase 3 -> Phase 4)", color="#C9D1D9", fontsize=9)
    ax3.set_ylabel("4-Axis Attribution Magnitude", color="#C9D1D9", fontsize=9)
    ax3.grid(color="#21262D", linestyle="--", linewidth=0.7, alpha=0.7)

    ax3_r = ax3.twinx()
    l5 = ax3_r.plot(xs, zooms, color="#D2A8FF", lw=1.9, ls=":", marker="D", ms=3.5, label="Zoom Scale")
    ax3_r.set_ylim(0.0, 2.3)
    ax3_r.set_ylabel("Scale Zoom Factor", color="#D2A8FF", fontsize=9)
    ax3_r.tick_params(colors="#D2A8FF", labelsize=8.5)
    for sp in ax3_r.spines.values():
        sp.set_color("#30363D")

    lines = l1 + l2 + l3 + l4 + l5
    labels = [ln.get_label() for ln in lines]
    ax3.legend(
        lines,
        labels,
        loc="upper left",
        ncol=2,
        fontsize=7.6,
        facecolor="#0D1117",
        edgecolor="#30363D",
        labelcolor="#E6EDF3",
    )

    # ------------------------------------------------------------------
    # Panel 4 (Bottom-Right): Analogical Foothold Reflection Accuracy
    # ------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    _style_ax(ax4, "4. Analogical Reflection: Water Foothold -> Invisible Physics")

    dom_names = [
        r.target_domain_id.replace("Invisible_", "").replace("_Conduction", " Conduct.")
        for r in report.analogical_reports
    ]
    naive_pct = [r.naive_analogy_accuracy * 100.0 for r in report.analogical_reports]
    reint_pct = [r.reinterpreted_accuracy * 100.0 for r in report.analogical_reports]

    x_pos = np.arange(len(dom_names), dtype=np.float64)
    bar_w = 0.32
    bars_n = ax4.bar(x_pos - bar_w / 2.0, naive_pct, width=bar_w, color="#F85149", alpha=0.88, label="Naive Water Analogy")
    bars_r = ax4.bar(x_pos + bar_w / 2.0, reint_pct, width=bar_w, color="#3FB950", alpha=0.92, label="Reinterpreted s=3")

    for b in list(bars_n) + list(bars_r):
        h = float(b.get_height())
        ax4.text(
            b.get_x() + b.get_width() / 2.0,
            h + 2.5,
            f"{h:.0f}%",
            ha="center",
            va="bottom",
            color="#F0F6FC",
            fontsize=9.5,
            fontweight="bold",
        )

    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(dom_names, fontsize=9.5)
    ax4.set_yticks([0, 25, 50, 75, 100])
    ax4.set_ylim(-58, 136)
    ax4.axhline(0.0, color="#30363D", linewidth=1.0, zorder=1)
    ax4.set_ylabel("Unseen Counterfactual Accuracy (%)", color="#C9D1D9", fontsize=9)
    ax4.grid(axis="y", color="#21262D", linestyle="--", linewidth=0.7, alpha=0.7)
    ax4.legend(
        loc="upper center",
        ncol=2,
        fontsize=8.2,
        facecolor="#0D1117",
        edgecolor="#30363D",
        labelcolor="#E6EDF3",
    )

    torn_lines = "\n".join(
        f"{r.target_domain_id.replace('Invisible_', '')} breaks={', '.join(r.breakdown_probes)}"
        for r in report.analogical_reports
    )
    footer_txt = (
        f"6/6 Language Grounding: {report.grounding_accuracy * 100:.0f}%  |  "
        f"Stream Coherence: {report.external_stream_summary.get('mean_beam_coherence', 0.0):.3f}\n"
        f"Shared CausalField: {report.total_causal_voxels} Voxels, {report.total_causal_beams} Beams, "
        f"{report.total_trinitarian_nodes} Rotors\n"
        f"{torn_lines}"
    )
    box = FancyBboxPatch(
        (0.03, 0.02),
        0.94,
        0.24,
        boxstyle="round,pad=0.02",
        transform=ax4.transAxes,
        facecolor="#0D1117",
        edgecolor="#30363D",
        alpha=0.92,
        zorder=4,
    )
    ax4.add_patch(box)
    ax4.text(
        0.50,
        0.14,
        footer_txt,
        transform=ax4.transAxes,
        ha="center",
        va="center",
        color="#C9D1D9",
        fontsize=7.1,
        zorder=5,
    )

    fig.savefig(out, facecolor=fig.get_facecolor(), dpi=150)
    plt.close(fig)
    return out
