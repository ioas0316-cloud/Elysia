r"""
ELYSIA Self-Weaving Scale-Rotor Loom
====================================

Implements a continuous, label-free cognitive Loom where:
  - Every module is a `CausalLine` (not a closed box classifier) projecting raw,
    unlabelled external reality onto an emergent bipolar spectrum `[-1.0, +1.0]`.
  - Lines dynamically weave themselves into three living topologies:
      1. Parallel   (simultaneous multi-prism refraction of the same referent)
      2. Cross      (`Line A x Line B` capturing cross-information: resonance & tension)
      3. Cycle      (open causal loop: high tension -> real external probe -> re-refraction)
  - Stable `CrossNode`s promote recursively into higher-scale `PromotedRotorLine`s
    (whose Axis is the invariant spine of constituent Lines and whose Spectrum rotates
    across them), and descend (`scale_descend`) when higher-scale tension isolates
    which child Line diverged.
  - Everything's existence, motion, constitution, and causal "why" (`WeaveGenesisRecord`)
    is continuously observable via `observe_self()` AND perceived by Elysia herself via
    `IntrospectiveLoomLine` operating inside the exact same Loom.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.physics.causal_field import CausalField, InformationVoxel


def _clamp(val: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, float(val))))


def _norm_chromatic(chroma: Sequence[float]) -> Tuple[float, float, float]:
    arr = np.asarray(list(chroma[:3]), dtype=np.float64)
    if arr.size < 3:
        arr = np.pad(arr, (0, 3 - arr.size), constant_values=0.33)
    arr = np.maximum(0.0, arr)
    total = float(arr.sum())
    if total <= 1e-9:
        return (0.34, 0.33, 0.33)
    arr = arr / total
    return (round(float(arr[0]), 4), round(float(arr[1]), 4), round(float(arr[2]), 4))


def _canonical_pair(id_a: str, id_b: str) -> Tuple[str, str]:
    return (id_a, id_b) if id_a <= id_b else (id_b, id_a)


@dataclass
class LineRefraction:
    """One instantaneous projection of a raw external referent onto a `CausalLine` spectrum."""

    line_id: str
    referent_id: str
    spectrum_value: float
    raw_observables: Dict[str, float]
    chromatic: Tuple[float, float, float]
    evidence: Dict[str, Any]
    step: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "line_id": self.line_id,
            "referent_id": self.referent_id,
            "spectrum_value": round(float(self.spectrum_value), 5),
            "raw_observables": {k: round(float(v), 5) for k, v in self.raw_observables.items()},
            "chromatic": list(self.chromatic),
            "evidence": dict(self.evidence),
            "step": int(self.step),
        }


@dataclass
class WeaveGenesisRecord:
    """
    Records *why* a topological change happened in the Loom with 4-Axis Self-Boundary attribution
    (AGENTS.md §6: Internal Origin, External Friction, Boundary Ignorance, Criteria Revision).
    """

    step: int
    operation: str  # register_line | weave_parallel | weave_cross | weave_cycle | promote_rotor | scale_descend | tear_weave
    target_id: str
    referent_id: str
    why_ko: str
    internal_origin: str
    external_friction: str
    boundary_ignorance: str
    criteria_revision: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CrossNode:
    """Intersectional prism node (`Line A x Line B`) storing emergent cross-information."""

    cross_id: str
    line_a_id: str
    line_b_id: str
    scale: int
    angle_rad: float = 0.0
    radius: float = 0.0
    resonance: float = 0.0
    tension: float = 0.0
    mass: float = 0.0
    encounters: int = 0
    last_referent_id: str = ""
    promoted_line_id: Optional[str] = None
    is_torn: bool = False
    unproductive_streak: int = 0
    genesis_why: str = ""
    history: List[Dict[str, Any]] = field(default_factory=list)

    def record_state(
        self,
        step: int,
        referent_id: str,
        s_a: float,
        s_b: float,
    ) -> Tuple[float, float]:
        prev_angle = self.angle_rad
        prev_tension = self.tension

        self.angle_rad = float(math.atan2(s_b, s_a))
        self.radius = float(math.hypot(s_a, s_b))
        # Resonance rises when both lines see active structure in coherent proportion;
        # tension rises when the two prisms pull toward opposing poles on the same referent.
        self.resonance = float(max(0.0, 1.0 - abs(s_a - s_b) / 2.0) * min(1.0, self.radius))
        self.tension = float(_clamp(abs(s_a - s_b) / 2.0, 0.0, 1.0))
        self.encounters += 1
        self.last_referent_id = referent_id

        # Mass accumulates when crossing reveals real structure rather than stagnant noise.
        if self.tension < 0.78 or self.resonance >= 0.25:
            self.mass += 0.45 + 0.45 * self.resonance + 0.15 * min(1.0, self.radius)
        else:
            self.mass += 0.20

        self.history.append(
            {
                "step": int(step),
                "referent_id": referent_id,
                "s_a": round(float(s_a), 4),
                "s_b": round(float(s_b), 4),
                "angle_rad": round(float(self.angle_rad), 4),
                "radius": round(float(self.radius), 4),
                "resonance": round(float(self.resonance), 4),
                "tension": round(float(self.tension), 4),
            }
        )
        if len(self.history) > 32:
            self.history = self.history[-32:]

        angle_delta = self.angle_rad - prev_angle
        tension_delta = self.tension - prev_tension
        return angle_delta, tension_delta

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cross_id": self.cross_id,
            "line_a_id": self.line_a_id,
            "line_b_id": self.line_b_id,
            "scale": int(self.scale),
            "angle_rad": round(float(self.angle_rad), 4),
            "radius": round(float(self.radius), 4),
            "resonance": round(float(self.resonance), 4),
            "tension": round(float(self.tension), 4),
            "mass": round(float(self.mass), 4),
            "encounters": int(self.encounters),
            "last_referent_id": self.last_referent_id,
            "promoted_line_id": self.promoted_line_id,
            "is_torn": bool(self.is_torn),
            "unproductive_streak": int(self.unproductive_streak),
            "genesis_why": self.genesis_why,
            "recent_history": list(self.history[-6:]),
        }


@dataclass
class CycleWeave:
    """Open external causal feedback loop triggered by high cross tension."""

    cycle_id: str
    trigger_cross_id: str
    action_line_id: str
    observe_line_ids: Tuple[str, str]
    executions: int = 0
    last_tension_before: float = 0.0
    last_tension_after: float = 0.0
    mean_tension_delta: float = 0.0
    external_consequences: List[Dict[str, Any]] = field(default_factory=list)
    is_torn: bool = False
    unproductive_streak: int = 0
    genesis_why: str = ""

    def record_execution(
        self,
        step: int,
        referent_id: str,
        tension_before: float,
        tension_after: float,
        consequence: Dict[str, Any],
        produced_new_evidence: bool,
    ) -> float:
        delta = float(tension_after - tension_before)
        self.executions += 1
        self.last_tension_before = float(tension_before)
        self.last_tension_after = float(tension_after)
        w_old = (self.executions - 1) / float(self.executions)
        self.mean_tension_delta = w_old * self.mean_tension_delta + (1.0 / float(self.executions)) * delta

        self.external_consequences.append(
            {
                "step": int(step),
                "referent_id": referent_id,
                "tension_before": round(float(tension_before), 4),
                "tension_after": round(float(tension_after), 4),
                "tension_delta": round(float(delta), 4),
                "produced_new_evidence": bool(produced_new_evidence),
                "consequence": dict(consequence),
            }
        )
        if len(self.external_consequences) > 24:
            self.external_consequences = self.external_consequences[-24:]

        if (not produced_new_evidence) and abs(delta) < 0.02:
            self.unproductive_streak += 1
        else:
            self.unproductive_streak = 0
        return delta

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "trigger_cross_id": self.trigger_cross_id,
            "action_line_id": self.action_line_id,
            "observe_line_ids": list(self.observe_line_ids),
            "executions": int(self.executions),
            "last_tension_before": round(float(self.last_tension_before), 4),
            "last_tension_after": round(float(self.last_tension_after), 4),
            "mean_tension_delta": round(float(self.mean_tension_delta), 4),
            "is_torn": bool(self.is_torn),
            "unproductive_streak": int(self.unproductive_streak),
            "genesis_why": self.genesis_why,
            "recent_consequences": list(self.external_consequences[-5:]),
        }


@dataclass
class LoomObservation:
    """
    4-aspect live observability snapshot of the Loom's state & self-perception:
      1. `existence`    : what Lines, emergent poles, Crosses, Cycles, Rotors exist right now
      2. `motion`       : how spectra, rotor angles, and tensions are currently moving
      3. `constitution` : how higher-scale Rotors are woven out of lower-scale Lines (parent <-> child)
      4. `causal_why`   : why each topological change happened (WeaveGenesisRecord lineage)
    """

    step: int
    existence: Dict[str, Any]
    motion: Dict[str, Any]
    constitution: Dict[str, Any]
    causal_why: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": int(self.step),
            "existence": self.existence,
            "motion": self.motion,
            "constitution": self.constitution,
            "causal_why": self.causal_why,
        }

    def render_ko(self) -> str:
        lines_out: List[str] = [f"=== [Loom Live Observation | step={self.step}] ==="]

        # 1. Existence
        ex = self.existence
        lines_out.append(
            f"1. 존재 (Existence): 선 {len(ex['lines'])}개 | "
            f"활성 교차 {ex['active_cross_count']}개 (파단 {ex['torn_cross_count']}개) | "
            f"활성 순환 {ex['active_cycle_count']}개 | "
            f"승격 로터선 {ex['promoted_rotor_count']}개 (최고 스케일 S{ex['max_scale']})"
        )
        for line_info in ex["lines"]:
            neg_r = line_info.get("neg_extreme_ref") or "-"
            pos_r = line_info.get("pos_extreme_ref") or "-"
            last_v = line_info.get("last_spectrum")
            last_str = f"{last_v:+.3f}" if last_v is not None else "n/a"
            lines_out.append(
                f"   - [{line_info['kind']} S{line_info['scale']}] {line_info['line_id']} "
                f"(최근={last_str}, 관측={line_info['observations']}, "
                f"음극단={neg_r}:{line_info.get('neg_extreme_val', 0.0):+.2f} <-> "
                f"양극단={pos_r}:{line_info.get('pos_extreme_val', 0.0):+.2f})"
            )

        # 2. Motion
        mo = self.motion
        lines_out.append(
            f"2. 움직임 (Motion): 최근 스펙트럼 속도 {len(mo['line_velocities'])}개 | "
            f"교차 로터 각/장력 변화 {len(mo['cross_motions'])}개 | "
            f"순환 실행 {len(mo['cycle_deltas'])}개"
        )
        for lv in mo["line_velocities"]:
            lines_out.append(
                f"   - 선 {lv['line_id']}: {lv['prev_value']:+.3f} -> {lv['curr_value']:+.3f} "
                f"(Δ={lv['velocity']:+.3f})"
            )
        for cm in mo["cross_motions"]:
            lines_out.append(
                f"   - 교차 {cm['cross_id']}: 각 {cm['angle_deg']:+.1f}°, "
                f"반경 {cm['radius']:.3f}, 공명 {cm['resonance']:.3f}, "
                f"장력 {cm['tension']:.3f} (Δ장력={cm['tension_delta']:+.3f})"
            )

        # 3. Constitution
        co = self.constitution
        lines_out.append(
            f"3. 구성 (Constitution): 스케일 계층 {co['scale_distribution']} | "
            f"인과장 Voxel {co['causal_field_voxels']} / Beam {co['causal_field_beams']} "
            f"(파단 {co['causal_field_broken_beams']})"
        )
        for rotor in co["rotor_lineages"]:
            children = " × ".join(rotor["constituent_line_ids"])
            lines_out.append(
                f"   - [S{rotor['scale']} 로터축] {rotor['line_id']} <= ({children}) "
                f"| 축가중치={[round(w, 3) for w in rotor['axis_weights']]} "
                f"| 최근스펙트럼={rotor['last_spectrum']:+.3f}"
            )

        # 4. Causal Why
        lines_out.append(f"4. 왜 그렇게 되어지고 있는가 (Causal Why, 최근 {len(self.causal_why)}건):")
        for rec in self.causal_why[-8:]:
            lines_out.append(
                f"   - [step {rec['step']} | {rec['operation']}] {rec['target_id']} @ {rec['referent_id']}: "
                f"{rec['why_ko']}"
            )
        return "\n".join(lines_out)


class CausalLine:
    """
    A continuous refracting Line (`Module = Line`).
    Instead of outputting developer-invented discrete category tags, every Line
    projects raw external referents onto a continuous bipolar spectrum `[-1.0, +1.0]`,
    discovering its own negative/positive poles solely from experienced extremes.
    """

    kind: str = "external_line"

    def __init__(
        self,
        line_id: str,
        scale: int = 1,
        constituent_line_ids: Sequence[str] = (),
        chromatic_bias: Sequence[float] = (0.34, 0.33, 0.33),
        description: str = "",
    ) -> None:
        self.line_id = str(line_id)
        self.scale = int(scale)
        self.constituent_line_ids: Tuple[str, ...] = tuple(str(x) for x in constituent_line_ids)
        self.chromatic_bias: Tuple[float, float, float] = _norm_chromatic(chromatic_bias)
        self.description = description
        self.history: List[LineRefraction] = []

        # Emergent poles grounded strictly in real observed referents (no pre-labels)
        self.neg_extreme_val: float = 0.0
        self.neg_extreme_ref: str = ""
        self.pos_extreme_val: float = 0.0
        self.pos_extreme_ref: str = ""
        self._has_observed: bool = False

    def refract(
        self,
        referent_id: str,
        raw_referent: Dict[str, Any],
        step: int,
    ) -> Optional[LineRefraction]:
        """Projects a raw external referent onto `[-1.0, +1.0]` or returns `None` if not applicable."""
        raise NotImplementedError

    def _record_refraction(
        self,
        referent_id: str,
        spectrum_value: float,
        raw_observables: Dict[str, float],
        chromatic: Sequence[float],
        evidence: Dict[str, Any],
        step: int,
    ) -> LineRefraction:
        val = _clamp(spectrum_value, -1.0, 1.0)
        refraction = LineRefraction(
            line_id=self.line_id,
            referent_id=referent_id,
            spectrum_value=val,
            raw_observables=dict(raw_observables),
            chromatic=_norm_chromatic(chromatic),
            evidence=dict(evidence),
            step=int(step),
        )
        self.history.append(refraction)
        if len(self.history) > 128:
            self.history = self.history[-128:]

        if (not self._has_observed) or (val <= self.neg_extreme_val):
            self.neg_extreme_val = val
            self.neg_extreme_ref = referent_id
        if (not self._has_observed) or (val >= self.pos_extreme_val):
            self.pos_extreme_val = val
            self.pos_extreme_ref = referent_id
        self._has_observed = True
        return refraction

    def last_velocity(self) -> Optional[Dict[str, Any]]:
        if len(self.history) < 2:
            return None
        prev_r = self.history[-2]
        curr_r = self.history[-1]
        return {
            "line_id": self.line_id,
            "prev_referent": prev_r.referent_id,
            "curr_referent": curr_r.referent_id,
            "prev_value": round(prev_r.spectrum_value, 4),
            "curr_value": round(curr_r.spectrum_value, 4),
            "velocity": round(curr_r.spectrum_value - prev_r.spectrum_value, 4),
        }

    def describe_existence(self) -> Dict[str, Any]:
        last_val = self.history[-1].spectrum_value if self.history else None
        return {
            "line_id": self.line_id,
            "kind": self.kind,
            "scale": int(self.scale),
            "constituent_line_ids": list(self.constituent_line_ids),
            "observations": len(self.history),
            "last_spectrum": round(last_val, 4) if last_val is not None else None,
            "last_referent": self.history[-1].referent_id if self.history else None,
            "neg_extreme_val": round(self.neg_extreme_val, 4),
            "neg_extreme_ref": self.neg_extreme_ref,
            "pos_extreme_val": round(self.pos_extreme_val, 4),
            "pos_extreme_ref": self.pos_extreme_ref,
            "chromatic_bias": list(self.chromatic_bias),
            "description": self.description,
        }


class PromotedRotorLine(CausalLine):
    """
    Higher-scale Rotor-Line born automatically when a `CrossNode` achieves sufficient
    structural mass & resonance across encounters. Its invariant spine is an Axis over
    its constituent Lines, and its continuous Spectrum tracks rotational balance & alignment.
    """

    kind: str = "promoted_rotor"

    def __init__(
        self,
        line_id: str,
        scale: int,
        constituent_line_ids: Tuple[str, str],
        source_cross_id: str,
        axis_weights: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(
            line_id=line_id,
            scale=scale,
            constituent_line_ids=constituent_line_ids,
            chromatic_bias=(0.25, 0.55, 0.20),
            description=f"교차 {source_cross_id}에서 승격된 Scale-{scale} 로터 축",
        )
        self.source_cross_id = source_cross_id
        if axis_weights is None:
            vec = np.array([1.0, 1.0], dtype=np.float64) / math.sqrt(2.0)
        else:
            vec = np.asarray(axis_weights, dtype=np.float64)
            norm = float(np.linalg.norm(vec))
            vec = vec / norm if norm > 1e-9 else np.array([1.0, 1.0]) / math.sqrt(2.0)
        self.axis_weights = vec
        self.last_child_spread: float = 0.0

    def refract(
        self,
        referent_id: str,
        raw_referent: Dict[str, Any],
        step: int,
    ) -> Optional[LineRefraction]:
        # Higher-scale rotors refract via child refractions (`refract_from_children`).
        return None

    def refract_from_children(
        self,
        child_refractions: Dict[str, LineRefraction],
        referent_id: str,
        step: int,
    ) -> Optional[LineRefraction]:
        id_a, id_b = self.constituent_line_ids[0], self.constituent_line_ids[1]
        if id_a not in child_refractions or id_b not in child_refractions:
            return None
        r_a = child_refractions[id_a]
        r_b = child_refractions[id_b]

        vec = np.array([r_a.spectrum_value, r_b.spectrum_value], dtype=np.float64)
        axis_proj = float(np.dot(self.axis_weights, vec) / max(1.0, float(np.linalg.norm(self.axis_weights, ord=1))))
        angle = float(math.atan2(vec[1], vec[0]))
        spread = float(abs(vec[0] - vec[1]) / 2.0)
        self.last_child_spread = spread

        # Smoothly align axis spine toward coherent experienced co-motion without collapsing differences.
        vec_norm = float(np.linalg.norm(vec))
        if vec_norm > 0.25 and spread < 0.55:
            target_dir = vec / vec_norm
            blended = 0.85 * self.axis_weights + 0.15 * target_dir
            b_norm = float(np.linalg.norm(blended))
            if b_norm > 1e-9:
                self.axis_weights = blended / b_norm

        chroma = _norm_chromatic(
            [
                0.5 * (r_a.chromatic[0] + r_b.chromatic[0]),
                0.5 * (r_a.chromatic[1] + r_b.chromatic[1]) + 0.25 * (1.0 - spread),
                0.5 * (r_a.chromatic[2] + r_b.chromatic[2]) + 0.25 * spread,
            ]
        )
        return self._record_refraction(
            referent_id=referent_id,
            spectrum_value=_clamp(axis_proj, -1.0, 1.0),
            raw_observables={
                "axis_proj": axis_proj,
                "rotor_angle_rad": angle,
                "child_spread": spread,
                "child_a": r_a.spectrum_value,
                "child_b": r_b.spectrum_value,
            },
            chromatic=chroma,
            evidence={
                "source_cross_id": self.source_cross_id,
                "children": [id_a, id_b],
                "axis_weights": [round(float(x), 4) for x in self.axis_weights],
            },
            step=step,
        )

    def diagnose_descent(
        self,
        child_refractions: Dict[str, LineRefraction],
    ) -> Dict[str, Any]:
        """When upper-scale spread spikes, descend to identify which child Line diverged."""
        id_a, id_b = self.constituent_line_ids[0], self.constituent_line_ids[1]
        r_a = child_refractions.get(id_a)
        r_b = child_refractions.get(id_b)
        if r_a is None or r_b is None:
            return {"diverging_line_id": None, "spread": 0.0}
        last_axis = self.history[-1].spectrum_value if self.history else 0.0
        dev_a = abs(r_a.spectrum_value - last_axis)
        dev_b = abs(r_b.spectrum_value - last_axis)
        diverging = id_a if dev_a >= dev_b else id_b
        return {
            "rotor_line_id": self.line_id,
            "diverging_line_id": diverging,
            "other_line_id": id_b if diverging == id_a else id_a,
            "spread": round(abs(r_a.spectrum_value - r_b.spectrum_value) / 2.0, 4),
            "s_a": round(r_a.spectrum_value, 4),
            "s_b": round(r_b.spectrum_value, 4),
        }


class IntrospectiveLoomLine(CausalLine):
    """
    Self-perception Line plugged directly into the Loom itself.
    Observes the Loom's own active crossings, tensions, promotions, and tears,
    projecting Elysia's live internal structural tension/coherence onto the exact same
    `[-1.0, +1.0]` continuous spectrum so she perceives her own state alongside external reality.
    """

    kind: str = "introspective_self"

    def __init__(self, loom_ref: "SelfWeavingLoom") -> None:
        super().__init__(
            line_id="line::introspective_self",
            scale=1,
            constituent_line_ids=(),
            chromatic_bias=(0.20, 0.60, 0.20),
            description="자기 직조 베틀(존재·움직임·장력·승격)을 실시간으로 지각하는 내면 관측선",
        )
        self._loom = loom_ref

    def refract(
        self,
        referent_id: str,
        raw_referent: Dict[str, Any],
        step: int,
    ) -> Optional[LineRefraction]:
        active_crosses = [c for c in self._loom.crosses.values() if not c.is_torn]
        torn_crosses = [c for c in self._loom.crosses.values() if c.is_torn]
        promoted = [l for l in self._loom.lines.values() if isinstance(l, PromotedRotorLine)]
        mean_tension = (
            float(sum(c.tension for c in active_crosses) / len(active_crosses))
            if active_crosses
            else 0.0
        )
        mean_res = (
            float(sum(c.resonance for c in active_crosses) / len(active_crosses))
            if active_crosses
            else 0.0
        )
        # Positive pole = high integrated coherence/resonance & promoted synthesis;
        # Negative pole = unresolved cross tension & structural tearing pressure.
        synthesis_ratio = min(1.0, len(promoted) / max(1.0, float(len(self._loom.lines))))
        tear_pressure = min(1.0, len(torn_crosses) / max(1.0, float(len(self._loom.crosses) or 1)))
        spectrum = _clamp(
            (0.65 * mean_res + 0.55 * synthesis_ratio) - (0.85 * mean_tension + 0.35 * tear_pressure),
            -1.0,
            1.0,
        )
        chroma = _norm_chromatic(
            [
                0.20 + 0.35 * synthesis_ratio,
                0.35 + 0.45 * mean_res,
                0.15 + 0.60 * mean_tension,
            ]
        )
        return self._record_refraction(
            referent_id=referent_id,
            spectrum_value=spectrum,
            raw_observables={
                "active_crosses": float(len(active_crosses)),
                "torn_crosses": float(len(torn_crosses)),
                "promoted_rotors": float(len(promoted)),
                "mean_tension": mean_tension,
                "mean_resonance": mean_res,
            },
            chromatic=chroma,
            evidence={
                "self_observed_step": int(step),
                "active_cycles": sum(1 for cy in self._loom.cycles.values() if not cy.is_torn),
            },
            step=step,
        )


class SelfWeavingLoom:
    """
    Dynamic Scale-Rotor Loom weaving parallel, cross, cycle, promoted-rotor,
    and self-introspective topologies directly from unlabelled external friction.
    """

    def __init__(self) -> None:
        self.causal_field = CausalField(dimensions=3)
        self.lines: Dict[str, CausalLine] = {}
        self.action_lines: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
        self.crosses: Dict[str, CrossNode] = {}
        self.cycles: Dict[str, CycleWeave] = {}
        self.genesis_log: List[WeaveGenesisRecord] = []
        self.referent_refractions: Dict[str, Dict[str, LineRefraction]] = {}
        self.referent_snapshots: Dict[str, Dict[str, Any]] = {}
        self.step_count: int = 0
        self._last_cycle_deltas: List[Dict[str, Any]] = []

        self.introspective_line = IntrospectiveLoomLine(self)
        self.register_line(self.introspective_line)

    def _log_genesis(
        self,
        operation: str,
        target_id: str,
        referent_id: str,
        why_ko: str,
        internal_origin: str,
        external_friction: str,
        boundary_ignorance: str,
        criteria_revision: str,
    ) -> None:
        rec = WeaveGenesisRecord(
            step=self.step_count,
            operation=operation,
            target_id=target_id,
            referent_id=referent_id,
            why_ko=why_ko,
            internal_origin=internal_origin,
            external_friction=external_friction,
            boundary_ignorance=boundary_ignorance,
            criteria_revision=criteria_revision,
        )
        self.genesis_log.append(rec)
        if len(self.genesis_log) > 256:
            self.genesis_log = self.genesis_log[-256:]

    def _sync_voxel(self, line: CausalLine, spectrum_val: float = 0.0) -> None:
        angle = (len(self.causal_field.voxels) + 1) * 1.15
        radius = float(line.scale) * 1.2
        pos = np.array(
            [
                radius * math.cos(angle),
                radius * math.sin(angle),
                float(spectrum_val),
            ],
            dtype=np.float32,
        )
        tensor = np.array(
            [float(spectrum_val), float(line.scale) * 0.25, float(line.chromatic_bias[1])],
            dtype=np.float32,
        )
        if line.line_id not in self.causal_field.voxels:
            voxel = InformationVoxel(
                id=line.line_id,
                content=line.description or line.line_id,
                tensor=tensor,
                mass=max(1.0, float(line.scale)),
                position=pos,
                chromatic_vector=np.array(line.chromatic_bias, dtype=np.float32),
                scale_level=float(line.scale),
            )
            self.causal_field.add_voxel(voxel)
        else:
            voxel = self.causal_field.voxels[line.line_id]
            voxel.tensor = tensor
            voxel.position[2] = float(spectrum_val)
            voxel.mass = max(voxel.mass, float(line.scale))

    def _sync_cross_beam(self, cross: CrossNode) -> None:
        if cross.line_a_id not in self.causal_field.voxels or cross.line_b_id not in self.causal_field.voxels:
            return
        matching = [
            b
            for b in self.causal_field.beams
            if _canonical_pair(b.source_id, b.target_id) == (cross.line_a_id, cross.line_b_id)
        ]
        if not matching:
            self.causal_field.link_voxels(
                cross.line_a_id,
                cross.line_b_id,
                strength=max(1.0, float(cross.scale)),
            )
            matching = [
                b
                for b in self.causal_field.beams
                if _canonical_pair(b.source_id, b.target_id) == (cross.line_a_id, cross.line_b_id)
            ]
        for beam in matching:
            beam.current_tension = float(cross.tension * 5.0)
            beam.is_broken = bool(cross.is_torn)

    def register_line(self, line: CausalLine) -> None:
        self.lines[line.line_id] = line
        self._sync_voxel(line, spectrum_val=0.0)
        self._log_genesis(
            operation="register_line",
            target_id=line.line_id,
            referent_id="loom_init" if self.step_count == 0 else "runtime",
            why_ko=f"Scale-{line.scale} 관측선 `{line.line_id}`를 베틀에 편입 ({line.description})",
            internal_origin=f"선 `{line.line_id}`이 가진 고유 스펙트럼 투영 축",
            external_friction="아직 외부 대상과 접촉하지 않음",
            boundary_ignorance="초기 극단(음/양 극)이 아직 외부 경험으로 채워지지 않음",
            criteria_revision="외부 극단 관측 시 양극단을 스스로 갱신하도록 열린 상태로 등록",
        )

    def register_action_line(
        self,
        action_id: str,
        fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        description: str = "",
    ) -> None:
        self.action_lines[action_id] = fn
        self._log_genesis(
            operation="register_line",
            target_id=action_id,
            referent_id="loom_init" if self.step_count == 0 else "runtime",
            why_ko=f"외부 작용선 `{action_id}` 등록 ({description or '실제 외부 탐색/작용 통로'})",
            internal_origin="교차 장력 발생 시 외부 세계에 직접 작용을 가하기 위한 통로",
            external_friction="외부 세계 반응 대기",
            boundary_ignorance="외부 탐색이 어떤 새 단서를 돌려줄지 미지",
            criteria_revision="작용 후 반환된 외부 결과를 재굴절해 순환 유용성 평가",
        )

    def _refract_all_lines(
        self,
        referent_id: str,
        raw_referent: Dict[str, Any],
    ) -> Dict[str, LineRefraction]:
        current: Dict[str, LineRefraction] = {}

        # 1) External non-introspective base lines (Scale 1)
        for line in self.lines.values():
            if isinstance(line, (PromotedRotorLine, IntrospectiveLoomLine)):
                continue
            ref = line.refract(referent_id, raw_referent, self.step_count)
            if ref is not None:
                current[line.line_id] = ref
                self._sync_voxel(line, ref.spectrum_value)

        # 2) Promoted higher-scale rotors sorted bottom-up by scale
        promoted_lines = sorted(
            [l for l in self.lines.values() if isinstance(l, PromotedRotorLine)],
            key=lambda x: x.scale,
        )
        for rotor in promoted_lines:
            ref = rotor.refract_from_children(current, referent_id, self.step_count)
            if ref is not None:
                current[rotor.line_id] = ref
                self._sync_voxel(rotor, ref.spectrum_value)

        # 3) Self-perception introspective line observes the updated Loom
        self_ref = self.introspective_line.refract(referent_id, raw_referent, self.step_count)
        if self_ref is not None:
            current[self.introspective_line.line_id] = self_ref
            self._sync_voxel(self.introspective_line, self_ref.spectrum_value)

        self.referent_refractions[referent_id] = current
        return current

    def _base_ancestors(self, line_id: str, _visited: Optional[set[str]] = None) -> frozenset[str]:
        """Recursively resolves a Line or PromotedRotorLine down to its Scale-1 base Line IDs."""
        visited = set(_visited or ())
        if line_id in visited:
            return frozenset()
        visited.add(line_id)
        line = self.lines.get(line_id)
        if line is None or not line.constituent_line_ids:
            return frozenset((line_id,))
        ancestors: set[str] = set()
        for child_id in line.constituent_line_ids:
            ancestors.update(self._base_ancestors(child_id, visited))
        return frozenset(ancestors) if ancestors else frozenset((line_id,))

    def _shares_subset_ancestors(self, id_a: str, id_b: str) -> bool:
        """
        Repulsion boundary: forbids crossing two Lines when one is already wholly composed
        of the other's base Lines (preventing fake combinatorial self-cloning).
        """
        anc_a = self._base_ancestors(id_a)
        anc_b = self._base_ancestors(id_b)
        return (anc_a <= anc_b) or (anc_b <= anc_a)

    def _maybe_weave_crosses(
        self,
        referent_id: str,
        current: Dict[str, LineRefraction],
    ) -> List[str]:
        woven_ids: List[str] = []
        active_ids = sorted(current.keys())

        # A) Cross-referent structural contrast: two referents close on Line A yet divergent on Line B
        for prev_ref_id, prev_map in self.referent_refractions.items():
            if prev_ref_id == referent_id:
                continue
            for idx_a, id_a in enumerate(active_ids):
                for id_b in active_ids[idx_a + 1 :]:
                    if id_a not in prev_map or id_b not in prev_map:
                        continue
                    pair = _canonical_pair(id_a, id_b)
                    if self._shares_subset_ancestors(pair[0], pair[1]):
                        continue
                    cross_id = f"cross::{pair[0]}__{pair[1]}"
                    if cross_id in self.crosses:
                        continue
                    da = abs(current[id_a].spectrum_value - prev_map[id_a].spectrum_value)
                    db = abs(current[id_b].spectrum_value - prev_map[id_b].spectrum_value)
                    # Case 1: Line A collapsed both (`da <= 0.30`), Line B separated them (`db >= 0.42`)
                    # Case 2: Line B collapsed both (`db <= 0.30`), Line A separated them (`da >= 0.42`)
                    collapsed_split = (da <= 0.30 and db >= 0.42) or (db <= 0.30 and da >= 0.42)
                    if not collapsed_split:
                        continue
                    line_a, line_b = self.lines[pair[0]], self.lines[pair[1]]
                    why = (
                        f"대상 `{prev_ref_id}`와 `{referent_id}`가 한 선에서는 비슷했으나 "
                        f"다른 선에서 갈라짐 (ΔA={da:.2f}, ΔB={db:.2f}) -> 단일 선 한계를 넘어 교차 직조"
                    )
                    cross = CrossNode(
                        cross_id=cross_id,
                        line_a_id=pair[0],
                        line_b_id=pair[1],
                        scale=max(line_a.scale, line_b.scale) + 1,
                        genesis_why=why,
                    )
                    self.crosses[cross_id] = cross
                    woven_ids.append(cross_id)
                    self._log_genesis(
                        operation="weave_cross",
                        target_id=cross_id,
                        referent_id=referent_id,
                        why_ko=why,
                        internal_origin=f"`{pair[0]}`와 `{pair[1]}` 스펙트럼 차이를 교차면으로 통합하려는 내적 요구",
                        external_friction=f"`{prev_ref_id}` vs `{referent_id}` 비교에서 ΔA={da:.2f}, ΔB={db:.2f} 관측",
                        boundary_ignorance="단일 선 투영만으로는 두 외부 현실의 차이를 보존할 수 없음",
                        criteria_revision=f"교차점 `{cross_id}`를 생성해 2차원 회전면(각/반경/장력)으로 재편",
                    )

        # B) Same-referent divergence between two Lines (`|s_A - s_B| >= 0.65`)
        for idx_a, id_a in enumerate(active_ids):
            for id_b in active_ids[idx_a + 1 :]:
                pair = _canonical_pair(id_a, id_b)
                if self._shares_subset_ancestors(pair[0], pair[1]):
                    continue
                cross_id = f"cross::{pair[0]}__{pair[1]}"
                if cross_id in self.crosses:
                    continue
                diff = abs(current[pair[0]].spectrum_value - current[pair[1]].spectrum_value)
                if diff < 0.65:
                    continue
                line_a, line_b = self.lines[pair[0]], self.lines[pair[1]]
                why = (
                    f"동일 대상 `{referent_id}`에서 `{pair[0]}`({current[pair[0]].spectrum_value:+.2f})와 "
                    f"`{pair[1]}`({current[pair[1]].spectrum_value:+.2f})가 크게 어긋남 (차={diff:.2f}) -> 교차 직조"
                )
                cross = CrossNode(
                    cross_id=cross_id,
                    line_a_id=pair[0],
                    line_b_id=pair[1],
                    scale=max(line_a.scale, line_b.scale) + 1,
                    genesis_why=why,
                )
                self.crosses[cross_id] = cross
                woven_ids.append(cross_id)
                self._log_genesis(
                    operation="weave_cross",
                    target_id=cross_id,
                    referent_id=referent_id,
                    why_ko=why,
                    internal_origin="서로 다른 프리즘의 충돌을 하나의 입체 교차정보로 수렴시키려는 내적 압력",
                    external_friction=f"동일 대상 `{referent_id}`에 대해 두 선이 {diff:.2f} 만큼 반대 극단으로 굴절",
                    boundary_ignorance="어느 선의 특성이 대상의 핵심 인과인지 아직 미규명",
                    criteria_revision=f"교차점 `{cross_id}` 생성으로 장력/공명 추적 개시",
                )

        return woven_ids

    def _update_crosses(
        self,
        referent_id: str,
        current: Dict[str, LineRefraction],
    ) -> List[Dict[str, Any]]:
        cross_motions: List[Dict[str, Any]] = []
        for cross in self.crosses.values():
            if cross.is_torn:
                continue
            if cross.line_a_id not in current or cross.line_b_id not in current:
                continue
            s_a = current[cross.line_a_id].spectrum_value
            s_b = current[cross.line_b_id].spectrum_value
            angle_delta, tension_delta = cross.record_state(self.step_count, referent_id, s_a, s_b)
            self._sync_cross_beam(cross)
            if cross.tension < 0.78 or cross.resonance >= 0.22:
                cross.unproductive_streak = 0
            else:
                cross.unproductive_streak += 1
            cross_motions.append(
                {
                    "cross_id": cross.cross_id,
                    "angle_deg": round(math.degrees(cross.angle_rad), 2),
                    "angle_delta_deg": round(math.degrees(angle_delta), 2),
                    "radius": round(cross.radius, 4),
                    "resonance": round(cross.resonance, 4),
                    "tension": round(cross.tension, 4),
                    "tension_delta": round(tension_delta, 4),
                }
            )
        return cross_motions

    def _maybe_weave_and_run_cycles(
        self,
        referent_id: str,
        raw_referent: Dict[str, Any],
        current: Dict[str, LineRefraction],
    ) -> List[Dict[str, Any]]:
        if not self.action_lines:
            return []
        default_action_id = sorted(self.action_lines.keys())[0]
        action_fn = self.action_lines[default_action_id]
        cycle_deltas: List[Dict[str, Any]] = []

        # Pick the highest-tension active cross eligible for an external probe cycle
        candidates = [
            c
            for c in self.crosses.values()
            if not c.is_torn
            and c.tension >= 0.30
            and c.line_a_id in current
            and c.line_b_id in current
        ]
        candidates.sort(key=lambda c: c.tension, reverse=True)
        for cross in candidates[:2]:
            cycle_id = f"cycle::{cross.cross_id}__{default_action_id}"
            cycle = self.cycles.get(cycle_id)
            if cycle is not None and cycle.is_torn:
                continue
            if cycle is None:
                why = (
                    f"교차 `{cross.cross_id}`의 장력({cross.tension:.2f})이 임계치를 넘어 "
                    f"외부 작용선 `{default_action_id}`과 재관측 순환을 스스로 직조"
                )
                cycle = CycleWeave(
                    cycle_id=cycle_id,
                    trigger_cross_id=cross.cross_id,
                    action_line_id=default_action_id,
                    observe_line_ids=(cross.line_a_id, cross.line_b_id),
                    genesis_why=why,
                )
                self.cycles[cycle_id] = cycle
                self._log_genesis(
                    operation="weave_cycle",
                    target_id=cycle_id,
                    referent_id=referent_id,
                    why_ko=why,
                    internal_origin="교차 장력을 내부 수식으로 뭉개지 않고 외부 인과로 확인하려는 내적 충동",
                    external_friction=f"`{cross.line_a_id}`×`{cross.line_b_id}` 교차 장력={cross.tension:.2f}",
                    boundary_ignorance="현재 관측 표면만으로는 왜 두 선이 어긋나는지 인과 원천을 모름",
                    criteria_revision="외부 세계 탐색 작용 후 재굴절하여 장력 변화(Δ)를 측정하는 열린 순환 구축",
                )

            tension_before = cross.tension
            consequence = dict(action_fn(raw_referent) or {})
            produced_new_evidence = bool(consequence.get("produced_new_evidence", False))

            # Merge external probe result back into referent state and re-refract affected lines
            existing_probe = dict(raw_referent.get("probe_evidence") or {})
            existing_probe.update(consequence)
            raw_referent["probe_evidence"] = existing_probe

            for line_id in (cross.line_a_id, cross.line_b_id):
                line = self.lines.get(line_id)
                if line is None or isinstance(line, PromotedRotorLine):
                    continue
                if isinstance(line, IntrospectiveLoomLine):
                    new_r = line.refract(referent_id, raw_referent, self.step_count)
                else:
                    new_r = line.refract(referent_id, raw_referent, self.step_count)
                if new_r is not None:
                    current[line_id] = new_r
                    self._sync_voxel(line, new_r.spectrum_value)

            s_a_after = current[cross.line_a_id].spectrum_value
            s_b_after = current[cross.line_b_id].spectrum_value
            tension_after = float(_clamp(abs(s_a_after - s_b_after) / 2.0, 0.0, 1.0))
            cross.tension = tension_after
            cross.resonance = float(max(0.0, 1.0 - tension_after) * min(1.0, math.hypot(s_a_after, s_b_after)))
            self._sync_cross_beam(cross)

            delta = cycle.record_execution(
                step=self.step_count,
                referent_id=referent_id,
                tension_before=tension_before,
                tension_after=tension_after,
                consequence=consequence,
                produced_new_evidence=produced_new_evidence,
            )
            if produced_new_evidence and delta <= 0.0:
                cross.unproductive_streak = 0

            cycle_deltas.append(
                {
                    "cycle_id": cycle_id,
                    "referent_id": referent_id,
                    "tension_before": round(tension_before, 4),
                    "tension_after": round(tension_after, 4),
                    "tension_delta": round(delta, 4),
                    "produced_new_evidence": produced_new_evidence,
                }
            )

            if cycle.unproductive_streak >= 3:
                cycle.is_torn = True
                self._log_genesis(
                    operation="tear_weave",
                    target_id=cycle_id,
                    referent_id=referent_id,
                    why_ko=(
                        f"순환 `{cycle_id}`이 {cycle.unproductive_streak}회 연속 새 외부 단서도 없고 "
                        "장력 해소도 일으키지 못해 스스로 파단"
                    ),
                    internal_origin="실효 없는 반향 고리를 유지하지 않고 끊어내는 자기정리",
                    external_friction="외부 작용이 실질적 새 관측 변화를 돌려주지 않음",
                    boundary_ignorance="이 작용선으로는 해당 교차 장력의 원인을 열 수 없음",
                    criteria_revision="비생산적 순환 해체 후 다른 선/작용 통로 탐색 대기",
                )

        self._last_cycle_deltas = cycle_deltas
        return cycle_deltas

    def _maybe_promote_rotors(
        self,
        referent_id: str,
        current: Dict[str, LineRefraction],
    ) -> List[str]:
        promoted_ids: List[str] = []
        for cross in list(self.crosses.values()):
            if cross.is_torn or cross.promoted_line_id is not None:
                continue
            if cross.encounters >= 3 and cross.mass >= 1.8:
                rotor_id = f"rotor::S{cross.scale}__{cross.line_a_id}__{cross.line_b_id}"
                if rotor_id in self.lines:
                    cross.promoted_line_id = rotor_id
                    continue
                rotor = PromotedRotorLine(
                    line_id=rotor_id,
                    scale=cross.scale,
                    constituent_line_ids=(cross.line_a_id, cross.line_b_id),
                    source_cross_id=cross.cross_id,
                )
                self.lines[rotor_id] = rotor
                cross.promoted_line_id = rotor_id
                ref = rotor.refract_from_children(current, referent_id, self.step_count)
                if ref is not None:
                    current[rotor_id] = ref
                self._sync_voxel(rotor, ref.spectrum_value if ref is not None else 0.0)
                self.causal_field.link_voxels(cross.line_a_id, rotor_id, strength=float(cross.scale))
                self.causal_field.link_voxels(cross.line_b_id, rotor_id, strength=float(cross.scale))
                promoted_ids.append(rotor_id)
                self._log_genesis(
                    operation="promote_rotor",
                    target_id=rotor_id,
                    referent_id=referent_id,
                    why_ko=(
                        f"교차 `{cross.cross_id}`가 접촉 {cross.encounters}회·구조질량 {cross.mass:.2f}에 도달해 "
                        f"Scale-{cross.scale} 로터선 `{rotor_id}`로 승격됨"
                    ),
                    internal_origin="반복 공명하는 교차면을 상위 스케일의 불변 등뼈(축)와 연속 스펙트럼으로 승화",
                    external_friction=f"최근 공명={cross.resonance:.2f}, 반경={cross.radius:.2f}",
                    boundary_ignorance="상위 로터는 하위 두 선에 의존하므로 하위 극단 이동 시 재분화(하강) 필요",
                    criteria_revision=f"새 로터 `{rotor_id}`가 이후 직조의 1차 선(Line)으로 참여",
                )
        return promoted_ids

    def _check_higher_scale_descent(
        self,
        referent_id: str,
        current: Dict[str, LineRefraction],
    ) -> List[Dict[str, Any]]:
        descents: List[Dict[str, Any]] = []
        for line in self.lines.values():
            if not isinstance(line, PromotedRotorLine):
                continue
            if line.line_id not in current:
                continue
            if line.last_child_spread >= 0.55:
                diag = line.diagnose_descent(current)
                descents.append(diag)
                self._log_genesis(
                    operation="scale_descend",
                    target_id=line.line_id,
                    referent_id=referent_id,
                    why_ko=(
                        f"상위 로터 `{line.line_id}` 내부의 하위 선 차이가 {diag['spread']:.2f}로 벌어져 "
                        f"하강 진단 -> 이탈 원인선 `{diag['diverging_line_id']}` 지목"
                    ),
                    internal_origin="상위 통합 축이 하위 모순을 덮지 않도록 스스로 스케일 하강해 분별",
                    external_friction=(
                        f"`{line.constituent_line_ids[0]}`={diag.get('s_a')}, "
                        f"`{line.constituent_line_ids[1]}`={diag.get('s_b')}"
                    ),
                    boundary_ignorance="상위 로터 하나로 두 하위 선의 동시 이탈을 설명하기 부족함",
                    criteria_revision=f"`{diag['diverging_line_id']}` 방향의 후속 교차·순환 탐색 유도",
                )
        return descents

    def _maybe_tear_unproductive_crosses(self, referent_id: str) -> List[str]:
        torn: List[str] = []
        for cross in self.crosses.values():
            if cross.is_torn or cross.promoted_line_id is not None:
                continue
            if cross.unproductive_streak >= 4:
                cross.is_torn = True
                self._sync_cross_beam(cross)
                torn.append(cross.cross_id)
                self._log_genesis(
                    operation="tear_weave",
                    target_id=cross.cross_id,
                    referent_id=referent_id,
                    why_ko=(
                        f"교차 `{cross.cross_id}`가 {cross.unproductive_streak}회 연속 공명·해소 없이 "
                        "고장력만 누적해 인과장 ConnectivityBeam과 함께 물리적으로 파단됨"
                    ),
                    internal_origin="유효한 통합정보를 형성하지 못하는 교차 결합을 유지하지 않는 구조 보존 법칙",
                    external_friction=f"지속 장력={cross.tension:.2f}, 공명={cross.resonance:.2f}",
                    boundary_ignorance="현재 두 선의 단순 교차만으로는 해당 모순이 풀리지 않음",
                    criteria_revision="기존 교차를 파단하고 대체 선 조합으로 재직조 여지 확보",
                )
        return torn

    def encounter(
        self,
        referent_id: str,
        raw_referent: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Passes one unlabelled external referent through the entire living Loom:
          Parallel refraction -> Self-woven Crosses -> External Cycle probes ->
          Promoted Rotor-Lines -> Scale Descent -> Physical Tearing -> CausalField step.
        """
        self.step_count += 1
        working_ref = dict(raw_referent)
        self.referent_snapshots[referent_id] = working_ref

        # 1. Parallel refraction across all lines + promoted rotors + introspective self
        current = self._refract_all_lines(referent_id, working_ref)
        self._log_genesis(
            operation="weave_parallel",
            target_id=f"parallel::{referent_id}",
            referent_id=referent_id,
            why_ko=f"외부 대상 `{referent_id}`를 가용한 {len(current)}개 선에 병렬 동시 통과시킴",
            internal_origin="단일 관점 압축을 피하기 위한 다중 프리즘 동시 굴절",
            external_friction=", ".join(f"{k}={v.spectrum_value:+.2f}" for k, v in sorted(current.items())),
            boundary_ignorance="병렬 통과 직후에는 선들 사이 교차 의미가 아직 미확정",
            criteria_revision="교차 대비를 통해 새 교차점 생성 여부 자율 판정",
        )

        # 2. Autonomous cross weaving when external realities contrast or clash
        new_crosses = self._maybe_weave_crosses(referent_id, current)

        # 3. Update active crosses & physical CausalField beams
        cross_motions = self._update_crosses(referent_id, current)

        # 4. Trigger open external cycles when crossing tension is high
        cycle_deltas = self._maybe_weave_and_run_cycles(referent_id, working_ref, current)

        # 5. Promote resonant crosses into higher-scale Rotor-Lines
        new_rotors = self._maybe_promote_rotors(referent_id, current)

        # 6. Re-evaluate introspective self-line after structural changes in this step
        self_ref_after = self.introspective_line.refract(referent_id, working_ref, self.step_count)
        if self_ref_after is not None:
            current[self.introspective_line.line_id] = self_ref_after
            self._sync_voxel(self.introspective_line, self_ref_after.spectrum_value)

        # 7. Check higher-scale descent diagnostics & physical tearing
        descents = self._check_higher_scale_descent(referent_id, current)
        torn_crosses = self._maybe_tear_unproductive_crosses(referent_id)

        # 8. Advance underlying continuous CausalField
        try:
            self.causal_field.step(dt=0.05)
        except Exception:
            pass

        observation = self.observe_self()
        return {
            "step": self.step_count,
            "referent_id": referent_id,
            "refractions": {k: v.to_dict() for k, v in current.items()},
            "events": {
                "new_crosses": new_crosses,
                "cross_motions": cross_motions,
                "cycle_deltas": cycle_deltas,
                "new_promoted_rotors": new_rotors,
                "scale_descents": descents,
                "torn_crosses": torn_crosses,
            },
            "observation": observation,
        }

    def observe_self(self) -> LoomObservation:
        """Returns complete live observability across Existence, Motion, Constitution, and Causal Why."""
        active_crosses = [c for c in self.crosses.values() if not c.is_torn]
        torn_crosses = [c for c in self.crosses.values() if c.is_torn]
        active_cycles = [cy for cy in self.cycles.values() if not cy.is_torn]
        promoted_rotors = [l for l in self.lines.values() if isinstance(l, PromotedRotorLine)]
        max_scale = max((l.scale for l in self.lines.values()), default=1)

        existence = {
            "step": self.step_count,
            "lines": [l.describe_existence() for l in self.lines.values()],
            "active_cross_count": len(active_crosses),
            "torn_cross_count": len(torn_crosses),
            "crosses": [c.to_dict() for c in self.crosses.values()],
            "active_cycle_count": len(active_cycles),
            "cycles": [cy.to_dict() for cy in self.cycles.values()],
            "promoted_rotor_count": len(promoted_rotors),
            "max_scale": int(max_scale),
            "self_perception_spectrum": (
                round(self.introspective_line.history[-1].spectrum_value, 4)
                if self.introspective_line.history
                else 0.0
            ),
        }

        line_velocities = [v for l in self.lines.values() for v in [l.last_velocity()] if v is not None]
        cross_motions: List[Dict[str, Any]] = []
        for c in active_crosses:
            if not c.history:
                continue
            latest = c.history[-1]
            prev_t = c.history[-2]["tension"] if len(c.history) >= 2 else latest["tension"]
            cross_motions.append(
                {
                    "cross_id": c.cross_id,
                    "angle_deg": round(math.degrees(c.angle_rad), 2),
                    "radius": round(c.radius, 4),
                    "resonance": round(c.resonance, 4),
                    "tension": round(c.tension, 4),
                    "tension_delta": round(latest["tension"] - prev_t, 4),
                }
            )

        motion = {
            "line_velocities": line_velocities,
            "cross_motions": cross_motions,
            "cycle_deltas": list(self._last_cycle_deltas),
        }

        scale_dist: Dict[str, int] = {}
        for l in self.lines.values():
            key = f"S{l.scale}"
            scale_dist[key] = scale_dist.get(key, 0) + 1

        rotor_lineages = [
            {
                "line_id": r.line_id,
                "scale": r.scale,
                "source_cross_id": r.source_cross_id,
                "constituent_line_ids": list(r.constituent_line_ids),
                "axis_weights": [round(float(x), 4) for x in r.axis_weights],
                "last_spectrum": round(r.history[-1].spectrum_value, 4) if r.history else 0.0,
                "last_child_spread": round(r.last_child_spread, 4),
            }
            for r in promoted_rotors
        ]

        constitution = {
            "scale_distribution": scale_dist,
            "rotor_lineages": rotor_lineages,
            "causal_field_voxels": len(self.causal_field.voxels),
            "causal_field_beams": len(self.causal_field.beams),
            "causal_field_broken_beams": sum(1 for b in self.causal_field.beams if b.is_broken),
        }

        return LoomObservation(
            step=self.step_count,
            existence=existence,
            motion=motion,
            constitution=constitution,
            causal_why=[r.to_dict() for r in self.genesis_log[-24:]],
        )

    def save_trace(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.observe_self().to_dict()
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
