r"""
ELYSIA Whole-System Scale-Rotor Prism & Brush Architecture (v3.0)
==================================================================

Unifies the entire system under the Master Integrated Document principles:
  1. Unchangeable External Invariants (`ExternalInvariant`): fixed outside anchors
     (Python grammar, OS filesystem, screen pixel optics, real human utterance/feedback)
     so "change" can be anchored against what does NOT move.
  2. Conventions as Tools (`ConventionPrism` & `FormBrush`): formal conventions (language,
     code, filesystem topology, optical screen) act as refracting Prisms and outward-
     projecting Form Brushes rather than sealed cages.
  3. Scale-Rotor Cycles:
       Convergence (Seed) -> Attraction (Resonance) -> Repulsion (Boundary/Limit) -> Expansion (Outward Brush)
     connected directly to `SelfWeavingLoom` (with ancestor repulsion preventing internal cloning).
  4. Principle-Habit Self-Perception (`PrincipleHabitObserver`): perceives *why* the system
     keeps repeating the same internal principle against immutable external anchors and points
     *outside* the program for genuine room to change (`external_room_for_change`).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from core.embodiment.external_reality_lines import (
    FileSurfaceLine,
    FilesystemTopologyLine,
    LiveScreenLine,
    PythonRuntimeLine,
    _read_referent_text,
    external_world_probe_action,
)
from core.topology.self_weaving_loom import (
    CausalLine,
    LineRefraction,
    SelfWeavingLoom,
    _clamp,
)


@dataclass(frozen=True)
class ExternalInvariant:
    """An unchangeable external anchor against which internal change becomes perceptible."""

    invariant_id: str
    domain: str
    description_ko: str
    check_fn: Callable[[Dict[str, Any]], Dict[str, Any]]
    immutable_by_system: bool = True

    def verify(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        res = dict(self.check_fn(payload) or {})
        res["invariant_id"] = self.invariant_id
        res["domain"] = self.domain
        res["immutable_by_system"] = True
        return res


def _check_python_grammar_invariant(payload: Dict[str, Any]) -> Dict[str, Any]:
    text, path_obj, _ = _read_referent_text(payload)
    if not text and path_obj is None:
        return {"applies": False, "accepted": None, "note": "텍스트/파일 입력 없음"}
    try:
        compile(text, str(path_obj or "world_input"), "exec")
        return {
            "applies": True,
            "accepted": True,
            "note": "파이썬 문법 불변식 통과(외부 실행기 기준)",
        }
    except SyntaxError as exc:
        return {
            "applies": True,
            "accepted": False,
            "note": f"파이썬 문법 불변식 거부: {exc.msg} (line {exc.lineno})",
        }
    except Exception as exc:
        return {"applies": True, "accepted": False, "note": f"컴파일 거부: {exc}"}


def _check_os_filesystem_invariant(payload: Dict[str, Any]) -> Dict[str, Any]:
    path_val = payload.get("path")
    if not path_val:
        return {"applies": False, "accepted": None, "note": "파일 경로 없음"}
    p = Path(path_val)
    if not p.exists():
        return {"applies": True, "accepted": False, "note": f"OS 파일 미존재: {p}"}
    try:
        st = p.stat()
        return {
            "applies": True,
            "accepted": True,
            "size_bytes": int(st.st_size),
            "note": f"OS 실물 파일({st.st_size} bytes, suffix={p.suffix or 'none'})",
        }
    except OSError as exc:
        return {"applies": True, "accepted": False, "note": f"OS 접근 마찰: {exc}"}


def _check_human_utterance_invariant(payload: Dict[str, Any]) -> Dict[str, Any]:
    utt = str(payload.get("human_utterance") or "").strip()
    fb = str(payload.get("human_feedback") or "").strip()
    combined = f"{utt} {fb}".strip()
    if not combined:
        return {"applies": False, "accepted": None, "note": "인간 발화/반론 없음"}
    rejection_markers = ("아니", "틀렸", "그게 아니", "무시", "갇혀", "답습", "밖에서", "못 보", "아직")
    is_pushback = any(m in combined for m in rejection_markers)
    return {
        "applies": True,
        "accepted": not is_pushback,
        "is_pushback": is_pushback,
        "utterance_excerpt": combined[:100],
        "note": (
            f"인간 상호작용 반론/경계 제시: '{combined[:64]}'"
            if is_pushback
            else f"인간 발화 유입: '{combined[:64]}'"
        ),
    }


def _check_screen_optics_invariant(payload: Dict[str, Any]) -> Dict[str, Any]:
    if payload.get("screen_rgb") is not None:
        arr = np.asarray(payload["screen_rgb"])
        return {
            "applies": True,
            "accepted": bool(arr.size > 0),
            "note": f"외부 시각 픽셀 단면({arr.shape})",
        }
    if payload.get("capture_screen"):
        return {
            "applies": True,
            "accepted": True,
            "note": "실시간 OS 디스플레이 광학 입력",
        }
    return {"applies": False, "accepted": None, "note": "시각 단면 입력 없음"}


def default_external_invariants() -> List[ExternalInvariant]:
    return [
        ExternalInvariant(
            invariant_id="inv::python_grammar",
            domain="code_runtime",
            description_ko="파이썬 컴파일러/문법 규약 (엘리시아 내부 논리로 바꿀 수 없는 외부 기준)",
            check_fn=_check_python_grammar_invariant,
        ),
        ExternalInvariant(
            invariant_id="inv::os_filesystem",
            domain="filesystem",
            description_ko="운영체제 파일 존재·바이트·경로 실재",
            check_fn=_check_os_filesystem_invariant,
        ),
        ExternalInvariant(
            invariant_id="inv::human_utterance",
            domain="human_interaction",
            description_ko="인간 발화·반박·수정 요구 (개발자 정답표가 아닌 실제 대화 상대의 반응)",
            check_fn=_check_human_utterance_invariant,
        ),
        ExternalInvariant(
            invariant_id="inv::screen_optics",
            domain="visual_media",
            description_ko="실시간 화면/이미지 광학 픽셀 장",
            check_fn=_check_screen_optics_invariant,
        ),
    ]


class LanguageMetaLine(CausalLine):
    """
    Top-level cognitive Convergence & Boundary Line (`language_meta`):
    Treats human utterances, contextual prompts, and semantic boundary cues as
    macro-scale refractions coupled into the same Loom as raw file/screen/runtime signals.
    """

    kind: str = "language_macro"

    def __init__(self) -> None:
        super().__init__(
            line_id="line::language_meta",
            scale=1,
            chromatic_bias=(0.30, 0.50, 0.20),
            description="오감과 사고가 수렴·확장하며 경계를 맺는 최상위 언어 사고선",
        )

    def refract(
        self,
        referent_id: str,
        raw_referent: Dict[str, Any],
        step: int,
    ) -> Optional[LineRefraction]:
        utt = str(raw_referent.get("human_utterance") or "").strip()
        fb = str(raw_referent.get("human_feedback") or "").strip()
        ctx = str(raw_referent.get("external_context") or "").strip()
        text_fallback = ""
        if not (utt or fb or ctx) and raw_referent.get("raw_text"):
            text_fallback = str(raw_referent["raw_text"])[:240]

        combined = " | ".join(part for part in (utt, fb, ctx, text_fallback) if part).strip()
        if not combined:
            return None

        question_density = float(combined.count("?") + combined.count("왜") + combined.count("어떻게"))
        contrast_density = float(
            sum(combined.count(tok) for tok in ("아니", "한계", "경계", "밖에", "다르", "모순", "틀렸"))
        )
        synthesis_density = float(
            sum(combined.count(tok) for tok in ("연결", "통합", "교차", "흐름", "씨앗", "확장", "함께"))
        )
        length_norm = min(1.0, len(combined) / 140.0)

        # Negative pole: questioning / boundary-challenging friction;
        # Positive pole: integrative / expansive conceptual articulation.
        raw_score = 0.35 * synthesis_density - 0.55 * contrast_density - 0.25 * question_density + 0.15 * length_norm
        spectrum = _clamp(math.tanh(raw_score * 0.65), -1.0, 1.0)
        return self._record_refraction(
            referent_id=referent_id,
            spectrum_value=spectrum,
            raw_observables={
                "char_len": float(len(combined)),
                "question_markers": question_density,
                "contrast_markers": contrast_density,
                "synthesis_markers": synthesis_density,
            },
            chromatic=(0.25 + 0.35 * length_norm, 0.30 + 0.30 * min(1.0, synthesis_density), 0.20 + 0.40 * min(1.0, contrast_density)),
            evidence={
                "has_human_utterance": bool(utt or fb),
                "excerpt": combined[:90],
            },
            step=step,
        )


@dataclass
class PrismSpectrum:
    """Output of refracting a raw WorldEvent through a single `ConventionPrism`."""

    convention_id: str
    domain: str
    scale_focus: str  # "micro" | "meso_static" | "meso_dynamic" | "macro_language"
    spectrum_value: float
    what_it_retracted_ko: str
    convention_limit_ko: str
    observables: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConventionPrism:
    """Instrumentalized formal convention used as an optical Prism (never an absolute cage)."""

    convention_id: str
    domain: str
    scale_focus: str
    refract_fn: Callable[[str, Dict[str, Any], Dict[str, LineRefraction]], Optional[PrismSpectrum]]


def _prism_micro_surface(
    event_id: str, payload: Dict[str, Any], loom_refs: Dict[str, LineRefraction]
) -> Optional[PrismSpectrum]:
    ref = loom_refs.get("line::file_surface")
    if ref is None:
        return None
    return PrismSpectrum(
        convention_id="prism::symbolic_surface",
        domain="surface_morphology",
        scale_focus="micro",
        spectrum_value=ref.spectrum_value,
        what_it_retracted_ko=(
            f"문자/기호/들여쓰기 표면 밀도 스펙트럼={ref.spectrum_value:+.3f}"
        ),
        convention_limit_ko="표면 기호 규약만으로는 실제 실행 의미나 인간 의도, 문서 맥락을 보지 못함",
        observables=dict(ref.raw_observables),
    )


def _prism_code_convention(
    event_id: str, payload: Dict[str, Any], loom_refs: Dict[str, LineRefraction]
) -> Optional[PrismSpectrum]:
    ref = loom_refs.get("line::python_runtime")
    if ref is None:
        return None
    compiles = bool(ref.evidence.get("compiles", False))
    return PrismSpectrum(
        convention_id="prism::python_compiler",
        domain="executable_grammar",
        scale_focus="meso_static",
        spectrum_value=ref.spectrum_value,
        what_it_retracted_ko=(
            f"파이썬 규약 굴절({'컴파일 수용' if compiles else '문법 거부'}, s={ref.spectrum_value:+.3f})"
        ),
        convention_limit_ko="컴파일러 규약만으로는 파이썬 밖의 자연어·문서·시각·정서적 가치를 판별할 수 없음",
        observables=dict(ref.raw_observables),
    )


def _prism_visual_static(
    event_id: str, payload: Dict[str, Any], loom_refs: Dict[str, LineRefraction]
) -> Optional[PrismSpectrum]:
    ref = loom_refs.get("line::live_screen")
    if ref is None:
        return None
    return PrismSpectrum(
        convention_id="prism::visual_optics",
        domain="visual_static_section",
        scale_focus="meso_static",
        spectrum_value=ref.spectrum_value,
        what_it_retracted_ko=(
            f"정적 시각 단면 광학 스펙트럼={ref.spectrum_value:+.3f} "
            f"(창: {ref.evidence.get('window_title', 'unknown')})"
        ),
        convention_limit_ko="정지된 픽셀 단면만으로는 시간에 따른 인과 변화와 코드·언어 구조를 알 수 없음",
        observables=dict(ref.raw_observables),
    )


def _prism_topology_dynamic(
    event_id: str, payload: Dict[str, Any], loom_refs: Dict[str, LineRefraction]
) -> Optional[PrismSpectrum]:
    ref = loom_refs.get("line::fs_topology")
    if ref is None:
        return None
    return PrismSpectrum(
        convention_id="prism::relational_topology",
        domain="neighborhood_coupling",
        scale_focus="meso_dynamic",
        spectrum_value=ref.spectrum_value,
        what_it_retracted_ko=(
            f"주변 환경 연결망 굴절={ref.spectrum_value:+.3f} "
            f"(이웃={int(ref.raw_observables.get('sibling_count', 0))}, 참조={int(ref.raw_observables.get('inbound_refs', 0))})"
        ),
        convention_limit_ko="디렉터리 이웃 관계만으로는 대상 내부 논리나 외부 사용자의 직접 피드백을 보지 못함",
        observables=dict(ref.raw_observables),
    )


def _prism_language_macro(
    event_id: str, payload: Dict[str, Any], loom_refs: Dict[str, LineRefraction]
) -> Optional[PrismSpectrum]:
    ref = loom_refs.get("line::language_meta")
    if ref is None:
        return None
    return PrismSpectrum(
        convention_id="prism::language_meta_unit",
        domain="linguistic_convergence",
        scale_focus="macro_language",
        spectrum_value=ref.spectrum_value,
        what_it_retracted_ko=(
            f"언어 사고 단위 굴절={ref.spectrum_value:+.3f} "
            f"(발췌: '{ref.evidence.get('excerpt', '')[:48]}')"
        ),
        convention_limit_ko="언어 한 문장만으로는 물리적 검증·파일 실체·광학 감각의 세부를 대신할 수 없음",
        observables=dict(ref.raw_observables),
    )


def default_convention_prisms() -> List[ConventionPrism]:
    return [
        ConventionPrism("prism::symbolic_surface", "surface_morphology", "micro", _prism_micro_surface),
        ConventionPrism("prism::python_compiler", "executable_grammar", "meso_static", _prism_code_convention),
        ConventionPrism("prism::visual_optics", "visual_static_section", "meso_static", _prism_visual_static),
        ConventionPrism("prism::relational_topology", "neighborhood_coupling", "meso_dynamic", _prism_topology_dynamic),
        ConventionPrism("prism::language_meta_unit", "linguistic_convergence", "macro_language", _prism_language_macro),
    ]


@dataclass
class ConvergenceSeed:
    """Converged cross-scale Seed shaped by Attraction (resonance) & Repulsion (limits)."""

    event_id: str
    active_prisms: List[str]
    scales_touched: List[str]
    attraction_coherence: float
    repulsion_boundaries: List[str]
    seed_summary_ko: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BrushProjection:
    """Outward Form-Brush stroke seeking change outside the system rather than inside."""

    brush_mode: str  # outward_question_to_human | external_convention_switch | external_artifact_probe | cross_scale_synthesis
    outward_expression_ko: str
    target_outside_anchor: str
    external_room_for_change: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FormBrush:
    """Projects the converged Seed & Repulsion limits outward onto external reality."""

    def project(
        self,
        seed: ConvergenceSeed,
        invariant_checks: List[Dict[str, Any]],
        prisms: List[PrismSpectrum],
    ) -> BrushProjection:
        human_inv = next(
            (inv for inv in invariant_checks if inv["invariant_id"] == "inv::human_utterance" and inv.get("applies")),
            None,
        )
        rejected_invs = [inv for inv in invariant_checks if inv.get("applies") and inv.get("accepted") is False]

        if human_inv is not None and human_inv.get("is_pushback"):
            excerpt = human_inv.get("utterance_excerpt", "")
            room = [
                f"인간이 직접 지적한 '{excerpt[:50]}'의 실제 사례·외부 대상을 사용자에게 되물어 확인하기",
                "현재 사용 중인 프리즘 밖의 감각/매질(예: 코드 외의 물리·시각·대화 맥락)로 동일 현상 다시 비춰보기",
            ]
            return BrushProjection(
                brush_mode="outward_question_to_human",
                outward_expression_ko=(
                    f"내부 규약({', '.join(seed.active_prisms)})이 인간 반론과 충돌함. "
                    f"기존 틀로 덮지 않고 바깥의 구체적 사례를 사용자에게 질의하는 붓질로 투사."
                ),
                target_outside_anchor="inv::human_utterance",
                external_room_for_change=room,
            )

        if rejected_invs:
            inv = rejected_invs[0]
            missing_scales = [
                s for s in ("micro", "meso_static", "meso_dynamic", "macro_language") if s not in seed.scales_touched
            ]
            room = [
                f"불변 기준 `{inv['invariant_id']}`의 거부 원인을 외부 문맥·인접 파일/문서에서 탐색하기",
                (
                    f"아직 닿지 않은 스케일({', '.join(missing_scales)})의 외부 자극 추가 유입"
                    if missing_scales
                    else "인간에게 이 불일치의 의도 확인 요청"
                ),
            ]
            return BrushProjection(
                brush_mode="external_convention_switch",
                outward_expression_ko=(
                    f"`{inv['invariant_id']}` 경계에서 척력 발생({inv.get('note', '')}). "
                    f"코드 규약의 한계를 인정하고 다른 스케일·외부 맥락으로 프리즘을 꺾는 붓질 투사."
                ),
                target_outside_anchor=inv["invariant_id"],
                external_room_for_change=room,
            )

        if len(seed.repulsion_boundaries) >= 2:
            return BrushProjection(
                brush_mode="external_artifact_probe",
                outward_expression_ko=(
                    f"프리즘 간 척력 경계 {len(seed.repulsion_boundaries)}개 포착. "
                    f"추가 외부 연결망과 실제 반응 데이터를 탐색해 씨앗을 밖으로 팽창시킴."
                ),
                target_outside_anchor="inv::os_filesystem",
                external_room_for_change=[
                    "동일 대상을 연결하는 작업공간 외부 파일·웹 문서·사용자 대화 추가 관찰",
                ],
            )

        return BrushProjection(
            brush_mode="cross_scale_synthesis",
            outward_expression_ko=(
                f"수렴 씨앗({len(prisms)}개 프리즘, 인력={seed.attraction_coherence:.2f})을 "
                f"상위 언어·감각 맥락으로 확장해 다음 외부 자극을 기다림."
            ),
            target_outside_anchor=prisms[0].convention_id if prisms else "outside_world",
            external_room_for_change=[
                "아직 경험하지 않은 이질적 외부 매질(소리·영상·실시간 인간 반론)과의 교차 대기",
            ],
        )


@dataclass
class HabitSelfPerception:
    """Perceives why the system repeats a principle against external anchors and looks outward for change."""

    is_repeating_principle: bool
    repeated_principle: str
    external_invariant_anchor: str
    repetition_count: int
    why_stuck_ko: str
    external_room_for_change: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PrincipleHabitObserver:
    """Watches whether the system keeps re-applying the same internal principle against fixed external reality."""

    def __init__(self, window: int = 8) -> None:
        self.window = max(3, int(window))
        self.trace: List[Dict[str, Any]] = []

    def observe(
        self,
        step: int,
        dominant_prism: str,
        anchor_invariant: str,
        has_unresolved_repulsion: bool,
        external_room: List[str],
    ) -> HabitSelfPerception:
        self.trace.append(
            {
                "step": int(step),
                "dominant_prism": dominant_prism,
                "anchor_invariant": anchor_invariant,
                "has_unresolved_repulsion": bool(has_unresolved_repulsion),
            }
        )
        if len(self.trace) > self.window:
            self.trace = self.trace[-self.window :]

        recent = self.trace[-4:]
        prism_counts = Counter(item["dominant_prism"] for item in recent)
        top_prism, top_count = prism_counts.most_common(1)[0]
        repeated_pushbacks = sum(1 for item in recent if item["has_unresolved_repulsion"])

        is_repeating = (top_count >= 3) or (repeated_pushbacks >= 2 and top_count >= 2)
        if is_repeating:
            why_ko = (
                f"최근 {len(recent)}회 동안 `{top_prism}` 원리·틀이 {top_count}회 연속 지배적이며, "
                f"변하지 않는 외부 기준 `{anchor_invariant}`과의 척력 마찰이 {repeated_pushbacks}회 반복됨. "
                f"내부 가중치나 조건문만 만져서는 외부 기준이 요구하는 새 맥락을 열 수 없어 같은 원리를 답습하고 있음."
            )
        else:
            why_ko = (
                f"현재 `{top_prism}` 중심으로 외부 기준 `{anchor_invariant}`과 교차 조율 중이며 "
                "고착적 단선 답습 징후는 아직 임계치 미만임."
            )

        return HabitSelfPerception(
            is_repeating_principle=bool(is_repeating),
            repeated_principle=top_prism,
            external_invariant_anchor=anchor_invariant,
            repetition_count=int(top_count),
            why_stuck_ko=why_ko,
            external_room_for_change=list(external_room),
        )


@dataclass
class RotorCycleReport:
    """Single full world-interaction cycle snapshot."""

    step: int
    event_id: str
    invariant_checks: List[Dict[str, Any]]
    prisms: List[PrismSpectrum]
    convergence_seed: ConvergenceSeed
    brush_projection: BrushProjection
    habit_perception: HabitSelfPerception
    loom_events: Dict[str, Any]
    self_perception_spectrum: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": int(self.step),
            "event_id": self.event_id,
            "invariant_checks": self.invariant_checks,
            "prisms": [p.to_dict() for p in self.prisms],
            "convergence_seed": self.convergence_seed.to_dict(),
            "brush_projection": self.brush_projection.to_dict(),
            "habit_perception": self.habit_perception.to_dict(),
            "loom_events": self.loom_events,
            "self_perception_spectrum": round(float(self.self_perception_spectrum), 4),
        }


class WholeSystemPrismRotor:
    """
    Whole-system Scale-Rotor combining:
      - Unchangeable `ExternalInvariant`s
      - `SelfWeavingLoom` (with ancestor repulsion + language/external lines)
      - Multi-scale `ConventionPrism`s + outward `FormBrush`
      - `PrincipleHabitObserver` (meta-perception of why a principle is repeated & where outside room lies)
    """

    def __init__(self, workspace_root: Optional[Path] = None) -> None:
        self.workspace_root = workspace_root
        self.loom = SelfWeavingLoom()
        self.loom.register_line(FileSurfaceLine())
        self.loom.register_line(PythonRuntimeLine())
        self.loom.register_line(FilesystemTopologyLine())
        self.loom.register_line(LiveScreenLine())
        self.loom.register_line(LanguageMetaLine())
        self.loom.register_action_line(
            "action::external_world_probe",
            lambda ref: external_world_probe_action(ref, workspace_root=workspace_root),
            description="실제 파일 AST·작업공간 참조·화면 확대 외부 탐색 작용",
        )

        self.invariants: Dict[str, ExternalInvariant] = {
            inv.invariant_id: inv for inv in default_external_invariants()
        }
        self.prisms: List[ConventionPrism] = default_convention_prisms()
        self.brush = FormBrush()
        self.habit_observer = PrincipleHabitObserver()
        self.cycle_reports: List[RotorCycleReport] = []

    def experience_world_event(
        self,
        event_id: str,
        payload: Dict[str, Any],
    ) -> RotorCycleReport:
        # 1. Ground against immutable external invariants first
        inv_checks = [inv.verify(payload) for inv in self.invariants.values()]

        # 2. Traverse SelfWeavingLoom (parallel, ancestor-repelling crosses, external cycles, rotors)
        loom_res = self.loom.encounter(event_id, payload)
        step = int(loom_res["step"])

        # Reconstruct typed LineRefraction map for prisms from Loom lines' latest history
        latest_refs: Dict[str, LineRefraction] = {}
        for lid, line in self.loom.lines.items():
            if line.history and line.history[-1].referent_id == event_id:
                latest_refs[lid] = line.history[-1]

        # 3. Refract through instrumentalized formal ConventionPrisms
        prism_outputs: List[PrismSpectrum] = []
        for prism in self.prisms:
            spec = prism.refract_fn(event_id, payload, latest_refs)
            if spec is not None:
                prism_outputs.append(spec)

        # 4. Convergence (Seed) + Attraction (Coherence) + Repulsion (Limits & Pushbacks)
        if len(prism_outputs) >= 2:
            vals = np.array([p.spectrum_value for p in prism_outputs], dtype=np.float64)
            spread = float(np.std(vals))
            attraction = float(max(0.0, 1.0 - spread))
        elif len(prism_outputs) == 1:
            attraction = 0.5
        else:
            attraction = 0.0

        repulsion_boundaries: List[str] = []
        for inv in inv_checks:
            if inv.get("applies") and inv.get("accepted") is False:
                repulsion_boundaries.append(str(inv.get("note", inv["invariant_id"])))
        for p in prism_outputs:
            repulsion_boundaries.append(f"{p.convention_id} 한계: {p.convention_limit_ko}")
        for c in self.loom.crosses.values():
            if not c.is_torn and c.tension >= 0.40:
                repulsion_boundaries.append(
                    f"교차 척력({c.line_a_id} × {c.line_b_id}): 장력={c.tension:.2f}"
                )

        scales_touched = list(dict.fromkeys(p.scale_focus for p in prism_outputs))
        seed = ConvergenceSeed(
            event_id=event_id,
            active_prisms=[p.convention_id for p in prism_outputs],
            scales_touched=scales_touched,
            attraction_coherence=round(attraction, 4),
            repulsion_boundaries=repulsion_boundaries[:8],
            seed_summary_ko=(
                f"`{event_id}`가 {len(prism_outputs)}개 프리즘({', '.join(scales_touched) or 'none'})에서 "
                f"인력 {attraction:.2f}로 수렴하며 {len(repulsion_boundaries)}개 경계·척력을 형성"
            ),
        )

        # 5. Expansion: FormBrush projects outward instead of sealing inside
        brush_proj = self.brush.project(seed, inv_checks, prism_outputs)

        # 6. Principle-Habit Self-Perception: inspect why we may be repeating an internal principle
        if prism_outputs:
            dominant_prism = max(prism_outputs, key=lambda p: abs(p.spectrum_value)).convention_id
        else:
            dominant_prism = "none"
        unresolved_repulsion = any(
            inv.get("applies") and inv.get("accepted") is False for inv in inv_checks
        ) or any(not c.is_torn and c.tension >= 0.45 for c in self.loom.crosses.values())

        habit = self.habit_observer.observe(
            step=step,
            dominant_prism=dominant_prism,
            anchor_invariant=brush_proj.target_outside_anchor,
            has_unresolved_repulsion=unresolved_repulsion,
            external_room=brush_proj.external_room_for_change,
        )

        self_val = float(
            loom_res["refractions"].get("line::introspective_self", {}).get("spectrum_value", 0.0)
        )
        report = RotorCycleReport(
            step=step,
            event_id=event_id,
            invariant_checks=inv_checks,
            prisms=prism_outputs,
            convergence_seed=seed,
            brush_projection=brush_proj,
            habit_perception=habit,
            loom_events=loom_res["events"],
            self_perception_spectrum=self_val,
        )
        self.cycle_reports.append(report)
        if len(self.cycle_reports) > 128:
            self.cycle_reports = self.cycle_reports[-128:]
        return report

    def observe_whole_system(self) -> Dict[str, Any]:
        loom_obs = self.loom.observe_self()
        latest_cycle = self.cycle_reports[-1].to_dict() if self.cycle_reports else None
        return {
            "invariants": [
                {
                    "invariant_id": inv.invariant_id,
                    "domain": inv.domain,
                    "description_ko": inv.description_ko,
                    "immutable_by_system": inv.immutable_by_system,
                }
                for inv in self.invariants.values()
            ],
            "loom": loom_obs.to_dict(),
            "latest_cycle": latest_cycle,
            "total_cycles": len(self.cycle_reports),
        }

    def render_ko(self) -> str:
        lines: List[str] = []
        lines.append("=== [변하지 않는 외부 기준 (Immutable External Invariants)] ===")
        for inv in self.invariants.values():
            lines.append(f"  - [{inv.domain}] {inv.invariant_id}: {inv.description_ko}")
        lines.append("")
        lines.append(self.loom.observe_self().render_ko())
        if self.cycle_reports:
            lc = self.cycle_reports[-1]
            lines.append("")
            lines.append("=== [수렴·인척력·팽창 & 원리 답습 자기지각 (Whole-System Cycle)] ===")
            lines.append(f"  - 수렴 씨앗: {lc.convergence_seed.seed_summary_ko}")
            lines.append(
                f"  - 척력/경계 ({len(lc.convergence_seed.repulsion_boundaries)}건): "
                + (" | ".join(lc.convergence_seed.repulsion_boundaries[:4]) or "없음")
            )
            lines.append(
                f"  - 브러쉬 외부 투사 [{lc.brush_projection.brush_mode}]: "
                f"{lc.brush_projection.outward_expression_ko}"
            )
            lines.append(
                f"  - 원리 답습 지각 (반복여부={lc.habit_perception.is_repeating_principle}, "
                f"기준={lc.habit_perception.external_invariant_anchor}): "
                f"{lc.habit_perception.why_stuck_ko}"
            )
            lines.append(
                "  - 밖에서 찾은 변화의 여지: "
                + " / ".join(lc.habit_perception.external_room_for_change)
            )
        return "\n".join(lines)

    def save_system_trace(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.observe_whole_system()
        payload["recent_cycles"] = [c.to_dict() for c in self.cycle_reports[-16:]]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
