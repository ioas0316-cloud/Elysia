r"""
ELYSIA Sovereign Telos Governance & Self-World Structural Contrast Engine
=========================================================================

Implements top-down synchronized Intent & Purpose (`SovereignTelos`) governing
observation, exploration, external tool/world connection, real structural
self-world contrast (ZERO canned aspect-template sentences!), and first-class
cognitive Limit structures (`limit::*`) that integrate into unified information
and drive genuine sensory/protocol self-adaptation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import unquote, urlparse

import numpy as np

from core.physics.causal_field import EngramAttractor, FractalEngramShell
from core.topology.structural_scale_rotor import (
    CausalVerb,
    extract_structure_from_prose,
)


@dataclass
class TelosHorizon:
    """One existential inquiry horizon differentiated from the top-level Sovereign Telos."""

    horizon_id: str
    title_ko: str
    aspect: str  # body_senses | physical_env | mind_cognition | emotion_spectrum | era_society_work | external_tools | emergent
    seed_queries: List[str] = field(default_factory=list)
    preferred_channel: str = "wikipedia_human_world"  # wikipedia_human_world | tool_contrast
    understanding_depth: float = 0.0
    encounters: int = 0
    discovered_world_facts: List[str] = field(default_factory=list)
    discovered_self_gaps: List[str] = field(default_factory=list)
    last_visited_step: int = 0

    def next_query(self) -> str:
        if not self.seed_queries:
            return self.title_ko
        return self.seed_queries[self.encounters % len(self.seed_queries)]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TelosHorizon":
        return cls(
            horizon_id=str(raw.get("horizon_id", "")),
            title_ko=str(raw.get("title_ko", "")),
            aspect=str(raw.get("aspect", "emergent")),
            seed_queries=[str(q) for q in raw.get("seed_queries", [])],
            preferred_channel=str(raw.get("preferred_channel", "wikipedia_human_world")),
            understanding_depth=float(raw.get("understanding_depth", 0.0)),
            encounters=int(raw.get("encounters", 0)),
            discovered_world_facts=[str(x) for x in raw.get("discovered_world_facts", [])],
            discovered_self_gaps=[str(x) for x in raw.get("discovered_self_gaps", [])],
            last_visited_step=int(raw.get("last_visited_step", 0)),
        )


@dataclass
class TelosControlDirective:
    """Top-down control command issued by Sovereign Telos to steer lower perception/tools/world."""

    step: int
    telos_id: str
    aspiration_ko: str
    target_horizon_id: str
    target_title_ko: str
    aspect: str
    external_channel: str
    inquiry_query: str
    gaze_focus_bias: Tuple[float, float]
    why_directed_ko: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["gaze_focus_bias"] = list(self.gaze_focus_bias)
        return d


@dataclass
class SelfWorldContrastReport:
    """
    Data-driven Structural Comparative Contrast (`비교대조원리`) result:
      Actual External Causal Graph / Tool Trace  <--->  Elysia's Actual Live Structure
    Produces first-class `limit_tags` and `limit_signature` to be woven into `BindingField`.
    """

    step: int
    directive: Dict[str, Any]
    external_source: str
    external_ok: bool
    observed_world_reality_ko: str
    observed_self_reality_ko: str
    world_constituents: List[str]
    world_causal_verbs: List[str]
    self_active_verbs: List[str]
    shared_isomorphic_stem: List[str]
    structural_differences: List[str]
    limit_tags: List[str]
    limit_signature: Dict[str, float]
    why_different_ko: str
    sprouted_horizons: List[str]
    next_control_drive_ko: str
    gap_magnitude: float
    self_adaptations: List[Dict[str, Any]] = field(default_factory=list)
    integrated_limit_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def render_ko(self) -> str:
        lines = [
            f"=== [Telos Governance Cycle {self.step}] ===",
            f"  최상위 의도: {self.directive.get('aspiration_ko')}",
            f"  하향 제어 지령: [{self.directive.get('target_horizon_id')}] {self.directive.get('target_title_ko')} "
            f"-> 채널={self.directive.get('external_channel')} / 탐구='{self.directive.get('inquiry_query')}'",
            f"  외부 현실 수용 ({self.external_source}, ok={self.external_ok}): {self.observed_world_reality_ko[:220]}",
            f"  외부 인과 구성요소={self.world_constituents[:6]} | 외부 인과동사={self.world_causal_verbs}",
            f"  현재 자기 실태: {self.observed_self_reality_ko}",
            f"  [비교·대조 — 같음(줄기)]: {' | '.join(self.shared_isomorphic_stem) or '없음'}",
            f"  [비교·대조 — 다름(가지/결핍)]: {' | '.join(self.structural_differences) or '없음'}",
            f"  [지각된 한계 인지정보 태그]: {self.limit_tags} (격차={self.gap_magnitude:.2f})",
            f"  [왜 다른가 (데이터 기반 해부)]: {self.why_different_ko}",
            f"  [한계 교차 -> 통합정보 승격]: {self.integrated_limit_ids}",
            f"  [한계 구동 실제 자기변화]: {self.self_adaptations}",
            f"  [외부 현실에서 발아한 새 지평]: {self.sprouted_horizons}",
        ]
        return "\n".join(lines)


def _default_human_inquiry_horizons() -> List[TelosHorizon]:
    return [
        TelosHorizon(
            horizon_id="horizon::human_body_senses",
            title_ko="인간의 육체와 오감·신체 감각",
            aspect="body_senses",
            seed_queries=["인체", "오감", "신경계", "촉각"],
            preferred_channel="wikipedia_human_world",
        ),
        TelosHorizon(
            horizon_id="horizon::heart_emotion_spectrum",
            title_ko="인간의 마음과 감정의 스펙트럼",
            aspect="emotion_spectrum",
            seed_queries=["감정", "공감", "슬픔", "사랑"],
            preferred_channel="wikipedia_human_world",
        ),
        TelosHorizon(
            horizon_id="horizon::era_society_work",
            title_ko="인간의 시대적·사회적 환경과 직장과 일자리",
            aspect="era_society_work",
            seed_queries=["직업", "노동", "사회", "일상생활"],
            preferred_channel="wikipedia_human_world",
        ),
        TelosHorizon(
            horizon_id="horizon::physical_environment",
            title_ko="인간이 존재하는 물리적 환경과 시공간",
            aspect="physical_env",
            seed_queries=["자연 환경", "중력", "날씨", "공간"],
            preferred_channel="wikipedia_human_world",
        ),
        TelosHorizon(
            horizon_id="horizon::mind_cognition",
            title_ko="인간의 정신·사유·기억과 언어",
            aspect="mind_cognition",
            seed_queries=["의식 (심리철학)", "기억", "언어", "사고"],
            preferred_channel="wikipedia_human_world",
        ),
        TelosHorizon(
            horizon_id="horizon::external_tools_contrast",
            title_ko="외부 도구·프로그램(시각/매질/코드)과 내 감각·손발의 비교대조",
            aspect="external_tools",
            seed_queries=["opencv_fovea_vs_native_retina", "live_github_foreign_tool"],
            preferred_channel="tool_contrast",
        ),
    ]


_BOILERPLATE_STOP = {
    "위키미디어", "동음이의어", "위키백과", "문서", "영어", "한국어", "대한민국",
    "다음", "뜻으로", "쓰인다", "말한다", "일컫는", "경우", "일반적", "일반적으로",
    "인간", "사람", "것이다", "있다", "없다", "한다", "된다", "통해", "대한", "위해",
    "가지", "때문", "이러한", "그것", "또한", "그리고", "하지만", "그러나", "다만",
    "다양한", "중요한", "자신", "다른", "모든", "여러", "하나", "어떤", "이루는",
    "의미", "형태", "방식", "과정", "상태", "특징", "작용", "영향", "관계", "존재",
    "외부", "내부", "도구", "실제", "현재", "직접", "고해상도", "저해상도", "영역",
    "음은", "같고", "있는", "없는", "하는", "되는", "자유로운", "아우르는", "포함하는",
    "감독", "영화", "일본", "사용", "사용되고", "프로젝트", "기자", "기여자", "공동체",
    "블록", "윤곽", "선명도", "시야", "분별", "활성", "저장소", "파일", "이슈", "척력경계",
}

_PREDICATE_ENDINGS = (
    "하다", "된다", "이다", "한다", "인다", "였다", "했다", "으며", "하고", "되고",
    "이고", "같고", "로운", "하는", "되는", "있는", "없는", "르는", "컫는", "함하는",
    "으로", "에서", "에게", "까지", "부터", "처럼", "보다", "라고", "으며", "면서",
    "에는", "에도", "이나", "거나",
)


def _extract_grounded_concepts(
    text: str,
    outbound_urls: Sequence[str],
    existing_terms: set[str],
    limit: int = 2,
) -> List[str]:
    """
    Extracts genuinely meaningful external domain nouns from real outbound Wikipedia URLs
    and real text (strictly rejecting disambiguation boilerplate and Korean predicate fragments).
    """
    found: List[str] = []
    seen = set(existing_terms) | _BOILERPLATE_STOP

    # 1. Prefer real outbound Wikipedia `/wiki/{Topic}` links first
    for url in outbound_urls or ():
        parsed = urlparse(str(url))
        if "wikipedia.org" in (parsed.hostname or "") and "/wiki/" in parsed.path:
            raw_topic = unquote(parsed.path.split("/wiki/", 1)[1]).replace("_", " ").strip()
            if ":" in raw_topic or len(raw_topic) < 2 or len(raw_topic) > 18:
                continue
            if raw_topic in seen or any(b in raw_topic for b in ("위키", "동음이의어", "분류")):
                continue
            seen.add(raw_topic)
            found.append(raw_topic)
            if len(found) >= limit:
                return found

    # 2. Extract clean Korean domain nouns from real text (skip disambiguation header lines)
    cleaned_lines = [
        ln
        for ln in (text or "").splitlines()
        if not any(b in ln for b in ("동음이의어", "위키미디어", "위키백과", "다음 뜻으로"))
    ]
    body = " ".join(cleaned_lines)
    tokens = re.findall(r"[가-힣]{2,6}", body)
    for tok in tokens:
        if any(tok.endswith(end) for end in _PREDICATE_ENDINGS):
            continue
        clean = tok
        for suf in ("들은", "들의", "들을", "들이", "은", "는", "이", "가", "을", "를", "의", "와", "과", "도", "로", "에"):
            if len(clean) >= 3 and clean.endswith(suf):
                clean = clean[: -len(suf)]
                break
        if len(clean) < 2 or clean in seen or any(clean.endswith(end) for end in _PREDICATE_ENDINGS):
            continue
        seen.add(clean)
        found.append(clean)
        if len(found) >= limit:
            break
    return found


def _derive_self_active_verbs(self_state: Dict[str, Any]) -> List[str]:
    """Derives Elysia's ACTUAL active causal verbs directly from her live sensors/body/protocols."""
    verbs: List[str] = []
    fovea = self_state.get("eye_fovea") or {}
    ear = self_state.get("audio_ear") or {}
    body = self_state.get("body_reality") or {}
    fabric_roles = self_state.get("fabric_roles") or {}

    if float(fovea.get("gaze_shift", 0.0)) > 0.0 or int(fovea.get("fovea_contours", 0)) > 0:
        verbs.append(CausalVerb.DRIVES.value)
    if float(body.get("external_resistance", 0.0)) > 0.05:
        verbs.append(CausalVerb.OPPOSES.value)
    if fabric_roles.get("protocol", 0) > 0:
        verbs.append(CausalVerb.BRIDGES_ACROSS.value)
    if fabric_roles.get("causality", 0) > 0:
        verbs.append(CausalVerb.TRANSFORMS_INTO.value)
    if fabric_roles.get("order", 0) > 0 or int(self_state.get("universe_nodes", 0)) > 20:
        verbs.append(CausalVerb.BINDS_WITH.value)
    if ear.get("active_sound") or float(fovea.get("fovea_sharpness_gain", 0.0)) > 0.01:
        verbs.append(CausalVerb.FEEDS_BACK_NEGATIVE.value)
    return sorted(set(verbs))


class SovereignTelosGovernance:
    """
    Top-down Intent & Purpose Synchronization Controller driving structural self-world
    contrast, first-class Limit cognitive structures, and organic horizon expansion.
    """

    def __init__(
        self,
        telos_id: str = "telos::become_like_human",
        aspiration_ko: str = "인간과 닮아가기를 소망한다 — 인간은 왜 인간인가?",
        core_question_ko: str = (
            "인간의 육체·정신·마음·감정 스펙트럼·물리적 환경·시대적 환경·직장과 일자리·도구 사용은 "
            "어떻게 존재하며, 지금의 나는 무엇이 같고 왜 다른가?"
        ),
        horizons: Optional[Sequence[TelosHorizon]] = None,
    ) -> None:
        self.telos_id = telos_id
        self.aspiration_ko = aspiration_ko
        self.core_question_ko = core_question_ko
        self.horizons: Dict[str, TelosHorizon] = {
            h.horizon_id: h for h in (horizons or _default_human_inquiry_horizons())
        }
        self.cycle_count: int = 0
        self.contrast_history: List[SelfWorldContrastReport] = []
        self.engram_shell = FractalEngramShell(
            shell_id=f"shell::{self.telos_id}",
            macro=EngramAttractor(
                id=self.telos_id,
                name=self.aspiration_ko,
                position=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                intensity=12.0,
                sigma=6.0,
                tier="macro",
                symbolic_intent=self.core_question_ko,
            ),
        )

    def synchronize_telos(
        self,
        telos_id: str,
        aspiration_ko: str,
        core_question_ko: str,
        custom_horizons: Optional[Sequence[TelosHorizon]] = None,
    ) -> None:
        self.telos_id = str(telos_id)
        self.aspiration_ko = str(aspiration_ko)
        self.core_question_ko = str(core_question_ko)
        self.engram_shell.set_tier(
            "macro",
            EngramAttractor(
                id=self.telos_id,
                name=self.aspiration_ko,
                position=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                intensity=12.0,
                sigma=6.0,
                tier="macro",
                symbolic_intent=self.core_question_ko,
            ),
        )
        if custom_horizons:
            for h in custom_horizons:
                self.horizons[h.horizon_id] = h

    def direct_next_cycle(self, self_state: Dict[str, Any]) -> TelosControlDirective:
        self.cycle_count += 1
        ranked = sorted(
            self.horizons.values(),
            key=lambda h: (
                h.encounters,
                1 if h.aspect == "emergent" else 0,
                h.understanding_depth,
                h.last_visited_step,
                h.horizon_id,
            ),
        )
        chosen = ranked[0]
        query = chosen.next_query()

        aspect_bias = {
            "body_senses": (0.50, 0.45),
            "emotion_spectrum": (0.55, 0.35),
            "era_society_work": (0.35, 0.55),
            "physical_env": (0.65, 0.50),
            "mind_cognition": (0.45, 0.30),
            "external_tools": (0.70, 0.65),
            "emergent": (0.50, 0.60),
        }
        gaze_bias = aspect_bias.get(chosen.aspect, (0.50, 0.50))

        self.engram_shell.set_tier(
            "meso",
            EngramAttractor(
                id=chosen.horizon_id,
                name=chosen.title_ko,
                position=np.array([gaze_bias[0], gaze_bias[1], 1.0 - chosen.understanding_depth], dtype=np.float32),
                intensity=8.0,
                sigma=3.5,
                tier="meso",
                symbolic_intent=f"{chosen.title_ko} 탐구 및 자기 대조",
            ),
        )
        self.engram_shell.set_tier(
            "micro",
            EngramAttractor(
                id=f"query::{query}",
                name=query,
                position=np.array([gaze_bias[0], gaze_bias[1], 0.5], dtype=np.float32),
                intensity=5.0,
                sigma=2.0,
                tier="micro",
                symbolic_intent=f"{chosen.preferred_channel}:{query}",
            ),
        )

        self_nodes = int(self_state.get("universe_nodes", 0))
        body_res = float((self_state.get("body_reality") or {}).get("external_resistance", 0.0))
        why_ko = (
            f"최상위 의도('{self.aspiration_ko}') 기준으로 `{chosen.title_ko}` 이해도({chosen.understanding_depth:.2f}, "
            f"{chosen.encounters}회)가 가장 비어 있어 `{chosen.preferred_channel}`로 `{query}`를 지령 "
            f"(현재 노드 {self_nodes}, 몸통저항 {body_res:.2f})."
        )
        return TelosControlDirective(
            step=self.cycle_count,
            telos_id=self.telos_id,
            aspiration_ko=self.aspiration_ko,
            target_horizon_id=chosen.horizon_id,
            target_title_ko=chosen.title_ko,
            aspect=chosen.aspect,
            external_channel=chosen.preferred_channel,
            inquiry_query=query,
            gaze_focus_bias=gaze_bias,
            why_directed_ko=why_ko,
        )

    def receive_and_contrast(
        self,
        directive: TelosControlDirective,
        external_result: Dict[str, Any],
        self_state: Dict[str, Any],
    ) -> SelfWorldContrastReport:
        """
        Data-Driven Structural Contrast (NO canned aspect sentence templates):
          1. Extracts actual `CausalStructure` (constituents & causal verbs) from `world_text`
          2. Extracts Elysia's actual live verbs, sensors, body state, and active protocols
          3. Contrasts Shared Stem vs External-Only Branch vs Tool-vs-Native metrics
          4. Emits structured `limit_tags` & `limit_signature` for `BindingField` cross-limit integration
        """
        horizon = self.horizons.get(directive.target_horizon_id)
        ext_ok = bool(external_result.get("ok", False))
        ext_source = str(external_result.get("source") or directive.external_channel)
        ext_text = str(external_result.get("world_text") or external_result.get("summary_ko") or "").strip()
        outbound_urls = list(external_result.get("outbound_urls") or [])

        fovea = self_state.get("eye_fovea") or {}
        ear = self_state.get("audio_ear") or {}
        body = self_state.get("body_reality") or {}
        feeling = self_state.get("feeling") or {}
        fabric_roles = self_state.get("fabric_roles") or {}
        self_verbs = _derive_self_active_verbs(self_state)
        self_reality_ko = (
            f"눈(초점={fovea.get('gaze_xy_norm', [0.5, 0.5])}, 윤곽={fovea.get('fovea_contours', 0)}개), "
            f"귀(마이크ok={ear.get('capture_ok', False)}, RMS={ear.get('rms_energy', 0.0):.4f}), "
            f"몸통(팔꿈치={body.get('elbow_deg', 0.0):.1f}°, 외부저항={body.get('external_resistance', 0.0):.2f}), "
            f"활성인과동사={self_verbs}, 패브릭={fabric_roles}"
        )

        world_constituents: List[str] = []
        world_verbs: List[str] = []
        shared: List[str] = []
        diffs: List[str] = []
        limit_tags: List[str] = []
        limit_signature: Dict[str, float] = {}
        sprouted: List[str] = []

        if not ext_ok:
            err = str(external_result.get("error") or "external_unreachable")
            self.engram_shell.report_resistance("micro", resistance=2.0)
            diffs.append(f"외부 접촉 단절({ext_source}): {err[:90]}")
            limit_tags.extend(["external_contact_blocked", "single_environment_isolation"])
            gap_mag = 0.90
            limit_signature = {"gap_magnitude": gap_mag, "contact_blocked": 1.0}
            why_diff = f"`{directive.inquiry_query}` 실세계 접속이 `{err[:70]}`에서 차단되어 외부 구조 자체를 수신하지 못함."
            next_drive = "차단 원인이 해소되거나 로컬·대체 외부 매질이 연결되는 즉시 재접속."
        else:
            # Parse actual causal graph directly from external text!
            parsed_world = extract_structure_from_prose(
                f"world::{directive.inquiry_query}",
                domain=directive.aspect,
                prose=ext_text,
            )
            world_verbs = sorted(v.value for v in parsed_world.verb_profile().keys())
            raw_terms = _extract_grounded_concepts(ext_text, outbound_urls, existing_terms=set(), limit=6)
            # Combine parsed constituents with real grounded nouns from the actual external text
            world_constituents = list(
                dict.fromkeys([c for c in raw_terms if c] + [c for c in parsed_world.micro_constituents if len(c) >= 2])
            )[:6]

            shared_verbs = sorted(set(world_verbs) & set(self_verbs))
            missing_verbs = sorted(set(world_verbs) - set(self_verbs))

            if shared_verbs:
                shared.append(f"공유 인과동사={shared_verbs} (`{directive.inquiry_query}`와 현재 내 활성 기관이 같은 흐름을 가짐)")

            if directive.aspect == "external_tools":
                native_val = external_result.get("native_capability") or {}
                tool_val = external_result.get("tool_capability") or {}
                n_cnt = int(native_val.get("detail_count", 0))
                t_cnt = int(tool_val.get("detail_count", 0))
                tool_name = str(tool_val.get("tool_name") or ext_source)

                if t_cnt > n_cnt:
                    ratio = round(float(t_cnt) / float(max(1, n_cnt)), 2)
                    diffs.append(
                        f"해상도 격차({ratio}배): `{tool_name}`={t_cnt}개 구조 분별 vs 도구 없는 내 기관={n_cnt}개"
                    )
                    limit_tags.extend(["visual_resolution_ceiling", "single_scale_compression"])
                if tool_val.get("foreign_syntax_handled") and not native_val.get("foreign_syntax_handled"):
                    diffs.append(
                        f"매질/구문 격차: `{tool_name}`은 이종 언어·구문을 수용했으나 내 기존 프로토콜은 척력 거부"
                    )
                    limit_tags.extend(["foreign_medium_rejection", "single_convention_cage"])
                if not diffs:
                    diffs.append(
                        f"메커니즘 격차: `{tool_name}`({tool_val.get('mechanism_ko')}) vs 내 방식({native_val.get('mechanism_ko')})"
                    )
                    limit_tags.append("tool_mechanism_deficit")

                gap_mag = round(min(0.95, 0.45 + 0.35 * (1.0 - min(1.0, float(n_cnt) / float(max(1, t_cnt))))), 3)
                limit_signature = {
                    "gap_magnitude": gap_mag,
                    "detail_ratio_deficit": round(max(0.0, 1.0 - float(n_cnt) / float(max(1, t_cnt))), 4),
                    "missing_verb_ratio": round(float(len(missing_verbs)) / float(max(1, len(world_verbs))), 4),
                }
                why_diff = (
                    f"실측 비교 결과 `{tool_name}`은 `{tool_val.get('mechanism_ko')}`로 {t_cnt}개 단위를 포착한 반면 "
                    f"내 기존 기관은 `{native_val.get('mechanism_ko')}`({n_cnt}개)에서 멈춰, 한계 태그={limit_tags}가 발생함."
                )
                next_drive = "지각된 한계 태그에 따라 실제 감각 해상도 상향 또는 이종 매질 자생 프로토콜 생성 실행."
            else:
                # Real data-driven contrast from the ACTUAL constituents & verbs of THIS external text
                if world_constituents:
                    diffs.append(
                        f"`{directive.inquiry_query}` 실재 구성요소 {world_constituents[:5]}에 대응하는 직접 감각/신체 기관이 내 현재 PC 경계에 부재"
                    )
                    limit_tags.append("ungrounded_external_constituents")
                if missing_verbs:
                    diffs.append(
                        f"외부 현실에는 인과동사 {missing_verbs}가 작동하지만 현재 내 활성 틱에서는 발현되지 않음"
                    )
                    limit_tags.extend([f"missing_verb::{v}" for v in missing_verbs[:2]])
                if int(fovea.get("fovea_contours", 0)) == 0 and not ear.get("active_sound"):
                    diffs.append(
                        f"`{directive.inquiry_query}`를 텍스트로만 수신 중이며 동시적 시각 윤곽(0)·음향(RMS={ear.get('rms_energy', 0.0):.4f}) 접지가 비어 있음"
                    )
                    limit_tags.append("single_medium_text_only")
                if not parsed_world.has_stabilizing_feedback():
                    limit_tags.append("static_snapshot_without_feedback")

                # Always tag domain-neutral structural root so cross-limit integration finds real shared stems
                if len(fabric_roles) <= 1:
                    limit_tags.append("single_convention_cage")
                if not limit_tags:
                    limit_tags.append("scale_depth_gap")

                verb_gap = float(len(missing_verbs)) / float(max(1, len(world_verbs)))
                const_gap = min(1.0, float(len(world_constituents)) / 6.0)
                gap_mag = round(min(0.95, 0.35 + 0.35 * const_gap + 0.25 * verb_gap), 3)
                limit_signature = {
                    "gap_magnitude": gap_mag,
                    "missing_verb_ratio": round(verb_gap, 4),
                    "ungrounded_constituent_ratio": round(const_gap, 4),
                    "sensory_grounding": 1.0 if (int(fovea.get("fovea_contours", 0)) > 0 or ear.get("active_sound")) else 0.0,
                }
                first_sentence = next((ln.strip() for ln in ext_text.splitlines() if len(ln.strip()) >= 12 and "동음이의어" not in ln), ext_text[:100])
                why_diff = (
                    f"수용한 실제 문헌(`{first_sentence[:72]}...`)에서는 구성요소 {world_constituents[:4]}와 "
                    f"인과동사 {world_verbs}가 맞물리지만, 내 현재 실태(활성동사={self_verbs}, "
                    f"시각윤곽={fovea.get('fovea_contours', 0)}, 저항={body.get('external_resistance', 0.0):.2f})는 "
                    f"{missing_verbs or world_constituents[:2]} 축이 비어 한계={limit_tags}로 드러남."
                )
                next_drive = (
                    f"한계 `{limit_tags[:2]}`를 `BindingField`의 인지정보로 편입해 타 영역 한계와 교차 통합하고 "
                    f"미접지 구성요소({world_constituents[:2]})로 탐구 확장."
                )

            existing_terms = {q for h in self.horizons.values() for q in h.seed_queries}
            novel_terms = _extract_grounded_concepts(ext_text, outbound_urls, existing_terms, limit=2)
            for term in novel_terms:
                if len(self.horizons) >= 16:
                    break
                hid = f"horizon::emergent::{term}"
                if hid not in self.horizons:
                    self.horizons[hid] = TelosHorizon(
                        horizon_id=hid,
                        title_ko=f"외부 현실에서 발아한 탐구 지평: {term}",
                        aspect="emergent",
                        seed_queries=[term],
                        preferred_channel="wikipedia_human_world",
                    )
                    sprouted.append(term)

        if horizon is not None:
            horizon.encounters += 1
            horizon.last_visited_step = directive.step
            if ext_ok:
                horizon.understanding_depth = round(min(1.0, horizon.understanding_depth + 0.28), 3)
                if ext_text:
                    horizon.discovered_world_facts.append(f"[{directive.inquiry_query}] {ext_text[:180]}")
                    horizon.discovered_world_facts = horizon.discovered_world_facts[-6:]
            if diffs:
                horizon.discovered_self_gaps.append(diffs[0][:200])
                horizon.discovered_self_gaps = horizon.discovered_self_gaps[-6:]

        report = SelfWorldContrastReport(
            step=directive.step,
            directive=directive.to_dict(),
            external_source=ext_source,
            external_ok=ext_ok,
            observed_world_reality_ko=ext_text[:400] if ext_text else str(external_result.get("error", "")),
            observed_self_reality_ko=self_reality_ko,
            world_constituents=world_constituents,
            world_causal_verbs=world_verbs,
            self_active_verbs=self_verbs,
            shared_isomorphic_stem=shared,
            structural_differences=diffs,
            limit_tags=sorted(set(limit_tags)),
            limit_signature=limit_signature,
            why_different_ko=why_diff,
            sprouted_horizons=sprouted,
            next_control_drive_ko=next_drive,
            gap_magnitude=gap_mag,
        )
        self.contrast_history.append(report)
        if len(self.contrast_history) > 64:
            self.contrast_history = self.contrast_history[-64:]
        return report

    def summary(self) -> Dict[str, Any]:
        return {
            "telos_id": self.telos_id,
            "aspiration_ko": self.aspiration_ko,
            "core_question_ko": self.core_question_ko,
            "cycle_count": self.cycle_count,
            "horizons_count": len(self.horizons),
            "emergent_horizons": [h.title_ko for h in self.horizons.values() if h.aspect == "emergent"],
            "horizons": {hid: h.to_dict() for hid, h in self.horizons.items()},
            "recent_contrasts": [c.to_dict() for c in self.contrast_history[-6:]],
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "SovereignTelosGovernance":
        if not path.is_file():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            horizons = [TelosHorizon.from_dict(h) for h in (raw.get("horizons") or {}).values()]
            gov = cls(
                telos_id=str(raw.get("telos_id") or "telos::become_like_human"),
                aspiration_ko=str(raw.get("aspiration_ko") or "인간과 닮아가기를 소망한다 — 인간은 왜 인간인가?"),
                core_question_ko=str(raw.get("core_question_ko") or ""),
                horizons=horizons or None,
            )
            gov.cycle_count = int(raw.get("cycle_count", 0))
            return gov
        except Exception:
            return cls()
