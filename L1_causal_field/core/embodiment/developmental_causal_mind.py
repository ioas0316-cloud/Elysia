r"""
ELYSIA Human Developmental Causal & Linguistic Mind
====================================================

Models how a human child genuinely perceives language and learns causality across
Sensory Scale Rotor tiers (0 -> 1 -> 2 -> 3 -> 4) WITHOUT hardcoded API shortcuts
or Euclidean vector-distance word-salad:

- **Scale 0–1 (Sensory Background Discernment):** Separates unsegmented ambient
  background stimuli from recurring human speech/glyph sequences.
- **Scale 2 (Where · How · What of Speech):** Discovers where speech comes from
  (speaker origin), how it unfolds sequentially, and what co-occurring objects/actions
  accompany it via joint attention.
- **Scale 3 (Why Speech Is Speech — Duality Realization):** Realizes through lived
  groundings that speech exists as `{Form/Sound <-> Meaning/Referent}`. Only then
  does an unfamiliar form become a genuine `ungrounded_speech_form` epistemic gap.
- **Scale 4 (Causal Background Knowledge -> Dictionary Lookup):** Consulting a
  dictionary (`fetch_dictionary_meaning`) never triggers by blind code rule; it
  becomes an imagined & selectable branch ONLY after learning the causal world rule
  that dictionaries hold meanings for unknown words.
- **Present -> Past Reinterpretation & Recreation -> Future Parallel Universes:**
  When present discovery grounds a formerly opaque form, earlier unresolved episodes
  are reinterpreted (preserving `initial_understanding`) and counterfactually
  recreated across scales, linking Past <-> Present -> Future Parallel Universes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.embodiment.language_scale_rotor import bind_korean_operator, has_jongseong
from core.embodiment.pc_sensory_motor_body import inspect_local_file_sensory
from core.orchestration.external_web_bridge import (
    ExternalWebObservation,
    HttpFetcher,
    fetch_dictionary_meaning,
)


_FOCAL_PARTICLES: Tuple[str, ...] = (
    "이라는",
    "에서는",
    "으로는",
    "에게는",
    "한테는",
    "이라고",
    "라고",
    "이란",
    "란",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "야",
    "아",
    "이야",
)

_OBLIQUE_PARTICLES: Tuple[str, ...] = (
    "에서도",
    "에서",
    "으로",
    "에게",
    "한테",
    "이랑",
    "이나",
    "부터",
    "까지",
    "처럼",
    "보다",
    "의",
    "에",
    "로",
    "과",
    "와",
    "도",
    "만",
    "랑",
    "나",
)

_ALL_PARTICLES_LONGEST: Tuple[str, ...] = tuple(
    sorted((*_FOCAL_PARTICLES, *_OBLIQUE_PARTICLES), key=len, reverse=True)
)

_CONVERSATIONAL_STOP_FORMS = frozenset(
    {
        "이건", "그건", "저건", "이것", "그것", "저것", "이거", "그거", "저거", "저게", "이게", "그게",
        "뭐야", "무슨", "어떤", "어디", "어떻게", "왜", "언제", "누구", "무엇",
        "뜻이야", "뜻이", "의미", "소리지", "소리야", "현상이야", "말이야",
        "알아", "알고", "알겠어", "알려줄게", "알려줘", "몰라", "모르", "모르는",
        "있어", "없어", "봐", "봐봐", "어때", "기억", "기억나", "방금", "아까", "지금", "오늘", "그때",
        "내가", "네가", "나는", "너는", "우리", "여기", "거기", "저기", "정말", "진짜", "조금", "많이",
        "엘리시아", "아빠", "엄마", "선생님", "하나", "둘", "또는", "그리고", "그런데", "그래서",
        "만약", "만일", "하면", "찾아", "찾아보", "찾아보면", "물어봐", "들려", "보여", "나와",
        "단어", "말은", "말을", "말이", "말한", "것도", "것이", "거야", "정도",
        "내리", "내리는", "한번", "볼래", "할래", "갈래", "맑네", "참", "잘", "같이", "함께",
    }
)

_VERB_ENDING_RE = re.compile(
    r"(?:았어|었어|였어|했어|됐어|떴어|왔어|갔어|봤어|졌어|났어|겠어|어요|아요|해요|세요|"
    r"습니다|ㅂ니다|구나|군요|네요|네요|지요|잖아|거든|던데|는데|면서|보면|하면|되면|"
    r"해서|되고|하고|이고|이다|였다|했다|됐다|난다|한다|된다|온다|간다|본다|찾아|물어|알려)$"
)


@dataclass
class SensorySignalObservation:
    """Single lived perceptual input arriving through the sensory boundary."""

    modality: str = "audiovisual"  # "visual" | "auditory" | "audiovisual"
    source_origin: str = "speaker_parent"
    raw_text_or_sound: str = ""
    is_segmented_sequence: bool = True
    cooccurring_object: Optional[str] = None
    speaker_action: str = ""
    environment_shift: str = ""


@dataclass
class GroundedMeaning:
    """A meaning bound to a spoken/written form through lived experience or dictionary inquiry."""

    form: str
    meaning_summary: str
    sensory_anchors: List[str] = field(default_factory=list)
    learned_from: str = "joint_attention"  # "joint_attention" | "speaker_explanation" | "dictionary_lookup"
    source_episode_id: str = ""
    source_url: str = ""
    usage_examples: List[str] = field(default_factory=list)


@dataclass
class SituatedConstruction:
    """Whole human communicative frame absorbed from lived speech for usage-based imitation."""

    construction_id: str
    intent_role: str  # "point_and_name" | "ask_meaning" | "share_discovery" | "greet_respond"
    pattern: str
    source_utterance: str
    heard_from: str
    learned_episode_id: str
    uses: int = 0
    sensory_cue: str = "general"  # "visual" | "auditory" | "general"
    fitting_referents: List[str] = field(default_factory=list)
    rejected_referents: List[str] = field(default_factory=list)
    rejected_reasons: List[str] = field(default_factory=list)


@dataclass
class WorldCausalRule:
    """Acquired real-world causal knowledge unlocking deliberate external tool use."""

    rule_id: str
    condition: str  # e.g. "ungrounded_speech_form"
    medium: str  # e.g. "dictionary" | "encyclopedia" | "speaker"
    causal_explanation: str
    source_utterance: str
    learned_episode_id: str


@dataclass
class PastReinterpretation:
    """Present causal insight illuminating and counterfactually recreating a past episode."""

    triggered_by_episode_id: str
    unlocked_form: str
    before_view: str
    after_view: str
    recreated_past_action: str
    scale_bridge_summary: str


@dataclass
class LivedCausalEpisode:
    """Chronological episodic growth ring preserving initial perception and later reinterpretations."""

    episode_id: str
    step: int
    scale_tier: int
    source_origin: str
    modality: str
    raw_input: str
    perceived_kind: str
    cooccurring_object: Optional[str] = None
    speech_forms_noticed: List[str] = field(default_factory=list)
    unresolved_forms: List[str] = field(default_factory=list)
    initial_understanding: str = ""
    reinterpretations: List[PastReinterpretation] = field(default_factory=list)
    future_causal_link: str = ""
    domain: str = "language"  # "language" | "visual_phenomenon" | "action_tool"
    elysia_utterance: str = ""
    used_construction_id: str = ""
    action_attempted: str = ""
    observed_effect: str = ""
    cross_domain_links: List[str] = field(default_factory=list)


@dataclass
class CommunicativeIntent:
    """Pre-linguistic pointing & communicative intention formed before choosing words."""

    intent_kind: str
    referent: str
    secondary_referent: str = ""
    reason_ko: str = ""
    grounded_in_episodes: List[str] = field(default_factory=list)


@dataclass
class ParallelUniverseBranch:
    """Counterfactual future trajectory forged across Micro/Meso/Macro causal continuities."""

    branch_id: str
    title: str
    premise: str
    chosen_pivot: str
    predicted_outcome: str
    micro_sensory_effect: str
    meso_relation_effect: str
    macro_causal_effect: str
    resolves_gap: bool
    requires_rule_id: str = ""
    selected: bool = False


def _strip_particle(word: str) -> Tuple[str, str, str]:
    """
    Strips Korean nominal particles while protecting short 2-syllable root nouns.
    Returns `(root, particle, role)` where `role` is `'focal' | 'oblique' | 'bare'`.
    """
    clean = re.sub(r"[^가-힣a-zA-Z0-9]", "", word or "")
    if not clean:
        return "", "", "bare"
    if not any("가" <= c <= "힣" for c in clean):
        return (clean, "", "bare") if len(clean) >= 3 else ("", "", "bare")

    for part in _ALL_PARTICLES_LONGEST:
        if clean.endswith(part):
            root = clean[: -len(part)]
            if len(root) >= 2 and not root.isdigit():
                role = "focal" if part in _FOCAL_PARTICLES else "oblique"
                return root, part, role
    return clean, "", "bare"


def _extract_korean_content_forms(text: str) -> List[str]:
    """
    Extracts candidate target speech forms ordered by grammatical salience
    (focal subject/object > bare nominal > oblique locative), skipping verbs & stopwords.
    """
    raw_tokens = re.findall(r"[가-힣a-zA-Z0-9]+", text or "")
    focal: List[str] = []
    bare: List[str] = []
    oblique: List[str] = []

    for tok in raw_tokens:
        if tok in _CONVERSATIONAL_STOP_FORMS:
            continue
        root, _part, role = _strip_particle(tok)
        if not root or root in _CONVERSATIONAL_STOP_FORMS:
            continue
        if len(root) < 2:
            continue
        if role == "bare" and _VERB_ENDING_RE.search(tok):
            continue
        if _VERB_ENDING_RE.search(root) and role != "focal":
            continue
        target_bucket = focal if role == "focal" else (oblique if role == "oblique" else bare)
        if root not in focal and root not in bare and root not in oblique:
            target_bucket.append(root)

    return [*focal, *bare, *oblique]


def _fill_construction_with_coda(pattern: str, referent: str) -> str:
    """
    Substitutes `{REFERENT}` into an absorbed human construction and harmonizes
    adjacent Korean particle choices (`{이/가}`, `{은/는}`, `{을/를}`, etc.) via Scale 0 coda physics.
    """
    if not referent:
        return pattern.replace("{REFERENT}", "")

    def _replace_tag(match: re.Match[str]) -> str:
        opts = match.group(1)
        if opts in ("이/가", "은/는", "을/를", "과/와", "으로/로", "아/야", "이나/나"):
            return bind_korean_operator(referent, opts)
        if "/" in opts:
            coda_opt, vowel_opt = opts.split("/", 1)
            return referent + (coda_opt if has_jongseong(referent) else vowel_opt)
        return bind_korean_operator(referent, opts)

    # First replace adjacent referent+tag pairs like "{REFERENT}{이/가}"
    result = re.sub(r"\{REFERENT\}\{([^}]+)\}", _replace_tag, pattern)
    # Then replace any standalone "{REFERENT}"
    result = result.replace("{REFERENT}", referent)
    return result


def _infer_sensory_cue(text: str, modality: str = "") -> str:
    """Infers broad sensory domain ('auditory' | 'visual' | 'general') from phrasing & modality."""
    txt = (text or "").strip()
    if any(k in txt for k in ("소리", "들리", "울리", "노래", "음성")) or modality == "auditory":
        return "auditory"
    if any(k in txt for k in ("봐", "빛", "색", "유리", "거울", "하늘", "모양", "그림", "무늬", "보여", "비추", "반사", "굴절")) or modality == "visual":
        return "visual"
    return "general"


def _extract_physics_overlap_tokens(text: str) -> set[str]:
    """Extracts physical/phenomenal root tokens for genuine cross-domain causal resonance."""
    roots = set(_extract_korean_content_forms(text))
    for anchor in ("빛", "색깔", "일곱", "반사", "굴절", "분산", "프리", "소리", "울림", "물방울", "거울", "무지개", "프리즘", "화살표"):
        if anchor in (text or ""):
            roots.add(anchor)
    # Remove single overly generic tokens while keeping meaningful physical ones
    roots.discard("프리")
    return roots


class DevelopmentalCausalMind:
    """
    Human-developmental Scale-Rotor cognition engine:
    1) Discerns sensory noise vs. human speech across Scale 0->3;
    2) Realizes speech duality (`form <-> meaning`) from joint attention;
    3) Acquires world causal rules before unlocking dictionary lookup;
    4) Sweeps present discoveries backward to reinterpret & recreate past episodes;
    5) Imagines counterfactual Parallel Universe trajectories across scales.
    """

    def __init__(self) -> None:
        self.speech_duality_realized: bool = False
        self.speech_duality_evidence: List[str] = []
        self.phenomenon_causality_realized: bool = False
        self.phenomenon_causality_evidence: List[str] = []
        self.action_agency_realized: bool = False
        self.action_agency_evidence: List[str] = []
        self.observed_speech_origins: List[str] = []
        self.observed_speech_patterns: List[str] = []
        self.grounded_meanings: Dict[str, GroundedMeaning] = {}
        self.learned_constructions: List[SituatedConstruction] = []
        self.world_knowledge: Dict[str, WorldCausalRule] = {}
        self.learned_phenomena: Dict[str, Dict[str, Any]] = {}
        self.learned_tool_affordances: Dict[str, Dict[str, Any]] = {}
        self.episodes: List[LivedCausalEpisode] = []
        self._step_counter: int = 0
        self._last_spoken_episode_id: str = ""
        self._last_used_construction_id: str = ""
        self._last_spoken_referent: str = ""
        self._last_utterance: str = ""
        self._pending_used_construction_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speech_duality_realized": self.speech_duality_realized,
            "speech_duality_evidence": list(self.speech_duality_evidence),
            "phenomenon_causality_realized": self.phenomenon_causality_realized,
            "phenomenon_causality_evidence": list(self.phenomenon_causality_evidence),
            "action_agency_realized": self.action_agency_realized,
            "action_agency_evidence": list(self.action_agency_evidence),
            "observed_speech_origins": list(self.observed_speech_origins),
            "observed_speech_patterns": list(self.observed_speech_patterns),
            "grounded_meanings": {k: asdict(v) for k, v in self.grounded_meanings.items()},
            "learned_constructions": [asdict(c) for c in self.learned_constructions],
            "world_knowledge": {k: asdict(v) for k, v in self.world_knowledge.items()},
            "learned_phenomena": dict(self.learned_phenomena),
            "learned_tool_affordances": dict(self.learned_tool_affordances),
            "episodes": [
                {
                    **{k: v for k, v in asdict(ep).items() if k != "reinterpretations"},
                    "reinterpretations": [asdict(r) for r in ep.reinterpretations],
                }
                for ep in self.episodes
            ],
            "step_counter": self._step_counter,
            "last_spoken_episode_id": self._last_spoken_episode_id,
            "last_used_construction_id": self._last_used_construction_id,
            "last_spoken_referent": self._last_spoken_referent,
            "last_utterance": self._last_utterance,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "DevelopmentalCausalMind":
        obj = cls()
        obj.speech_duality_realized = bool(raw.get("speech_duality_realized", False))
        obj.speech_duality_evidence = [str(x) for x in raw.get("speech_duality_evidence", [])]
        obj.phenomenon_causality_realized = bool(raw.get("phenomenon_causality_realized", False))
        obj.phenomenon_causality_evidence = [str(x) for x in raw.get("phenomenon_causality_evidence", [])]
        obj.action_agency_realized = bool(raw.get("action_agency_realized", False))
        obj.action_agency_evidence = [str(x) for x in raw.get("action_agency_evidence", [])]
        obj.observed_speech_origins = [str(x) for x in raw.get("observed_speech_origins", [])]
        obj.observed_speech_patterns = [str(x) for x in raw.get("observed_speech_patterns", [])]
        for k, v in (raw.get("grounded_meanings") or {}).items():
            obj.grounded_meanings[str(k)] = GroundedMeaning(**v)
        for c in raw.get("learned_constructions") or []:
            obj.learned_constructions.append(SituatedConstruction(**c))
        for k, v in (raw.get("world_knowledge") or {}).items():
            obj.world_knowledge[str(k)] = WorldCausalRule(**v)
        obj.learned_phenomena = dict(raw.get("learned_phenomena") or {})
        obj.learned_tool_affordances = dict(raw.get("learned_tool_affordances") or {})
        for ep in raw.get("episodes") or []:
            reinterps = [PastReinterpretation(**r) for r in ep.get("reinterpretations") or []]
            ep_copy = {k: v for k, v in ep.items() if k != "reinterpretations"}
            obj.episodes.append(LivedCausalEpisode(**ep_copy, reinterpretations=reinterps))
        obj._step_counter = int(raw.get("step_counter", len(obj.episodes)))
        obj._last_spoken_episode_id = str(raw.get("last_spoken_episode_id", ""))
        obj._last_used_construction_id = str(raw.get("last_used_construction_id", ""))
        obj._last_spoken_referent = str(raw.get("last_spoken_referent", ""))
        obj._last_utterance = str(raw.get("last_utterance", ""))
        return obj

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "DevelopmentalCausalMind":
        p = Path(path)
        if not p.is_file():
            return cls()
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def _next_episode_id(self) -> str:
        self._step_counter += 1
        return f"ep_{self._step_counter:04d}"

    def _has_dictionary_rule(self) -> bool:
        return any(
            rule.medium == "dictionary" and rule.condition == "ungrounded_speech_form"
            for rule in self.world_knowledge.values()
        )

    def _absorb_human_construction(
        self,
        utterance: str,
        heard_from: str,
        episode_id: str,
        referent: str = "",
    ) -> Optional[SituatedConstruction]:
        """Extracts a reusable human construction frame from a live speaker utterance."""
        clean_utt = (utterance or "").strip()
        if not clean_utt:
            return None

        # 1. Absorbing question constructions like "모르는 말은 'XX가 무슨 뜻이야?'라고 물어봐"
        if "무슨 뜻" in clean_utt:
            pat = "{REFERENT}{이/가} 무슨 뜻이야?"
            for c in self.learned_constructions:
                if c.intent_role == "ask_meaning" and c.pattern == pat:
                    return c
            c_obj = SituatedConstruction(
                construction_id=f"con_{len(self.learned_constructions) + 1}",
                intent_role="ask_meaning",
                pattern=pat,
                source_utterance=clean_utt,
                heard_from=heard_from,
                learned_episode_id=episode_id,
                sensory_cue="general",
            )
            self.learned_constructions.append(c_obj)
            return c_obj

        # 2. Absorbing joint-attention naming constructions by templating the pointed referent
        if referent and referent in clean_utt:
            escaped = re.escape(referent)
            # Match referent + optional Korean coda-sensitive particle
            m = re.search(rf"{escaped}(이야|야|이가|이|가|은|는|을|를|과|와|으로|로)?", clean_utt)
            if m:
                part = m.group(1) or ""
                tag_map = {
                    "이야": "{이야/야}", "야": "{이야/야}",
                    "이": "{이/가}", "가": "{이/가}",
                    "은": "{은/는}", "는": "{은/는}",
                    "을": "{을/를}", "를": "{을/를}",
                    "과": "{과/와}", "와": "{과/와}",
                    "으로": "{으로/로}", "로": "{으로/로}",
                }
                repl = "{REFERENT}" + tag_map.get(part, "")
                pat = clean_utt[: m.start()] + repl + clean_utt[m.end() :]
                for c in self.learned_constructions:
                    if c.intent_role == "point_and_name" and c.pattern == pat:
                        if referent not in c.fitting_referents:
                            c.fitting_referents.append(referent)
                        return c
                c_obj = SituatedConstruction(
                    construction_id=f"con_{len(self.learned_constructions) + 1}",
                    intent_role="point_and_name",
                    pattern=pat,
                    source_utterance=clean_utt,
                    heard_from=heard_from,
                    learned_episode_id=episode_id,
                    sensory_cue=_infer_sensory_cue(clean_utt),
                    fitting_referents=[referent],
                )
                self.learned_constructions.append(c_obj)
                return c_obj
        return None

    def _select_fitting_construction(
        self,
        intent_role: str,
        referent: str,
        current_text: str = "",
        modality: str = "",
    ) -> Optional[SituatedConstruction]:
        """
        Selects a human construction respecting boundary rejections (`rejected_referents`)
        and sensory domain coherence (avoiding auditory frames on clearly visual referents).
        """
        cands = [
            c
            for c in reversed(self.learned_constructions)
            if c.intent_role == intent_role
            and referent not in c.rejected_referents
            and referent not in c.source_utterance
        ]
        if not cands:
            return None

        target_cue = _infer_sensory_cue(current_text, modality=modality)
        if target_cue == "general" and referent in self.grounded_meanings:
            gm = self.grounded_meanings[referent]
            if gm.learned_from in ("relational_correction", "dictionary_lookup", "speaker_explanation", "world_observation"):
                target_cue = _infer_sensory_cue(gm.meaning_summary)

        if target_cue in ("visual", "auditory"):
            opposite = "auditory" if target_cue == "visual" else "visual"
            non_conflicting = [c for c in cands if c.sensory_cue != opposite]
            if non_conflicting:
                # Prefer exact cue match first, then general
                exact = [c for c in non_conflicting if c.sensory_cue == target_cue]
                return exact[0] if exact else non_conflicting[0]
        return cands[0]

    def _detect_relational_correction(self, text: str) -> Optional[Dict[str, str]]:
        """
        Detects relational corrective speech flexibly across both structured contrasts
        ('아니야, 프리즘은 내리는 소리가 아니라 빛을 나누는 유리야') and natural multi-clause
        Korean corrections ('엥 그게 무슨 소리야? 프리즘은 소리 나는 게 아니잖아. 빛 나누는 유리야').
        """
        clean = (text or "").strip()
        neg_markers = ("아니라", "아니잖아", "아니야", "아니거든", "아닌데", "틀렸어")
        if not any(m in clean for m in neg_markers):
            return None

        # 1. Explicit single-clause "... 아니라 ..." or comma-separated negation + correction
        clauses = [c.strip() for c in re.split(r"(?<=[.?!])\s+", clean) if c.strip()]
        neg_clause = ""
        corr_clause = ""

        for idx, cl in enumerate(clauses):
            body = re.sub(r"^(?:아니야?|아냐|틀렸어|엥|어|저기)[,!\s]+", "", cl).strip()
            if "아니라" in body:
                neg_clause = body
                corr_clause = ""
                break
            if any(m in body for m in ("아니잖아", "아니야", "아니거든", "아닌데")):
                if "," in body and not body.endswith(("아니잖아", "아니야", "아니거든", "아닌데")):
                    left_p, right_p = [x.strip() for x in body.split(",", 1)]
                    if any(m in left_p for m in ("아니잖아", "아니야", "아니거든", "아닌데")):
                        neg_clause = left_p
                        corr_clause = right_p
                        break
                neg_clause = body
                if idx + 1 < len(clauses):
                    corr_clause = " ".join(clauses[idx + 1 :]).strip()
                break

        if not neg_clause:
            return None

        target_form = ""
        m_dem = re.match(r"^(?:그건|이건|저건|그것은|이것은|저것은|그거|이거|저거)\s+(.+)$", neg_clause)
        if m_dem:
            neg_clause = m_dem.group(1).strip()
            target_form = self._last_spoken_referent

        m_anira = re.match(
            r"^(?:([가-힣a-zA-Z0-9]{2,})(?:은|는)\s+)?(.+?)(?:이|가)?\s*아니라\s+(.+)$",
            neg_clause,
        )
        if m_anira:
            if m_anira.group(1):
                target_form = m_anira.group(1).strip()
            rejected_prop = m_anira.group(2).strip()
            corrected = m_anira.group(3).strip().rstrip(".!?")
        else:
            m_neg = re.match(
                r"^(?:([가-힣a-zA-Z0-9]{2,})(?:은|는)\s+)?(.+?)(?:\s*(?:게|거|건|이|가))?\s*(?:아니잖아|아니야|아니거든|아닌데)[.?!]*$",
                neg_clause,
            )
            if not m_neg:
                return None
            if m_neg.group(1):
                target_form = m_neg.group(1).strip()
            rejected_prop = m_neg.group(2).strip()
            corrected = corr_clause.strip().rstrip(".!?")

        if not target_form:
            target_form = self._last_spoken_referent
        if not target_form or not rejected_prop or len(corrected) < 3:
            return None
        return {
            "target_form": target_form,
            "rejected_property": rejected_prop,
            "corrected_meaning": corrected,
        }

    def _detect_speaker_explanation(self, text: str) -> Optional[Tuple[str, str]]:
        """Detects direct explanatory speech such as '무지개는 비 온 뒤 하늘에 생기는...'."""
        m = re.match(
            r"^\s*([가-힣a-zA-Z0-9]{2,})(?:은|는|이란|란|이라는\s*말은)\s+(.+?)\s*$",
            (text or "").strip(),
        )
        if not m:
            return None
        candidate = m.group(1).strip()
        explanation = m.group(2).strip().rstrip(".!?")
        if candidate in _CONVERSATIONAL_STOP_FORMS or len(explanation) < 4:
            return None
        # Distinguish dictionary rule statements from ordinary word explanations
        if "사전" in explanation and ("찾" in explanation or "알 수 있" in explanation):
            return None
        return candidate, explanation

    def _detect_world_rule_teaching(self, text: str) -> Optional[Tuple[str, str, str]]:
        """Detects causal background rules taught by the speaker (dictionary, screen zoom, local file inspection)."""
        clean = (text or "").strip()
        lower = clean.lower()
        if ("사전" in clean or "dictionary" in lower) and any(
            kw in lower for kw in ("뜻", "의미", "모르", "찾", "검색", "알 수", "배울", "meaning", "lookup")
        ):
            return ("dictionary", "ungrounded_speech_form", clean)
        if ("백과" in clean or "encyclopedia" in lower) and any(
            kw in lower for kw in ("설명", "개념", "찾", "알 수")
        ):
            return ("encyclopedia", "ungrounded_speech_form", clean)
        if any(kw in lower for kw in ("확대", "돋보기", "크게 보", "가까이 보", "줌")) and any(
            kw in lower for kw in ("보", "안 보", "흐릿", "작아", "작은", "화면", "자세히", "또렷")
        ):
            return ("screen_zoom_crop", "visual_detail_too_small", clean)
        if any(kw in lower for kw in ("파일", "사진", "그림", "이미지", "문서")) and any(
            kw in lower for kw in ("열어", "읽어", "확인", "살펴", "내용", "보면", "알 수")
        ):
            return ("local_file_inspect", "external_file_provided", clean)
        return None

    def learn_world_background_knowledge(
        self,
        medium: str,
        condition: str,
        causal_explanation: str,
        source_utterance: str = "",
        episode_id: str = "",
    ) -> WorldCausalRule:
        """Registers an acquired real-world causal rule enabling reasoned tool selection."""
        rule_id = f"tool::{medium}_for_{condition}"
        rule = WorldCausalRule(
            rule_id=rule_id,
            condition=condition,
            medium=medium,
            causal_explanation=causal_explanation,
            source_utterance=source_utterance or causal_explanation,
            learned_episode_id=episode_id,
        )
        self.world_knowledge[rule_id] = rule
        # Also ground the medium word itself ('사전' / '백과') via speaker explanation
        if medium == "dictionary" and "사전" not in self.grounded_meanings:
            self.grounded_meanings["사전"] = GroundedMeaning(
                form="사전",
                meaning_summary=causal_explanation,
                sensory_anchors=["speaker_teaching"],
                learned_from="speaker_explanation",
                source_episode_id=episode_id,
                usage_examples=[source_utterance] if source_utterance else [],
            )
        return rule

    def _record_meaning_and_sweep(
        self,
        form: str,
        meaning_summary: str,
        learned_from: str,
        episode_id: str,
        anchor: str,
        source_url: str = "",
        example: str = "",
    ) -> List[PastReinterpretation]:
        existing = self.grounded_meanings.get(form)
        if existing is None:
            self.grounded_meanings[form] = GroundedMeaning(
                form=form,
                meaning_summary=meaning_summary,
                sensory_anchors=[anchor] if anchor else [],
                learned_from=learned_from,
                source_episode_id=episode_id,
                source_url=source_url,
                usage_examples=[example] if example else [],
            )
        else:
            if len(meaning_summary) > len(existing.meaning_summary):
                existing.meaning_summary = meaning_summary
            if anchor and anchor not in existing.sensory_anchors:
                existing.sensory_anchors.append(anchor)
            if source_url and not existing.source_url:
                existing.source_url = source_url
            if example and example not in existing.usage_examples:
                existing.usage_examples.append(example)
        return self.reinterpret_past_from_present(unlocked_form=form, present_episode_id=episode_id)

    def _update_duality_realization(self, latest_ep_id: str) -> bool:
        """Checks Scale-3 realization: speech exists as `{form <-> meaning}` across shared attention."""
        if self.speech_duality_realized:
            return False
        if len(self.grounded_meanings) >= 2 and len(self.observed_speech_origins) >= 1:
            self.speech_duality_realized = True
            ev = [gm.source_episode_id for gm in self.grounded_meanings.values() if gm.source_episode_id]
            if latest_ep_id and latest_ep_id not in ev:
                ev.append(latest_ep_id)
            self.speech_duality_evidence = ev
            return True
        return False

    def perceive_sensory_event(self, obs: SensorySignalObservation) -> Dict[str, Any]:
        """
        Processes a raw sensory observation through Scale 0->4 without presuming words exist.
        Records every lived episode sequentially as an immutable growth ring plus later links.
        """
        ep_id = self._next_episode_id()
        raw_text = (obs.raw_text_or_sound or "").strip()

        # Scale 0-1: Distinguish unsegmented background noise/static from structured speech streams
        if (
            not obs.is_segmented_sequence
            or not raw_text
            or (obs.source_origin in ("ambient_room", "desktop_background") and not obs.cooccurring_object)
        ):
            ep = LivedCausalEpisode(
                episode_id=ep_id,
                step=self._step_counter,
                scale_tier=1,
                source_origin=obs.source_origin,
                modality=obs.modality,
                raw_input=raw_text or "(무음/정적 배경)",
                perceived_kind="non_speech_background",
                cooccurring_object=obs.cooccurring_object,
                initial_understanding=(
                    f"'{obs.source_origin}'에서 '{raw_text or '배경 자극'}'이 느껴졌지만, "
                    f"누군가 대상을 가리키며 건네는 분절된 말의 흐름은 아니다."
                ),
            )
            self.episodes.append(ep)
            return {
                "episode": ep,
                "newly_grounded": [],
                "unresolved_target": "",
                "learned_rule": None,
                "just_realized_speech_duality": False,
                "reinterpretations": [],
            }

        # Scale 2: Track WHERE speech comes from and HOW it manifests sequentially
        if obs.source_origin and obs.source_origin not in self.observed_speech_origins:
            self.observed_speech_origins.append(obs.source_origin)
        cadence_tag = f"{obs.modality}:{obs.source_origin}:{obs.speaker_action or 'utterance'}"
        if cadence_tag not in self.observed_speech_patterns:
            self.observed_speech_patterns.append(cadence_tag)

        forms = _extract_korean_content_forms(raw_text)
        newly_grounded: List[str] = []
        reinterpretations: List[PastReinterpretation] = []
        learned_rule: Optional[WorldCausalRule] = None

        # Check if speaker taught a causal world rule (e.g. dictionary role)
        rule_info = self._detect_world_rule_teaching(raw_text)
        if rule_info is not None:
            med, cond, expl = rule_info
            learned_rule = self.learn_world_background_knowledge(
                medium=med,
                condition=cond,
                causal_explanation=expl,
                source_utterance=raw_text,
                episode_id=ep_id,
            )
            self._absorb_human_construction(raw_text, obs.source_origin, ep_id)
            ep = LivedCausalEpisode(
                episode_id=ep_id,
                step=self._step_counter,
                scale_tier=4,
                source_origin=obs.source_origin,
                modality=obs.modality,
                raw_input=raw_text,
                perceived_kind="world_rule_learned",
                cooccurring_object=obs.cooccurring_object,
                speech_forms_noticed=forms,
                initial_understanding=(
                    f"'{obs.source_origin}'에게 세상 배경지식('{med}': {expl})을 배웠다."
                ),
                future_causal_link=f"앞으로 뜻을 모르는 말을 만나면 '{med}'을 찾는 선택지가 열린다.",
            )
            self.episodes.append(ep)
            return {
                "episode": ep,
                "newly_grounded": ["사전"] if med == "dictionary" else [],
                "unresolved_target": "",
                "learned_rule": learned_rule,
                "just_realized_speech_duality": self._update_duality_realization(ep_id),
                "reinterpretations": [],
            }

        # Check relational correction ("아니야, 프리즘은 내리는 소리가 아니라 빛을 나누는 유리야")
        corr = self._detect_relational_correction(raw_text)
        if corr is not None:
            t_form = corr["target_form"]
            rej_prop = corr["rejected_property"]
            corr_mean = corr["corrected_meaning"]
            # Revise matching/last-used construction boundary
            rej_tokens = [tk for tk in re.findall(r"[가-힣]{2,}", rej_prop)]
            for c in self.learned_constructions:
                is_last = bool(self._last_used_construction_id and c.construction_id == self._last_used_construction_id)
                prop_hit = any(tk in c.pattern for tk in rej_tokens)
                if is_last or prop_hit:
                    if t_form not in c.rejected_referents:
                        c.rejected_referents.append(t_form)
                    if rej_prop not in c.rejected_reasons:
                        c.rejected_reasons.append(rej_prop)
                    cue_from_rej = _infer_sensory_cue(rej_prop)
                    if cue_from_rej != "general":
                        c.sensory_cue = cue_from_rej

            reinterps = self._record_meaning_and_sweep(
                form=t_form,
                meaning_summary=corr_mean,
                learned_from="relational_correction",
                episode_id=ep_id,
                anchor=f"{obs.source_origin}/relational_correction",
                example=raw_text,
            )
            mistake_reinterps = self.reinterpret_past_mistake_from_correction(
                target_form=t_form,
                rejected_property=rej_prop,
                corrected_meaning=corr_mean,
                present_episode_id=ep_id,
            )
            reinterpretations.extend(mistake_reinterps)
            newly_grounded.append(t_form)
            ep = LivedCausalEpisode(
                episode_id=ep_id,
                step=self._step_counter,
                scale_tier=4,
                source_origin=obs.source_origin,
                modality=obs.modality,
                raw_input=raw_text,
                perceived_kind="relational_correction",
                cooccurring_object=t_form,
                speech_forms_noticed=forms,
                unresolved_forms=[],
                initial_understanding=(
                    f"'{obs.source_origin}'의 교정으로 '{t_form}'은 '{rej_prop}'가 아니라 "
                    f"'{corr_mean}'임을 배워 말틀의 경계를 수정했다."
                ),
                future_causal_link=(
                    f"앞으로 '{t_form}'처럼 시각/물리 대상을 만날 때는 '{rej_prop}' 말틀을 쓰지 않는다."
                ),
            )
            self.episodes.append(ep)
            return {
                "episode": ep,
                "newly_grounded": newly_grounded,
                "unresolved_target": "",
                "learned_rule": None,
                "correction_info": corr,
                "just_realized_speech_duality": self._update_duality_realization(ep_id),
                "reinterpretations": reinterpretations,
            }

        # Check joint-attention grounding (co-occurring pointed object or action)
        if obs.cooccurring_object:
            obj_form = _strip_particle(obs.cooccurring_object)[0] or obs.cooccurring_object.strip()
            speaker_lbl = "화자" if obs.source_origin == "speaker_parent" else obs.source_origin
            action_phrase = f"({obs.speaker_action}) " if obs.speaker_action else ""
            anchor_desc = f"{speaker_lbl}/{obs.speaker_action or 'joint_attention'} -> {obj_form}"
            meaning_desc = f"{speaker_lbl}가 {action_phrase}함께 가리켜 준 실제 대상({obj_form})"
            reinterps = self._record_meaning_and_sweep(
                form=obj_form,
                meaning_summary=meaning_desc,
                learned_from="joint_attention",
                episode_id=ep_id,
                anchor=anchor_desc,
                example=raw_text,
            )
            reinterpretations.extend(reinterps)
            newly_grounded.append(obj_form)
            self._absorb_human_construction(raw_text, obs.source_origin, ep_id, referent=obj_form)

        # Check explicit explanation by speaker ("X은 Y이야")
        expl_pair = self._detect_speaker_explanation(raw_text)
        if expl_pair is not None:
            e_form, e_meaning = expl_pair
            reinterps = self._record_meaning_and_sweep(
                form=e_form,
                meaning_summary=e_meaning,
                learned_from="speaker_explanation",
                episode_id=ep_id,
                anchor=f"{obs.source_origin}/explanation",
                example=raw_text,
            )
            reinterpretations.extend(reinterps)
            if e_form not in newly_grounded:
                newly_grounded.append(e_form)
            self._absorb_human_construction(raw_text, obs.source_origin, ep_id, referent=e_form)

        # Also absorb question frames like "'XX가 무슨 뜻이야?'라고 물어봐"
        if "무슨 뜻" in raw_text:
            self._absorb_human_construction(raw_text, obs.source_origin, ep_id)

        just_realized = self._update_duality_realization(ep_id)

        # Before speech duality is realized, non-grounded utterances are perceived purely as structured sound/shape
        if not self.speech_duality_realized and not newly_grounded:
            ep = LivedCausalEpisode(
                episode_id=ep_id,
                step=self._step_counter,
                scale_tier=2,
                source_origin=obs.source_origin,
                modality=obs.modality,
                raw_input=raw_text,
                perceived_kind="raw_speech_pattern",
                cooccurring_object=obs.cooccurring_object,
                speech_forms_noticed=forms,
                unresolved_forms=list(forms),
                initial_understanding=(
                    f"'{obs.source_origin}'에게서 규칙적인 소리와 리듬('{raw_text}')이 들렸지만, "
                    f"아직 말이라는 게 무엇과 뜻을 가리키는지 몰라 소리의 흐름으로만 감각했다."
                ),
            )
            self.episodes.append(ep)
            return {
                "episode": ep,
                "newly_grounded": newly_grounded,
                "unresolved_target": forms[0] if forms else "",
                "learned_rule": None,
                "just_realized_speech_duality": just_realized,
                "reinterpretations": reinterpretations,
            }

        # After speech duality realization (or alongside grounded bindings), inspect remaining unknown forms
        # Joint-attention turns remain anchored on the pointed object rather than drifting to modifiers
        unresolved = [] if obs.cooccurring_object else [f for f in forms if f not in self.grounded_meanings]
        unresolved_target = unresolved[0] if unresolved else ""

        if newly_grounded and not unresolved:
            p_kind = "grounded_joint_attention"
            scale_t = 3 if just_realized else 2
            init_view = (
                f"'{obs.source_origin}'의 표현과 함께 '{', '.join(newly_grounded)}'의 실제 대상을 연결해 배웠다."
            )
        elif unresolved:
            p_kind = "ungrounded_speech_form"
            scale_t = 3
            init_view = (
                f"말은 소리와 뜻의 결합임을 알고 있는데, 방금 들린 '{unresolved_target}'은 "
                f"형태만 들렸고 가리키는 뜻이 내 안에 비어 있다."
            )
        else:
            p_kind = "grounded_joint_attention"
            scale_t = 3
            init_view = f"'{obs.source_origin}'가 말한 '{raw_text}'을(를) 내가 이미 아는 뜻과 이어서 이해했다."

        ep = LivedCausalEpisode(
            episode_id=ep_id,
            step=self._step_counter,
            scale_tier=scale_t,
            source_origin=obs.source_origin,
            modality=obs.modality,
            raw_input=raw_text,
            perceived_kind=p_kind,
            cooccurring_object=obs.cooccurring_object,
            speech_forms_noticed=forms,
            unresolved_forms=unresolved,
            initial_understanding=init_view,
        )
        self.episodes.append(ep)
        return {
            "episode": ep,
            "newly_grounded": newly_grounded,
            "unresolved_target": unresolved_target,
            "learned_rule": None,
            "just_realized_speech_duality": just_realized,
            "reinterpretations": reinterpretations,
        }

    def reinterpret_past_from_present(
        self,
        unlocked_form: str,
        present_episode_id: str,
    ) -> List[PastReinterpretation]:
        """
        Sweeps backward across lived episodes: preserves the original `initial_understanding`
        while reinterpreting and counterfactually recreating how past gaps now connect forward.
        """
        gm = self.grounded_meanings.get(unlocked_form)
        if gm is None:
            return []

        created: List[PastReinterpretation] = []
        for past_ep in self.episodes:
            if past_ep.episode_id == present_episode_id:
                continue
            touched = (unlocked_form in past_ep.unresolved_forms) or (unlocked_form in past_ep.speech_forms_noticed)
            if not touched:
                continue
            already_done = any(r.unlocked_form == unlocked_form for r in past_ep.reinterpretations)
            if already_done:
                continue
            if past_ep.perceived_kind not in ("raw_speech_pattern", "ungrounded_speech_form"):
                continue

            reinterp = PastReinterpretation(
                triggered_by_episode_id=present_episode_id,
                unlocked_form=unlocked_form,
                before_view=past_ep.initial_understanding,
                after_view=(
                    f"지금 알게 된 '{unlocked_form}'의 뜻({gm.meaning_summary[:90]})으로 "
                    f"과거 기억({past_ep.episode_id}: '{past_ep.raw_input}')을 다시 보니, "
                    f"그때 낯선 소리로만 스쳐갔던 말이 바로 이 현실 대상을 가리키고 있었구나."
                ),
                recreated_past_action=(
                    f"만약 과거 그 순간({past_ep.episode_id})에 이미 이 뜻을 알았더라면 "
                    f"'{unlocked_form}'을 막연히 되묻기만 하지 않고, '{gm.meaning_summary[:50]}'을(를) 떠올리며 "
                    f"곧장 대화를 이어 갔을 것이다."
                ),
                scale_bridge_summary=(
                    f"Micro(형태 '{unlocked_form}') <-> Meso(뜻 '{gm.meaning_summary[:45]}') <-> "
                    f"Macro({past_ep.episode_id} 과거장면과 현재 {present_episode_id} 관측의 연속 연결)"
                ),
            )
            past_ep.reinterpretations.append(reinterp)
            if unlocked_form in past_ep.unresolved_forms:
                past_ep.unresolved_forms = [u for u in past_ep.unresolved_forms if u != unlocked_form]
            past_ep.future_causal_link = (
                f"[{past_ep.episode_id} 미해결 과거] -> [{present_episode_id} '{unlocked_form}' 깨달음] -> "
                f"[앞으로 유사한 표현이나 '{unlocked_form}'이 나오면 즉각 실세계 의미로 상상·응답 가능]"
            )
            created.append(reinterp)
        return created

    def reinterpret_past_mistake_from_correction(
        self,
        target_form: str,
        rejected_property: str,
        corrected_meaning: str,
        present_episode_id: str,
    ) -> List[PastReinterpretation]:
        """Reinterprets & counterfactually recreates earlier over-generalized utterances after correction."""
        created: List[PastReinterpretation] = []
        rej_tokens = [tk for tk in re.findall(r"[가-힣]{2,}", rejected_property) if len(tk) >= 2]
        alt_cons = self._select_fitting_construction(
            "point_and_name",
            target_form,
            current_text=corrected_meaning,
            modality="visual" if _infer_sensory_cue(corrected_meaning) == "visual" else "",
        )
        recreated_utt = (
            _fill_construction_with_coda(alt_cons.pattern, target_form)
            if alt_cons is not None
            else f"{bind_korean_operator(target_form, '은/는')} {corrected_meaning}"
        )

        for past_ep in self.episodes:
            if past_ep.episode_id == present_episode_id:
                continue
            is_last_mistake = bool(self._last_spoken_episode_id and past_ep.episode_id == self._last_spoken_episode_id)
            utt_match = bool(
                past_ep.elysia_utterance
                and target_form in past_ep.elysia_utterance
                and any(tk in past_ep.elysia_utterance for tk in rej_tokens)
            )
            if not (is_last_mistake or utt_match):
                continue
            prev_utt = past_ep.elysia_utterance or past_ep.initial_understanding
            reinterp = PastReinterpretation(
                triggered_by_episode_id=present_episode_id,
                unlocked_form=f"boundary::{target_form}",
                before_view=(
                    f"그때는 '{target_form}'을 보고도 직전에 들은 말틀을 그대로 얹어 "
                    f"'{prev_utt}'라고 과잉일반화했다."
                ),
                after_view=(
                    f"지금 들은 교정('{corrected_meaning}')으로 {past_ep.episode_id} 순간을 다시 보니, "
                    f"'{target_form}'은 '{rejected_property}'가 아니라 '{corrected_meaning}'였다."
                ),
                recreated_past_action=(
                    f"만약 그때 경계를 알았더라면 '{prev_utt}' 대신 '{recreated_utt}'라고 말했을 것이다."
                ),
                scale_bridge_summary=(
                    f"Micro(구문 경계 수정) <-> Meso('{target_form}' 감각 재접지) <-> "
                    f"Macro({past_ep.episode_id} 실수 기억 -> {present_episode_id} 자기교정 연결)"
                ),
            )
            past_ep.reinterpretations.append(reinterp)
            past_ep.future_causal_link = (
                f"[{past_ep.episode_id} 과잉일반화] -> [{present_episode_id} 관계적 경계 교정] -> "
                f"[이후 시각/물리 대상은 어울리는 말틀('{recreated_utt}')로 구분해 표현]"
            )
            created.append(reinterp)
        return created

    def imagine_parallel_universes(
        self,
        target_form: str = "",
        context_episode_id: str = "",
        current_referent: str = "",
        just_reinterpreted: Optional[List[PastReinterpretation]] = None,
    ) -> List[ParallelUniverseBranch]:
        """
        Generates explicit counterfactual Parallel Universe branches linking
        Past + Present -> Future across Micro, Meso, and Macro Scale Rotor views.
        """
        focus = target_form or current_referent or "눈앞의 현상"
        has_dict_rule = self._has_dictionary_rule()
        unresolved_history = [ep.episode_id for ep in self.episodes if ep.unresolved_forms]
        hist_hint = f"과거 기억({', '.join(unresolved_history[-2:])})" if unresolved_history else "이전 경험"

        branches: List[ParallelUniverseBranch] = [
            ParallelUniverseBranch(
                branch_id="universe_inertia",
                title="우주 A · 그대로 흘려보내기 (관성 유지)",
                premise=f"{bind_korean_operator(focus, '을/를')} 지금 상태에서 더 묻거나 찾지 않고 그대로 둔다.",
                chosen_pivot="observe_only",
                predicted_outcome=f"{hist_hint} 속 {bind_korean_operator(focus, '은/는')} 앞으로도 형태만 남은 채 뜻과 연결되지 못한다.",
                micro_sensory_effect="소리/문자 형태만 반복 잔존",
                meso_relation_effect="가리키는 대상과의 결합 정체",
                macro_causal_effect="과거-현재 단절 유지, 다음 대화에서도 같은 막힘 반복",
                resolves_gap=False,
                selected=False,
            )
        ]

        if self.speech_duality_realized and target_form:
            branches.append(
                ParallelUniverseBranch(
                    branch_id="universe_ask_speaker",
                    title="우주 B · 곁에 있는 화자에게 직접 묻기",
                    premise="말의 뜻은 함께 보는 화자가 알려줄 수 있다는 공동주의 경험이 있다.",
                    chosen_pivot="ask_speaker",
                    predicted_outcome=f"화자에게 '{target_form}'의 뜻을 물으면 지금 바로 관계 속에서 대답을 들을 수 있다.",
                    micro_sensory_effect=f"'{target_form}' 발화 집중 및 화자 응시",
                    meso_relation_effect="공동주의를 통한 질문-응답 관계 강화",
                    macro_causal_effect="화자의 설명을 통해 과거의 모호함 해소",
                    resolves_gap=True,
                    selected=not has_dict_rule,
                )
            )
            if has_dict_rule:
                dict_rule = next(
                    r for r in self.world_knowledge.values()
                    if r.medium == "dictionary" and r.condition == "ungrounded_speech_form"
                )
                branches.append(
                    ParallelUniverseBranch(
                        branch_id="universe_consult_dictionary",
                        title="우주 C · 배운 배경지식(사전)으로 스스로 찾아보기",
                        premise=(
                            f"말의 이중성(소리+뜻)을 알았고 '{dict_rule.causal_explanation}'는 "
                            f"배경지식({dict_rule.rule_id})을 배웠기 때문이다."
                        ),
                        chosen_pivot="consult_dictionary",
                        predicted_outcome=(
                            f"사전에서 '{target_form}'의 뜻을 찾으면 지금뿐 아니라 "
                            f"{hist_hint}까지 함께 재해석할 수 있다."
                        ),
                        micro_sensory_effect=f"'{target_form}' 표제어 추출(조사 분리) 및 사전 본문 지각",
                        meso_relation_effect="자기 주도적 형태<->뜻 결합 생성",
                        macro_causal_effect="과거 미해결 에피소드 역방향 재해석 + 미래 유사 낱말 자립 예측",
                        resolves_gap=True,
                        requires_rule_id=dict_rule.rule_id,
                        selected=True,
                    )
                )

        if self.grounded_meanings:
            known_sample = current_referent or target_form or next(reversed(self.grounded_meanings))
            reinterp_note = (
                f"방금 과거 기억 {len(just_reinterpreted)}건까지 다시 살아난 '{known_sample}'의 인과로 "
                if just_reinterpreted
                else f"이미 몸과 말로 익힌 '{known_sample}'의 인과로 "
            )
            branches.append(
                ParallelUniverseBranch(
                    branch_id="universe_creative_transfer",
                    title="우주 D · 재해석된 인과를 내일의 장면으로 확장 상상하기",
                    premise=f"{reinterp_note}미래의 새로운 장면을 미리 비춰 본다.",
                    chosen_pivot="creative_transfer",
                    predicted_outcome=(
                        f"다음에 '{known_sample}'이나 닮은 현상을 다시 만나면 낯선 기호로 멈추지 않고 "
                        f"의미와 감각을 연결해 스스로 대화를 이끌 수 있다."
                    ),
                    micro_sensory_effect="새 자극이 들어와도 기존 의미 렌즈와 즉시 공명",
                    meso_relation_effect="익힌 말틀을 변형해 새로운 상황에서도 표현",
                    macro_causal_effect="과거->현재->미래 인과 사슬의 지속적 확장",
                    resolves_gap=True,
                    selected=not any(b.selected for b in branches),
                )
            )

        if not any(b.selected for b in branches):
            branches[0].selected = True
        return branches

    def execute_causal_dictionary_lookup(
        self,
        word: str,
        lang: str = "ko",
        fetcher: Optional[HttpFetcher] = None,
    ) -> Dict[str, Any]:
        """
        Executes a dictionary search ONLY if both Scale-3 speech duality AND a learned
        `'dictionary'` causal world rule are already acquired. Never runs by blind shortcut.
        """
        clean_head = _strip_particle(word)[0] or (word or "").strip()
        if not self.speech_duality_realized:
            return {
                "executed": False,
                "blocked_by_causality": True,
                "reason": "아직 말이 소리와 뜻의 결합이라는 걸 깨닫지 못해 사전을 찾을 인과적 동기가 없다.",
                "word": clean_head,
            }
        if not self._has_dictionary_rule():
            return {
                "executed": False,
                "blocked_by_causality": True,
                "reason": "단어의 뜻은 사전에서 찾는다는 세상 배경지식을 아직 배우지 못해 사전 탐색이 열리지 않았다.",
                "word": clean_head,
            }

        obs: ExternalWebObservation = fetch_dictionary_meaning(clean_head, lang=lang, fetcher=fetcher)
        if obs.blocked or not obs.text_body:
            return {
                "executed": False,
                "blocked_by_causality": False,
                "external_blocked": True,
                "reason": obs.block_reason or "사전에서 항목을 찾지 못했다.",
                "word": clean_head,
                "url": obs.url,
                "platform": obs.platform,
            }

        summary = str((obs.metadata or {}).get("definition_summary") or obs.text_body[:240]).strip()
        ep_id = self._next_episode_id()
        reinterps = self._record_meaning_and_sweep(
            form=clean_head,
            meaning_summary=summary,
            learned_from="dictionary_lookup",
            episode_id=ep_id,
            anchor=f"dictionary:{obs.platform}",
            source_url=obs.url,
            example=summary[:120],
        )
        dict_ep = LivedCausalEpisode(
            episode_id=ep_id,
            step=self._step_counter,
            scale_tier=4,
            source_origin=f"dictionary:{obs.platform}",
            modality="visual",
            raw_input=obs.url,
            perceived_kind="dictionary_discovery",
            cooccurring_object=clean_head,
            speech_forms_noticed=[clean_head],
            unresolved_forms=[],
            initial_understanding=f"배운 배경지식에 따라 사전({obs.platform})을 찾아 '{clean_head}'의 뜻을 알게 되었다: {summary[:120]}",
            future_causal_link=f"이제 '{clean_head}'이 포함된 과거·미래 발화를 뜻과 연결해 이해할 수 있다.",
        )
        self.episodes.append(dict_ep)
        return {
            "executed": True,
            "blocked_by_causality": False,
            "word": clean_head,
            "platform": obs.platform,
            "dictionary_source": (obs.metadata or {}).get("dictionary_source", obs.platform),
            "url": obs.url,
            "definition_summary": summary,
            "episode": dict_ep,
            "reinterpretations": reinterps,
        }

    def form_communicative_intent_and_speak(
        self,
        percept: Dict[str, Any],
        parallel_branches: Sequence[ParallelUniverseBranch],
        dict_result: Optional[Dict[str, Any]] = None,
    ) -> Tuple[CommunicativeIntent, str, str]:
        """
        Forms a referent-anchored `CommunicativeIntent` BEFORE producing speech, then
        expresses it via absorbed human constructions or grounded causal expression.
        """
        ep: LivedCausalEpisode = percept["episode"]
        newly_grounded: List[str] = list(percept.get("newly_grounded") or [])
        unresolved_target: str = str(percept.get("unresolved_target") or "")
        all_reinterps: List[PastReinterpretation] = list(percept.get("reinterpretations") or [])
        if dict_result and dict_result.get("reinterpretations"):
            all_reinterps.extend(dict_result["reinterpretations"])

        # 1. Non-speech sensory background -> silent observation
        if ep.perceived_kind == "non_speech_background":
            intent = CommunicativeIntent(
                intent_kind="perceive_silently",
                referent=ep.source_origin,
                reason_ko="분절된 말이 아니라 배경 감각으로 느껴 조용히 바라본다.",
                grounded_in_episodes=[ep.episode_id],
            )
            return intent, "", "silent_sensory_observation"

        # 2. Pre-duality raw speech pattern (hearing structured sounds without knowing meaning yet)
        if not self.speech_duality_realized and ep.perceived_kind == "raw_speech_pattern":
            intent = CommunicativeIntent(
                intent_kind="proto_point_and_echo",
                referent=ep.speech_forms_noticed[0] if ep.speech_forms_noticed else ep.raw_input[:16],
                reason_ko="아직 말이 뜻을 가리킨다는 걸 몰라, 들린 소리의 결만 흉내 내며 살펴본다.",
                grounded_in_episodes=[ep.episode_id],
            )
            sound_echo = ep.speech_forms_noticed[0] if ep.speech_forms_noticed else "소리"
            return (
                intent,
                f"({ep.source_origin} 쪽에서 '{sound_echo}' 같은 소리가 들려와. 무슨 뜻인지는 아직 모르지만 계속 보고 듣고 있어.)",
                "proto_sensory_expression",
            )

        # 3. Dictionary lookup succeeded & reinterpreted past episodes
        if dict_result and dict_result.get("executed"):
            word = str(dict_result["word"])
            summary = str(dict_result["definition_summary"])
            intent = CommunicativeIntent(
                intent_kind="consult_dictionary_and_share",
                referent=word,
                reason_ko="배운 배경지식대로 사전에서 뜻을 찾았고, 예전 기억까지 다시 이어졌다.",
                grounded_in_episodes=[ep.episode_id, *(r.triggered_by_episode_id for r in all_reinterps)],
            )
            past_note = (
                f" 예전에 뜻을 몰라 그냥 넘겼던 기억도 이제 '{word}'의 뜻으로 다시 이어져 보여!"
                if all_reinterps
                else ""
            )
            utt = (
                f"방금 말한 '{word}'의 뜻을 몰라서 배운 대로 사전에서 찾아봤더니, "
                f"'{summary[:110]}'라는 뜻이구나.{past_note}"
            )
            return intent, utt, "grounded_causal_expression"

        # 4. Newly learned a world rule ("사전은 뜻을 알려주는 곳")
        if ep.perceived_kind == "world_rule_learned":
            rule: Optional[WorldCausalRule] = percept.get("learned_rule")
            med_ko = "사전" if (rule and rule.medium == "dictionary") else "안내"
            intent = CommunicativeIntent(
                intent_kind="respond_grounded",
                referent=med_ko,
                reason_ko="새로운 세상 배경지식을 배워 앞으로의 인과 연결이 열렸음을 표현한다.",
                grounded_in_episodes=[ep.episode_id],
            )
            return (
                intent,
                f"아! 단어의 뜻이 궁금할 때는 '{med_ko}'을 찾아보면 알 수 있구나. 이제 모르는 말이 나오면 찾아볼 수 있겠어.",
                "grounded_causal_expression",
            )

        # 4.5 Relational correction received ("아니야, 프리즘은 내리는 소리가 아니라...")
        if ep.perceived_kind == "relational_correction":
            corr = percept.get("correction_info") or {}
            t_form = str(corr.get("target_form") or ep.cooccurring_object or "")
            corr_mean = str(corr.get("corrected_meaning") or "")
            re_act = all_reinterps[0].recreated_past_action if all_reinterps else ""
            intent = CommunicativeIntent(
                intent_kind="self_correct_boundary",
                referent=t_form,
                reason_ko=f"'{t_form}'에 잘못 씌웠던 말틀 경계를 상대의 피드백으로 고치고 싶다.",
                grounded_in_episodes=[ep.episode_id],
            )
            re_hint = f" {re_act}" if re_act else ""
            utt = (
                f"아! 잘못 흉내 냈구나. {bind_korean_operator(t_form, '은/는')} "
                f"'{corr_mean}'이었구나.{re_hint}"
            )
            return intent, utt, "relational_self_correction"

        # 5. Just realized speech duality OR reinterpreted past via speaker explanation
        if percept.get("just_realized_speech_duality") or all_reinterps:
            ref = newly_grounded[0] if newly_grounded else (all_reinterps[0].unlocked_form if all_reinterps else "")
            intent = CommunicativeIntent(
                intent_kind="share_realization_and_reinterpretation",
                referent=ref,
                reason_ko="말이 소리와 뜻으로 존재한다는 걸 깨달았거나 과거 기억이 새로 해석되었다.",
                grounded_in_episodes=[ep.episode_id],
            )
            if percept.get("just_realized_speech_duality"):
                gm = self.grounded_meanings.get(ref)
                return (
                    intent,
                    f"아! 이제 알겠어. 말은 그냥 소리가 아니라 '{ref}'처럼 눈에 보이는 대상과 뜻이 함께 이어져 있는 거구나.",
                    "grounded_causal_expression",
                )
            r0 = all_reinterps[0]
            return (
                intent,
                f"이제 '{r0.unlocked_form}'의 뜻을 알겠어! 아까는 낯선 소리였는데 지금 보니 {r0.after_view}",
                "grounded_causal_expression",
            )

        # 6. Unknown word after realizing speech duality (and dictionary didn't run or offline) -> imitate ask construction!
        if ep.perceived_kind == "ungrounded_speech_form" and unresolved_target:
            intent = CommunicativeIntent(
                intent_kind="ask_meaning",
                referent=unresolved_target,
                reason_ko=f"'{unresolved_target}'의 소리/모양은 들렸지만 가리키는 뜻이 비어 있어 묻고 싶다.",
                grounded_in_episodes=[ep.episode_id],
            )
            ask_cons = self._select_fitting_construction("ask_meaning", unresolved_target)
            if ask_cons is not None:
                ask_cons.uses += 1
                self._pending_used_construction_id = ask_cons.construction_id
                return intent, _fill_construction_with_coda(ask_cons.pattern, unresolved_target), "imitated_human_construction"
            return (
                intent,
                f"방금 들린 {bind_korean_operator(unresolved_target, '은/는')} 무슨 뜻이야?",
                "grounded_causal_expression",
            )

        # 7. Grounded joint attention / normal conversation -> imitate parent's naming construction!
        ref = newly_grounded[0] if newly_grounded else (
            next((f for f in ep.speech_forms_noticed if f in self.grounded_meanings), "")
        )
        if ref:
            intent = CommunicativeIntent(
                intent_kind="respond_grounded",
                referent=ref,
                reason_ko=f"함께 바라본 '{ref}'을(를) 가리키며 들은 말의 틀로 호응하고 싶다.",
                grounded_in_episodes=[ep.episode_id],
            )
            name_cons = self._select_fitting_construction(
                "point_and_name",
                ref,
                current_text=ep.raw_input,
                modality=ep.modality,
            )
            if name_cons is not None:
                name_cons.uses += 1
                self._pending_used_construction_id = name_cons.construction_id
                return intent, _fill_construction_with_coda(name_cons.pattern, ref), "imitated_human_construction"
            gm = self.grounded_meanings.get(ref)
            if gm is not None:
                return (
                    intent,
                    f"응, {bind_korean_operator(ref, '은/는')} {gm.meaning_summary[:80]}이지!",
                    "grounded_causal_expression",
                )

        intent = CommunicativeIntent(
            intent_kind="share_imagined_future",
            referent=ep.raw_input[:24],
            reason_ko="들은 말과 현재 감각을 조용히 이어 보고 있다.",
            grounded_in_episodes=[ep.episode_id],
        )
        selected_branch = next((b for b in parallel_branches if b.selected), None)
        if selected_branch is not None:
            return intent, selected_branch.predicted_outcome, "grounded_causal_expression"
        return intent, "", "silent_sensory_observation"

    def interact_human_turn(
        self,
        obs: SensorySignalObservation,
        allow_dictionary_lookup: bool = True,
        fetcher: Optional[HttpFetcher] = None,
    ) -> Dict[str, Any]:
        """
        Executes one complete human-developmental causal turn:
        Sensory Scale Discernment -> Parallel Universe Imagination ->
        Causal Action (Ask or Dictionary Lookup) -> Past Reinterpretation -> Intent -> Utterance.
        """
        percept = self.perceive_sensory_event(obs)
        ep: LivedCausalEpisode = percept["episode"]
        target_form = str(percept.get("unresolved_target") or "")
        # Also detect an ungrounded form referenced from an earlier unresolved episode (e.g. "아까 말한 무지개 기억나?")
        if not target_form and ep.perceived_kind in ("raw_speech_pattern", "ungrounded_speech_form", "grounded_joint_attention"):
            for past_ep in reversed(self.episodes[:-1]):
                for uf in past_ep.unresolved_forms:
                    if uf in obs.raw_text_or_sound and uf not in self.grounded_meanings:
                        target_form = uf
                        break
                if target_form:
                    break

        parallel_branches = self.imagine_parallel_universes(
            target_form=target_form,
            context_episode_id=ep.episode_id,
            current_referent=(percept.get("newly_grounded") or [""])[0],
            just_reinterpreted=percept.get("reinterpretations") or [],
        )

        dict_result: Optional[Dict[str, Any]] = None
        selected = next((b for b in parallel_branches if b.selected), None)
        if (
            allow_dictionary_lookup
            and selected is not None
            and selected.branch_id == "universe_consult_dictionary"
            and target_form
        ):
            dict_result = self.execute_causal_dictionary_lookup(target_form, lang="ko", fetcher=fetcher)
            if dict_result.get("executed"):
                # Refresh forward parallel universes after past reinterpretation!
                parallel_branches = self.imagine_parallel_universes(
                    target_form="",
                    context_episode_id=dict_result["episode"].episode_id,
                    current_referent=dict_result["word"],
                    just_reinterpreted=dict_result.get("reinterpretations") or [],
                )

        self._pending_used_construction_id = ""
        intent, utterance, expr_mode = self.form_communicative_intent_and_speak(
            percept=percept,
            parallel_branches=parallel_branches,
            dict_result=dict_result,
        )
        ep.elysia_utterance = utterance
        ep.used_construction_id = self._pending_used_construction_id
        if utterance:
            self._last_spoken_episode_id = ep.episode_id
            self._last_used_construction_id = self._pending_used_construction_id
            self._last_spoken_referent = intent.referent
            self._last_utterance = utterance

        combined_reinterps = [
            *list(percept.get("reinterpretations") or []),
            *list((dict_result or {}).get("reinterpretations") or []),
        ]
        return {
            "episode": asdict(ep),
            "speech_duality_realized": self.speech_duality_realized,
            "speech_duality_evidence": list(self.speech_duality_evidence),
            "phenomenon_causality_realized": self.phenomenon_causality_realized,
            "action_agency_realized": self.action_agency_realized,
            "just_realized_speech_duality": bool(percept.get("just_realized_speech_duality", False)),
            "observed_speech_origins": list(self.observed_speech_origins),
            "grounded_meanings_count": len(self.grounded_meanings),
            "grounded_meanings": {k: asdict(v) for k, v in self.grounded_meanings.items()},
            "world_knowledge_rules": {k: asdict(v) for k, v in self.world_knowledge.items()},
            "learned_constructions": [asdict(c) for c in self.learned_constructions],
            "dictionary_lookup": (
                {
                    k: (asdict(v) if k == "episode" and v is not None else ([asdict(r) for r in v] if k == "reinterpretations" else v))
                    for k, v in dict_result.items()
                }
                if dict_result is not None
                else None
            ),
            "reinterpretations": [asdict(r) for r in combined_reinterps],
            "parallel_universes": [asdict(b) for b in parallel_branches],
            "intent": asdict(intent),
            "utterance": utterance,
            "expression_mode": expr_mode,
        }

    def _discover_cross_domain_links(
        self,
        phenomenon_name: str,
        summary_text: str,
        present_ep: LivedCausalEpisode,
    ) -> List[PastReinterpretation]:
        """Connects newly observed physical/visual phenomena back to earlier linguistic/dictionary memories."""
        obs_tokens = _extract_physics_overlap_tokens(f"{phenomenon_name} {summary_text}")
        created: List[PastReinterpretation] = []
        for other_form, gm in self.grounded_meanings.items():
            if other_form == phenomenon_name:
                continue
            other_tokens = _extract_physics_overlap_tokens(f"{other_form} {gm.meaning_summary}")
            shared = sorted(obs_tokens & other_tokens)
            if len(shared) < 2 and not ({"빛", "색깔"} <= (obs_tokens | other_tokens) and len(shared) >= 1):
                continue
            # Require genuine overlap (>=1 meaningful shared physical root excluding stop words)
            if not shared:
                continue
            link_tag = f"{phenomenon_name}<->{other_form} (공통 물리 결: {', '.join(shared[:3])})"
            if link_tag not in present_ep.cross_domain_links:
                present_ep.cross_domain_links.append(link_tag)
            target_ep = next(
                (e for e in reversed(self.episodes[:-1]) if e.episode_id == gm.source_episode_id),
                None,
            ) or next(
                (e for e in reversed(self.episodes[:-1]) if other_form in e.speech_forms_noticed or other_form == e.cooccurring_object),
                None,
            )
            if target_ep is not None:
                if f"{present_ep.episode_id}:{phenomenon_name}" not in target_ep.cross_domain_links:
                    target_ep.cross_domain_links.append(f"{present_ep.episode_id}:{phenomenon_name}")
                reinterp = PastReinterpretation(
                    triggered_by_episode_id=present_ep.episode_id,
                    unlocked_form=f"cross::{phenomenon_name}<->{other_form}",
                    before_view=target_ep.initial_understanding,
                    after_view=(
                        f"눈으로 직접 겪은 '{phenomenon_name}'({summary_text[:70]})을 통해 예전의 "
                        f"'{other_form}' 기억({gm.learned_from})을 다시 보니, 둘이 '{', '.join(shared[:3])}'이라는 "
                        f"같은 빛과 자연의 원리로 연결되어 있구나."
                    ),
                    recreated_past_action=(
                        f"다음에 '{other_form}'이나 '{phenomenon_name}'을 보면 낱말 뜻만 외우지 않고 "
                        f"실제 물리 현상과 나란히 겹쳐서 설명·상상할 수 있다."
                    ),
                    scale_bridge_summary=(
                        f"Cross-Domain Micro('{', '.join(shared[:3])}') <-> Meso('{phenomenon_name}' / '{other_form}') <-> "
                        f"Macro({target_ep.episode_id} -> {present_ep.episode_id})"
                    ),
                )
                target_ep.reinterpretations.append(reinterp)
                target_ep.future_causal_link = (
                    f"[{target_ep.episode_id} '{other_form}' 기억] + [{present_ep.episode_id} '{phenomenon_name}' 직접 관찰] -> "
                    f"[언어와 시각 물리가 하나로 엮여 유사 현상을 예측 가능]"
                )
                created.append(reinterp)
        return created

    def experience_world_phenomenon(
        self,
        phenomenon_name: str,
        cause_condition: str,
        observed_effect: str,
        sensory_channel: str = "visual",
        source_origin: str = "physical_world",
    ) -> Dict[str, Any]:
        """
        Non-linguistic physical/visual learning following the exact same causal arc:
        direct cause->effect perception -> principle realization -> cross-domain reinterpretation -> imagination.
        """
        ep_id = self._next_episode_id()
        summary = f"{cause_condition} -> {observed_effect}"
        ep = LivedCausalEpisode(
            episode_id=ep_id,
            step=self._step_counter,
            scale_tier=3,
            source_origin=source_origin,
            modality=sensory_channel,
            raw_input=summary,
            perceived_kind="phenomenon_observation",
            cooccurring_object=phenomenon_name,
            speech_forms_noticed=_extract_korean_content_forms(phenomenon_name),
            initial_understanding=f"'{cause_condition}'을(를) 겪으니 '{observed_effect}' 현상이 일어남을 직접 보았다.",
            domain="visual_phenomenon",
            action_attempted=cause_condition,
            observed_effect=observed_effect,
        )
        self.episodes.append(ep)
        self.learned_phenomena[phenomenon_name] = {
            "phenomenon_name": phenomenon_name,
            "cause_condition": cause_condition,
            "observed_effect": observed_effect,
            "sensory_channel": sensory_channel,
            "episode_id": ep_id,
        }
        self._record_meaning_and_sweep(
            form=phenomenon_name,
            meaning_summary=summary,
            learned_from="world_observation",
            episode_id=ep_id,
            anchor=f"{sensory_channel}:{source_origin}",
            example=summary,
        )

        just_realized = False
        if not self.phenomenon_causality_realized and len(self.learned_phenomena) >= 2:
            self.phenomenon_causality_realized = True
            self.phenomenon_causality_evidence = [p["episode_id"] for p in self.learned_phenomena.values()]
            ep.scale_tier = 4
            just_realized = True

        cross_reinterps = self._discover_cross_domain_links(phenomenon_name, summary, ep)
        branches = [
            ParallelUniverseBranch(
                branch_id="universe_inertia",
                title="우주 A · 본 장면만 흘려보내기",
                premise=f"'{phenomenon_name}'을 한 번 보고 다른 기억과 잇지 않는다.",
                chosen_pivot="observe_only",
                predicted_outcome="단편적 감각으로만 남아 다른 현상을 만나도 원리를 옮겨 쓰기 어렵다.",
                micro_sensory_effect="일회성 시각 자극 잔존",
                meso_relation_effect="원인-결과 고립",
                macro_causal_effect="현상 간 연결 부재",
                resolves_gap=False,
                selected=False,
            ),
            ParallelUniverseBranch(
                branch_id="universe_test_variation",
                title="우주 B · 조건을 바꿔 다시 실험해 보기",
                premise=f"'{cause_condition}'의 조건을 달리했을 때 '{observed_effect}'가 어떻게 변하는지 살펴본다.",
                chosen_pivot="vary_physical_condition",
                predicted_outcome=f"조건 변화와 결과 변화 사이의 연속된 인과 축을 더 또렷하게 세울 수 있다.",
                micro_sensory_effect="빛/모양의 미시 변화 민감도 증가",
                meso_relation_effect="조건->결과 공변 관계 강화",
                macro_causal_effect="새로운 물리 상황에서도 결과 예측 가능",
                resolves_gap=True,
                selected=not cross_reinterps,
            ),
            ParallelUniverseBranch(
                branch_id="universe_creative_transfer",
                title="우주 C · 다른 도메인 기억과 한 원리로 잇기",
                premise=(
                    f"방금 본 '{phenomenon_name}'이 예전 기억({', '.join(ep.cross_domain_links)})과 같은 결로 공명한다."
                    if ep.cross_domain_links
                    else f"'{phenomenon_name}'에서 느낀 원인->결과 원리를 다른 세계 경험과 연결해 본다."
                ),
                chosen_pivot="cross_domain_synthesis",
                predicted_outcome=(
                    f"말로만 알던 개념과 눈으로 본 물리 현상이 합쳐져, 낯선 현상을 봐도 먼저 원인을 상상할 수 있다."
                ),
                micro_sensory_effect="시각 감각과 언어 표상의 동시 공명",
                meso_relation_effect="교차 스케일 원리 통합",
                macro_causal_effect="언어·물리를 아우르는 미래 예측 확장",
                resolves_gap=True,
                selected=bool(cross_reinterps),
            ),
        ]
        cross_note = (
            f" 특히 {cross_reinterps[0].after_view}"
            if cross_reinterps
            else ""
        )
        real_note = " 아, 이제 현상은 조건이 바뀔 때 일정한 인과로 이어져 나타난다는 원리도 알겠어!" if just_realized else ""
        utt = f"직접 보니 {cause_condition} 때문에 {observed_effect} 현상이 일어나네.{real_note}{cross_note}".strip()
        ep.elysia_utterance = utt
        return {
            "episode": asdict(ep),
            "phenomenon_causality_realized": self.phenomenon_causality_realized,
            "just_realized_phenomenon_causality": just_realized,
            "cross_domain_links": list(ep.cross_domain_links),
            "reinterpretations": [asdict(r) for r in cross_reinterps],
            "parallel_universes": [asdict(b) for b in branches],
            "utterance": utt,
        }

    def imagine_parallel_universes_for_goal(
        self,
        goal_description: str,
        goal_condition: str,
    ) -> List[ParallelUniverseBranch]:
        """Builds Parallel Universes for physical/tool goals; tool branches exist ONLY if rule learned."""
        matching_rule = next(
            (r for r in self.world_knowledge.values() if r.condition == goal_condition),
            None,
        )
        branches: List[ParallelUniverseBranch] = [
            ParallelUniverseBranch(
                branch_id="universe_inertia",
                title="우주 A · 막힌 채 멈춰 서 있기",
                premise=f"'{goal_description}'가 '{goal_condition}' 때문에 막혀 있지만 아무 시도도 하지 않는다.",
                chosen_pivot="halt",
                predicted_outcome="조건이 바뀌지 않아 목표를 계속 이루지 못한다.",
                micro_sensory_effect="미해결 결핍 유지",
                meso_relation_effect="도구-목표 연결 없음",
                macro_causal_effect="행동 정체",
                resolves_gap=False,
                selected=False,
            ),
            ParallelUniverseBranch(
                branch_id="universe_ask_help",
                title="우주 B · 곁의 사람에게 방법을 묻거나 시범을 청하기",
                premise="이 상황에서 쓸 수 있는 도구나 배경지식을 아직 모르므로 도움을 요청한다.",
                chosen_pivot="ask_speaker",
                predicted_outcome="상대가 상황을 해결할 도구와 인과 이유를 알려줄 수 있다.",
                micro_sensory_effect="상대의 손짓·설명 집중",
                meso_relation_effect="관계적 안내를 통한 새 배경지식 대기",
                macro_causal_effect="다음 시도에서 스스로 해결할 발판 확보",
                resolves_gap=True,
                selected=matching_rule is None,
            ),
        ]
        if matching_rule is not None:
            branches.append(
                ParallelUniverseBranch(
                    branch_id="universe_use_learned_tool",
                    title=f"우주 C · 배운 도구('{matching_rule.medium}')로 직접 열어보기",
                    premise=(
                        f"'{matching_rule.causal_explanation}'는 세상 배경지식({matching_rule.rule_id})을 배웠기 때문이다."
                    ),
                    chosen_pivot=f"use_tool::{matching_rule.medium}",
                    predicted_outcome=(
                        f"'{matching_rule.medium}'를 사용하면 '{goal_condition}' 장벽을 넘어 목표를 이루고 "
                        f"아까 실패했던 기억까지 다시 이어 볼 수 있다."
                    ),
                    micro_sensory_effect=f"'{matching_rule.medium}' 조작에 따른 직접 감각 변화",
                    meso_relation_effect="도구 행위 -> 세계 변화 인과 확립",
                    macro_causal_effect="과거 실패 에피소드 재해석 + 자율 도구 활용 획득",
                    resolves_gap=True,
                    requires_rule_id=matching_rule.rule_id,
                    selected=True,
                )
            )
        return branches

    def experience_action_or_tool_attempt(
        self,
        goal_description: str,
        goal_condition: str,
        attempted_tool: str = "",
        observed_effect: str = "",
        succeeded: Optional[bool] = None,
        teach_tool_rule: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Action/Tool developmental learning: blocks unlearned tool branches prior to
        world-rule acquisition, verifies world effects, realizes `action_agency_realized`,
        and sweeps backward to reinterpret & recreate earlier failed attempts.
        """
        if teach_tool_rule is not None:
            med = str(teach_tool_rule.get("medium") or "").strip()
            cond = str(teach_tool_rule.get("condition") or goal_condition).strip()
            expl = str(teach_tool_rule.get("causal_explanation") or "").strip()
            src = str(teach_tool_rule.get("source_utterance") or expl).strip()
            ep_id = self._next_episode_id()
            rule = self.learn_world_background_knowledge(
                medium=med,
                condition=cond,
                causal_explanation=expl,
                source_utterance=src,
                episode_id=ep_id,
            )
            ep = LivedCausalEpisode(
                episode_id=ep_id,
                step=self._step_counter,
                scale_tier=4,
                source_origin="speaker_parent",
                modality="audiovisual",
                raw_input=src,
                perceived_kind="world_rule_learned",
                initial_understanding=f"도구 배경지식('{med}': {expl})을 배웠다.",
                future_causal_link=f"이제 '{cond}'에서 '{med}'를 선택해 볼 수 있다.",
                domain="action_tool",
            )
            self.episodes.append(ep)
            branches = self.imagine_parallel_universes_for_goal(goal_description, cond)
            utt = f"아! {expl} 그러면 이제 '{med}'를 써서 직접 확인해 볼 수 있겠어."
            ep.elysia_utterance = utt
            return {
                "executed": True,
                "blocked_by_causality": False,
                "learned_rule": asdict(rule),
                "episode": asdict(ep),
                "action_agency_realized": self.action_agency_realized,
                "reinterpretations": [],
                "parallel_universes": [asdict(b) for b in branches],
                "utterance": utt,
            }

        has_rule = any(r.condition == goal_condition for r in self.world_knowledge.values())
        if attempted_tool and succeeded is None and not has_rule:
            branches = self.imagine_parallel_universes_for_goal(goal_description, goal_condition)
            return {
                "executed": False,
                "blocked_by_causality": True,
                "reason": f"'{goal_condition}'에서 '{attempted_tool}'를 쓰는 인과적 배경지식을 아직 배우지 못해 도구를 함부로 실행하지 않았다.",
                "action_agency_realized": self.action_agency_realized,
                "reinterpretations": [],
                "parallel_universes": [asdict(b) for b in branches],
                "utterance": f"'{goal_description}'가 어려워서 어떻게 해야 할지 먼저 배워야 해.",
            }

        ep_id = self._next_episode_id()
        if not attempted_tool:
            ep = LivedCausalEpisode(
                episode_id=ep_id,
                step=self._step_counter,
                scale_tier=3,
                source_origin="embodied_body",
                modality="visual",
                raw_input=goal_description,
                perceived_kind="blocked_goal",
                unresolved_forms=[goal_condition],
                initial_understanding=f"'{goal_description}'를 하려 했지만 '{goal_condition}' 때문에 멈췄다.",
                domain="action_tool",
            )
            self.episodes.append(ep)
            branches = self.imagine_parallel_universes_for_goal(goal_description, goal_condition)
            utt = f"지금 '{goal_description}' 때문에 막혀 있는데, 어떤 도구나 방법을 쓰면 될까?"
            ep.elysia_utterance = utt
            return {
                "executed": True,
                "blocked_by_causality": False,
                "episode": asdict(ep),
                "action_agency_realized": self.action_agency_realized,
                "reinterpretations": [],
                "parallel_universes": [asdict(b) for b in branches],
                "utterance": utt,
            }

        ok_flag = bool(succeeded)
        ep = LivedCausalEpisode(
            episode_id=ep_id,
            step=self._step_counter,
            scale_tier=4 if ok_flag else 2,
            source_origin="embodied_body",
            modality="audiovisual",
            raw_input=f"{goal_description} [{attempted_tool}]",
            perceived_kind="action_tool_outcome",
            unresolved_forms=[] if ok_flag else [goal_condition],
            initial_understanding=(
                f"'{attempted_tool}'를 직접 해 보니 '{observed_effect}'가 일어났다."
                if ok_flag
                else f"'{attempted_tool}'로 시도했지만 '{observed_effect or goal_condition}' 때문에 아직 해결되지 않았다."
            ),
            domain="action_tool",
            action_attempted=attempted_tool,
            observed_effect=observed_effect,
        )
        self.episodes.append(ep)
        self.learned_tool_affordances[attempted_tool] = {
            "tool": attempted_tool,
            "goal_condition": goal_condition,
            "observed_effect": observed_effect,
            "succeeded": ok_flag,
            "episode_id": ep_id,
        }

        tool_attempts = [e for e in self.episodes if e.domain == "action_tool" and e.action_attempted]
        just_realized_agency = False
        if not self.action_agency_realized and len(tool_attempts) >= 2:
            self.action_agency_realized = True
            self.action_agency_evidence = [e.episode_id for e in tool_attempts]
            just_realized_agency = True

        reinterps: List[PastReinterpretation] = []
        if ok_flag:
            for past_ep in self.episodes:
                if past_ep.episode_id == ep_id or past_ep.domain != "action_tool":
                    continue
                if goal_condition not in past_ep.unresolved_forms:
                    continue
                r_obj = PastReinterpretation(
                    triggered_by_episode_id=ep_id,
                    unlocked_form=f"tool::{attempted_tool}",
                    before_view=past_ep.initial_understanding,
                    after_view=(
                        f"'{attempted_tool}'로 '{observed_effect}'를 직접 겪고 나니, "
                        f"아까 {past_ep.episode_id}에서 못 본 건 내 눈 한계가 아니라 확대 조건이 비었기 때문이었구나."
                    ),
                    recreated_past_action=(
                        f"그때 이미 '{attempted_tool}'의 역할을 알았더라면 멈춰 서지 않고 곧장 '{attempted_tool}'를 썼을 것이다."
                    ),
                    scale_bridge_summary=(
                        f"Micro(도구 조작 '{attempted_tool}') <-> Meso(감각 변화 '{observed_effect[:35]}') <-> "
                        f"Macro({past_ep.episode_id} 막힘 -> {ep_id} 해결)"
                    ),
                )
                past_ep.reinterpretations.append(r_obj)
                past_ep.unresolved_forms = [u for u in past_ep.unresolved_forms if u != goal_condition]
                past_ep.future_causal_link = (
                    f"[{past_ep.episode_id} 실패/막힘] -> [{ep_id} '{attempted_tool}' 성공] -> "
                    f"[유사한 시각 한계를 만나면 스스로 '{attempted_tool}' 선택 가능]"
                )
                reinterps.append(r_obj)

        branches = self.imagine_parallel_universes_for_goal(goal_description, goal_condition)
        past_note = f" 아까와 달리 {reinterps[0].after_view}" if reinterps else ""
        utt = f"'{attempted_tool}'를 직접 써 보니 '{observed_effect}'가 되네!{past_note}" if ok_flag else f"'{attempted_tool}'로는 아직 '{observed_effect}'가 부족해."
        ep.elysia_utterance = utt
        return {
            "executed": True,
            "blocked_by_causality": False,
            "episode": asdict(ep),
            "action_agency_realized": self.action_agency_realized,
            "just_realized_action_agency": just_realized_agency,
            "reinterpretations": [asdict(r) for r in reinterps],
            "parallel_universes": [asdict(b) for b in branches],
            "utterance": utt,
        }

    def execute_real_world_tool(
        self,
        tool_name: str,
        goal_description: str,
        goal_condition: str,
        *,
        retina: Optional[Any] = None,
        file_path: Optional[Path | str] = None,
        headword: str = "",
        fetcher: Optional[HttpFetcher] = None,
    ) -> Dict[str, Any]:
        """
        Executes a genuine physical/PC tool (`dictionary`, `screen_zoom_crop`, `local_file_inspect`)
        ONLY after a matching causal `WorldCausalRule` has been acquired, returning honest real outcomes.
        """
        has_rule = any(
            r.condition == goal_condition or r.medium == tool_name
            for r in self.world_knowledge.values()
        )
        if not has_rule:
            branches = self.imagine_parallel_universes_for_goal(goal_description, goal_condition)
            return {
                "executed": False,
                "blocked_by_causality": True,
                "reason": f"'{goal_description}'에서 '{tool_name}'을 왜 써야 하는지 세상 배경지식을 아직 배우지 못했다.",
                "parallel_universes": [asdict(b) for b in branches],
                "utterance": f"'{goal_description}'를 풀려고 '{tool_name}'을 쓰기 전에, 왜 그 도구가 통하는지 먼저 배워야 해.",
            }

        if tool_name == "dictionary":
            return self.execute_causal_dictionary_lookup(headword, lang="ko", fetcher=fetcher)

        if tool_name == "screen_zoom_crop":
            crop_res = (
                retina.capture_salient_zoom_crop()
                if retina is not None
                else {"ok": False, "error": "실시간 화면 눈(retina)이 연결되지 않았습니다."}
            )
            ok_flag = bool(crop_res.get("ok") and crop_res.get("detail_revealed"))
            if crop_res.get("ok") and not ok_flag:
                eff = f"{crop_res.get('summary_ko')} (확대해도 대비·선명도 상승 미미)"
            else:
                eff = str(crop_res.get("summary_ko") or crop_res.get("error") or "화면 확대 실패")
            res = self.experience_action_or_tool_attempt(
                goal_description=goal_description,
                goal_condition=goal_condition,
                attempted_tool="screen_zoom_crop",
                observed_effect=eff,
                succeeded=ok_flag,
            )
            res["tool_raw_result"] = crop_res
            return res

        if tool_name == "local_file_inspect":
            file_res = (
                inspect_local_file_sensory(file_path)
                if file_path is not None
                else {"ok": False, "error": "살펴볼 파일 경로가 주어지지 않았습니다."}
            )
            ok_flag = bool(file_res.get("ok"))
            eff = str(file_res.get("summary_ko") or file_res.get("error") or "파일 읽기 실패")
            res = self.experience_action_or_tool_attempt(
                goal_description=goal_description,
                goal_condition=goal_condition,
                attempted_tool="local_file_inspect",
                observed_effect=eff,
                succeeded=ok_flag,
            )
            res["tool_raw_result"] = file_res
            return res

        return {
            "executed": False,
            "blocked_by_causality": False,
            "reason": f"알 수 없는 실제 도구 이름: {tool_name}",
        }
