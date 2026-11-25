r"""
ELYSIA Self-Inspecting, Self-Contrasting & Self-Rewriting Cognitive Architecture
==============================================================================

Instead of tweaking numeric weights along fixed developer rails, Elysia executes
her cognition through inspectable, versioned Python organs (`CognitiveOrgan`):

1. **Execute (`run_organ`):** Runs live Python source (`def process(percept, memory):`)
   inside an AST-validated sandbox.
2. **Self-Inspect & Contrast (`inspect_and_contrast_self`):** Walks her own Python
   AST to locate exact lines responsible for outcome divergence against external/human
   criteria, identifying `must_stay_same` (invariants) vs `must_change`.
3. **Synthesize, Regression-Verify & Hot-Swap (`attempt_self_rewrite`):** Rewrites
   an organ (`v1 -> v2`) or spawns a brand-new organ, rejects unsafe code or any
   rewrite breaking `must_stay_same` invariants, and adopts verified rewrites.
4. **Retroactive Re-Execution (`reinterpret_past_episodes`):** Re-runs earlier lived
   episodes through the rewritten cognitive organ without erasing original history.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


_SAFE_BUILTINS: Dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}

_FORBIDDEN_AST_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.Delete,
    ast.With,
    ast.AsyncWith,
)

_FORBIDDEN_NAMES = frozenset(
    {
        "__import__",
        "compile",
        "delattr",
        "eval",
        "exec",
        "getattr",
        "globals",
        "locals",
        "open",
        "setattr",
        "vars",
    }
)


def validate_organ_source(source_code: str) -> Tuple[bool, str]:
    """Validates candidate cognitive organ Python source via AST inspection before compilation."""
    clean = (source_code or "").strip()
    if not clean:
        return False, "Empty organ source code"
    try:
        tree = ast.parse(clean)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"

    has_process_fn = False
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_AST_NODES):
            return False, f"Forbidden AST statement: {type(node).__name__}"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False, f"Forbidden dunder attribute access: {node.attr}"
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            return False, f"Forbidden builtin name: {node.id}"
        if isinstance(node, ast.FunctionDef) and node.name == "process":
            if len(node.args.args) != 2:
                return False, "Function 'process' must accept exactly 2 arguments (percept, memory)"
            has_process_fn = True

    if not has_process_fn:
        return False, "Missing required entry point: def process(percept, memory):"
    return True, "ok"


def _compile_organ_callable(source_code: str) -> Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]:
    ok, reason = validate_organ_source(source_code)
    if not ok:
        raise ValueError(f"Unsafe or invalid organ code: {reason}")
    ns: Dict[str, Any] = {}
    exec(compile(source_code, "<cognitive_organ>", "exec"), {"__builtins__": _SAFE_BUILTINS}, ns)
    fn = ns.get("process")
    if not callable(fn):
        raise ValueError("Compiled organ lacks callable 'process'")
    return fn


_NAIVE_PERCEIVER_V1 = """def process(percept, memory):
    content = str(percept.get("content", ""))
    known = memory.get("known_items", [])
    if content and content in known:
        return {
            "stance": "familiar",
            "action": "keep_observing",
            "differentiation": "already_known",
        }
    return {
        "stance": "unfamiliar_noise",
        "action": "avoid",
        "differentiation": "rejected_as_unknown",
    }
"""

_DIFFERENTIATING_PERCEIVER_V2 = """def process(percept, memory):
    content = str(percept.get("content", ""))
    known = memory.get("known_items", [])
    if content and content in known:
        return {
            "stance": "familiar",
            "action": "keep_observing",
            "differentiation": "already_known",
        }

    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    lower = content.lower()
    is_code = any(ln.lower().startswith(("def ", "class ", "import ", "from ", "async def ")) for ln in lines)
    is_doc = any(ln.startswith("#") for ln in lines)
    is_cfg = ("[" in content and "=" in content) and not is_code
    is_visual = percept.get("kind") == "visual_screen"

    if is_visual:
        kind = "visual_scene"
    elif is_code:
        kind = "code_structure"
    elif is_doc:
        kind = "structured_document"
    elif is_cfg:
        kind = "config_spec"
    else:
        kind = "novel_text"

    words = [w for w in content.replace("\\n", " ").split(" ") if len(w) >= 2]
    prev_cats = memory.get("categories", {})
    prev_words = set()
    for vals in prev_cats.values():
        for w in vals:
            prev_words.add(w)

    shared = sorted(set(words) & prev_words)[:5]
    novel = [w for w in words if w not in prev_words][:6]
    diff_summary = (
        kind
        + " ("
        + str(len(lines))
        + " lines; novel="
        + ", ".join(novel[:4])
        + "; shared="
        + (", ".join(shared[:3]) if shared else "none")
        + ")"
    )
    return {
        "stance": "differentiated_curiosity",
        "action": "analyze_and_differentiate",
        "category": kind,
        "differentiation": diff_summary,
        "shared_with_past": shared,
        "novel_elements": novel,
        "question": "What causal role does this " + kind + " serve?",
        "memory_updates": {
            "known_items": [content] if content else [],
            "categories": {kind: list(dict.fromkeys(words[:15]))},
        },
    }
"""

_CROSS_CASE_COMPARATOR_V1 = """def process(percept, memory):
    a = percept.get("case_a", {})
    b = percept.get("case_b", {})
    txt_a = str(a.get("content", ""))
    txt_b = str(b.get("content", ""))
    toks_a = set([w for w in txt_a.replace("\\n", " ").split(" ") if len(w) >= 2])
    toks_b = set([w for w in txt_b.replace("\\n", " ").split(" ") if len(w) >= 2])
    same_tokens = sorted(toks_a & toks_b)[:8]
    only_a = sorted(toks_a - toks_b)[:6]
    only_b = sorted(toks_b - toks_a)[:6]

    out_a = a.get("output", {})
    out_b = b.get("output", {})
    cat_a = str(out_a.get("category", out_a.get("stance", "")))
    cat_b = str(out_b.get("category", out_b.get("stance", "")))

    return {
        "comparison_ok": True,
        "label_a": str(a.get("label", "A")),
        "label_b": str(b.get("label", "B")),
        "same_features": same_tokens,
        "only_in_a": only_a,
        "only_in_b": only_b,
        "categories_same": cat_a == cat_b,
        "category_a": cat_a,
        "category_b": cat_b,
        "why_different": (
            str(a.get("label", "A"))
            + "("
            + cat_a
            + ") vs "
            + str(b.get("label", "B"))
            + "("
            + cat_b
            + "): distinct structures="
            + ", ".join(only_b[:3])
        ),
    }
"""


@dataclass
class CognitiveOrgan:
    """Versioned live executable Python cognitive organ owned by Elysia."""

    organ_id: str
    version: int
    parent_version: int
    purpose: str
    source_code: str
    reason_for_change: str
    verified_cases_passed: int = 0
    created_at_step: int = 0


@dataclass
class StructuralOutcomeContrast:
    """AST-grounded comparison showing What matched/differed, Where in code, Must-Stay vs Must-Change."""

    organ_id: str
    organ_version: int
    what_matched: List[str]
    what_differed: Dict[str, Tuple[Any, Any]]
    where_in_own_code: List[str]
    must_stay_same: List[str]
    must_change: List[str]
    recommended_evolution: str  # "keep" | "rewrite_organ" | "create_new_organ"
    explanation_ko: str


@dataclass
class SelfRewriteAttempt:
    """Audit record of an attempted executable code rewrite or new organ creation."""

    attempt_id: str
    organ_id: str
    from_version: int
    to_version: int
    evolution_kind: str  # "rewrite_organ" | "create_new_organ"
    adopted: bool
    rejection_reason: str
    source_diff_summary: str
    regression_passed: int
    regression_total: int
    past_episodes_reinterpreted: int = 0


@dataclass
class CognitiveEpisode:
    """Episodic record preserving initial output alongside later reinterpreted outputs."""

    episode_id: str
    step: int
    percept: Dict[str, Any]
    organ_id: str
    organ_version: int
    initial_output: Dict[str, Any]
    reinterpreted_outputs: List[Dict[str, Any]] = field(default_factory=list)


class SelfRewritingMind:
    """
    Cognitive system that executes inspectable Python organs, locates flaws in its own
    AST against human/world expectations, rewrites or spawns new executable organs,
    guards regression invariants (`must_stay_same`), and retroactively reinterprets history.
    """

    def __init__(self) -> None:
        self.memory: Dict[str, Any] = {
            "known_items": ["known_baseline_anchor"],
            "categories": {},
        }
        v1 = CognitiveOrgan(
            organ_id="primary_perceiver",
            version=1,
            parent_version=0,
            purpose="Perceive incoming reality items and decide cognitive response",
            source_code=_NAIVE_PERCEIVER_V1,
            reason_for_change="Initial naive baseline",
            verified_cases_passed=1,
            created_at_step=0,
        )
        self.active_organs: Dict[str, CognitiveOrgan] = {"primary_perceiver": v1}
        self.organ_history: List[CognitiveOrgan] = []
        self.verified_cases: List[Dict[str, Any]] = [
            {
                "organ_id": "primary_perceiver",
                "percept": {"content": "known_baseline_anchor"},
                "memory_snapshot": {"known_items": ["known_baseline_anchor"], "categories": {}},
                "expected_checks": {"stance": "familiar", "action": "keep_observing"},
            }
        ]
        self.episodes: List[CognitiveEpisode] = []
        self.rewrite_log: List[SelfRewriteAttempt] = []
        self._compiled_cache: Dict[Tuple[str, int], Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = {}
        self._step_counter: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory": copy.deepcopy(self.memory),
            "active_organs": {k: asdict(v) for k, v in self.active_organs.items()},
            "organ_history": [asdict(o) for o in self.organ_history],
            "verified_cases": copy.deepcopy(self.verified_cases),
            "episodes": [asdict(e) for e in self.episodes],
            "rewrite_log": [asdict(r) for r in self.rewrite_log],
            "step_counter": self._step_counter,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "SelfRewritingMind":
        obj = cls()
        obj.memory = copy.deepcopy(raw.get("memory") or {"known_items": [], "categories": {}})
        obj.active_organs = {
            str(k): CognitiveOrgan(**v) for k, v in (raw.get("active_organs") or {}).items()
        }
        obj.organ_history = [CognitiveOrgan(**o) for o in (raw.get("organ_history") or [])]
        obj.verified_cases = copy.deepcopy(raw.get("verified_cases") or [])
        obj.episodes = [CognitiveEpisode(**e) for e in (raw.get("episodes") or [])]
        obj.rewrite_log = [SelfRewriteAttempt(**r) for r in (raw.get("rewrite_log") or [])]
        obj._step_counter = int(raw.get("step_counter", len(obj.episodes)))
        obj._compiled_cache.clear()
        return obj

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "SelfRewritingMind":
        p = Path(path)
        if not p.is_file():
            return cls()
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def _get_callable(self, organ: CognitiveOrgan) -> Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]:
        key = (organ.organ_id, organ.version)
        fn = self._compiled_cache.get(key)
        if fn is None:
            fn = _compile_organ_callable(organ.source_code)
            self._compiled_cache[key] = fn
        return fn

    def _apply_memory_updates(self, mem_upd: Dict[str, Any]) -> None:
        for item in mem_upd.get("known_items") or []:
            if item not in self.memory.setdefault("known_items", []):
                self.memory["known_items"].append(item)
        cats = self.memory.setdefault("categories", {})
        for ck, vals in (mem_upd.get("categories") or {}).items():
            existing = cats.setdefault(ck, [])
            for v in vals:
                if v not in existing:
                    existing.append(v)

    def run_organ(
        self,
        organ_id: str,
        percept: Dict[str, Any],
        *,
        record_episode: bool = True,
    ) -> Dict[str, Any]:
        """Executes an active Python cognitive organ safely and records the episode."""
        organ = self.active_organs.get(organ_id)
        if organ is None:
            raise KeyError(f"Cognitive organ not found: {organ_id}")
        fn = self._get_callable(organ)
        out = dict(fn(copy.deepcopy(percept), copy.deepcopy(self.memory)) or {})
        if isinstance(out.get("memory_updates"), dict):
            self._apply_memory_updates(out["memory_updates"])
        if record_episode:
            self._step_counter += 1
            ep = CognitiveEpisode(
                episode_id=f"cog_{self._step_counter:04d}",
                step=self._step_counter,
                percept=copy.deepcopy(percept),
                organ_id=organ.organ_id,
                organ_version=organ.version,
                initial_output=copy.deepcopy(out),
            )
            self.episodes.append(ep)
        return out

    def _locate_diverging_lines_in_ast(self, source_code: str, diverging_values: List[Any]) -> List[str]:
        lines = source_code.splitlines()
        targets = {v for v in diverging_values if isinstance(v, str) and v}
        hit_linenos: set[int] = set()
        try:
            tree = ast.parse(source_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in targets:
                    if hasattr(node, "lineno"):
                        hit_linenos.add(int(node.lineno))
        except SyntaxError:
            pass
        return [
            f"L{ln}: {lines[ln - 1]}"
            for ln in sorted(hit_linenos)
            if 1 <= ln <= len(lines)
        ]

    def inspect_and_contrast_self(
        self,
        organ_id: str,
        percept: Dict[str, Any],
        my_output: Dict[str, Any],
        external_reference: Dict[str, Any],
        *,
        requires_new_organ_id: str = "",
    ) -> StructuralOutcomeContrast:
        """
        Inspects active organ's Python AST alongside external outcome expectations:
        reports what matched, what differed, where in own source code, must-stay vs must-change.
        """
        if requires_new_organ_id and requires_new_organ_id not in self.active_organs:
            return StructuralOutcomeContrast(
                organ_id=requires_new_organ_id,
                organ_version=0,
                what_matched=[],
                what_differed={"missing_organ": (None, requires_new_organ_id)},
                where_in_own_code=["(현재 활성 기관 중 두 대상의 공통점/차이점을 비교하는 기관이 부재함)"],
                must_stay_same=[f"{c['organ_id']}:{c['expected_checks']}" for c in self.verified_cases],
                must_change=[f"create_organ:{requires_new_organ_id}"],
                recommended_evolution="create_new_organ",
                explanation_ko=(
                    f"기존 기관 '{organ_id}'은 개별 대상을 볼 뿐 두 경험을 나란히 비교하는 능력이 없어 "
                    f"새 인지 기관 '{requires_new_organ_id}' 생성이 필요하다."
                ),
            )

        organ = self.active_organs[organ_id]
        what_matched: List[str] = []
        what_differed: Dict[str, Tuple[Any, Any]] = {}
        for k, ref_v in external_reference.items():
            my_v = my_output.get(k)
            if my_v == ref_v:
                what_matched.append(k)
            else:
                what_differed[k] = (my_v, ref_v)

        div_vals = [pair[0] for pair in what_differed.values()]
        where = self._locate_diverging_lines_in_ast(organ.source_code, div_vals)
        must_stay = [f"{c['organ_id']}:{c['expected_checks']}" for c in self.verified_cases if c["organ_id"] == organ_id]
        for mk in what_matched:
            must_stay.append(f"{mk}=={my_output.get(mk)}")
        must_change = [f"{k}: {pair[0]} -> {pair[1]}" for k, pair in what_differed.items()]

        evo = "keep" if not what_differed else "rewrite_organ"
        expl = (
            f"내 코드({organ.organ_id} v{organ.version})는 {where or ['(해당 분기)']}에서 "
            f"{must_change} 방향으로 작동해 바깥 기준과 달라졌고, {must_stay[:2]} 성질은 그대로 유지해야 한다."
            if what_differed
            else f"내 코드({organ.organ_id} v{organ.version})가 기대 결과와 일치한다."
        )
        return StructuralOutcomeContrast(
            organ_id=organ.organ_id,
            organ_version=organ.version,
            what_matched=what_matched,
            what_differed=what_differed,
            where_in_own_code=where,
            must_stay_same=must_stay,
            must_change=must_change,
            recommended_evolution=evo,
            explanation_ko=expl,
        )

    def synthesize_candidate_code(
        self,
        contrast: StructuralOutcomeContrast,
        *,
        target_organ_id: str = "",
    ) -> Tuple[str, str]:
        """Synthesizes executable Python source for rewriting an existing organ or creating a new one."""
        if contrast.recommended_evolution == "create_new_organ" or (target_organ_id and target_organ_id not in self.active_organs):
            return (
                _CROSS_CASE_COMPARATOR_V1,
                "Create new cognitive organ extracting shared features, unique features, and causal outcome contrast",
            )
        return (
            _DIFFERENTIATING_PERCEIVER_V2,
            "Preserve familiar invariant while replacing avoid/noise rejection with structural differentiation & memory learning",
        )

    def _check_expected_output(self, actual_out: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        for k, exp_v in expected.items():
            if actual_out.get(k) != exp_v:
                return False
        return True

    def attempt_self_rewrite(
        self,
        contrast: StructuralOutcomeContrast,
        trigger_percept: Dict[str, Any],
        external_reference: Dict[str, Any],
        *,
        candidate_source_override: Optional[str] = None,
        new_organ_id: str = "",
    ) -> SelfRewriteAttempt:
        """
        Tests candidate organ code against AST safety, ALL `must_stay_same` regression cases,
        and the new `must_change` reality before hot-swapping; rejects unsafe/regressing code.
        """
        is_new = contrast.recommended_evolution == "create_new_organ" or bool(new_organ_id and new_organ_id not in self.active_organs)
        target_id = new_organ_id or contrast.organ_id
        old_organ = self.active_organs.get(target_id)
        from_ver = old_organ.version if old_organ is not None else 0
        to_ver = from_ver + 1
        evo_kind = "create_new_organ" if is_new else "rewrite_organ"
        attempt_id = f"rw_{len(self.rewrite_log) + 1:04d}"

        if candidate_source_override is not None:
            cand_src = candidate_source_override
            diff_summary = "Custom candidate rewrite"
        else:
            cand_src, diff_summary = self.synthesize_candidate_code(contrast, target_organ_id=target_id)

        # 1. AST security & syntactic validity check
        ok_ast, ast_reason = validate_organ_source(cand_src)
        if not ok_ast:
            att = SelfRewriteAttempt(
                attempt_id=attempt_id,
                organ_id=target_id,
                from_version=from_ver,
                to_version=to_ver,
                evolution_kind=evo_kind,
                adopted=False,
                rejection_reason=f"ast_validation_failed: {ast_reason}",
                source_diff_summary=diff_summary,
                regression_passed=0,
                regression_total=sum(1 for c in self.verified_cases if c["organ_id"] == target_id),
            )
            self.rewrite_log.append(att)
            return att

        try:
            cand_fn = _compile_organ_callable(cand_src)
        except Exception as exc:
            att = SelfRewriteAttempt(
                attempt_id=attempt_id,
                organ_id=target_id,
                from_version=from_ver,
                to_version=to_ver,
                evolution_kind=evo_kind,
                adopted=False,
                rejection_reason=f"compile_error: {exc}",
                source_diff_summary=diff_summary,
                regression_passed=0,
                regression_total=sum(1 for c in self.verified_cases if c["organ_id"] == target_id),
            )
            self.rewrite_log.append(att)
            return att

        # 2. Regression verification on ALL prior verified cases (`must_stay_same`)
        relevant_cases = [c for c in self.verified_cases if c["organ_id"] == target_id]
        reg_passed = 0
        for case in relevant_cases:
            try:
                res = dict(cand_fn(copy.deepcopy(case["percept"]), copy.deepcopy(case["memory_snapshot"])) or {})
            except Exception as exc:
                att = SelfRewriteAttempt(
                    attempt_id=attempt_id,
                    organ_id=target_id,
                    from_version=from_ver,
                    to_version=to_ver,
                    evolution_kind=evo_kind,
                    adopted=False,
                    rejection_reason=f"regression_runtime_error: {exc}",
                    source_diff_summary=diff_summary,
                    regression_passed=reg_passed,
                    regression_total=len(relevant_cases),
                )
                self.rewrite_log.append(att)
                return att
            if not self._check_expected_output(res, case["expected_checks"]):
                att = SelfRewriteAttempt(
                    attempt_id=attempt_id,
                    organ_id=target_id,
                    from_version=from_ver,
                    to_version=to_ver,
                    evolution_kind=evo_kind,
                    adopted=False,
                    rejection_reason=(
                        f"regression_on_prior_verified_case: expected {case['expected_checks']}, got {res}"
                    ),
                    source_diff_summary=diff_summary,
                    regression_passed=reg_passed,
                    regression_total=len(relevant_cases),
                )
                self.rewrite_log.append(att)
                return att
            reg_passed += 1

        # 3. Verify candidate fixes the `must_change` mismatch on trigger_percept
        try:
            trig_mem = {"known_items": ["known_baseline_anchor"], "categories": {}}
            trig_out = dict(cand_fn(copy.deepcopy(trigger_percept), copy.deepcopy(trig_mem)) or {})
        except Exception as exc:
            att = SelfRewriteAttempt(
                attempt_id=attempt_id,
                organ_id=target_id,
                from_version=from_ver,
                to_version=to_ver,
                evolution_kind=evo_kind,
                adopted=False,
                rejection_reason=f"trigger_runtime_error: {exc}",
                source_diff_summary=diff_summary,
                regression_passed=reg_passed,
                regression_total=len(relevant_cases),
            )
            self.rewrite_log.append(att)
            return att

        if not self._check_expected_output(trig_out, external_reference):
            att = SelfRewriteAttempt(
                attempt_id=attempt_id,
                organ_id=target_id,
                from_version=from_ver,
                to_version=to_ver,
                evolution_kind=evo_kind,
                adopted=False,
                rejection_reason=(
                    f"failed_target_expectation: expected {external_reference}, got {trig_out}"
                ),
                source_diff_summary=diff_summary,
                regression_passed=reg_passed,
                regression_total=len(relevant_cases),
            )
            self.rewrite_log.append(att)
            return att

        # 4. Hot-swap adopted cognitive organ and archive old version
        if old_organ is not None:
            self.organ_history.append(old_organ)
        new_organ = CognitiveOrgan(
            organ_id=target_id,
            version=to_ver,
            parent_version=from_ver,
            purpose=old_organ.purpose if old_organ else diff_summary,
            source_code=cand_src,
            reason_for_change="; ".join(contrast.must_change) or diff_summary,
            verified_cases_passed=reg_passed + 1,
            created_at_step=self._step_counter,
        )
        self.active_organs[target_id] = new_organ
        self.verified_cases.append(
            {
                "organ_id": target_id,
                "percept": copy.deepcopy(trigger_percept),
                "memory_snapshot": copy.deepcopy(trig_mem),
                "expected_checks": copy.deepcopy(external_reference),
            }
        )

        # 5. Retroactively reinterpret all prior episodes for this organ
        reinterp_count = self.reinterpret_past_episodes(target_id)
        att = SelfRewriteAttempt(
            attempt_id=attempt_id,
            organ_id=target_id,
            from_version=from_ver,
            to_version=to_ver,
            evolution_kind=evo_kind,
            adopted=True,
            rejection_reason="",
            source_diff_summary=diff_summary,
            regression_passed=reg_passed,
            regression_total=len(relevant_cases),
            past_episodes_reinterpreted=reinterp_count,
        )
        self.rewrite_log.append(att)
        return att

    def reinterpret_past_episodes(self, organ_id: str) -> int:
        """
        Re-executes past episodes through the rewritten organ using clean pre-learning memory
        so previously avoided episodes are genuinely re-understood without overwriting history.
        """
        organ = self.active_organs.get(organ_id)
        if organ is None:
            return 0
        fn = self._get_callable(organ)
        changed_cnt = 0
        replay_mem: Dict[str, Any] = {"known_items": ["known_baseline_anchor"], "categories": {}}

        for ep in self.episodes:
            if ep.organ_id != organ_id:
                continue
            new_out = dict(fn(copy.deepcopy(ep.percept), copy.deepcopy(replay_mem)) or {})
            if isinstance(new_out.get("memory_updates"), dict):
                for item in new_out["memory_updates"].get("known_items") or []:
                    if item not in replay_mem["known_items"]:
                        replay_mem["known_items"].append(item)
                for ck, vals in (new_out["memory_updates"].get("categories") or {}).items():
                    replay_mem["categories"].setdefault(ck, []).extend(
                        [v for v in vals if v not in replay_mem["categories"].get(ck, [])]
                    )

            diff_keys = [
                k for k in ("stance", "action", "differentiation", "category")
                if ep.initial_output.get(k) != new_out.get(k)
            ]
            if diff_keys:
                ep.reinterpreted_outputs.append(
                    {
                        "by_version": organ.version,
                        "changed_fields": diff_keys,
                        "new_output": new_out,
                    }
                )
                changed_cnt += 1
        return changed_cnt
