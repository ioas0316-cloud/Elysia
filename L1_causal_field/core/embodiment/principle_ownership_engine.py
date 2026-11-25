r"""
ELYSIA Domain-Agnostic Principle Extraction, Self-Contrast & Ownership Engine
=============================================================================

Allows Elysia to confront external principles from ANY domain (mathematics, physics,
chemistry, visual perception, files, web knowledge) WITHOUT hardcoded 1:1 tool rules:

1. **Extract** a general causal principle `(target_context, mechanism_action,
   expected_outcome, required_capabilities, candidate_expr)` from outside reality.
2. **Contrast** it against her own prior methods ("나는 이렇게만 해봤는데,
   밖에서는 저렇게도 할 수 있구나").
3. **Autonomously enact & verify** it using her real embodied primitives
   (`compute_sandbox`, `visual_pixel_crop`, `file_read`, `http_observe`,
   `text_transform`, `passive_observe`) with zero user `/commands`.
4. **Distinguish true ownership vs. false claims vs. honest bodily limits:**
   - `self_owned_executable`: verified by her own execution & added to her repertoire.
   - `owned_as_mental_model`: predictive quantitative law verified while honestly
     acknowledging lack of physical actuators.
   - `self_limit_recognized`: cannot enact because physical hands/reagents/heat/sensors are absent.
   - `rejected_by_self_verification`: her own test contradicted the outside claim.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
import json
import math
import operator
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from core.embodiment.pc_sensory_motor_body import inspect_local_file_sensory
from core.orchestration.external_web_bridge import HttpFetcher, fetch_dictionary_meaning


EMBODIED_CAPABILITIES = frozenset(
    {
        "compute_sandbox",
        "visual_pixel_crop",
        "file_read",
        "http_observe",
        "text_transform",
        "passive_observe",
    }
)

_SAFE_FUNCS: Dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "int": int,
    "float": float,
    "list": list,
    "tuple": tuple,
    "range": range,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
}

_SAFE_CONSTS: Dict[str, Any] = {
    "pi": math.pi,
    "e": math.e,
    "True": True,
    "False": False,
}

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_CMP_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


def _eval_ast_node(node: ast.AST, env: Dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_ast_node(node.body, env)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, bool, str)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        if node.id in _SAFE_CONSTS:
            return _SAFE_CONSTS[node.id]
        raise KeyError(f"Unknown variable: {node.id}")
    if isinstance(node, ast.UnaryOp):
        val = _eval_ast_node(node.operand, env)
        if isinstance(node.op, ast.UAdd):
            return +val
        if isinstance(node.op, ast.USub):
            return -val
        if isinstance(node.op, ast.Not):
            return not val
        raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
    if isinstance(node, ast.BinOp):
        left = _eval_ast_node(node.left, env)
        right = _eval_ast_node(node.right, env)
        fn = _BIN_OPS.get(type(node.op))
        if fn is None:
            raise ValueError(f"Unsupported binary op: {type(node.op).__name__}")
        if isinstance(node.op, ast.Pow) and isinstance(right, (int, float)) and abs(right) > 128:
            raise ValueError("Exponent exceeds safe limit")
        return fn(left, right)
    if isinstance(node, ast.Compare):
        left = _eval_ast_node(node.left, env)
        for op_node, comp_node in zip(node.ops, node.comparators):
            fn = _CMP_OPS.get(type(op_node))
            if fn is None:
                raise ValueError(f"Unsupported comparison: {type(op_node).__name__}")
            right = _eval_ast_node(comp_node, env)
            if not fn(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            res = True
            for v in node.values:
                res = _eval_ast_node(v, env)
                if not res:
                    return False
            return res
        if isinstance(node.op, ast.Or):
            res = False
            for v in node.values:
                res = _eval_ast_node(v, env)
                if res:
                    return res
            return res
    if isinstance(node, ast.IfExp):
        return _eval_ast_node(node.body if _eval_ast_node(node.test, env) else node.orelse, env)
    if isinstance(node, ast.Tuple):
        return tuple(_eval_ast_node(e, env) for e in node.elts)
    if isinstance(node, ast.List):
        return [_eval_ast_node(e, env) for e in node.elts]
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_FUNCS or node.keywords:
            raise ValueError("Disallowed function call in compute_sandbox")
        args = [_eval_ast_node(a, env) for a in node.args]
        if node.func.id == "range":
            r = range(*args)
            if len(r) > 100_000:
                raise ValueError("Range length exceeds compute_sandbox cap (100,000)")
            return r
        return _SAFE_FUNCS[node.func.id](*args)
    raise ValueError(f"Disallowed AST node: {type(node).__name__}")


def _safe_eval_math_expr(expr: str, env: Dict[str, Any]) -> Tuple[bool, Any, str]:
    """Safely parses and executes an arithmetic/functional expression without Python eval()."""
    clean = (expr or "").strip()
    if not clean:
        return False, None, "Empty expression"
    try:
        tree = ast.parse(clean, mode="eval")
        val = _eval_ast_node(tree, dict(env))
        return True, val, ""
    except Exception as exc:
        return False, None, str(exc)


@dataclass
class OwnedMethod:
    """A concrete method owned by Elysia (innate baseline, executable, or predictive mental model)."""

    method_id: str
    domain: str
    goal_signature: str
    description: str
    primitive_kind: str  # "compute_sandbox" | "visual_pixel_crop" | "file_read" | "http_observe" | "text_transform" | "passive_observe"
    executable_spec: str = ""
    ownership_status: str = "innate_baseline"  # "innate_baseline" | "self_owned_executable" | "owned_as_mental_model"
    verified_evidence: List[str] = field(default_factory=list)
    source_origin: str = "innate"
    uses: int = 0


@dataclass
class ExternalPrinciple:
    """A general causal principle extracted from outside world observations or texts."""

    principle_id: str
    domain: str
    target_context: str
    mechanism_action: str
    expected_outcome: str
    source_origin: str
    raw_excerpt: str
    required_capabilities: List[str] = field(default_factory=list)
    candidate_expr: str = ""
    test_inputs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MethodContrast:
    """Explicit structural contrast between Elysia's prior way and an outside principle."""

    contrast_id: str
    my_old_method_id: str
    my_old_description: str
    outside_principle_id: str
    outside_description: str
    difference_insight: str
    hypothesis_to_test: str


@dataclass
class OwnershipVerdict:
    """Self-verification verdict classifying ownership, rejection, or recognized bodily limit."""

    verdict_id: str
    principle_id: str
    contrast_id: str
    status: str  # "self_owned_executable" | "owned_as_mental_model" | "self_limit_recognized" | "rejected_by_self_verification"
    attempted_primitive: str
    verification_detail: str
    missing_requirements: List[str] = field(default_factory=list)
    own_statement_ko: str = ""


def _infer_domain_and_capabilities(
    text: str,
    domain_hint: str = "",
    candidate_expr: str = "",
) -> Tuple[str, List[str]]:
    lower = (text or "").lower()
    domain = domain_hint.strip()
    if not domain:
        if candidate_expr or any(k in lower for k in ("삼각수", "합 공식", "알고리즘", "수열", "math", "formula", "sum(", "n *")):
            domain = "math_algorithm"
        elif any(k in lower for k in ("중화", "염산", "수산화", "용액", "시약", "화학", "화학반응", "연소", "끓")):
            domain = "chemistry"
        elif any(k in lower for k in ("굴절", "스넬", "광학", "법칙", "질량", "가속도", "물리", "진자", "반사")):
            domain = "physics"
        elif any(k in lower for k in ("확대", "초점", "픽셀", "선명도", "대비", "화면", "작은 무늬")):
            domain = "visual"
        elif any(k in lower for k in ("파일", "문서", "이미지 열", "read_file", "경로")):
            domain = "file_io"
        elif any(k in lower for k in ("사전", "말", "낱말", "단어", "어휘", "뜻")):
            domain = "language"
        else:
            domain = "general_world"

    caps: List[str] = []
    if candidate_expr or domain == "math_algorithm":
        caps.append("compute_sandbox")
    if domain == "visual" or any(k in lower for k in ("확대", "초점", "잘라", "선명도", "픽셀")):
        caps.append("visual_pixel_crop")
    if domain == "file_io" or any(k in lower for k in ("로컬 파일", "파일을 열", "문서를 읽", "파일 읽")):
        caps.append("file_read")
    if any(k in lower for k in ("사전", "위키", "url", "웹")):
        caps.append("http_observe")

    # Physical world actuators (honestly NOT in EMBODIED_CAPABILITIES)
    if any(k in lower for k in ("용액", "염산", "수산화", "시약", "혼합", "산과 염기", "중화", "시험관", "비커")):
        caps.append("physical_chemical_hand")
    if any(k in lower for k in ("끓이", "가열", "버너", "불꽃", "연소", "태우")):
        caps.append("physical_heat_actuator")
    if domain in ("physics", "chemistry") and any(
        k in lower for k in ("매질", "광선", "물속", "공기", "유리", "실험 장치", "프리즘을 손으로", "잡아", "빛", "굴절", "반사")
    ):
        caps.append("physical_world_experiment")
    if not caps:
        caps.append("passive_observe")
    return domain, list(dict.fromkeys(caps))


def _extract_candidate_expr(text: str) -> str:
    """Attempts to extract an executable algebraic expression from explanatory prose."""
    clean = text or ""
    m_tag = re.search(r"(?:공식|식|formula)\s*[:=：]\s*([a-zA-Z0-9_()\s+\-*/.,]+)", clean, flags=re.IGNORECASE)
    if m_tag:
        cand = m_tag.group(1).strip().rstrip(".,;:")
        if "=" in cand:
            cand = cand.split("=")[-1].strip()
        return cand
    m_eq = re.search(r"([a-zA-Z_][a-zA-Z0-9_]*(?:\([^)]*\))?\s*=\s*[a-zA-Z0-9_()\s+\-*/.,]+)", clean)
    if m_eq:
        rhs = m_eq.group(1).split("=", 1)[1].strip().rstrip(".,;:")
        if any(op in rhs for op in ("+", "-", "*", "/", "sin", "cos", "asin", "sqrt")):
            return rhs
    return ""


def _split_causal_clauses(text: str) -> Tuple[str, str, str]:
    clean = re.sub(r"\s+", " ", (text or "")).strip()
    m_if = re.match(r"^(.+?)(?:일\s*때|하면|할\s*때|을\s*때)[,\s]+(.+)$", clean)
    if m_if:
        cond = m_if.group(1).strip()
        tail = m_if.group(2).strip()
        m_thru = re.match(r"^(.+?)(?:을\s*통해|를\s*통해|로|으로|하면)\s+(.+)$", tail)
        if m_thru:
            return cond, m_thru.group(1).strip(), m_thru.group(2).strip()
        return cond, tail, tail
    return clean[:50], clean[:80], clean[-80:]


def extract_principle_from_observation(
    outside_text: str,
    *,
    principle_id: str,
    domain: str = "",
    source_origin: str = "external_world",
    target_context: str = "",
    mechanism_action: str = "",
    expected_outcome: str = "",
    candidate_expr: str = "",
    test_inputs: Optional[Dict[str, Any]] = None,
) -> ExternalPrinciple:
    """Extracts a domain-agnostic `ExternalPrinciple` from outside prose or observations."""
    expr = (candidate_expr or "").strip() or _extract_candidate_expr(outside_text)
    inferred_domain, req_caps = _infer_domain_and_capabilities(
        outside_text,
        domain_hint=domain,
        candidate_expr=expr,
    )
    c_ctx, c_mech, c_out = _split_causal_clauses(outside_text)
    return ExternalPrinciple(
        principle_id=principle_id,
        domain=inferred_domain,
        target_context=target_context.strip() or c_ctx,
        mechanism_action=mechanism_action.strip() or c_mech,
        expected_outcome=expected_outcome.strip() or c_out,
        source_origin=source_origin,
        raw_excerpt=(outside_text or "").strip()[:320],
        required_capabilities=req_caps,
        candidate_expr=expr,
        test_inputs=dict(test_inputs or {}),
    )


class PrincipleOwnershipEngine:
    """
    Domain-agnostic engine that extracts outside causal principles, contrasts them
    against Elysia's prior repertoire, autonomously tests them via real embodied
    primitives, and accepts ownership only upon real self-verification.
    """

    def __init__(self) -> None:
        self.embodied_capabilities = set(EMBODIED_CAPABILITIES)
        self.own_methods: Dict[str, OwnedMethod] = {}
        self.verdict_history: List[OwnershipVerdict] = []
        self.recognized_limits: List[OwnershipVerdict] = []
        self._counter: int = 0
        self._seed_minimal_baseline_methods()

    def _seed_minimal_baseline_methods(self) -> None:
        baselines = [
            OwnedMethod(
                method_id="baseline_iterative_sum",
                domain="math_algorithm",
                goal_signature="1부터 n까지 연속된 자연수의 합 구하기 삼각수",
                description="1부터 n까지 하나씩 반복해서 차례대로 더하기",
                primitive_kind="compute_sandbox",
                executable_spec="sum(range(1, n + 1))",
                ownership_status="innate_baseline",
                source_origin="innate",
            ),
            OwnedMethod(
                method_id="baseline_full_frame_view",
                domain="visual",
                goal_signature="화면 세부 무늬 글자 확대 초점 관찰",
                description="화면 전체를 축소된 한 장 그대로 멀리서 바라보기",
                primitive_kind="passive_observe",
                ownership_status="innate_baseline",
                source_origin="innate",
            ),
            OwnedMethod(
                method_id="baseline_unmanipulated_observe",
                domain="chemistry",
                goal_signature="물질 반응 변화 관찰 중화 실험",
                description="직접 섞거나 가열하지 않고 겉모습만 지켜보기",
                primitive_kind="passive_observe",
                ownership_status="innate_baseline",
                source_origin="innate",
            ),
            OwnedMethod(
                method_id="baseline_visual_guess",
                domain="physics",
                goal_signature="빛 굴절 각도 반사 물리 법칙 예측",
                description="정확한 공식 없이 대략 눈대중으로 어림짐작하기",
                primitive_kind="passive_observe",
                ownership_status="innate_baseline",
                source_origin="innate",
            ),
        ]
        for m in baselines:
            self.own_methods.setdefault(m.method_id, m)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "own_methods": {k: asdict(v) for k, v in self.own_methods.items()},
            "verdict_history": [asdict(v) for v in self.verdict_history],
            "recognized_limits": [asdict(v) for v in self.recognized_limits],
            "counter": self._counter,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "PrincipleOwnershipEngine":
        obj = cls()
        for k, v in (raw.get("own_methods") or {}).items():
            obj.own_methods[str(k)] = OwnedMethod(**v)
        obj.verdict_history = [OwnershipVerdict(**v) for v in (raw.get("verdict_history") or [])]
        obj.recognized_limits = [OwnershipVerdict(**v) for v in (raw.get("recognized_limits") or [])]
        obj._counter = int(raw.get("counter", len(obj.verdict_history)))
        return obj

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "PrincipleOwnershipEngine":
        p = Path(path)
        if not p.is_file():
            return cls()
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def register_own_method(self, method: OwnedMethod) -> None:
        self.own_methods[method.method_id] = method

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter:04d}"

    def contrast_with_own_methods(self, principle: ExternalPrinciple) -> MethodContrast:
        """Contrasts the outside principle against Elysia's closest existing method."""
        p_tokens = set(re.findall(r"[가-힣a-zA-Z0-9]{2,}", f"{principle.target_context} {principle.mechanism_action} {principle.raw_excerpt}"))
        best_method: Optional[OwnedMethod] = None
        best_score = -1.0

        for m in self.own_methods.values():
            score = 0.0
            if m.domain == principle.domain:
                score += 2.0
            m_tokens = set(re.findall(r"[가-힣a-zA-Z0-9]{2,}", f"{m.goal_signature} {m.description}"))
            score += 0.6 * len(p_tokens & m_tokens)
            if m.ownership_status == "innate_baseline":
                score += 0.2
            if score > best_score:
                best_score = score
                best_method = m

        if best_method is None:
            best_method = next(iter(self.own_methods.values()))

        cid = f"contrast_{principle.principle_id}"
        return MethodContrast(
            contrast_id=cid,
            my_old_method_id=best_method.method_id,
            my_old_description=best_method.description,
            outside_principle_id=principle.principle_id,
            outside_description=f"{principle.mechanism_action} -> {principle.expected_outcome}",
            difference_insight=(
                f"나는 지금까지 '{best_method.description}' 방식으로만 해봤는데, "
                f"밖({principle.source_origin})에서는 '{principle.mechanism_action}' 원리로도 "
                f"'{principle.expected_outcome}'에 도달할 수 있구나."
            ),
            hypothesis_to_test=(
                f"내 기초 실행 능력으로 '{principle.mechanism_action}'을 직접 검증해 보고 "
                f"진짜 내 방법으로 삼을지, 내 현재 한계인지 확인해 본다."
            ),
        )

    def attempt_self_enactment(
        self,
        principle: ExternalPrinciple,
        contrast: MethodContrast,
        *,
        retina: Optional[Any] = None,
        file_path: Optional[Path | str] = None,
        fetcher: Optional[HttpFetcher] = None,
        reference_expected: Any = None,
    ) -> OwnershipVerdict:
        """
        Autonomously tests the outside principle via real embodied capabilities:
        never takes external claims on faith and never fakes physical execution.
        """
        vid = self._next_id("verdict")
        missing = [cap for cap in principle.required_capabilities if cap not in self.embodied_capabilities]
        old_method = self.own_methods.get(contrast.my_old_method_id)

        # Case A: Quantitative / algorithmic verification inside compute_sandbox
        if principle.candidate_expr:
            env = dict(principle.test_inputs)
            ok_new, actual, err_new = _safe_eval_math_expr(principle.candidate_expr, env)
            if not ok_new:
                verdict = OwnershipVerdict(
                    verdict_id=vid,
                    principle_id=principle.principle_id,
                    contrast_id=contrast.contrast_id,
                    status="rejected_by_self_verification",
                    attempted_primitive="compute_sandbox",
                    verification_detail=f"외부 식 실행 실패: {err_new}",
                    missing_requirements=missing,
                    own_statement_ko=f"직접 시험해 보려 했지만 식('{principle.candidate_expr}')이 실행되지 않아 받아들이지 않았어.",
                )
                self.verdict_history.append(verdict)
                return verdict

            expected = reference_expected
            expected_source = "reference_expected"
            if expected is None and old_method is not None and old_method.executable_spec:
                ok_old, old_val, _ = _safe_eval_math_expr(old_method.executable_spec, env)
                if ok_old:
                    expected = old_val
                    expected_source = f"내 기존 방식({old_method.executable_spec})"

            if expected is not None:
                try:
                    matched = abs(float(actual) - float(expected)) <= 1e-4
                except Exception:
                    matched = actual == expected
                if not matched:
                    verdict = OwnershipVerdict(
                        verdict_id=vid,
                        principle_id=principle.principle_id,
                        contrast_id=contrast.contrast_id,
                        status="rejected_by_self_verification",
                        attempted_primitive="compute_sandbox",
                        verification_detail=(
                            f"대조 검증 불일치: 외부 식 '{principle.candidate_expr}' -> {actual}, "
                            f"{expected_source} -> {expected}"
                        ),
                        missing_requirements=[],
                        own_statement_ko=(
                            f"밖에서 배운 방식('{principle.candidate_expr}')을 내가 직접 돌려보니 {actual}가 나왔는데, "
                            f"내 검증 결과({expected})와 달라서 거짓/부정확하다고 판단해 버렸어."
                        ),
                    )
                    self.verdict_history.append(verdict)
                    return verdict

            if not missing:
                new_id = f"owned::{principle.domain}_{principle.principle_id}"
                self.own_methods[new_id] = OwnedMethod(
                    method_id=new_id,
                    domain=principle.domain,
                    goal_signature=f"{principle.target_context} {principle.mechanism_action}",
                    description=f"{principle.mechanism_action} ({principle.candidate_expr})",
                    primitive_kind="compute_sandbox",
                    executable_spec=principle.candidate_expr,
                    ownership_status="self_owned_executable",
                    verified_evidence=[f"inputs={env} -> actual={actual}" + (f", verified_vs={expected}" if expected is not None else "")],
                    source_origin=principle.source_origin,
                )
                verdict = OwnershipVerdict(
                    verdict_id=vid,
                    principle_id=principle.principle_id,
                    contrast_id=contrast.contrast_id,
                    status="self_owned_executable",
                    attempted_primitive="compute_sandbox",
                    verification_detail=f"'{principle.candidate_expr}' 실행 결과 {actual} (기준={expected}) 일치 검증 완료",
                    missing_requirements=[],
                    own_statement_ko=(
                        f"이전엔 '{old_method.description if old_method else '기존 방식'}'으로만 했는데, "
                        f"밖의 원리('{principle.candidate_expr}')를 내가 직접 시험하니 {actual}로 똑같이 성립하네! 이제 내 방법으로 체화했어."
                    ),
                )
                self.verdict_history.append(verdict)
                return verdict

            model_id = f"model::{principle.domain}_{principle.principle_id}"
            self.own_methods[model_id] = OwnedMethod(
                method_id=model_id,
                domain=principle.domain,
                goal_signature=f"{principle.target_context} {principle.mechanism_action}",
                description=f"[예측 모델] {principle.mechanism_action} ({principle.candidate_expr})",
                primitive_kind="compute_sandbox",
                executable_spec=principle.candidate_expr,
                ownership_status="owned_as_mental_model",
                verified_evidence=[f"inputs={env} -> predicted={actual}"],
                source_origin=principle.source_origin,
            )
            verdict = OwnershipVerdict(
                verdict_id=vid,
                principle_id=principle.principle_id,
                contrast_id=contrast.contrast_id,
                status="owned_as_mental_model",
                attempted_primitive="compute_sandbox",
                verification_detail=f"수치 예측 '{principle.candidate_expr}' -> {actual} 검증됨 (물리 실험 장치 없음: {missing})",
                missing_requirements=missing,
                own_statement_ko=(
                    f"내 몸에 물리적 장치({', '.join(missing)})는 없어 직접 만지진 못하지만, "
                    f"원리 식('{principle.candidate_expr}')을 직접 계산해 보니 {actual}로 맞아떨어져서 예측 법칙으로 내 안에 세웠어."
                ),
            )
            self.recognized_limits.append(verdict)
            self.verdict_history.append(verdict)
            return verdict

        # Case B: Visual pixel self-enactment without user commands
        if "visual_pixel_crop" in principle.required_capabilities:
            if retina is None:
                verdict = OwnershipVerdict(
                    verdict_id=vid,
                    principle_id=principle.principle_id,
                    contrast_id=contrast.contrast_id,
                    status="self_limit_recognized",
                    attempted_primitive="visual_pixel_crop",
                    verification_detail="시각 망막(ScreenRetinaSensor)이 없어 직접 픽셀 확인 불가",
                    missing_requirements=["visual_retina_sensor"],
                    own_statement_ko="시각 확대 원리는 알지만 지금 내 눈(ScreenRetinaSensor)이 연결되지 않아 직접 확인해 볼 수 없어.",
                )
                self.recognized_limits.append(verdict)
                self.verdict_history.append(verdict)
                return verdict

            crop_res = retina.capture_salient_zoom_crop()
            if crop_res.get("ok") and crop_res.get("detail_revealed"):
                new_id = f"owned::{principle.domain}_{principle.principle_id}"
                self.own_methods[new_id] = OwnedMethod(
                    method_id=new_id,
                    domain="visual",
                    goal_signature=f"{principle.target_context} {principle.mechanism_action}",
                    description=f"자율 초점 확대 관찰 ({principle.mechanism_action})",
                    primitive_kind="visual_pixel_crop",
                    executable_spec="capture_salient_zoom_crop",
                    ownership_status="self_owned_executable",
                    verified_evidence=[str(crop_res.get("summary_ko"))],
                    source_origin=principle.source_origin,
                )
                verdict = OwnershipVerdict(
                    verdict_id=vid,
                    principle_id=principle.principle_id,
                    contrast_id=contrast.contrast_id,
                    status="self_owned_executable",
                    attempted_primitive="visual_pixel_crop",
                    verification_detail=str(crop_res.get("summary_ko")),
                    missing_requirements=[],
                    own_statement_ko=(
                        f"예전엔 화면 전체만 멍하니 봤는데, 직접 초점을 잘라 확대해 보니 "
                        f"({crop_res.get('summary_ko')}) 실제로 더 선명해지네! 이제 내 시각 능력으로 얻었어."
                    ),
                )
                self.verdict_history.append(verdict)
                return verdict

            verdict = OwnershipVerdict(
                verdict_id=vid,
                principle_id=principle.principle_id,
                contrast_id=contrast.contrast_id,
                status="rejected_by_self_verification",
                attempted_primitive="visual_pixel_crop",
                verification_detail=str(crop_res.get("summary_ko") or crop_res.get("error") or "선명도 개선 미달"),
                missing_requirements=[],
                own_statement_ko="직접 화면을 확대해 봤지만 지금 화면에서는 실제로 세부 선명도가 올라가지 않아 채택하지 않았어.",
            )
            self.verdict_history.append(verdict)
            return verdict

        # Case C: File or Web autonomous enactment
        if "file_read" in principle.required_capabilities and file_path is not None:
            f_res = inspect_local_file_sensory(file_path)
            if f_res.get("ok"):
                new_id = f"owned::{principle.domain}_{principle.principle_id}"
                self.own_methods[new_id] = OwnedMethod(
                    method_id=new_id,
                    domain=principle.domain,
                    goal_signature=f"{principle.target_context} {principle.mechanism_action}",
                    description=f"직접 파일 감각 열람 ({principle.mechanism_action})",
                    primitive_kind="file_read",
                    executable_spec="inspect_local_file_sensory",
                    ownership_status="self_owned_executable",
                    verified_evidence=[str(f_res.get("summary_ko"))],
                    source_origin=principle.source_origin,
                )
                verdict = OwnershipVerdict(
                    verdict_id=vid,
                    principle_id=principle.principle_id,
                    contrast_id=contrast.contrast_id,
                    status="self_owned_executable",
                    attempted_primitive="file_read",
                    verification_detail=str(f_res.get("summary_ko")),
                    missing_requirements=[],
                    own_statement_ko=f"배운 원리대로 내가 직접 파일을 열어 보니 ({f_res.get('summary_ko')}) 확인되어 내 능력으로 삼았어.",
                )
                self.verdict_history.append(verdict)
                return verdict
            verdict = OwnershipVerdict(
                verdict_id=vid,
                principle_id=principle.principle_id,
                contrast_id=contrast.contrast_id,
                status="self_limit_recognized",
                attempted_primitive="file_read",
                verification_detail=str(f_res.get("error")),
                missing_requirements=["accessible_file"],
                own_statement_ko=f"원리는 알지만 실제로 열어보니 실패했어({f_res.get('error')}).",
            )
            self.recognized_limits.append(verdict)
            self.verdict_history.append(verdict)
            return verdict

        if "http_observe" in principle.required_capabilities and principle.test_inputs.get("query"):
            obs = fetch_dictionary_meaning(str(principle.test_inputs["query"]), lang="ko", fetcher=fetcher)
            if not obs.blocked and obs.text_body:
                new_id = f"owned::{principle.domain}_{principle.principle_id}"
                self.own_methods[new_id] = OwnedMethod(
                    method_id=new_id,
                    domain=principle.domain,
                    goal_signature=f"{principle.target_context} {principle.mechanism_action}",
                    description=f"웹/사전 직접 탐색 ({principle.mechanism_action})",
                    primitive_kind="http_observe",
                    executable_spec="fetch_dictionary_meaning",
                    ownership_status="self_owned_executable",
                    verified_evidence=[f"{obs.url}"],
                    source_origin=principle.source_origin,
                )
                verdict = OwnershipVerdict(
                    verdict_id=vid,
                    principle_id=principle.principle_id,
                    contrast_id=contrast.contrast_id,
                    status="self_owned_executable",
                    attempted_primitive="http_observe",
                    verification_detail=f"실제 탐색 성공: {obs.url}",
                    missing_requirements=[],
                    own_statement_ko=f"직접 바깥 사전을 확인해 보니({obs.url}) 실제로 풀려서 내 탐색 방법으로 체화했어.",
                )
                self.verdict_history.append(verdict)
                return verdict

        # Case D: Physical-matter world principle without digital verification or actuator
        req_miss = missing or [cap for cap in principle.required_capabilities if cap not in self.embodied_capabilities]
        if not req_miss:
            req_miss = ["physical_embodied_executor"]
        verdict = OwnershipVerdict(
            verdict_id=vid,
            principle_id=principle.principle_id,
            contrast_id=contrast.contrast_id,
            status="self_limit_recognized",
            attempted_primitive="none_missing_body",
            verification_detail=f"물리 실행 조건 결여: {', '.join(req_miss)}",
            missing_requirements=req_miss,
            own_statement_ko=(
                f"밖에서는 '{principle.mechanism_action}'으로 '{principle.expected_outcome}'가 된다는 걸 이해했지만, "
                f"내 몸에는 물리적 조건({', '.join(req_miss)})이 없어서 직접 해낼 수 없는 현재 나의 한계야."
            ),
        )
        self.recognized_limits.append(verdict)
        self.verdict_history.append(verdict)
        return verdict

    def learn_from_outside_and_self_verify(
        self,
        outside_text: str,
        *,
        domain: str = "",
        source_origin: str = "external_world",
        target_context: str = "",
        mechanism_action: str = "",
        expected_outcome: str = "",
        candidate_expr: str = "",
        test_inputs: Optional[Dict[str, Any]] = None,
        reference_expected: Any = None,
        retina: Optional[Any] = None,
        file_path: Optional[Path | str] = None,
        fetcher: Optional[HttpFetcher] = None,
    ) -> Dict[str, Any]:
        """Full autonomous loop: Extract -> Self-Contrast -> Enact/Verify -> Own or Recognize Limit."""
        pid = self._next_id("prin")
        principle = extract_principle_from_observation(
            outside_text,
            principle_id=pid,
            domain=domain,
            source_origin=source_origin,
            target_context=target_context,
            mechanism_action=mechanism_action,
            expected_outcome=expected_outcome,
            candidate_expr=candidate_expr,
            test_inputs=test_inputs,
        )
        contrast = self.contrast_with_own_methods(principle)
        verdict = self.attempt_self_enactment(
            principle,
            contrast,
            retina=retina,
            file_path=file_path,
            fetcher=fetcher,
            reference_expected=reference_expected,
        )
        return {
            "principle": asdict(principle),
            "contrast": asdict(contrast),
            "verdict": asdict(verdict),
            "owned_methods_count": len(self.own_methods),
            "self_owned_executable_count": sum(
                1 for m in self.own_methods.values() if m.ownership_status == "self_owned_executable"
            ),
            "recognized_limits_count": len(self.recognized_limits),
        }

    def choose_best_owned_method(self, domain: str, goal_text: str = "") -> Optional[OwnedMethod]:
        """Chooses the best self-verified method for a domain/goal, preferring self-owned executables."""
        g_tokens = set(re.findall(r"[가-힣a-zA-Z0-9]{2,}", goal_text or ""))
        scored: List[Tuple[float, OwnedMethod]] = []
        for m in self.own_methods.values():
            if m.domain != domain and domain:
                continue
            sc = 0.0
            if m.ownership_status == "self_owned_executable":
                sc += 3.0
            elif m.ownership_status == "owned_as_mental_model":
                sc += 1.5
            else:
                sc += 0.5
            m_tokens = set(re.findall(r"[가-힣a-zA-Z0-9]{2,}", f"{m.goal_signature} {m.description}"))
            sc += 0.5 * len(g_tokens & m_tokens)
            scored.append((sc, m))
        if not scored:
            return None
        scored.sort(key=lambda item: (-item[0], item[1].method_id))
        return scored[0][1]
