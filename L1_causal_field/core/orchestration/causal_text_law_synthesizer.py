"""
ELYSIA Living Causal Text Law Synthesizer, Dimensional Mirror & Metaphor Reflector (Phase 12)
=============================================================================================

Bridges natural Korean/English scientific prose, explicit equations, dimensional
mirrors, false internet myths, and poetic metaphors into `ExtendedCognitiveOrganism`:

1. Safe AST & SI Dimensional Mirror (`DimensionVector`, `SafeMathASTCompiler`):
   Strictly forbids `eval()`/`exec()`. Validates every expression against an
   allowlisted math AST and verifies physical SI dimensions BEFORE running any
   empirical test (e.g., `파장 + 온도 = 500` fails immediately as `[L]` + `[Theta]`).
2. Equation-Free Relational Archetypes (`CausalTextParser`):
   Transforms natural-language causal laws ("온도가 높아질수록 파장은 반비례한다",
   "거리가 멀어질수록 소리 세기는 거리의 제곱에 반비례한다") into live
   `EqualityBond` structures with an open unknown constant (`PinMode.FREE`).
3. Multi-World Reproducibility & Autonomous Constant Discovery:
   Evaluates every dimensionally sound claim across >= 2 distinct lived calibration
   worlds simultaneously via Least Action (dA = 0):
   - Single-point overfitting myths ("파장은 온도에 비례") collapse under multi-world
     contradiction (`EMPIRICALLY_FALSIFIED`).
   - True relational laws converge to an invariant physical constant
     (e.g., autonomously discovering `b_wien = 2,897,771.955 nm·K` purely from
     equation-free Korean prose + embodied perception) and permanently enter
     `ExtendedCognitiveOrganism.internalized_disciplines`.
4. Honest Literal Limits & Sky-to-Earth Metaphor Back-Tracing:
   Detects subjective/abstract sky targets ("분노", "침묵", "희망", "기억") bound to
   concrete physical source domains ("용광로/뜨겁게", "무거운 돌/내려앉다"), refuses
   to hallucinate fake literal equations, reflects on why literal parsing cannot
   grasp the sky directly, and back-traces the grounded physical isomorphism &
   Red/Blue/Yellow chromatic transfer explaining why the speaker chose the metaphor.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
import hashlib
import math
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.orchestration.extended_cognitive_organism import (
    AcademicLawCandidate,
    ExtendedCognitiveOrganism,
    ExtendedWorldState,
)
from core.physics.trinitarian_scale_binding_protocol import (
    CoupledQuantity,
    EqualityBond,
    PinMode,
    ScaleTier,
    TrinitarianScaleRotorSystem,
)
from core.topology.hangul_topological_engine import HangulTopologicalEngine

if TYPE_CHECKING:
    from core.physics.causal_field import CausalField


class UnsafeExpressionError(ValueError):
    """Raised when input attempts disallowed Python AST constructs or code injection."""


class DimensionalMismatchError(ValueError):
    """Raised when an equation combines incompatible physical SI dimensions."""


class TextSynthesisOutcome(str, Enum):
    """Classification of natural text processed through the Living Causal Synthesizer."""

    VERIFIED_PHYSICAL_LAW = "verified_physical_law"
    DIMENSIONAL_CONTRADICTION = "dimensional_contradiction"
    EMPIRICALLY_FALSIFIED = "empirically_falsified"
    ABSTRACT_METAPHOR_REFLECTION = "abstract_metaphor_reflection"
    UNPARSEABLE_OR_UNSAFE = "unparseable_or_unsafe"


@dataclass(frozen=True)
class DimensionVector:
    """SI dimensional exponent vector `[L, M, T, Theta, Angle]`."""

    L: int = 0
    M: int = 0
    T: int = 0
    Theta: int = 0
    Angle: int = 0

    def __mul__(self, other: "DimensionVector") -> "DimensionVector":
        return DimensionVector(
            L=self.L + other.L,
            M=self.M + other.M,
            T=self.T + other.T,
            Theta=self.Theta + other.Theta,
            Angle=self.Angle + other.Angle,
        )

    def __truediv__(self, other: "DimensionVector") -> "DimensionVector":
        return DimensionVector(
            L=self.L - other.L,
            M=self.M - other.M,
            T=self.T - other.T,
            Theta=self.Theta - other.Theta,
            Angle=self.Angle - other.Angle,
        )

    def pow(self, exp: int) -> "DimensionVector":
        e = int(exp)
        return DimensionVector(
            L=self.L * e,
            M=self.M * e,
            T=self.T * e,
            Theta=self.Theta * e,
            Angle=self.Angle * e,
        )

    def is_dimensionless(self) -> bool:
        return self.L == 0 and self.M == 0 and self.T == 0 and self.Theta == 0 and self.Angle == 0

    def format_label(self) -> str:
        parts: List[str] = []
        for sym, exp in (
            ("L", self.L),
            ("M", self.M),
            ("T", self.T),
            ("Theta", self.Theta),
            ("Angle", self.Angle),
        ):
            if exp == 1:
                parts.append(sym)
            elif exp != 0:
                parts.append(f"{sym}^{exp}")
        return "[" + "·".join(parts) + "]" if parts else "[1]"


VARIABLE_DIMENSIONS: Dict[str, DimensionVector] = {
    "emission_wavelength_nm": DimensionVector(L=1),
    "distance": DimensionVector(L=1),
    "source_pos_x": DimensionVector(L=1),
    "source_temp_k": DimensionVector(Theta=1),
    "ear_intensity": DimensionVector(M=1, T=-3),
    "source_power_w": DimensionVector(L=2, M=1, T=-3),
    "source_freq_hz": DimensionVector(T=-1),
    "source_vel_x": DimensionVector(L=1, T=-1),
    "touch_radiant_flux": DimensionVector(L=-2, Theta=4),
    "prism_angle_deg": DimensionVector(Angle=1),
}


# ASCII/Greek exact identifiers inside AST (looked up safely in ast.Name nodes)
AST_ALIAS_TO_CANONICAL: Dict[str, str] = {
    "lambda": "emission_wavelength_nm",
    "lam": "emission_wavelength_nm",
    "wavelength": "emission_wavelength_nm",
    "emission_wavelength_nm": "emission_wavelength_nm",
    "T": "source_temp_k",
    "temp": "source_temp_k",
    "temperature": "source_temp_k",
    "source_temp_k": "source_temp_k",
    "d": "distance",
    "dist": "distance",
    "distance": "distance",
    "source_pos_x": "source_pos_x",
    "I": "ear_intensity",
    "intensity": "ear_intensity",
    "ear_intensity": "ear_intensity",
    "P": "source_power_w",
    "power": "source_power_w",
    "source_power_w": "source_power_w",
    "v": "source_vel_x",
    "vel": "source_vel_x",
    "speed": "source_vel_x",
    "source_vel_x": "source_vel_x",
    "f": "source_freq_hz",
    "freq": "source_freq_hz",
    "source_freq_hz": "source_freq_hz",
    "touch_radiant_flux": "touch_radiant_flux",
    "prism_angle_deg": "prism_angle_deg",
    "c": "law_const",
    "const": "law_const",
    "k": "law_const",
    "law_const": "law_const",
}

# Ordered longest-first phrase rules for prose / natural language parsing
PROSE_PHRASE_TO_CANONICAL: Tuple[Tuple[str, str], ...] = (
    ("emission_wavelength_nm", "emission_wavelength_nm"),
    ("최대 방출 파장", "emission_wavelength_nm"),
    ("방출 파장", "emission_wavelength_nm"),
    ("방출파장", "emission_wavelength_nm"),
    ("wavelength", "emission_wavelength_nm"),
    ("파장", "emission_wavelength_nm"),
    ("source_temp_k", "source_temp_k"),
    ("절대온도", "source_temp_k"),
    ("절대 온도", "source_temp_k"),
    ("temperature", "source_temp_k"),
    ("온도", "source_temp_k"),
    ("ear_intensity", "ear_intensity"),
    ("소리의 세기", "ear_intensity"),
    ("음향 세기", "ear_intensity"),
    ("소리 세기", "ear_intensity"),
    ("소리세기", "ear_intensity"),
    ("intensity", "ear_intensity"),
    ("source_power_w", "source_power_w"),
    ("음원 출력", "source_power_w"),
    ("출력", "source_power_w"),
    ("power", "source_power_w"),
    ("distance", "distance"),
    ("거리", "distance"),
    ("source_freq_hz", "source_freq_hz"),
    ("진동수", "source_freq_hz"),
    ("주파수", "source_freq_hz"),
    ("frequency", "source_freq_hz"),
    ("source_vel_x", "source_vel_x"),
    ("상대 속도", "source_vel_x"),
    ("속도", "source_vel_x"),
    ("velocity", "source_vel_x"),
)

_ALLOWED_MATH_FUNCS: Dict[str, Callable[[float], float]] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "sin": math.sin,
    "cos": math.cos,
    "log": math.log,
    "exp": math.exp,
}


@dataclass
class MetaphorReflection:
    """Self-reflection explaining literal sensory limits & grounded physical isomorphism."""

    abstract_targets: List[str]
    physical_source_anchors: List[str]
    grounded_domain: str
    chromatic_transfer: Dict[str, float]
    literal_limit_reflection_ko: str
    backtraced_isomorphism_ko: str


@dataclass
class ParsedCausalClaim:
    """Structured intermediate representation synthesized from raw text."""

    raw_text: str
    kind: str  # "explicit_equation" | "relational_prose" | "metaphor_abstract" | "unsafe_or_unknown"
    lhs_expr: str
    rhs_expr: str
    variables: Tuple[str, ...]
    has_unknown_constant: bool
    archetype: str  # "explicit" | "inverse_square" | "square_proportional" | "inversely_proportional" | "proportional" | "metaphor" | "invalid"
    dimension_ok: bool
    dimension_error: str
    dependent_var: Optional[str] = None
    independent_var: Optional[str] = None
    metaphor_reflection: Optional[MetaphorReflection] = None
    lhs_eval: Optional[Callable[[Dict[str, float]], float]] = None
    rhs_eval: Optional[Callable[[Dict[str, float]], float]] = None


@dataclass
class TextSynthesisResult:
    """Outcome of grounding, dimension-checking, falsifying/verifying, or reflecting on text."""

    raw_text: str
    outcome: TextSynthesisOutcome
    claim: ParsedCausalClaim
    experiment_skipped: bool
    multi_world_action: float
    discovered_constant: Optional[float]
    internalized_law_name: Optional[str]
    metaphor_reflection: Optional[MetaphorReflection]
    hangul_reflection: str
    hangul_wave_coherence: float
    explanation: str


@dataclass
class DocumentIngestionReport:
    """Batch report across multi-sentence documents mixing laws, myths, & metaphors."""

    results: List[TextSynthesisResult]
    verified_laws: List[str]
    dimensional_rejections: List[str]
    falsified_claims: List[str]
    metaphor_reflections: List[MetaphorReflection]
    unsafe_or_unparsed: List[str]


# ---------------------------------------------------------------------------
# Safe AST Compiler & Dimensional Analyzer (Zero eval/exec!)
# ---------------------------------------------------------------------------


def _resolve_var(vals: Dict[str, float], name: str) -> float:
    if name == "distance" and "distance" not in vals:
        return float(vals["source_pos_x"]) - float(vals.get("body_pos_x", 0.0))
    return float(vals[name])


class SafeMathASTCompiler:
    """Parses and dimension-checks math expressions via strict AST allowlist."""

    @staticmethod
    def _preprocess_unicode_equation(expr: str) -> str:
        s = expr.strip()
        s = s.replace("λ", "lambda").replace("×", "*").replace("÷", "/").replace("−", "-")
        # Protect Python's reserved keyword `lambda` by mapping identifier token to `lam`
        s = re.sub(r"\blambda\b", "lam", s)
        # Map Korean/multi-word variable names inside algebraic formulas only
        for phrase, canon in PROSE_PHRASE_TO_CANONICAL:
            if any("\uac00" <= ch <= "\ud7a3" for ch in phrase):
                s = s.replace(phrase, f" {canon} ")
        # Replace single `=` outside `<=`, `>=`, `==`, `!=`
        s = s.replace("^", "**")
        return s.strip()

    def _validate_and_analyze_node(
        self,
        node: ast.AST,
        seen_vars: List[str],
    ) -> Tuple[Optional[DimensionVector], bool]:
        """Returns `(dimension_or_None, contains_const_factor)` or raises Unsafe/Dim errors."""
        if isinstance(node, ast.Expression):
            return self._validate_and_analyze_node(node.body, seen_vars)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return None, True
            raise UnsafeExpressionError(f"Unsupported constant literal {node.value!r}")

        if isinstance(node, ast.Name):
            raw_id = node.id
            if raw_id == "pi":
                return None, True
            if raw_id not in AST_ALIAS_TO_CANONICAL:
                raise UnsafeExpressionError(f"Unrecognized identifier '{raw_id}'")
            canon = AST_ALIAS_TO_CANONICAL[raw_id]
            if canon not in seen_vars:
                seen_vars.append(canon)
            if canon == "law_const":
                return None, True
            return VARIABLE_DIMENSIONS.get(canon), False

        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.UAdd, ast.USub)):
                raise UnsafeExpressionError(f"Disallowed unary operator {type(node.op).__name__}")
            return self._validate_and_analyze_node(node.operand, seen_vars)

        if isinstance(node, ast.BinOp):
            l_dim, l_hc = self._validate_and_analyze_node(node.left, seen_vars)
            r_dim, r_hc = self._validate_and_analyze_node(node.right, seen_vars)

            if isinstance(node.op, (ast.Add, ast.Sub)):
                if l_dim is not None and r_dim is not None and l_dim != r_dim:
                    raise DimensionalMismatchError(
                        f"Cannot add/subtract incompatible physical dimensions "
                        f"{l_dim.format_label()} and {r_dim.format_label()}"
                    )
                if (l_dim is not None) ^ (r_dim is not None):
                    concrete = l_dim if l_dim is not None else r_dim
                    return concrete, (l_hc or r_hc)
                return l_dim, (l_hc or r_hc)

            if isinstance(node.op, ast.Mult):
                if l_dim is None and r_dim is None:
                    return None, (l_hc or r_hc)
                d = (l_dim or DimensionVector()) * (r_dim or DimensionVector())
                return (None if d.is_dimensionless() else d), (l_hc or r_hc)

            if isinstance(node.op, ast.Div):
                if l_dim is None and r_dim is None:
                    return None, (l_hc or r_hc)
                d = (l_dim or DimensionVector()) / (r_dim or DimensionVector())
                return (None if d.is_dimensionless() else d), (l_hc or r_hc)

            if isinstance(node.op, ast.Pow):
                if r_dim is not None:
                    raise DimensionalMismatchError("Exponent must be dimensionless")
                if not isinstance(node.right, ast.Constant) or not isinstance(
                    node.right.value, (int, float)
                ):
                    if isinstance(node.right, ast.UnaryOp) and isinstance(
                        node.right.operand, ast.Constant
                    ):
                        exp_val = (
                            -float(node.right.operand.value)
                            if isinstance(node.right.op, ast.USub)
                            else float(node.right.operand.value)
                        )
                    else:
                        raise UnsafeExpressionError("Non-constant exponent not permitted")
                else:
                    exp_val = float(node.right.value)
                if abs(exp_val - round(exp_val)) > 1e-9:
                    raise DimensionalMismatchError("Fractional variable power requires sqrt()")
                if l_dim is None:
                    return None, l_hc
                d = l_dim.pow(int(round(exp_val)))
                return (None if d.is_dimensionless() else d), l_hc

            raise UnsafeExpressionError(f"Disallowed binary operator {type(node.op).__name__}")

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_MATH_FUNCS:
                raise UnsafeExpressionError("Function call outside math allowlist")
            if node.keywords or len(node.args) != 1:
                raise UnsafeExpressionError("Math functions take exactly 1 positional argument")
            arg_dim, arg_hc = self._validate_and_analyze_node(node.args[0], seen_vars)
            fname = node.func.id
            if fname == "abs":
                return arg_dim, arg_hc
            if fname == "sqrt":
                if arg_dim is None:
                    return None, arg_hc
                if any(
                    v % 2 != 0
                    for v in (arg_dim.L, arg_dim.M, arg_dim.T, arg_dim.Theta, arg_dim.Angle)
                ):
                    raise DimensionalMismatchError(
                        f"Square root produces fractional dimension on {arg_dim.format_label()}"
                    )
                d = DimensionVector(
                    L=arg_dim.L // 2,
                    M=arg_dim.M // 2,
                    T=arg_dim.T // 2,
                    Theta=arg_dim.Theta // 2,
                    Angle=arg_dim.Angle // 2,
                )
                return (None if d.is_dimensionless() else d), arg_hc
            if arg_dim is not None:
                raise DimensionalMismatchError(
                    f"Transcendental '{fname}' requires dimensionless argument, got {arg_dim.format_label()}"
                )
            return None, arg_hc

        raise UnsafeExpressionError(f"Disallowed AST node {type(node).__name__}")

    def _compile_node(self, node: ast.AST) -> Callable[[Dict[str, float]], float]:
        if isinstance(node, ast.Expression):
            return self._compile_node(node.body)
        if isinstance(node, ast.Constant):
            c_val = float(node.value)
            return lambda _v, c=c_val: c
        if isinstance(node, ast.Name):
            if node.id == "pi":
                return lambda _v: math.pi
            canon = AST_ALIAS_TO_CANONICAL[node.id]
            return lambda v, key=canon: _resolve_var(v, key)
        if isinstance(node, ast.UnaryOp):
            inner = self._compile_node(node.operand)
            if isinstance(node.op, ast.USub):
                return lambda v, f=inner: -f(v)
            return inner
        if isinstance(node, ast.BinOp):
            l_fn = self._compile_node(node.left)
            r_fn = self._compile_node(node.right)
            if isinstance(node.op, ast.Add):
                return lambda v, l=l_fn, r=r_fn: l(v) + r(v)
            if isinstance(node.op, ast.Sub):
                return lambda v, l=l_fn, r=r_fn: l(v) - r(v)
            if isinstance(node.op, ast.Mult):
                return lambda v, l=l_fn, r=r_fn: l(v) * r(v)
            if isinstance(node.op, ast.Div):
                return lambda v, l=l_fn, r=r_fn: l(v) / max(1e-12, r(v)) if abs(r(v)) < 1e-12 else l(v) / r(v)
            if isinstance(node.op, ast.Pow):
                return lambda v, l=l_fn, r=r_fn: l(v) ** r(v)
        if isinstance(node, ast.Call):
            arg_fn = self._compile_node(node.args[0])
            fn = _ALLOWED_MATH_FUNCS[node.func.id]  # type: ignore[union-attr]
            return lambda v, a=arg_fn, f=fn: float(f(a(v)))
        raise UnsafeExpressionError(f"Cannot compile AST node {type(node).__name__}")

    def compile_equation(
        self,
        lhs_raw: str,
        rhs_raw: str,
    ) -> Tuple[
        Callable[[Dict[str, float]], float],
        Callable[[Dict[str, float]], float],
        Tuple[str, ...],
        bool,
    ]:
        lhs_clean = self._preprocess_unicode_equation(lhs_raw)
        rhs_clean = self._preprocess_unicode_equation(rhs_raw)
        try:
            lhs_tree = ast.parse(lhs_clean, mode="eval")
            rhs_tree = ast.parse(rhs_clean, mode="eval")
        except SyntaxError as exc:
            raise UnsafeExpressionError(f"Invalid syntax: {exc}") from exc

        seen_vars: List[str] = []
        l_dim, l_hc = self._validate_and_analyze_node(lhs_tree, seen_vars)
        r_dim, r_hc = self._validate_and_analyze_node(rhs_tree, seen_vars)

        if l_dim is not None and r_dim is not None and not (l_hc or r_hc):
            if l_dim != r_dim:
                raise DimensionalMismatchError(
                    f"Dimensional equality mismatch: LHS {l_dim.format_label()} != RHS {r_dim.format_label()}"
                )

        if not seen_vars or all(v == "law_const" for v in seen_vars):
            raise UnsafeExpressionError("Equation contains no physical organism variables")

        has_const = "law_const" in seen_vars
        phys_vars = tuple(v for v in seen_vars if v != "law_const")
        return self._compile_node(lhs_tree), self._compile_node(rhs_tree), phys_vars, has_const


# ---------------------------------------------------------------------------
# Natural Language, Relational Archetype & Poetic Metaphor Parser
# ---------------------------------------------------------------------------


_ABSTRACT_SKY_TARGETS: Tuple[str, ...] = (
    "분노",
    "침묵",
    "슬픔",
    "희망",
    "고독",
    "마음",
    "사랑",
    "기억",
    "불안",
    "그리움",
    "anger",
    "silence",
    "grief",
    "hope",
    "loneliness",
    "memory",
)

_METAPHOR_DOMAINS: Tuple[Dict[str, Any], ...] = (
    {
        "domain": "thermal_radiant_source",
        "anchors": ("용광로", "뜨겁", "타오르", "불꽃", "열기", "붉은", "화염", "furnace", "fire", "burning", "hot"),
        "chromatic": {"red_flux": 0.72, "blue_order": 0.13, "yellow_entropy": 0.15},
        "source_desc_ko": "고온 열복사·화염 에너지 발산(높은 Red Flux)",
        "reason_ko": "내면 감정('분노')의 급격한 에너지 축적과 외부 분출 강도가 고온 열원의 복사 상승과 위상 동형을 이루기 때문입니다.",
    },
    {
        "domain": "gravitational_inertia",
        "anchors": ("무거운", "돌", "내려앉", "짓누르", "바위", "침전", "heavy", "stone", "weight", "gravity"),
        "chromatic": {"red_flux": 0.12, "blue_order": 0.76, "yellow_entropy": 0.12},
        "source_desc_ko": "중력 하강·고밀도 질량 정지(높은 Blue Order)",
        "reason_ko": "움직임이 멎은 '침묵'의 정적 구속감과 심리적 압박이 중력 아래 무거운 돌의 하강 관성과 위상 동형을 이루기 때문입니다.",
    },
    {
        "domain": "cryogenic_fracture",
        "anchors": ("얼음", "차갑", "날카롭", "칼날", "얼어붙", "ice", "cold", "sharp", "blade"),
        "chromatic": {"red_flux": 0.18, "blue_order": 0.58, "yellow_entropy": 0.24},
        "source_desc_ko": "저온 경직·날카로운 경계 단절(냉각 Blue Order)",
        "reason_ko": "관계적 온기가 식고 경계가 날 선 상태가 열 손실 및 결정 고착과 동형을 이루기 때문입니다.",
    },
    {
        "domain": "fluid_wave_momentum",
        "anchors": ("파도", "밀려오", "폭풍", "흐르", "물결", "wave", "storm", "tide"),
        "chromatic": {"red_flux": 0.55, "blue_order": 0.18, "yellow_entropy": 0.27},
        "source_desc_ko": "유체 파동 전파 및 확산 운동(Red-Yellow 간섭)",
        "reason_ko": "시간을 타고 밀려드는 정서의 주기적 가속과 확산이 유체 파동의 운동량 전달과 동형을 이루기 때문입니다.",
    },
)


class CausalTextParser:
    """Routes prose cleanly into equations, relational archetypes, metaphors, or unsafe/unknown."""

    def __init__(self) -> None:
        self.ast_compiler = SafeMathASTCompiler()

    @staticmethod
    def _extract_prose_variables(text: str) -> List[Tuple[int, str, int]]:
        """Returns sorted `(start_idx, canon_var, end_idx)` hits without duplicate overlaps."""
        lower = text.lower()
        spans: List[Tuple[int, str, int]] = []
        occupied: List[Tuple[int, int]] = []

        for phrase, canon in PROSE_PHRASE_TO_CANONICAL:
            p_low = phrase.lower()
            start = 0
            while True:
                idx = lower.find(p_low, start)
                if idx < 0:
                    break
                end = idx + len(p_low)
                # Word boundary check for pure ASCII phrases
                if phrase.isascii():
                    ok_l = idx == 0 or not (lower[idx - 1].isalnum() or lower[idx - 1] == "_")
                    ok_r = end == len(lower) or not (lower[end].isalnum() or lower[end] == "_")
                    if not (ok_l and ok_r):
                        start = end
                        continue
                overlaps = any(not (end <= s or idx >= e) for s, e in occupied)
                if not overlaps:
                    spans.append((idx, canon, end))
                    occupied.append((idx, end))
                start = end

        spans.sort(key=lambda item: item[0])
        return spans

    @staticmethod
    def _detect_metaphor(text: str) -> Optional[MetaphorReflection]:
        lower = text.lower()
        targets = [t for t in _ABSTRACT_SKY_TARGETS if t in lower]
        if not targets:
            return None

        best_dom: Optional[Dict[str, Any]] = None
        best_hits: List[str] = []
        for dom in _METAPHOR_DOMAINS:
            hits = [a for a in dom["anchors"] if a in lower]
            if len(hits) > len(best_hits):
                best_hits = hits
                best_dom = dom

        if best_dom is None or not best_hits:
            return None

        t_str = ", ".join(targets)
        a_str = ", ".join(best_hits)
        literal_refl = (
            f"나는 '{t_str}' 같은 내면·추상 영역(하늘)을 직접 측정하는 감각 기관이 없어 "
            f"이 문장을 문자 그대로의 물리 방정식으로 받아들이면 모순이 생깁니다. "
            f"대신 발화자가 끌어온 지상 감각 기반('{a_str}': {best_dom['source_desc_ko']})을 통해 "
            f"그 의미를 역추적합니다."
        )
        return MetaphorReflection(
            abstract_targets=targets,
            physical_source_anchors=best_hits,
            grounded_domain=str(best_dom["domain"]),
            chromatic_transfer=dict(best_dom["chromatic"]),
            literal_limit_reflection_ko=literal_refl,
            backtraced_isomorphism_ko=str(best_dom["reason_ko"]),
        )

    @staticmethod
    def _extract_equation_pair(text: str) -> Optional[Tuple[str, str]]:
        # Check backtick inline equation first: e.g. `lambda * T = 2897771.955`
        bt = re.search(r"`([^`]+)`", text)
        candidate = bt.group(1) if bt is not None else text
        if "=" not in candidate:
            return None
        # Split on single `=` or `==`
        eq_norm = candidate.replace("==", "=")
        parts = eq_norm.split("=")
        if len(parts) != 2:
            return None
        lhs, rhs = parts[0].strip(), parts[1].strip()
        # Strip leading/trailing Korean/English prose before colon if no backticks
        if ":" in lhs and bt is None:
            lhs = lhs.split(":")[-1].strip()
        return lhs, rhs

    def parse(self, raw_text: str) -> ParsedCausalClaim:
        text = raw_text.strip()
        if not text:
            return ParsedCausalClaim(
                raw_text=raw_text,
                kind="unsafe_or_unknown",
                lhs_expr="",
                rhs_expr="",
                variables=(),
                has_unknown_constant=False,
                archetype="invalid",
                dimension_ok=False,
                dimension_error="Empty input",
            )

        # 1. Explicit mathematical equation check
        eq_pair = self._extract_equation_pair(text)
        if eq_pair is not None:
            lhs_raw, rhs_raw = eq_pair
            try:
                lhs_fn, rhs_fn, phys_vars, has_const = self.ast_compiler.compile_equation(
                    lhs_raw, rhs_raw
                )
                return ParsedCausalClaim(
                    raw_text=raw_text,
                    kind="explicit_equation",
                    lhs_expr=lhs_raw,
                    rhs_expr=rhs_raw,
                    variables=phys_vars,
                    has_unknown_constant=has_const,
                    archetype="explicit",
                    dimension_ok=True,
                    dimension_error="",
                    lhs_eval=lhs_fn,
                    rhs_eval=rhs_fn,
                )
            except DimensionalMismatchError as exc:
                return ParsedCausalClaim(
                    raw_text=raw_text,
                    kind="explicit_equation",
                    lhs_expr=lhs_raw,
                    rhs_expr=rhs_raw,
                    variables=(),
                    has_unknown_constant=False,
                    archetype="explicit",
                    dimension_ok=False,
                    dimension_error=str(exc),
                )
            except UnsafeExpressionError as exc:
                return ParsedCausalClaim(
                    raw_text=raw_text,
                    kind="unsafe_or_unknown",
                    lhs_expr=lhs_raw,
                    rhs_expr=rhs_raw,
                    variables=(),
                    has_unknown_constant=False,
                    archetype="invalid",
                    dimension_ok=False,
                    dimension_error=str(exc),
                )

        # 2. Check poetic / abstract sky metaphor BEFORE rejecting prose
        metaphor = self._detect_metaphor(text)
        if metaphor is not None:
            return ParsedCausalClaim(
                raw_text=raw_text,
                kind="metaphor_abstract",
                lhs_expr="",
                rhs_expr="",
                variables=(),
                has_unknown_constant=False,
                archetype="metaphor",
                dimension_ok=True,
                dimension_error="",
                metaphor_reflection=metaphor,
            )

        # 3. Equation-free causal relational prose
        spans = self._extract_prose_variables(text)
        # Preserve unique canonical order while preserving occurrences for syntactic binding
        ordered_unique: List[str] = []
        for _, canon, _ in spans:
            if canon not in ordered_unique:
                ordered_unique.append(canon)

        if len(ordered_unique) >= 2:
            lower = text.lower()

            def _find_var_before_cue(cue_idx: int) -> Optional[str]:
                best: Optional[Tuple[int, str]] = None
                for start_i, canon, end_i in spans:
                    if end_i <= cue_idx:
                        best = (start_i, canon)
                return best[1] if best is not None else None

            if "제곱에 반비례" in lower or "제곱의 반비례" in lower or "inversely proportional to the square" in lower:
                cue_pos = max(
                    lower.find("제곱에 반비례"),
                    lower.find("제곱의 반비례"),
                    lower.find("inversely proportional to the square"),
                )
                x_var = _find_var_before_cue(cue_pos) or ordered_unique[0]
                y_var = next((v for v in reversed(ordered_unique) if v != x_var), ordered_unique[1])
                return ParsedCausalClaim(
                    raw_text=raw_text,
                    kind="relational_prose",
                    lhs_expr=f"{y_var} * ({x_var} ** 2)",
                    rhs_expr="law_const",
                    variables=(y_var, x_var),
                    has_unknown_constant=True,
                    archetype="inverse_square",
                    dimension_ok=True,
                    dimension_error="",
                    dependent_var=y_var,
                    independent_var=x_var,
                    lhs_eval=lambda v, y=y_var, x=x_var: _resolve_var(v, y) * (_resolve_var(v, x) ** 2),
                )

            if "제곱에 비례" in lower or "proportional to the square" in lower:
                cue_pos = max(lower.find("제곱에 비례"), lower.find("proportional to the square"))
                x_var = _find_var_before_cue(cue_pos) or ordered_unique[0]
                y_var = next((v for v in reversed(ordered_unique) if v != x_var), ordered_unique[1])
                return ParsedCausalClaim(
                    raw_text=raw_text,
                    kind="relational_prose",
                    lhs_expr=y_var,
                    rhs_expr=f"law_const * ({x_var} ** 2)",
                    variables=(y_var, x_var),
                    has_unknown_constant=True,
                    archetype="square_proportional",
                    dimension_ok=True,
                    dimension_error="",
                    dependent_var=y_var,
                    independent_var=x_var,
                )

            if "반비례" in lower or "inversely proportional" in lower:
                cue_pos = max(lower.find("반비례"), lower.find("inversely proportional"))
                x_var = _find_var_before_cue(cue_pos) or ordered_unique[0]
                y_var = next((v for v in ordered_unique if v != x_var), ordered_unique[-1])
                return ParsedCausalClaim(
                    raw_text=raw_text,
                    kind="relational_prose",
                    lhs_expr=f"{y_var} * {x_var}",
                    rhs_expr="law_const",
                    variables=(y_var, x_var),
                    has_unknown_constant=True,
                    archetype="inversely_proportional",
                    dimension_ok=True,
                    dimension_error="",
                    dependent_var=y_var,
                    independent_var=x_var,
                    lhs_eval=lambda v, y=y_var, x=x_var: _resolve_var(v, y) * _resolve_var(v, x),
                )

            if "비례" in lower or "proportional" in lower:
                cue_pos = max(lower.find("비례"), lower.find("proportional"))
                x_var = _find_var_before_cue(cue_pos) or ordered_unique[0]
                y_var = next((v for v in ordered_unique if v != x_var), ordered_unique[-1])
                return ParsedCausalClaim(
                    raw_text=raw_text,
                    kind="relational_prose",
                    lhs_expr=y_var,
                    rhs_expr=f"law_const * {x_var}",
                    variables=(y_var, x_var),
                    has_unknown_constant=True,
                    archetype="proportional",
                    dimension_ok=True,
                    dimension_error="",
                    dependent_var=y_var,
                    independent_var=x_var,
                )

        return ParsedCausalClaim(
            raw_text=raw_text,
            kind="unsafe_or_unknown",
            lhs_expr="",
            rhs_expr="",
            variables=tuple(ordered_unique),
            has_unknown_constant=False,
            archetype="invalid",
            dimension_ok=False,
            dimension_error="No grounded causal relation or safe equation detected.",
        )


# ---------------------------------------------------------------------------
# Multi-World Verifier, Constant Self-Discoverer & Organism Binder
# ---------------------------------------------------------------------------


class CausalTextLawSynthesizer:
    """
    Synthesizes live physical equality bonds from raw text, enforces dimensional
    consistency, verifies laws across multiple calibration worlds to reject
    overfitting myths, discovers hidden constants autonomously, and internalizes
    verified laws permanently into `ExtendedCognitiveOrganism`.
    """

    def __init__(self) -> None:
        self.parser = CausalTextParser()
        self.hangul_engine = HangulTopologicalEngine(target_dim=64)

    def _encode_reflection(self, text: str) -> float:
        _, metrics = self.hangul_engine.encode_text(text)
        return float(np.mean([m["coherence"] for m in metrics])) if metrics else 0.0

    def _observe_calibration_worlds(
        self,
        organism: ExtendedCognitiveOrganism,
        worlds: Sequence[ExtendedWorldState],
        active_organs: Sequence[str],
        use_spectrometer: bool,
    ) -> List[Dict[str, float]]:
        observed_list: List[Dict[str, float]] = []
        for w in worlds:
            perc = organism.perceive_extended(
                world=w,
                active_organs=active_organs,
                use_spectrometer=use_spectrometer,
            )
            vals = dict(perc.inferred_world)
            vals.update(perc.reconstructed_organs)
            vals["body_pos_x"] = perc.body_pos_x
            vals["distance"] = float(vals["source_pos_x"]) - float(perc.body_pos_x)
            observed_list.append(vals)
        return observed_list

    def _build_internalizable_candidate(
        self,
        claim: ParsedCausalClaim,
        discovered_const: Optional[float],
    ) -> AcademicLawCandidate:
        digest = hashlib.sha256(claim.raw_text.encode("utf-8")).hexdigest()[:8]
        law_name = f"text_law_{claim.archetype}_{digest}"

        concrete_vars = tuple(
            v
            for kv in claim.variables
            for v in (("source_pos_x", "body_pos_x") if kv == "distance" else (kv,))
        )
        # Preserve deterministic unique order
        unique_vars = tuple(dict.fromkeys(concrete_vars))

        if claim.archetype in ("inversely_proportional", "inverse_square") and discovered_const is not None:
            const_val = float(discovered_const)
            scale = max(1e-9, abs(const_val))
            y_v = claim.dependent_var or claim.variables[0]
            x_v = claim.independent_var or claim.variables[1]
            pwr = 2 if claim.archetype == "inverse_square" else 1

            def _builder(suf: str, y=y_v, x=x_v, p=pwr, c=const_val, sc=scale, uvars=unique_vars, ln=law_name) -> EqualityBond:
                return EqualityBond(
                    name=f"{ln}{suf}",
                    tier=ScaleTier.MECHANISM,
                    variables=uvars,
                    lhs_fn=lambda v, yk=y, xk=x, pk=p, s=sc: (_resolve_var(v, yk) * (_resolve_var(v, xk) ** pk)) / s,
                    rhs_fn=lambda _v, ck=c, s=sc: ck / s,
                    lhs_expr=f"({y} * {x}^{p}) / {sc:.4g}",
                    rhs_expr=f"{c:.4g} / {sc:.4g}",
                )

            expr_desc = f"{y_v} * ({x_v}^{pwr}) == {const_val:.6g}"
        elif claim.archetype in ("proportional", "square_proportional") and discovered_const is not None:
            const_val = float(discovered_const)
            y_v = claim.dependent_var or claim.variables[0]
            x_v = claim.independent_var or claim.variables[1]
            pwr = 2 if claim.archetype == "square_proportional" else 1
            scale = max(1e-9, abs(const_val))

            def _builder(suf: str, y=y_v, x=x_v, p=pwr, c=const_val, sc=scale, uvars=unique_vars, ln=law_name) -> EqualityBond:
                return EqualityBond(
                    name=f"{ln}{suf}",
                    tier=ScaleTier.MECHANISM,
                    variables=uvars,
                    lhs_fn=lambda v, yk=y, xk=x, pk=p, ck=c, s=sc: _resolve_var(v, yk) / max(1e-9, abs(ck * (_resolve_var(v, xk) ** pk))),
                    rhs_fn=lambda _v: 1.0,
                    lhs_expr=f"{y} / ({c:.4g} * {x}^{p})",
                    rhs_expr="1",
                )

            expr_desc = f"{y_v} == {const_val:.6g} * ({x_v}^{pwr})"
        else:
            # Explicit equation (potentially substituting solved `law_const`)
            lhs_f = claim.lhs_eval
            rhs_f = claim.rhs_eval
            c_sub = float(discovered_const) if discovered_const is not None else 1.0

            def _builder(suf: str, lf=lhs_f, rf=rhs_f, cs=c_sub, uvars=unique_vars, ln=law_name, le=claim.lhs_expr, re_s=claim.rhs_expr) -> EqualityBond:
                def _eval_l(v: Dict[str, float]) -> float:
                    vd = dict(v)
                    vd["law_const"] = cs
                    val_r = rf(vd)
                    val_l = lf(vd)
                    norm = max(1.0, abs(val_r), abs(val_l))
                    return val_l / norm

                def _eval_r(v: Dict[str, float]) -> float:
                    vd = dict(v)
                    vd["law_const"] = cs
                    val_r = rf(vd)
                    val_l = lf(vd)
                    norm = max(1.0, abs(val_r), abs(val_l))
                    return val_r / norm

                return EqualityBond(
                    name=f"{ln}{suf}",
                    tier=ScaleTier.MECHANISM,
                    variables=uvars,
                    lhs_fn=_eval_l,
                    rhs_fn=_eval_r,
                    lhs_expr=le,
                    rhs_expr=re_s,
                )

            expr_desc = f"{claim.lhs_expr} == {claim.rhs_expr}"

        return AcademicLawCandidate(
            name=law_name,
            discipline="text_synthesized_science",
            description=f"Verified from prose: {claim.raw_text.strip()}",
            expression=expr_desc,
            source="natural_language_synthesis",
            build_bond=_builder,
        )

    def synthesize_and_verify(
        self,
        text: str,
        organism: ExtendedCognitiveOrganism,
        calibration_worlds: Sequence[ExtendedWorldState],
        active_organs: Sequence[str] = ("eye", "ear", "hand"),
        use_spectrometer: bool = True,
        auto_internalize: bool = True,
    ) -> TextSynthesisResult:
        claim = self.parser.parse(text)

        if claim.kind == "unsafe_or_unknown":
            refl = "안전한 물리 어휘·수식 구조로 환원할 수 없어 실행을 차단했습니다."
            return TextSynthesisResult(
                raw_text=text,
                outcome=TextSynthesisOutcome.UNPARSEABLE_OR_UNSAFE,
                claim=claim,
                experiment_skipped=True,
                multi_world_action=float("nan"),
                discovered_constant=None,
                internalized_law_name=None,
                metaphor_reflection=None,
                hangul_reflection=refl,
                hangul_wave_coherence=self._encode_reflection(refl),
                explanation=f"[Unsafe/Unparsed] {claim.dimension_error}",
            )

        if claim.kind == "metaphor_abstract" and claim.metaphor_reflection is not None:
            mr = claim.metaphor_reflection
            organism.cognitive_growth_rings.append(
                {
                    "event": "metaphor_reflection",
                    "abstract_targets": list(mr.abstract_targets),
                    "grounded_domain": mr.grounded_domain,
                }
            )
            return TextSynthesisResult(
                raw_text=text,
                outcome=TextSynthesisOutcome.ABSTRACT_METAPHOR_REFLECTION,
                claim=claim,
                experiment_skipped=True,
                multi_world_action=0.0,
                discovered_constant=None,
                internalized_law_name=None,
                metaphor_reflection=mr,
                hangul_reflection=mr.literal_limit_reflection_ko,
                hangul_wave_coherence=self._encode_reflection(mr.literal_limit_reflection_ko),
                explanation=f"{mr.literal_limit_reflection_ko}\n  -> 동형 역추적: {mr.backtraced_isomorphism_ko}",
            )

        if not claim.dimension_ok:
            refl = f"차원 거울에서 모순({claim.dimension_error})이 발견되어 실험 전에 거절했습니다."
            organism.cognitive_growth_rings.append(
                {
                    "event": "dimensional_rejection",
                    "raw_text": text,
                    "error": claim.dimension_error,
                }
            )
            return TextSynthesisResult(
                raw_text=text,
                outcome=TextSynthesisOutcome.DIMENSIONAL_CONTRADICTION,
                claim=claim,
                experiment_skipped=True,
                multi_world_action=float("inf"),
                discovered_constant=None,
                internalized_law_name=None,
                metaphor_reflection=None,
                hangul_reflection=refl,
                hangul_wave_coherence=self._encode_reflection(refl),
                explanation=f"[Dimensional Mirror Rejected] {claim.dimension_error}",
            )

        # Multi-World Embodied Reproducibility Check
        worlds_obs = self._observe_calibration_worlds(
            organism=organism,
            worlds=calibration_worlds,
            active_organs=active_organs,
            use_spectrometer=use_spectrometer,
        )

        verify_sys = TrinitarianScaleRotorSystem(name="multi_world_text_verifier")
        discovered_const: Optional[float] = None
        scale_ref: float = 1.0

        for w_idx, w_vals in enumerate(worlds_obs):
            for v_name in set(claim.variables):
                q_name = f"{v_name}_w{w_idx}"
                val = _resolve_var(w_vals, v_name)
                verify_sys.add_quantity(
                    CoupledQuantity(
                        q_name,
                        val,
                        ScaleTier.VALUE,
                        "",
                        "fixed",
                        PinMode.FIXED_BOUNDARY,
                    )
                )

        if claim.has_unknown_constant:
            w0 = worlds_obs[0]
            if claim.archetype in ("inversely_proportional", "inverse_square"):
                y_v = claim.dependent_var or claim.variables[0]
                x_v = claim.independent_var or claim.variables[1]
                pwr = 2 if claim.archetype == "inverse_square" else 1
                scale_ref = max(1e-9, abs(_resolve_var(w0, y_v) * (_resolve_var(w0, x_v) ** pwr)))
                verify_sys.add_quantity(
                    CoupledQuantity("law_const_norm", 1.0, ScaleTier.MECHANISM, "", "state", PinMode.FREE, -1e6, 1e6)
                )
                for w_idx in range(len(worlds_obs)):
                    yn = f"{y_v}_w{w_idx}"
                    xn = f"{x_v}_w{w_idx}"
                    verify_sys.add_bond(
                        EqualityBond(
                            name=f"world_check_w{w_idx}",
                            tier=ScaleTier.MECHANISM,
                            variables=(yn, xn, "law_const_norm"),
                            lhs_fn=lambda v, yk=yn, xk=xn, pk=pwr, s=scale_ref: (v[yk] * (v[xk] ** pk)) / s,
                            rhs_fn=lambda v: v["law_const_norm"],
                            lhs_expr=f"({yn} * {xn}^{pwr}) / {scale_ref:.4g}",
                            rhs_expr="law_const_norm",
                        )
                    )
            elif claim.archetype in ("proportional", "square_proportional"):
                y_v = claim.dependent_var or claim.variables[0]
                x_v = claim.independent_var or claim.variables[1]
                pwr = 2 if claim.archetype == "square_proportional" else 1
                y0 = max(1e-9, abs(_resolve_var(w0, y_v)))
                x0_p = max(1e-9, abs(_resolve_var(w0, x_v) ** pwr))
                scale_ref = y0 / x0_p
                verify_sys.add_quantity(
                    CoupledQuantity("law_const_norm", 1.0, ScaleTier.MECHANISM, "", "state", PinMode.FREE, -1e6, 1e6)
                )
                for w_idx in range(len(worlds_obs)):
                    yn = f"{y_v}_w{w_idx}"
                    xn = f"{x_v}_w{w_idx}"
                    verify_sys.add_bond(
                        EqualityBond(
                            name=f"world_check_w{w_idx}",
                            tier=ScaleTier.MECHANISM,
                            variables=(yn, xn, "law_const_norm"),
                            lhs_fn=lambda v, yk=yn, yref=y0: v[yk] / yref,
                            rhs_fn=lambda v, xk=xn, pk=pwr, xref=x0_p: v["law_const_norm"] * ((v[xk] ** pk) / xref),
                            lhs_expr=f"{yn} / y0",
                            rhs_expr=f"c_norm * ({xn}^{pwr} / x0^{pwr})",
                        )
                    )
            else:
                # Explicit equation with free `law_const`
                scale_ref = 1.0
                init_l = float(claim.lhs_eval(w0)) if claim.lhs_eval else 1.0
                scale_ref = max(1e-6, abs(init_l))
                verify_sys.add_quantity(
                    CoupledQuantity("law_const", scale_ref, ScaleTier.MECHANISM, "", "state", PinMode.FREE, -1e9, 1e9)
                )
                for w_idx in range(len(worlds_obs)):
                    v_map = {vn: f"{vn}_w{w_idx}" for vn in set(claim.variables)}
                    var_tuple = tuple(v_map.values()) + ("law_const",)
                    lf = claim.lhs_eval
                    rf = claim.rhs_eval
                    verify_sys.add_bond(
                        EqualityBond(
                            name=f"world_check_w{w_idx}",
                            tier=ScaleTier.MECHANISM,
                            variables=var_tuple,
                            lhs_fn=lambda v, vm=v_map, f=lf, s=scale_ref: f({k: v[qn] for k, qn in vm.items()} | {"law_const": v["law_const"]}) / s,
                            rhs_fn=lambda v, vm=v_map, f=rf, s=scale_ref: f({k: v[qn] for k, qn in vm.items()} | {"law_const": v["law_const"]}) / s,
                            lhs_expr=claim.lhs_expr,
                            rhs_expr=claim.rhs_expr,
                        )
                    )
        else:
            # Pure explicit equation with numeric constant
            w0 = worlds_obs[0]
            l0 = abs(float(claim.lhs_eval(w0))) if claim.lhs_eval else 1.0
            r0 = abs(float(claim.rhs_eval(w0))) if claim.rhs_eval else 1.0
            scale_ref = max(1.0, l0, r0)
            for w_idx in range(len(worlds_obs)):
                v_map = {vn: f"{vn}_w{w_idx}" for vn in set(claim.variables)}
                lf = claim.lhs_eval
                rf = claim.rhs_eval
                verify_sys.add_bond(
                    EqualityBond(
                        name=f"world_check_w{w_idx}",
                        tier=ScaleTier.MECHANISM,
                        variables=tuple(v_map.values()),
                        lhs_fn=lambda v, vm=v_map, f=lf, s=scale_ref: f({k: v[qn] for k, qn in vm.items()}) / s,
                        rhs_fn=lambda v, vm=v_map, f=rf, s=scale_ref: f({k: v[qn] for k, qn in vm.items()}) / s,
                        lhs_expr=claim.lhs_expr,
                        rhs_expr=claim.rhs_expr,
                    )
                )

        rep = verify_sys.relax_least_action(active_tiers=[ScaleTier.MECHANISM], max_iter=150, tol=1e-11)
        action = float(rep.final_action)

        if "law_const_norm" in verify_sys.quantities:
            discovered_const = float(verify_sys.quantities["law_const_norm"].value) * scale_ref
        elif "law_const" in verify_sys.quantities:
            discovered_const = float(verify_sys.quantities["law_const"].value)

        if action < 1e-6:
            cand = self._build_internalizable_candidate(claim, discovered_const)
            if auto_internalize:
                organism.internalized_disciplines[cand.name] = cand
            organism.cognitive_growth_rings.append(
                {
                    "event": "verified_text_law",
                    "law_name": cand.name,
                    "expression": cand.expression,
                    "multi_world_action": action,
                    "discovered_constant": discovered_const,
                }
            )
            refl = "서로 다른 세계 경험에서 법칙과 상수가 흔들림 없이 재현되어 신경망에 새겼습니다."
            return TextSynthesisResult(
                raw_text=text,
                outcome=TextSynthesisOutcome.VERIFIED_PHYSICAL_LAW,
                claim=claim,
                experiment_skipped=False,
                multi_world_action=action,
                discovered_constant=discovered_const,
                internalized_law_name=cand.name,
                metaphor_reflection=None,
                hangul_reflection=refl,
                hangul_wave_coherence=self._encode_reflection(refl),
                explanation=f"[VERIFIED across {len(worlds_obs)} worlds | action={action:.2e}] {cand.expression}",
            )

        organism.cognitive_growth_rings.append(
            {
                "event": "falsified_text_claim",
                "raw_text": text,
                "multi_world_action": action,
            }
        )
        refl = "다중 현실 대조에서 일관된 법칙으로 수렴하지 않아 거짓 가설로 기각했습니다."
        return TextSynthesisResult(
            raw_text=text,
            outcome=TextSynthesisOutcome.EMPIRICALLY_FALSIFIED,
            claim=claim,
            experiment_skipped=False,
            multi_world_action=action,
            discovered_constant=None,
            internalized_law_name=None,
            metaphor_reflection=None,
            hangul_reflection=refl,
            hangul_wave_coherence=self._encode_reflection(refl),
            explanation=f"[FALSIFIED across {len(worlds_obs)} worlds | action={action:.4f}] {text}",
        )

    def ingest_document_text(
        self,
        document_text: str,
        organism: ExtendedCognitiveOrganism,
        calibration_worlds: Sequence[ExtendedWorldState],
        active_organs: Sequence[str] = ("eye", "ear", "hand"),
        use_spectrometer: bool = True,
        auto_internalize: bool = True,
    ) -> DocumentIngestionReport:
        lines = [ln.strip() for ln in document_text.splitlines() if ln.strip()]
        results: List[TextSynthesisResult] = []
        verified: List[str] = []
        dim_rej: List[str] = []
        falsified: List[str] = []
        metaphors: List[MetaphorReflection] = []
        unsafe: List[str] = []

        for line in lines:
            res = self.synthesize_and_verify(
                text=line,
                organism=organism,
                calibration_worlds=calibration_worlds,
                active_organs=active_organs,
                use_spectrometer=use_spectrometer,
                auto_internalize=auto_internalize,
            )
            results.append(res)
            if res.outcome == TextSynthesisOutcome.VERIFIED_PHYSICAL_LAW:
                verified.append(res.internalized_law_name or line)
            elif res.outcome == TextSynthesisOutcome.DIMENSIONAL_CONTRADICTION:
                dim_rej.append(line)
            elif res.outcome == TextSynthesisOutcome.EMPIRICALLY_FALSIFIED:
                falsified.append(line)
            elif res.outcome == TextSynthesisOutcome.ABSTRACT_METAPHOR_REFLECTION and res.metaphor_reflection is not None:
                metaphors.append(res.metaphor_reflection)
            else:
                unsafe.append(line)

        return DocumentIngestionReport(
            results=results,
            verified_laws=verified,
            dimensional_rejections=dim_rej,
            falsified_claims=falsified,
            metaphor_reflections=metaphors,
            unsafe_or_unparsed=unsafe,
        )


def sync_synthesis_to_causal_field(
    report: DocumentIngestionReport,
    causal_field: Optional["CausalField"] = None,
    prefix: str = "txt_law",
) -> Dict[str, Any]:
    """Projects verified prose laws, dimensional rejections, falsifications & metaphors into `CausalField`."""
    from core.physics.causal_field import CausalField, EngramAttractor, InformationVoxel

    field = causal_field if causal_field is not None else CausalField(dimensions=3)
    n_ver = len(report.verified_laws)
    n_dim = len(report.dimensional_rejections)
    n_fal = len(report.falsified_claims)
    n_met = len(report.metaphor_reflections)

    sky_curiosity = 1.0 if n_met > 0 else 0.0
    specs = [
        ("text_source", [float(len(report.results)) / 10.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0], 0.0),
        ("dimensional_mirror", [1.0, float(n_dim), 0.0, 0.5], [-1.0, 0.5, 0.5], 0.0),
        ("verified_laws_node", [float(n_ver), 1.0, 0.0, 1.0], [1.0, 0.5, 1.0], 0.0),
        ("falsified_node", [float(n_fal), float(n_dim), 0.0, 0.2], [0.0, -1.0, 0.5], 0.0),
        ("abstract_sky_metaphor", [float(n_met), sky_curiosity, 0.5, 1.0], [0.0, 1.2, 1.8], sky_curiosity),
    ]

    vids: Dict[str, str] = {}
    for key, vec4, pos3, cur in specs:
        vid = f"{prefix}_{key}"
        vids[key] = vid
        field.add_voxel(
            InformationVoxel(
                id=vid,
                content={"role": key},
                tensor=np.asarray(vec4, dtype=np.float32),
                mass=max(1.0, 1.0 + 0.5 * vec4[0]),
                position=np.asarray(pos3[: field.dimensions], dtype=np.float32),
                curiosity_charge=float(cur),
            )
        )

    field.link_voxels(vids["text_source"], vids["dimensional_mirror"], strength=0.95)
    field.link_voxels(vids["dimensional_mirror"], vids["verified_laws_node"], strength=0.95 if n_ver > 0 else 0.4)
    field.link_voxels(vids["dimensional_mirror"], vids["falsified_node"], strength=0.12 if (n_fal + n_dim) > 0 else 0.8)
    field.link_voxels(vids["text_source"], vids["abstract_sky_metaphor"], strength=0.90 if n_met > 0 else 0.3)

    field.register_engram(
        EngramAttractor(
            id=f"{prefix}_engram_text_mirror",
            name="TextCausalSynthesis",
            position=np.array([0.0, 1.2, 1.8][: field.dimensions], dtype=np.float32),
            intensity=2.0,
            tier="macro",
            symbolic_intent=f"verified={n_ver}, dim_rej={n_dim}, fal={n_fal}, meta={n_met}",
        )
    )

    return {
        "causal_field": field,
        "voxels": vids,
        "abstract_sky_curiosity": float(sky_curiosity),
    }
